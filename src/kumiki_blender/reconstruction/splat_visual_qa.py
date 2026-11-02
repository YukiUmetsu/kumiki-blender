"""Calibrated, private multi-view photo comparisons of real GPU splat renders.

COLMAP camera poses remain paired with their *own* rectified photographs. This
does not infer a camera from a filename or assign a photo-faithfulness score.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .cameras import _quat_matrix, _read_intrinsics, _read_poses
from .dense import _check_regular, _sha256
from .references import ReferenceError
from .splat_video import _find_splat


@dataclass(frozen=True)
class SplatViewQASettings:
    max_views: int = 12
    max_dimension: int = 1280
    splat_transform: str = "splat-transform"
    timeout_seconds: int = 900
    min_valid_fraction: float = 0.70

    def validate(self) -> "SplatViewQASettings":
        if type(self.max_views) is not int or not 3 <= self.max_views <= 48:
            raise ReferenceError("Visual QA requires 3–48 views")
        if type(self.max_dimension) is not int or not 256 <= self.max_dimension <= 4096:
            raise ReferenceError("Visual QA maximum image dimension must be 256–4096")
        if (not isinstance(self.splat_transform, str) or
                not self.splat_transform.strip() or "\0" in self.splat_transform):
            raise ReferenceError("Invalid SplatTransform executable")
        if type(self.timeout_seconds) is not int or not 30 <= self.timeout_seconds <= 86_400:
            raise ReferenceError("Visual QA timeout must be 30–86,400 seconds")
        if (type(self.min_valid_fraction) not in (float, int) or
                not math.isfinite(self.min_valid_fraction) or
                not 0.3 <= self.min_valid_fraction <= 1):
            raise ReferenceError("Visual QA valid fraction must be 0.3–1")
        return self


def _camera_pose(quaternion: tuple[float, ...],
                 translation: tuple[float, ...]) -> tuple[list[float], list[float], list[float]]:
    """COLMAP world-to-camera (+Z forward, +Y down) -> PLY engine coordinates.

    SplatTransform's PLY import applies Transform.PLY: 180 degrees about Z.
    Apply that to camera position and orientation, not merely to the model.
    """
    r = _quat_matrix(quaternion)
    center = [-sum(r[row][axis] * translation[row] for row in range(3))
              for axis in range(3)]
    forward = list(r[2])
    up = [-value for value in r[1]]

    def flip(v: list[float]) -> list[float]:
        return [round(-v[0], 9), round(-v[1], 9), round(v[2], 9)]

    position = flip(center)
    direction = flip(forward)
    return position, [round(x+y, 9) for x, y in zip(position, direction)], flip(up)


def _select_spread(views: list[dict], limit: int) -> list[dict]:
    """Greedy deterministic farthest-pose sampling; not alphabetical-only picks."""
    if len(views) <= limit:
        return views
    centers = [view["camera_position"] for view in views]
    span = max(math.dist(a, b) for a in centers for b in centers)
    span = max(span, 1e-9)
    picked = [0]
    remaining = set(range(1, len(views)))
    while len(picked) < limit:
        def separation(i: int) -> float:
            # Positional baseline reveals unseen sides when all views focus on
            # the same target; direction protects non-orbit data collections.
            def pair(j: int) -> float:
                a, b = views[i], views[j]
                forward_a = [t-p for t, p in zip(a["camera_target"], a["camera_position"])]
                forward_b = [t-p for t, p in zip(b["camera_target"], b["camera_position"])]
                dot = sum(x*y for x, y in zip(forward_a, forward_b))
                dot /= max(1e-12, math.sqrt(sum(x*x for x in forward_a) *
                                             sum(y*y for y in forward_b)))
                return math.dist(centers[i], centers[j])/span + 0.5*(1-max(-1., min(1., dot)))
            return min(pair(j) for j in picked)
        next_index = max(sorted(remaining), key=lambda i: separation(i))
        picked.append(next_index)
        remaining.remove(next_index)
    return [views[index] for index in picked]


def _photo_plan(source: Path, settings: SplatViewQASettings) -> tuple[list[dict], list[str]]:
    report_file = source / "splat-report.json"
    _check_regular(report_file, source)
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
        dataset = report["dataset"]
        photo_hashes = {item["name"]: item["sha256"] for item in dataset["photo_hashes"]}
        camera_hashes = dataset["viewpoint_qa"]["text_model_hashes"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ReferenceError("Splat revision lacks pinned training photos and cameras") from error
    if not photo_hashes or len(photo_hashes) != len(dataset["photo_hashes"]):
        raise ReferenceError("Dataset photo filenames must be unique")
    dataset_dir = source / "dataset"
    _check_regular(dataset_dir / "dataset-report.json", source)
    for name in ("cameras.txt", "images.txt"):
        if name not in camera_hashes:
            raise ReferenceError(f"Missing registered camera hash: {name}")
    staged_cameras = dataset_dir / "cameras"
    if staged_cameras.is_symlink():
        raise ReferenceError("Staged camera directory cannot be a symlink")
    # Older revisions can use the source reconstruction, but only after
    # verifying the hashes recorded at Gaussian training time.
    if not staged_cameras.is_dir():
        reconstruction = Path(dataset.get("source_revision", ""))
        staged_cameras = reconstruction / "undistorted" / "text-model"
        if staged_cameras.is_symlink() or not staged_cameras.is_dir():
            raise ReferenceError("Registered camera text is unavailable; reconstruct or retrain with pinned camera text")
    else:
        if not staged_cameras.resolve().is_relative_to(source):
            raise ReferenceError("Staged camera directory must stay inside the splat revision")
    for name, expected in camera_hashes.items():
        camera_file = staged_cameras / name
        if camera_file.is_symlink() or not camera_file.is_file() or _sha256(camera_file) != expected:
            raise ReferenceError(f"Registered camera text changed since training: {name}")
    poses = _read_poses(staged_cameras / "images.txt")
    intrinsics = _read_intrinsics(staged_cameras / "cameras.txt", require_square_pixels=False)
    if len(poses) < 3 or set(item[2] for item in poses) != set(photo_hashes):
        raise ReferenceError("Pinned registered images do not match camera poses")

    views = []
    for image_id, camera_id, filename, quaternion, translation in sorted(poses, key=lambda p: p[2]):
        camera = intrinsics.get(camera_id)
        if camera is None:
            raise ReferenceError(f"Registered photo {filename} has no intrinsics")
        photo = dataset_dir / "images" / filename
        _check_regular(photo, dataset_dir)
        if _sha256(photo) != photo_hashes[filename]:
            raise ReferenceError(f"Pinned reference photo changed since training: {filename}")
        # Decode and inspect dimensions only in the worker; planning remains
        # fast and does not force image libraries or large RGBA allocations.
        position, target, up = _camera_pose(quaternion, translation)
        width, height = camera["width"], camera["height"]
        scale = min(1.0, settings.max_dimension / max(width, height))
        render_width = max(2, round(width * scale))
        render_height = max(2, round(height * scale))
        fx = camera["fx_px"] * render_width / width
        fy = camera["fy_px"] * render_height / height
        cx = camera["cx_px"] * render_width / width
        cy = camera["cy_px"] * render_height / height
        fov = math.degrees(2 * math.atan(render_height / (2 * fy)))
        if not 1 < fov < 179:
            raise ReferenceError(f"Unrenderable COLMAP focal length: {filename}")
        mask = dataset_dir / "masks" / (Path(filename).stem + ".png")
        if mask.is_symlink():
            raise ReferenceError(f"Staged photo mask cannot be a symlink: {filename}")
        if mask.exists():
            _check_regular(mask, dataset_dir)
        views.append({
            "image_id": image_id, "photo_name": filename,
            "photo_path": str(photo), "photo_sha256": photo_hashes[filename],
            "mask_path": str(mask) if mask.is_file() else None,
            "mask_sha256": _sha256(mask) if mask.is_file() else None,
            "resolution": [render_width, render_height],
            "source_resolution": [width, height],
            "camera_position": position, "camera_target": target, "camera_up": up,
            "vertical_fov_degrees": round(fov, 8),
            # Output pixel -> rendered pixel (centered principal point and
            # square-pixel renderer). Supports different fx/fy and cx/cy.
            "pixel_warp": [fy / fx, 0, render_width / 2 - (fy / fx)*cx,
                           0, 1, render_height / 2 - cy],
            "camera_model": camera["model"]
        })
    selected = _select_spread(views, settings.max_views)
    warnings = []
    if len(selected) < len(views):
        warnings.append(f"Selected {len(selected)} of {len(views)} registered views by pose spread; unselected angles need separate review.")
    if not any(view["mask_path"] for view in selected):
        warnings.append("No foreground masks: pixel errors include background, shadows and lighting differences.")
    if any(abs(v["pixel_warp"][0]-1) > 0.01 or abs(v["pixel_warp"][2]) > 2 or
           abs(v["pixel_warp"][5]) > 2 for v in selected):
        warnings.append("Some camera intrinsics need off-center/non-square-pixel warping; invalid borders will be excluded.")
    return selected, warnings


def plan_splat_view_qa(splat: str | Path, output: str | Path,
                       settings: SplatViewQASettings) -> dict:
    """Verify source immutability, renderer and paired COLMAP camera evidence."""
    settings.validate()
    source = Path(splat)
    if source.is_symlink() or not source.is_dir():
        raise ReferenceError("Visual QA requires a complete trained splat revision directory")
    source = source.resolve()
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists() or target.is_relative_to(source):
        raise ReferenceError("Visual QA output must be a new directory outside the splat revision")
    artifact = _find_splat(source)
    # Re-verify the recorded Gaussian checksum; _find_splat rejects a tampered
    # candidate even if its filename is still present.
    transform = shutil.which(settings.splat_transform)
    if transform is None:
        raise ReferenceError(f"SplatTransform GPU renderer is unavailable: {settings.splat_transform}")
    views, warnings = _photo_plan(source, settings)
    return {"schema_version": 1, "source_revision": str(source),
            "splat_file": str(artifact), "splat_sha256": _sha256(artifact),
            "output": str(target), "renderer": str(Path(transform).resolve()),
            "selected_views": views, "registered_views": len(views),
            "warnings": warnings, "distribution_allowed": False,
            "status": "preflight-only / GPU-visual-comparison-required",
            "limitation": "Registered cameras and image error diagnostics cannot certify object likeness or hidden geometry."}


def _compare_one(view: dict, rendered: Path, destination: Path,
                 min_valid_fraction: float) -> dict:
    """Resample only where the renderer's centered pinhole actually overlaps."""
    try:
        from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat
    except ImportError as error:
        raise ReferenceError("Visual comparison requires Pillow: pip install Pillow") from error
    dimensions = tuple(view["resolution"])
    source_photo = Path(view["photo_path"])
    if source_photo.is_symlink() or _sha256(source_photo) != view["photo_sha256"]:
        raise ReferenceError(f"Reference photo changed after preflight: {view['photo_name']}")
    try:
        with Image.open(source_photo) as image:
            if image.size != tuple(view["source_resolution"]):
                raise ReferenceError(f"COLMAP/photo dimensions differ: {view['photo_name']}")
            reference = image.convert("RGB").resize(dimensions, Image.Resampling.LANCZOS)
        with Image.open(rendered) as image:
            if image.format != "WEBP":
                raise ReferenceError(f"GPU renderer did not produce WebP for {view['photo_name']}")
            if image.size != dimensions:
                raise ReferenceError(f"GPU renderer emitted wrong dimensions for {view['photo_name']}")
            raster = image.convert("RGB")
    except (OSError, ValueError) as error:
        raise ReferenceError(f"Could not decode registered reference/render pair: {view['photo_name']}") from error
    affine = tuple(view["pixel_warp"])
    warped = raster.transform(dimensions, Image.Transform.AFFINE, affine,
                              resample=Image.Resampling.BICUBIC)
    valid = Image.new("L", dimensions, 255).transform(
        dimensions, Image.Transform.AFFINE, affine,
        resample=Image.Resampling.NEAREST, fillcolor=0)
    projection_fraction = ImageStat.Stat(valid).mean[0]/255
    if projection_fraction < min_valid_fraction:
        raise ReferenceError(f"Only {projection_fraction:.1%} projected pixels for {view['photo_name']}; check camera intrinsics")
    if view["mask_path"]:
        mask_path = Path(view["mask_path"])
        if mask_path.is_symlink() or _sha256(mask_path) != view["mask_sha256"]:
            raise ReferenceError(f"Foreground mask changed after preflight: {view['photo_name']}")
        with Image.open(mask_path) as image:
            if image.size != tuple(view["source_resolution"]):
                raise ReferenceError(f"Foreground mask dimensions differ: {view['photo_name']}")
            mask = image.convert("L").resize(dimensions, Image.Resampling.NEAREST)
        mask = mask.point(lambda pixel: 255 if pixel >= 128 else 0)
        valid = ImageChops.multiply(valid, mask)
    fraction = ImageStat.Stat(valid).mean[0]/255
    # A tight object foreground mask can cover a small fraction of the photo.
    # Do not reject a good mask for having lots of background; just reject
    # empty/negligible measurement regions.
    if fraction < 0.001 or fraction * dimensions[0] * dimensions[1] < 64:
        raise ReferenceError(f"Too few valid foreground pixels for {view['photo_name']}")
    difference = ImageChops.difference(reference, warped)
    stats = ImageStat.Stat(difference, valid)
    mean_absolute_error = sum(stats.mean)/3
    root_mean_square_error = math.sqrt(sum(channel*channel for channel in stats.rms)/3)
    psnr = (None if root_mean_square_error == 0 else
            20 * math.log10(255/root_mean_square_error))

    destination.mkdir()
    reference.save(destination / "reference.png")
    warped.save(destination / "registered-render.png")
    valid.save(destination / "valid-pixels.png")
    overlay = Image.blend(reference, warped, 0.5)
    overlay.save(destination / "overlay.png")
    heat = difference.convert("L").point(lambda pixel: min(255, pixel*3))
    ImageOps.colorize(heat, black="#130d17", white="#fa522a").save(destination / "difference.png")
    # One inspectable card: reference, matched render, 50/50 overlay, heatmap.
    thumb_size = (320, round(320 * dimensions[1] / dimensions[0]))
    card = Image.new("RGB", (320*4, thumb_size[1] + 32), (21, 22, 27))
    drawer = ImageDraw.Draw(card)
    for i, (label, panel) in enumerate((
            ("Reference", reference), ("Matched splat", warped),
            ("Overlay", overlay), ("Difference x3", ImageOps.colorize(heat, "#130d17", "#fa522a")))):
        card.paste(panel.resize(thumb_size, Image.Resampling.LANCZOS), (i*320, 32))
        drawer.text((i*320+8, 9), label, fill="white")
    card.save(destination / "comparison.jpg", quality=90)
    return {"photo_name": view["photo_name"], "image_id": view["image_id"],
            "valid_pixel_fraction": round(fraction, 5),
            "projection_pixel_fraction": round(projection_fraction, 5),
            "mean_absolute_error_rgb": round(mean_absolute_error, 5),
            "rmse_rgb": round(root_mean_square_error, 5),
            "psnr_db": round(psnr, 4) if psnr is not None else None,
            "mask_applied": bool(view["mask_path"]),
            "files": {name: str(destination.relative_to(destination.parent.parent.parent) / name) for name in (
                "reference.png", "registered-render.png", "valid-pixels.png",
                "overlay.png", "difference.png", "comparison.jpg")},
            "review_status": "human-visual-review-required"}


def render_splat_view_qa(splat: str | Path, output: str | Path,
                         settings: SplatViewQASettings) -> dict:
    """GPU render every selected registered pose, compare and archive evidence."""
    plan = plan_splat_view_qa(splat, output, settings)
    target = Path(plan["output"])
    target.mkdir(parents=True, exist_ok=False)
    (target / "views").mkdir()
    (target / "logs").mkdir()
    report = {**plan, "status": "in-progress / GPU-render", "completed_views": [],
              "failed_view": None, "failure": None}

    def checkpoint() -> dict:
        (target / "visual-qa-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    checkpoint()
    try:
        from PIL import Image
        cards = []
        for number, view in enumerate(plan["selected_views"], start=1):
            folder = target / "views" / f"{number:03d}"
            folder.mkdir()
            width, height = view["resolution"]
            raster = folder / "gpu-render.webp"
            center = view["camera_position"]
            aim = view["camera_target"]
            up = view["camera_up"]
            command = [plan["renderer"], plan["splat_file"], str(raster),
                       "--camera-pos", ",".join(map(str, center)),
                       "--camera-target", ",".join(map(str, aim)),
                       "--camera-up", ",".join(map(str, up)),
                       "--camera-fov", str(view["vertical_fov_degrees"]),
                       "--camera-near", "0.001",
                       "--resolution", f"{width}x{height}"]
            log_path = target / "logs" / f"{number:03d}.log"
            report["failed_view"] = view["photo_name"]
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=settings.timeout_seconds, check=False)
                    code = completed.returncode
                except subprocess.TimeoutExpired:
                    code = 124
                except OSError as error:
                    log.write(f"Failed to launch GPU renderer: {error}\n")
                    code = 127
            if code:
                raise ReferenceError(f"GPU rasterizer failed on {view['photo_name']} (exit {code}); see {log_path}")
            if raster.is_symlink() or not raster.is_file() or raster.stat().st_size < 16:
                raise ReferenceError(f"GPU rasterizer emitted no valid WebP: {view['photo_name']}")
            if _sha256(Path(plan["splat_file"])) != plan["splat_sha256"]:
                raise ReferenceError("Gaussian splat artifact changed during visual QA")
            result = _compare_one(view, raster, folder / "comparison",
                                  settings.min_valid_fraction)
            result.update(index=number, camera_position=center, camera_target=aim,
                          camera_up=up, vertical_fov_degrees=view["vertical_fov_degrees"],
                          camera_model=view["camera_model"], source_resolution=view["source_resolution"],
                          raster_sha256=_sha256(raster), log=str(log_path.relative_to(target)))
            report["completed_views"].append(result)
            with Image.open(folder / "comparison" / "comparison.jpg") as card:
                cards.append(card.convert("RGB").copy())
            checkpoint()
        if not cards:
            raise ReferenceError("No rendered registered view comparisons were produced")
        columns = 1 if len(cards) < 2 else 2
        rows = (len(cards) + columns - 1) // columns
        montage = Image.new("RGB", (cards[0].width*columns, cards[0].height*rows), "#15161b")
        for index, card in enumerate(cards):
            montage.paste(card, ((index % columns)*card.width, (index // columns)*card.height))
        montage.save(target / "multi-angle-contact-sheet.jpg", quality=90)
        sorted_views = sorted(report["completed_views"],
                              key=lambda entry: entry["mean_absolute_error_rgb"], reverse=True)
        report.update(status="candidate / human-multi-angle-visual-review-required",
                      failed_view=None,
                      worst_views=[item["photo_name"] for item in sorted_views[:min(3, len(sorted_views))]],
                      contact_sheet="multi-angle-contact-sheet.jpg",
                      comparison_count=len(sorted_views),
                      limitation="Pixel metrics are exposure/background-sensitive and may overfit training photos; inspect silhouettes, profiles and held-out views manually.")
    except (ReferenceError, ImportError, OSError, ValueError) as error:
        report.update(status="failed / visual-review-required", failure=str(error))
    return checkpoint()
