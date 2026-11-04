"""Feed-forward camera evidence from MapAnything, as a COLMAP-compatible revision.

Alternative to incremental SfM for photo sets where COLMAP registers too few
views. MapAnything predicts pose, intrinsics and depth for every photo in one
pass. This module keeps only the largest group of views whose predicted 3D
points agree in depth with at least one other view, then writes the same
undistorted COLMAP-text revision that the dense (OpenMVS) and camera-plan
stages already consume.

MapAnything resizes and crops every view in one call to a shared aspect ratio,
so each photo is instead scaled to fit a square canvas and pasted without
cropping. Predictions are read back only inside the photo rectangle, and the
principal point is shifted by the paste offset, so the saved photo and the
COLMAP intrinsics describe the same pixels.

A kept view means "its prediction is mutually consistent with other views",
not "the camera is verified". Output stays a private candidate that needs
visual review, like every other photo-reconstruction revision.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cameras import MAPANYTHING_MAX_FOCAL_ASYMMETRY
from .colmap import _prepare_images, _sha256, _staged_images, _write_json
from .references import ReferenceError, ReferenceSet, audit_reference_set

MAPANYTHING_DEFAULT_MODEL = "facebook/map-anything-apache"
# Model cards: the -apache checkpoint is Apache-2.0; the other is CC-BY-NC-4.0.
MODEL_LICENSES = {
    "facebook/map-anything-apache": "Apache-2.0",
    "facebook/map-anything": "CC-BY-NC-4.0",
}
CANVAS = 518  # MapAnything's native resolution set; a multiple of its 14-pixel patch
MIN_REGISTERED_VIEWS = 3
MIN_COVISIBLE_FRACTION = 0.30  # share of a view's samples re-observed in another view
# Relative depth agreement required for re-observation. 5% left the Sphinx
# 26-photo set with 3 mutually consistent views, none of COLMAP's 6; 30% kept 21,
# 5 of which COLMAP also registers. Tune per photo set with --mapanything-depth-tolerance.
DEPTH_TOLERANCE = 0.30
MIN_OVERLAP_POINTS = 100
SAMPLES_PER_VIEW = 4000
MAX_SPARSE_POINTS = 50_000
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_ORTHONORMAL_TOLERANCE = 1e-3


def is_valid_model_id(model_id: object) -> bool:
    """A Hugging Face repo id such as facebook/map-anything-apache (no local paths)."""
    return isinstance(model_id, str) and _MODEL_ID.fullmatch(model_id) is not None


def intrinsics_are_usable(k: np.ndarray) -> bool:
    """Finite, positive focals within the per-axis asymmetry cap."""
    if k.shape != (3, 3) or not np.isfinite(k).all():
        return False
    fx, fy = float(k[0, 0]), float(k[1, 1])
    if fx <= 0 or fy <= 0:
        return False
    return abs(fx - fy) / max(fx, fy) <= MAPANYTHING_MAX_FOCAL_ASYMMETRY


def canvas_placement(width: int, height: int, canvas: int = CANVAS) -> tuple[int, int, int, int]:
    """Return (w, h, x0, y0): the photo scaled to fit ``canvas``, centred on it."""
    scale = canvas / max(width, height)
    w = max(1, min(canvas, round(width * scale)))
    h = max(1, min(canvas, round(height * scale)))
    return w, h, (canvas - w) // 2, (canvas - h) // 2


def quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to COLMAP (w, x, y, z) quaternion, w >= 0."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    quaternion = np.asarray(q, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return -quaternion if quaternion[0] < 0 else quaternion


def colmap_pose_from_cam2world(cam2world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV cam2world (MapAnything) to COLMAP world2cam (qvec, tvec)."""
    rotation = np.asarray(cam2world[:3, :3], dtype=np.float64)
    center = np.asarray(cam2world[:3, 3], dtype=np.float64)
    world_to_cam = rotation.T
    return quaternion_from_rotation(world_to_cam), -world_to_cam @ center


def pose_is_valid(cam2world: np.ndarray) -> bool:
    if cam2world.shape != (4, 4) or not np.isfinite(cam2world).all():
        return False
    rotation = cam2world[:3, :3]
    orthonormal = np.abs(rotation.T @ rotation - np.eye(3)).max() < _ORTHONORMAL_TOLERANCE
    return bool(orthonormal and np.linalg.det(rotation) > 0)


def covisibility_matrices(views: list[dict[str, Any]], *,
                          tolerance: float = DEPTH_TOLERANCE) -> tuple[np.ndarray, np.ndarray]:
    """Return (fraction, count) with [i, j] = how view i's samples land in view j.

    Each view dict has ``K`` (canvas intrinsics), ``depth`` (canvas z-depth, NaN
    where invalid), ``rect`` (x0, y0, w, h of the photo on the canvas),
    ``points`` (sampled world points) and ``world_to_cam`` (4x4).
    ``fraction[i, j]`` is the share of view i's samples that project inside
    view j's photo rectangle with depth agreeing with view j's prediction.
    """
    n = len(views)
    fraction = np.zeros((n, n), dtype=np.float64)
    count = np.zeros((n, n), dtype=np.int64)
    for i, source in enumerate(views):
        points = source["points"]
        if len(points) == 0:
            continue
        for j, target in enumerate(views):
            if i == j:
                continue
            pose = target["world_to_cam"]
            cam = points @ pose[:3, :3].T + pose[:3, 3]
            z = cam[:, 2]
            k = target["K"]
            x0, y0, w, h = target["rect"]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = k[0, 0] * cam[:, 0] / z + k[0, 2]
                v = k[1, 1] * cam[:, 1] / z + k[1, 2]
            inside = ((z > 1e-6) & np.isfinite(u) & np.isfinite(v)
                      & (u >= x0) & (u < x0 + w) & (v >= y0) & (v < y0 + h))
            total = int(inside.sum())
            if total == 0:
                continue
            rows = np.floor(v[inside]).astype(np.int64)
            cols = np.floor(u[inside]).astype(np.int64)
            predicted = target["depth"][rows, cols]
            agree = (np.isfinite(predicted) & (predicted > 0)
                     & (np.abs(z[inside] - predicted) <= tolerance * predicted))
            count[i, j] = total
            fraction[i, j] = float(agree.sum()) / total
    return fraction, count


def largest_component(adjacency: np.ndarray) -> list[int]:
    """Largest connected set; ties resolved toward the lowest view index."""
    seen: set[int] = set()
    best: list[int] = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        seen.add(start)
        stack, component = [start], []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in np.flatnonzero(adjacency[node]):
                neighbour = int(neighbour)
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        component.sort()
        if len(component) > len(best):
            best = component
    return best


def _sample_pixels(valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pixels = np.argwhere(valid)
    if len(pixels) > SAMPLES_PER_VIEW:
        pixels = pixels[rng.choice(len(pixels), SAMPLES_PER_VIEW, replace=False)]
    return pixels


def _write_colmap_text_model(directory: Path, cameras: list[str], images: list[str],
                             points: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    header = "# Generated from MapAnything feed-forward predictions (private candidate)\n"
    (directory / "cameras.txt").write_text(
        header + "".join(line + "\n" for line in cameras), encoding="utf-8")
    # COLMAP pairs every image line with a (here empty) 2D observation line.
    (directory / "images.txt").write_text(
        header + "".join(line + "\n\n" for line in images), encoding="utf-8")
    (directory / "points3D.txt").write_text(
        header + "".join(line + "\n" for line in points), encoding="utf-8")


def _run_worker(command: list[str], log: Path, timeout: int) -> int:
    try:
        with log.open("w", encoding="utf-8") as stream:
            return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False).returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nMapAnything worker did not finish: {error}\n")
        return 124 if isinstance(error, subprocess.TimeoutExpired) else 127


def build_feedforward_sparse_reconstruction(
    reference_set: ReferenceSet, output: str | Path, *,
    model_id: str = MAPANYTHING_DEFAULT_MODEL,
    device: str = "auto",
    timeout_seconds: int = 3600,
    depth_tolerance: float = DEPTH_TOLERANCE,
    runner: Callable[[list[str], Path, int], int] | None = None,
) -> dict[str, Any]:
    """Build a COLMAP-compatible candidate revision from MapAnything predictions.

    Failures are checkpointed in ``reconstruction-report.json`` using the same
    keys as the sparse builder, so the photo-model orchestrator treats both
    engines the same way.
    """
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Output already exists: {target}. Choose an unused revision directory")
    if not is_valid_model_id(model_id):
        raise ReferenceError("MapAnything model must be a Hugging Face repo id such as facebook/map-anything-apache")
    if device not in DEVICE_CHOICES:
        raise ReferenceError(f"MapAnything device must be one of {', '.join(DEVICE_CHOICES)}")
    if type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 86400:
        raise ReferenceError("MapAnything timeout_seconds must be 30–86400")
    if (type(depth_tolerance) not in (int, float) or not math.isfinite(depth_tolerance)
            or not 0.01 <= depth_tolerance <= 1.0):
        raise ReferenceError("MapAnything depth_tolerance must be in [0.01, 1.0]")
    if any(image.mask is not None for image in reference_set.images):
        raise ReferenceError("The MapAnything engine does not rectify object masks yet; remove masks or use --engine pycolmap")
    audit = audit_reference_set(reference_set)
    references = _staged_images(reference_set, audit)
    if len(references) < MIN_REGISTERED_VIEWS:
        raise ReferenceError(f"MapAnything needs at least {MIN_REGISTERED_VIEWS} independent photographs")

    target.mkdir(parents=True, exist_ok=False)
    (target / "logs").mkdir()
    report: dict[str, Any] = {
        "schema_version": 1, "engine": "mapanything",
        "asset_id": reference_set.asset_id,
        "manifest_sha256": reference_set.manifest_sha256,
        "source_images": len(reference_set.images),
        "staged_independent_images": len(references),
        "registered_images": 0, "undistorted_images": 0,
        "model_id": model_id,
        "model_license": MODEL_LICENSES.get(model_id, "unknown-review-required"),
        "canvas": CANVAS,
        "depth_tolerance": float(depth_tolerance),
        "undistortion_requested": True,
        "commands": [], "failed_stage": None, "exit_code": None,
        "status": "review-required", "historical_fidelity_verified": False,
        "distribution_allowed": False,
        "limitations": ("Feed-forward predictions are not verified camera poses. Registered means "
                        "mutually depth-consistent in MapAnything's output; visual review is required."),
    }

    def checkpoint() -> dict[str, Any]:
        _write_json(target / "reconstruction-report.json", report)
        return report

    def fail(stage: str, reason: str) -> dict[str, Any]:
        report.update(status="review-required", failed_stage=stage, error=reason)
        return checkpoint()

    checkpoint()
    _write_json(target / "reference-audit.json", audit)
    try:
        _prepare_images(references, target, normalize_for_colmap=True)
    except (ReferenceError, OSError, ValueError) as error:
        return fail("staging", str(error))
    staged = json.loads((target / "staged-reference-index.json").read_text(encoding="utf-8"))["images"]
    placements = [canvas_placement(*entry["dimensions"]) for entry in staged]

    request = {"schema_version": 1, "model_id": model_id, "device": device, "canvas": CANVAS,
               "images": [{"path": str(target / "images" / entry["filename"]),
                           "width": w, "height": h, "x0": x0, "y0": y0}
                          for entry, (w, h, x0, y0) in zip(staged, placements)]}
    input_path = target / "mapanything-input.json"
    _write_json(input_path, request)
    worker_dir = target / "mapanything"
    worker_dir.mkdir()
    predictions_path = worker_dir / "predictions.npz"
    meta_path = worker_dir / "worker-meta.json"
    command = [sys.executable, "-m", "kumiki_blender.reconstruction.mapanything_worker",
               "--input", str(input_path), "--output", str(predictions_path),
               "--meta", str(meta_path)]
    code = (runner or _run_worker)(command, target / "logs" / "mapanything.log", timeout_seconds)
    report["exit_code"] = code
    if code:
        return fail("mapanything-inference",
                    f"MapAnything worker exited with {code}; inspect logs/mapanything.log")
    if meta_path.is_file():
        report["worker"] = json.loads(meta_path.read_text(encoding="utf-8"))
    checkpoint()

    try:
        with np.load(predictions_path, allow_pickle=False) as data:
            predictions = []
            for index in range(len(staged)):
                # float32 halves the memory of 256-photo runs; poses stay float64.
                pose = np.asarray(data[f"pose_{index}"], dtype=np.float64)
                k = np.asarray(data[f"K_{index}"], dtype=np.float64)
                points = np.asarray(data[f"pts_{index}"], dtype=np.float32)
                depth = np.asarray(data[f"depth_{index}"], dtype=np.float32)
                mask = np.asarray(data[f"mask_{index}"], dtype=bool)
                if (k.shape != (3, 3) or points.shape != (CANVAS, CANVAS, 3)
                        or depth.shape != (CANVAS, CANVAS) or mask.shape != (CANVAS, CANVAS)):
                    return fail("mapanything-output",
                                f"Prediction {index} is not on the {CANVAS}x{CANVAS} canvas")
                predictions.append({"pose": pose, "K": k, "points": points, "depth": depth,
                                    "mask": mask})
    except (KeyError, ValueError, OSError) as error:
        return fail("mapanything-output", f"Cannot read predictions: {error}")

    # Restrict every prediction to the photo rectangle; padding is never evidence.
    usable = [index for index, item in enumerate(predictions)
              if pose_is_valid(item["pose"]) and intrinsics_are_usable(item["K"])]
    rng = np.random.default_rng(0)
    views: list[dict[str, Any]] = []
    for index in usable:
        item = predictions[index]
        w, h, x0, y0 = placements[index]
        rect_valid = np.zeros((CANVAS, CANVAS), dtype=bool)
        rect_valid[y0:y0 + h, x0:x0 + w] = True
        valid = item["mask"] & rect_valid & np.isfinite(item["points"]).all(axis=-1) \
            & np.isfinite(item["depth"]) & (item["depth"] > 0)
        pixels = _sample_pixels(valid, rng)
        depth = np.where(valid, item["depth"], np.nan)
        views.append({"K": item["K"], "depth": depth, "rect": (x0, y0, w, h), "pixels": pixels,
                      "points": item["points"][pixels[:, 0], pixels[:, 1]],
                      "world_to_cam": np.linalg.inv(item["pose"])})
    fraction, count = covisibility_matrices(views, tolerance=depth_tolerance)
    adjacency = ((np.minimum(fraction, fraction.T) >= MIN_COVISIBLE_FRACTION)
                 & (np.minimum(count, count.T) >= MIN_OVERLAP_POINTS))
    component = [usable[position] for position in largest_component(adjacency)]
    report["covisibility"] = {
        "min_covisible_fraction": MIN_COVISIBLE_FRACTION,
        "depth_tolerance": float(depth_tolerance),
        "min_overlap_points": MIN_OVERLAP_POINTS,
        "views_with_valid_pose": len(usable),
        "graph_edges": int(adjacency.sum() // 2),
        "largest_component": len(component),
    }
    registered = sorted(component)
    report["registered_images"] = len(registered)
    report["registered_filenames"] = [staged[index]["filename"] for index in registered]
    report["unregistered_filenames"] = sorted(
        entry["filename"] for index, entry in enumerate(staged) if index not in registered)
    if len(registered) < MIN_REGISTERED_VIEWS:
        return fail("insufficient_registered_views",
                    "Fewer than three mutually depth-consistent views; inspect predictions and photos")
    checkpoint()

    try:
        from PIL import Image

        rectified = target / "undistorted"
        image_root = rectified / "images"
        image_root.mkdir(parents=True)
        position = {index: slot for slot, index in enumerate(usable)}
        cameras, images, index_entries = [], [], []
        point_xyz, point_rgb = [], []
        asymmetry = 0.0
        for camera_id, index in enumerate(registered, start=1):
            entry = staged[index]
            w, h, x0, y0 = placements[index]
            with Image.open(target / "images" / entry["filename"]) as source:
                photo = source.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
            photo.save(image_root / entry["filename"], format="PNG")
            colours = np.asarray(photo)
            item = predictions[index]
            k = item["K"]
            focal_x, focal_y = float(k[0, 0]), float(k[1, 1])
            asymmetry = max(asymmetry, abs(focal_x - focal_y) / max(focal_x, focal_y))
            # Shift the principal point from canvas to photo coordinates.
            cx, cy = float(k[0, 2]) - x0, float(k[1, 2]) - y0
            # PINHOLE stores fx and fy separately; the plan and Blender readers accept
            # the asymmetry only for mapanything revisions, up to the stated cap.
            cameras.append(f"{camera_id} PINHOLE {w} {h} {focal_x:.9g} {focal_y:.9g} "
                           f"{cx:.9g} {cy:.9g}")
            quaternion, translation = colmap_pose_from_cam2world(item["pose"])
            images.append(f"{camera_id} " + " ".join(f"{value:.12g}" for value in quaternion)
                          + " " + " ".join(f"{value:.12g}" for value in translation)
                          + f" {camera_id} {entry['filename']}")
            index_entries.append({**entry, "dimensions": [w, h],
                                  "sha256": _sha256(image_root / entry["filename"]),
                                  "mask_requires_rectification": True})
            view = views[position[index]]
            pixels = view["pixels"]
            point_xyz.append(view["points"])
            point_rgb.append(colours[pixels[:, 0] - y0, pixels[:, 1] - x0])

        all_points = np.concatenate(point_xyz)
        all_colours = np.concatenate(point_rgb)
        if len(all_points) > MAX_SPARSE_POINTS:
            chosen = np.random.default_rng(1).choice(len(all_points), MAX_SPARSE_POINTS, replace=False)
            all_points, all_colours = all_points[chosen], all_colours[chosen]
        points = [f"{number} {x:.9g} {y:.9g} {z:.9g} {int(r)} {int(g)} {int(b)} 0"
                  for number, ((x, y, z), (r, g, b)) in enumerate(zip(all_points, all_colours), start=1)]

        _write_colmap_text_model(rectified / "sparse", cameras, images, points)
        _write_colmap_text_model(rectified / "text-model", cameras, images, points)
        _write_json(target / "undistorted-reference-index.json", {"images": index_entries})
    except (ReferenceError, OSError, ValueError, KeyError) as error:
        return fail("undistortion_validation", str(error))

    names = [entry["filename"] for entry in index_entries]
    report.update(
        status="candidate / visual-review-required",
        undistorted_images=len(index_entries),
        undistorted_model="undistorted/text-model",
        source_sparse_model="0",
        models=[{"model": "0", "registered_images": len(index_entries), "registered_filenames": names}],
        sparse_points=len(points),
        max_focal_asymmetry=round(asymmetry, 6),
        failed_stage=None, error=None,
    )
    return checkpoint()
