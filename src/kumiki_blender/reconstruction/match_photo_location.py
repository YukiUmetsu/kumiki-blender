"""Locate new photographs against a refined camera model.

Adding a photo through full reconstruction solves all cameras again and changes
rectified pixels, which invalidates reviewed masks. This module keeps the model.
A dense matcher connects a new photo to registered anchors, and the existing 3D
tracks of those anchors give 2D-3D correspondences. A robust pose solve with an
unknown focal length then places the photo.

Correspondences from different anchors are independent evidence, so each pose
is tested on anchors that the solve did not use. The poses stay candidates, and
`calibrate_on_registered_cameras` measures the true error of this method on
cameras with a known pose.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

from .match_camera_refinement import _camera_table
from .references import ReferenceError


SCHEMA = "match-located-photos-v1"
REPORT = "photo-location-report.json"


@dataclass(frozen=True)
class LocationSettings:
    """Limits for a photo location. Pixel limits use a 2000 px long side."""

    anchor_stride: int = 2
    anchors_used: int = 10
    min_certainty: float = .4
    max_pose_error_px: float = 12.0
    min_inliers: int = 40
    min_inlier_anchors: int = 3
    max_holdout_error_px: float = 10.0
    max_focal_ratio: float = 60.0
    min_focal_ratio: float = .3

    def __post_init__(self) -> None:
        for name in ("anchor_stride", "anchors_used", "min_inliers", "min_inlier_anchors"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ReferenceError(f"Location setting {name} must be a positive integer")
        if not 0 < self.min_focal_ratio < self.max_focal_ratio:
            raise ReferenceError("Location focal ratio limits must be positive and ordered")


def sample_warp(warp: Any, certainty: Any, pixels: Any, anchor_size: tuple[int, int],
                target_size: tuple[int, int]) -> tuple[Any, Any]:
    """Read target pixels and certainty at anchor pixels from a dense warp."""
    import numpy as np

    rows, columns = warp.shape[:2]
    column = np.clip((pixels[:, 0] / anchor_size[0] * columns).astype(int), 0, columns - 1)
    row = np.clip((pixels[:, 1] / anchor_size[1] * rows).astype(int), 0, rows - 1)
    sampled = warp[row, column]
    target = np.c_[(sampled[:, 2] + 1) / 2 * target_size[0], (sampled[:, 3] + 1) / 2 * target_size[1]]
    return target, certainty[row, column]


def check_location(result: dict[str, Any], settings: LocationSettings) -> list[str]:
    """List the reasons that a solved location cannot be used."""
    reasons = []
    if result["inliers"] < settings.min_inliers:
        reasons.append("too-few-inliers")
    if result["inlier_anchors"] < settings.min_inlier_anchors:
        reasons.append("too-few-anchors-agree")
    if not settings.min_focal_ratio <= result["focal_ratio"] <= settings.max_focal_ratio:
        reasons.append("implausible-focal-length")
    if result["holdout_error_px"] is None or result["holdout_error_px"] > settings.max_holdout_error_px:
        reasons.append("held-out-anchors-disagree")
    return reasons


def fitted_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    """Return a photo size within the rectified long-side limit of a revision."""
    scale = min(1.0, max_side / max(size))
    return max(1, round(size[0] * scale)), max(1, round(size[1] * scale))


def solve_pose(pixels: Any, points: Any, size: tuple[int, int], limit_px: float) -> dict[str, Any] | None:
    """Solve a pose and a focal length, and return None if no pose fits."""
    import numpy as np
    import pycolmap

    camera = pycolmap.Camera(model="SIMPLE_PINHOLE", width=size[0], height=size[1],
                             params=[1.2 * max(size), size[0] / 2, size[1] / 2])
    options = pycolmap.AbsolutePoseEstimationOptions()
    options.estimate_focal_length = True
    options.ransac.max_error = limit_px
    refinement = pycolmap.AbsolutePoseRefinementOptions()
    refinement.refine_focal_length = True
    answer = pycolmap.estimate_and_refine_absolute_pose(pixels, points, camera, options, refinement)
    if not answer:
        return None
    pose = answer["cam_from_world"]
    return {"R": pose.rotation.matrix(), "t": np.asarray(pose.translation), "camera": camera,
            "inlier_mask": np.asarray(answer["inlier_mask"]).astype(bool),
            "K": camera.calibration_matrix()}


def reprojection_errors(solution: dict[str, Any], pixels: Any, points: Any) -> Any:
    import numpy as np

    projected = (solution["K"] @ (solution["R"] @ points.T + solution["t"][:, None])).T
    errors = np.linalg.norm(projected[:, :2] / np.maximum(projected[:, 2:], 1e-9) - pixels, axis=1)
    errors[projected[:, 2] <= 0] = np.inf
    return errors


def anchor_observations(model: Any, excluded: str | None) -> dict[str, tuple[Any, Any, Any]]:
    """Return tracked pixels, 3D points and point IDs per registered camera."""
    import numpy as np

    excluded_id = None
    if excluded is not None:
        excluded_id = model.find_image_with_name(excluded).image_id
    observations = {}
    for image in model.images.values():
        if not image.has_pose or image.name == excluded:
            continue
        pixels, points, ids = [], [], []
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            point = model.points3D[point2D.point3D_id]
            # Hold a known camera out: its own observations must not support it.
            others = [e for e in point.track.elements if e.image_id != excluded_id]
            if len(others) >= 2:
                pixels.append(point2D.xy)
                points.append(point.xyz)
                ids.append(point2D.point3D_id)
        if pixels:
            observations[image.name] = (np.array(pixels), np.array(points), np.array(ids))
    return observations


def locate_photo(predictor: Any, photo: Path, size: tuple[int, int], images: Path,
                 observations: dict[str, tuple[Any, Any, Any]], anchor_sizes: dict[str, tuple[int, int]],
                 settings: LocationSettings) -> dict[str, Any] | None:
    """Locate one photo, and test the pose on anchors that the solve did not use."""
    import numpy as np

    scale = max(size) / 2000
    candidates = []
    for name in sorted(observations)[::settings.anchor_stride]:
        pixels, points, ids = observations[name]
        warp, certainty = predictor.dense_warp(images / name, photo)
        target, score = sample_warp(warp, certainty, pixels, anchor_sizes[name], size)
        keep = (score >= settings.min_certainty) & (target >= 0).all(axis=1) & (
            target[:, 0] < size[0]) & (target[:, 1] < size[1])
        if keep.sum() >= 6:
            candidates.append((int(keep.sum()), name, target[keep], points[keep], ids[keep]))
    candidates = sorted(candidates, key=lambda row: (-row[0], row[1]))[:settings.anchors_used]
    if len(candidates) < settings.min_inlier_anchors:
        return None
    pixels = np.concatenate([row[2] for row in candidates])
    points = np.concatenate([row[3] for row in candidates])
    point_ids = np.concatenate([row[4] for row in candidates])
    labels = np.concatenate([[index] * len(row[2]) for index, row in enumerate(candidates)])
    solution = solve_pose(pixels, points, size, settings.max_pose_error_px * scale)
    if solution is None:
        return None
    inliers = solution["inlier_mask"]
    # Solve on half the anchors and measure on the other half.
    used = np.arange(len(candidates)) % 2 == 0
    train = used[labels]
    holdout_error = None
    partial = solve_pose(pixels[train], points[train], size, settings.max_pose_error_px * scale)
    if partial is not None and (~train).any():
        holdout_error = round(float(np.median(
            reprojection_errors(partial, pixels[~train], points[~train])) / scale), 2)
    focal = float(solution["camera"].params[0])
    return {**solution, "inliers": int(inliers.sum()), "correspondences": len(pixels),
            "inlier_pixels": pixels[inliers], "inlier_point_ids": point_ids[inliers],
            "inlier_anchors": len(set(labels[inliers].tolist())),
            "anchors": [row[1] for row in candidates], "focal_ratio": round(focal / max(size), 3),
            "holdout_error_px": holdout_error}


def unique_track_observations(pixels: Any, point_ids: Any) -> list[tuple[Any, int]]:
    """Keep one observation per 3D point, because a photo sees a point once."""
    seen: dict[int, Any] = {}
    for xy, point_id in zip(pixels, point_ids.tolist()):
        seen.setdefault(point_id, xy)
    return [(xy, point_id) for point_id, xy in seen.items()]


def pose_difference(solution: dict[str, Any], known: dict[str, Any]) -> tuple[float, float]:
    """Return the rotation error in degrees and the camera center error."""
    import numpy as np

    relative = solution["R"].T @ known["R"]
    angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    center = -solution["R"].T @ solution["t"]
    return float(angle), float(np.linalg.norm(center - known["center"]))


def calibrate_on_registered_cameras(
        revision: str | Path, names: list[str], *, settings: LocationSettings | None = None,
        device: str = "auto", predictor: Any = None,
        log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Hide cameras with a known pose, locate them, and report the true pose error."""
    import numpy as np

    if predictor is None and find_spec("torch") is not None:
        import torch  # noqa: F401
    import pycolmap

    settings = settings or LocationSettings()
    source = Path(revision).resolve() / "undistorted"
    model = pycolmap.Reconstruction(str(source / "sparse"))
    cameras = _camera_table(model)
    spread = float(np.median(np.linalg.norm(np.array([c["center"] for c in cameras.values()]) -
                                            np.median([c["center"] for c in cameras.values()], axis=0),
                                            axis=1)))
    if predictor is None:
        from .roma_predictions import RoMaPredictor
        predictor = RoMaPredictor(device=device, coarse_resolution=560, refine_resolution=864)
    sizes = {name: (camera["width"], camera["height"]) for name, camera in cameras.items()}
    rows = []
    for name in names:
        solution = locate_photo(predictor, source / "images" / name, sizes[name], source / "images",
                                anchor_observations(model, name), sizes, settings)
        row: dict[str, Any] = {"filename": name, "located": solution is not None}
        if solution is not None:
            angle, distance = pose_difference(solution, cameras[name])
            row.update(rotation_error_degrees=round(angle, 3),
                       center_error_relative_to_spread=round(distance / spread, 4),
                       focal_ratio=solution["focal_ratio"],
                       known_focal_ratio=round(float(cameras[name]["K"][0, 0]) / max(sizes[name]), 3),
                       inliers=solution["inliers"], inlier_anchors=solution["inlier_anchors"],
                       holdout_error_px=solution["holdout_error_px"],
                       reasons=check_location(solution, settings))
        rows.append(row)
        if log:
            log(json.dumps(row))
    return {"schema": SCHEMA, "calibration": rows, "settings": asdict(settings)}


def add_located_photos(
        revision: str | Path, photos: list[dict[str, Any]], output: str | Path, *,
        settings: LocationSettings | None = None, device: str = "auto", predictor: Any = None,
        log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Write a camera revision with each new photo that passes all location checks.

    Each photo entry has `path` and `filename`, plus optional source and licence
    fields, which are copied to the report.
    """
    if predictor is None and find_spec("torch") is not None:
        import torch  # noqa: F401
    import numpy as np
    import pycolmap
    from PIL import Image

    from .dense import _sha256
    from .registered_camera_recovery import _read_json

    settings = settings or LocationSettings()
    source = Path(revision).resolve()
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists() or target.is_relative_to(source):
        raise ReferenceError("Photo location needs an unused output outside the camera revision")
    rectified = source / "undistorted"
    model = pycolmap.Reconstruction(str(rectified / "sparse"))
    cameras = _camera_table(model)
    if set(item["filename"] for item in photos) & set(image.name for image in model.images.values()):
        raise ReferenceError("New photo file names must differ from the photos of this revision")
    sizes = {name: (camera["width"], camera["height"]) for name, camera in cameras.items()}
    observations = anchor_observations(model, None)
    if predictor is None:
        from .roma_predictions import RoMaPredictor
        predictor = RoMaPredictor(device=device, coarse_resolution=560, refine_resolution=864)
    target.mkdir(parents=True)
    (target / "new-photos").mkdir()
    results, accepted = [], []
    max_side = _read_json(source / "reconstruction-report.json").get("max_image_size", 4000)
    for item in photos:
        path = target / "new-photos" / item["filename"]
        with Image.open(item["path"]) as photo:
            # Keep new photos in the size range of the other rectified photos.
            size = fitted_size(photo.size, max_side)
            photo.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(path, format="PNG")
        solution = locate_photo(predictor, path, size, rectified / "images", observations, sizes, settings)
        row = {key: value for key, value in item.items() if key != "path"}
        row["located"] = solution is not None
        if solution is not None:
            row.update({key: solution[key] for key in ("inliers", "correspondences", "inlier_anchors",
                                                       "anchors", "focal_ratio", "holdout_error_px")},
                       reasons=check_location(solution, settings))
            if not row["reasons"]:
                accepted.append((item, path, size, solution))
        results.append(row)
        if log:
            log(json.dumps({key: row.get(key) for key in ("filename", "located", "inliers",
                                                          "inlier_anchors", "focal_ratio",
                                                          "holdout_error_px", "reasons")}))
    report = {"schema": SCHEMA, "status": "candidate / located-photos-need-visual-review",
              "source_revision": str(source), "settings": asdict(settings), "photos": results,
              "accepted": [item["filename"] for item, _, _, _ in accepted],
              "approved_for_reconstruction": False, "distribution_allowed": False}
    if not accepted:
        report["status"] = "no-located-photos"
        (target / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    used_camera_ids = max(model.cameras) + 1
    used_image_ids = max(model.images) + 1
    index = json.loads((source / "undistorted-reference-index.json").read_text(encoding="utf-8"))
    (target / "undistorted").mkdir()
    shutil.copytree(rectified / "images", target / "undistorted/images")
    for offset, (item, path, size, solution) in enumerate(accepted):
        camera = solution["camera"]
        camera.camera_id = used_camera_ids + offset
        model.add_camera_with_trivial_rig(camera)
        image = pycolmap.Image(name=item["filename"], camera_id=camera.camera_id,
                               image_id=used_image_ids + offset,
                               points2D=pycolmap.Point2DList())
        model.add_image_with_trivial_frame(image, pycolmap.Rigid3d(
            pycolmap.Rotation3d(solution["R"]), solution["t"]))
        # Depth stereo chooses neighbors from shared tracks, so a camera without
        # tracks gets no depth map. Join the tracks that the pose solve confirmed.
        registered = model.images[image.image_id]
        for xy, point_id in unique_track_observations(solution["inlier_pixels"],
                                                      solution["inlier_point_ids"]):
            registered.points2D.append(pycolmap.Point2D(np.asarray(xy, dtype=float)))
            model.add_observation(point_id, pycolmap.TrackElement(
                image.image_id, registered.num_points2D() - 1))
        shutil.copyfile(path, target / "undistorted/images" / item["filename"])
        index["images"].append({"filename": item["filename"], "sha256": _sha256(path),
                                "dimensions": list(size), "mask_requires_rectification": True,
                                "source_url": item.get("source_page_url"),
                                "license": item.get("licence"), "usage": "reference-only"})
    (target / "undistorted/sparse").mkdir()
    (target / "undistorted/text-model").mkdir()
    model.write(str(target / "undistorted/sparse"))
    model.write_text(str(target / "undistorted/text-model"))
    (target / "undistorted-reference-index.json").write_text(json.dumps(index, indent=2) + "\n",
                                                             encoding="utf-8")
    shutil.copyfile(source / "staged-reference-index.json", target / "staged-reference-index.json")
    source_report = _read_json(source / "reconstruction-report.json")
    names = sorted(item["filename"] for item in index["images"])
    (target / "reconstruction-report.json").write_text(json.dumps({
        **source_report, "registered_images": len(names), "registered_filenames": names,
        "undistorted_images": len(names), "located_photos": report["accepted"],
        "source_images": source_report.get("source_images", 0) + len(accepted),
        "staged_independent_images": source_report.get("staged_independent_images", 0) + len(accepted),
        "status": "candidate / visual-review-required"}, indent=2) + "\n", encoding="utf-8")
    (target / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
