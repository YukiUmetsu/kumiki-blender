"""Refine registered cameras with dense matches across the whole image.

Sparse features cluster on textured surfaces, so a registered camera can fit
those areas well and still be wrong elsewhere. Dense stereo then fails on the
other surfaces. This module matches registered photo pairs densely, verifies
each pair with a fundamental matrix that does not use the current cameras, and
runs a robust bundle adjustment that includes the new matches.

The refinement uses rectified photos and keeps their pixels, so masks for these
photos stay valid. A share of the matches is held out of the refinement. The
report compares the cameras on those matches, and the revision is written only
if they improve. Matches come from a learned model, so the cameras that use
them stay candidates that need review.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, replace
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

from .references import ReferenceError


SCHEMA = "match-refined-cameras-v1"
REPORT = "match-refinement-report.json"
STAGED_STATUS = "candidate / match-refined-cameras-need-visual-review"
MAX_ALIGNMENT_ANCHORS = 64  # The limit of cameras.fit_similarity.


@dataclass(frozen=True)
class RefinementSettings:
    """Limits for matching and refinement. Pixel limits use a 2000 px long side."""

    pairs_per_camera: int = 6
    coarse_resolution: int = 560
    refine_resolution: int = 864
    lattice_step: int = 12
    min_certainty: float = .4
    max_verified_epipolar_px: float = 1.5
    min_verified_matches: int = 60
    holdout_every: int = 5
    track_cell_px: float = 4.0
    reprojection_schedule_px: tuple[float, ...] = (48.0, 24.0, 12.0, 6.0, 4.0)
    # Off by default. On a long lens a principal point shift looks like a small
    # rotation, so it moved outside the image on 3 of 54 Sphinx cameras.
    refine_principal_point: bool = False
    min_track_length: int = 2
    max_focal_change: float = 3.0

    def __post_init__(self) -> None:
        for name in ("pairs_per_camera", "coarse_resolution", "refine_resolution", "lattice_step",
                     "min_verified_matches", "min_track_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ReferenceError(f"Refinement setting {name} must be a positive integer")
        if type(self.holdout_every) is not int or self.holdout_every < 2:
            raise ReferenceError("Refinement setting holdout_every must be 2 or more; "
                                 "the comparison needs held-out matches")
        if not 0 < self.min_certainty < 1:
            raise ReferenceError("Refinement setting min_certainty must be between 0 and 1")
        if not self.reprojection_schedule_px or min(self.reprojection_schedule_px) <= 0:
            raise ReferenceError("Refinement setting reprojection_schedule_px needs positive limits")


def select_pairs(shared_tracks: dict[tuple[str, str], int], per_camera: int) -> list[tuple[str, str]]:
    """Keep each camera's strongest partners, so a weakly connected camera keeps its pairs."""
    partners: dict[str, list[tuple[int, str]]] = {}
    for (first, second), count in shared_tracks.items():
        partners.setdefault(first, []).append((count, second))
        partners.setdefault(second, []).append((count, first))
    selected = set()
    for name, candidates in partners.items():
        for _, other in sorted(candidates, key=lambda item: (-item[0], item[1]))[:per_camera]:
            selected.add(tuple(sorted((name, other))))
    return sorted(selected)


def lattice_cells(rows: Any, columns: Any, holdout_every: int) -> Any:
    """Mark held-out lattice cells. The choice is fixed, so reruns hold out the same matches."""
    return (rows + columns) % holdout_every == 0


def epipolar_distance(fundamental: Any, first: Any, second: Any) -> Any:
    """Return the pixel distance of each second point from its epipolar line."""
    import numpy as np

    lines = (fundamental @ np.c_[first, np.ones(len(first))].T).T
    values = np.abs((lines * np.c_[second, np.ones(len(second))]).sum(axis=1))
    return values / np.maximum(np.linalg.norm(lines[:, :2], axis=1), 1e-12)


def fundamental_from_cameras(first: dict[str, Any], second: dict[str, Any]) -> Any:
    """Compute the fundamental matrix that the current cameras imply."""
    import numpy as np

    rotation = second["R"] @ first["R"].T
    translation = second["t"] - rotation @ first["t"]
    x, y, z = translation
    essential = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float) @ rotation
    return np.linalg.inv(second["K"]).T @ essential @ np.linalg.inv(first["K"])


def merge_tracks(pair_matches: list[tuple[str, str, Any, Any]], cell_px: float
                 ) -> list[list[tuple[str, float, float]]]:
    """Join matches that share an image cell into tracks.

    A track with two different cells in one image is ambiguous, so it is dropped.
    """
    import numpy as np

    parent: dict[tuple[str, int, int], tuple[str, int, int]] = {}
    positions: dict[tuple[str, int, int], tuple[float, float]] = {}

    def find(node: tuple[str, int, int]) -> tuple[str, int, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first_name, second_name, first_points, second_points in pair_matches:
        first_cells = np.floor(first_points / cell_px).astype(int).tolist()
        second_cells = np.floor(second_points / cell_px).astype(int).tolist()
        for cell_a, cell_b, xy_a, xy_b in zip(first_cells, second_cells,
                                              first_points.tolist(), second_points.tolist()):
            node_a, node_b = (first_name, *cell_a), (second_name, *cell_b)
            positions.setdefault(node_a, tuple(xy_a))
            positions.setdefault(node_b, tuple(xy_b))
            root_a, root_b = find(node_a), find(node_b)
            if root_a != root_b:
                parent[root_a] = root_b
    tracks: dict[tuple[str, int, int], list[tuple[str, int, int]]] = {}
    for node in parent:
        tracks.setdefault(find(node), []).append(node)
    merged = []
    for nodes in tracks.values():
        names = [node[0] for node in nodes]
        if len(names) >= 2 and len(set(names)) == len(names):
            merged.append([(node[0], *positions[node]) for node in nodes])
    return merged


def triangulate_rays(centers: Any, directions: Any) -> Any:
    """Return the point nearest to all rays, or None if the rays are almost parallel."""
    import numpy as np

    system = np.zeros((3, 3))
    target = np.zeros(3)
    for center, direction in zip(centers, directions):
        projector = np.eye(3) - np.outer(direction, direction)
        system += projector
        target += projector @ center
    if np.linalg.cond(system) > 1e8:
        return None
    return np.linalg.solve(system, target)


def use_square_pixels(model: Any) -> None:
    """Give each camera one focal length, so refinement cannot make pixels non-square."""
    import numpy as np
    import pycolmap

    for camera in model.cameras.values():
        if camera.model.name == "PINHOLE":
            fx, fy, cx, cy = camera.params
            camera.model = pycolmap.CameraModelId.SIMPLE_PINHOLE
            camera.params = np.array([(fx + fy) / 2, cx, cy])


def _camera_table(model: Any) -> dict[str, dict[str, Any]]:
    import numpy as np

    table = {}
    for image in model.images.values():
        if not image.has_pose:
            continue
        camera = model.cameras[image.camera_id]
        pose = image.cam_from_world()
        rotation, translation = pose.rotation.matrix(), np.asarray(pose.translation)
        table[image.name] = {
            "image_id": image.image_id, "K": camera.calibration_matrix(), "R": rotation,
            "t": translation, "center": -rotation.T @ translation,
            "width": camera.width, "height": camera.height}
    return table


def _shared_tracks(model: Any) -> dict[tuple[str, str], int]:
    names = {image.image_id: image.name for image in model.images.values() if image.has_pose}
    shared: dict[tuple[str, str], int] = {}
    for point in model.points3D.values():
        members = sorted({names[element.image_id] for element in point.track.elements
                          if element.image_id in names})
        for index, first in enumerate(members):
            for second in members[index + 1:]:
                shared[(first, second)] = shared.get((first, second), 0) + 1
    return shared


def _load_mask(masks: Path, name: str) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(masks / (name + ".mask.png")) as mask:
        return np.asarray(mask.convert("L")) >= 128


def match_pair(predictor: Any, images: Path, masks: Path, first: str, second: str,
               settings: RefinementSettings, sizes: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """Sample lattice matches on the subject and verify them without the current cameras."""
    import cv2
    import numpy as np

    width_a, height_a = sizes[first]
    width_b, height_b = sizes[second]
    warp, certainty = predictor.dense_warp(images / first, images / second)
    step = settings.lattice_step
    rows, columns = np.meshgrid(np.arange(step // 2, warp.shape[0], step),
                                np.arange(step // 2, warp.shape[1], step), indexing="ij")
    rows, columns = rows.ravel(), columns.ravel()
    sampled = warp[rows, columns]
    points_a = np.c_[(sampled[:, 0] + 1) / 2 * width_a, (sampled[:, 1] + 1) / 2 * height_a]
    points_b = np.c_[(sampled[:, 2] + 1) / 2 * width_b, (sampled[:, 3] + 1) / 2 * height_b]
    keep = (certainty[rows, columns] >= settings.min_certainty) & (
        (points_b >= 0).all(axis=1) & (points_b[:, 0] < width_b) & (points_b[:, 1] < height_b))
    mask_a, mask_b = _load_mask(masks, first), _load_mask(masks, second)
    keep[keep] &= mask_a[np.clip(points_a[keep, 1].astype(int), 0, height_a - 1),
                         np.clip(points_a[keep, 0].astype(int), 0, width_a - 1)]
    keep[keep] &= mask_b[points_b[keep, 1].astype(int), points_b[keep, 0].astype(int)]
    empty = np.zeros((0, 2))
    result = {"first": first, "second": second, "train_a": empty, "train_b": empty,
              "holdout_a": empty, "holdout_b": empty, "candidates": int(keep.sum())}
    if keep.sum() < max(8, settings.min_verified_matches):
        return result
    points_a, points_b, rows, columns = points_a[keep], points_b[keep], rows[keep], columns[keep]
    scale = max(width_a, height_a, width_b, height_b) / 2000
    fundamental, _ = cv2.findFundamentalMat(
        points_a, points_b, cv2.USAC_MAGSAC, settings.max_verified_epipolar_px * scale, .9999, 20000)
    if fundamental is None or fundamental.shape != (3, 3):
        return result
    verified = epipolar_distance(fundamental, points_a, points_b) <= (
        settings.max_verified_epipolar_px * scale)
    if verified.sum() < settings.min_verified_matches:
        return result
    held = lattice_cells(rows // step, columns // step, settings.holdout_every)
    result.update(train_a=points_a[verified & ~held], train_b=points_b[verified & ~held],
                  holdout_a=points_a[verified & held], holdout_b=points_b[verified & held],
                  verified=int(verified.sum()))
    return result


def holdout_error(cameras: dict[str, dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure the current cameras on matches that the refinement never used."""
    import numpy as np

    per_pair, values = {}, []
    for pair in pairs:
        if not len(pair["holdout_a"]):
            continue
        first, second = cameras[pair["first"]], cameras[pair["second"]]
        scale = 2000 / max(first["width"], first["height"], second["width"], second["height"])
        error = scale * epipolar_distance(fundamental_from_cameras(first, second),
                                          pair["holdout_a"], pair["holdout_b"])
        values.append(error)
        per_pair[f"{pair['first']}|{pair['second']}"] = round(float(np.median(error)), 3)
    joined = np.concatenate(values) if values else np.zeros(1)
    return {"matches": int(sum(map(len, values))), "median_px": round(float(np.median(joined)), 3),
            "p90_px": round(float(np.percentile(joined, 90)), 3),
            # One wrong camera leaves most pairs unchanged, so the overall median cannot
            # show it. Average the pairs: each pair counts, and one bad match does not.
            "pair_mean_px": round(float(np.mean(list(per_pair.values()) or [0])), 3),
            "per_pair": per_pair}


def add_match_tracks(model: Any, cameras: dict[str, dict[str, Any]],
                     tracks: list[list[tuple[str, float, float]]]) -> int:
    """Append keypoints and triangulated tracks to a model, and return the track count."""
    import numpy as np
    import pycolmap

    added = 0
    pending: dict[int, list[tuple[float, float]]] = {}
    first_free = {image.image_id: image.num_points2D() for image in model.images.values()}
    planned: list[tuple[Any, list[tuple[int, int]]]] = []
    for track in tracks:
        centers, directions, members = [], [], []
        for name, x, y in track:
            camera = cameras[name]
            ray = camera["R"].T @ (np.linalg.inv(camera["K"]) @ np.array([x, y, 1.]))
            centers.append(camera["center"])
            directions.append(ray / np.linalg.norm(ray))
            members.append((camera["image_id"], (x, y)))
        point = triangulate_rays(centers, directions)
        if point is None or not np.isfinite(point).all():
            continue
        elements = []
        for image_id, xy in members:
            rows = pending.setdefault(image_id, [])
            elements.append((image_id, first_free[image_id] + len(rows)))
            rows.append(xy)
        planned.append((point, elements))
    for image_id, rows in pending.items():
        # Append in place: pycolmap refuses to replace the list of a filled image,
        # and existing tracks refer to the current keypoint indexes.
        keypoints = model.images[image_id].points2D
        for xy in rows:
            keypoints.append(pycolmap.Point2D(np.asarray(xy, dtype=float)))
    for point, elements in planned:
        track = pycolmap.Track()
        track.add_elements([pycolmap.TrackElement(*element) for element in elements])
        model.add_point3D(point, track)
        added += 1
    return added


def find_unusable_focal_lengths(before: dict[str, float], after: dict[str, float],
                                max_change: float) -> list[str]:
    """Name cameras whose focal length became impossible during refinement.

    A camera with a very wrong start can pass through zero to a negative focal
    length. Its reprojection error can stay small, so held-out matches do not
    show this, and OpenMVS later crashes on the negative scale.
    """
    return sorted(name for name, focal in after.items()
                  if focal <= 0 or not 1 / max_change <= focal / before[name] <= max_change)


def _focal_lengths(model: Any) -> dict[str, float]:
    return {image.name: float(model.cameras[image.camera_id].params[0])
            for image in model.images.values() if image.has_pose}


def refine_model(model: Any, settings: RefinementSettings,
                 log: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Run bundle adjustment with a falling reprojection limit, so early errors can recover."""
    import pycolmap

    stages = []
    manager = pycolmap.ObservationManager(model)
    for limit in settings.reprojection_schedule_px:
        options = pycolmap.BundleAdjustmentOptions()
        options.refine_focal_length = True
        options.refine_principal_point = settings.refine_principal_point
        options.refine_extra_params = False
        options.min_track_length = settings.min_track_length
        options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
        options.ceres.loss_function_scale = limit / 4
        pycolmap.bundle_adjustment(model, options)
        removed = manager.filter_all_points3D(limit, 0.5)
        stage = {"reprojection_limit_px": limit, "observations_removed": int(removed),
                 "points3D": model.num_points3D(),
                 "mean_reprojection_px": round(float(model.compute_mean_reprojection_error()), 3)}
        stages.append(stage)
        if log:
            log(json.dumps(stage))
    return stages


def _align_to_source(refined: Any, source_cameras: dict[str, dict[str, Any]]) -> None:
    """Return a refined model to the coordinates of its source, so meshes stay comparable."""
    import numpy as np
    import pycolmap

    from .cameras import fit_similarity

    current = _camera_table(refined)
    names = sorted(current)
    anchors = [(tuple(current[name]["center"]), tuple(source_cameras[name]["center"]))
               for name in names[::max(1, len(names) // MAX_ALIGNMENT_ANCHORS + 1)]]
    scale, rotation, translation, _ = fit_similarity(anchors)
    refined.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(np.array(rotation)),
                                     np.array(translation)))


def _pair_cache_path(cache: Path, first: str, second: str, settings: RefinementSettings,
                     content: dict[str, str]) -> Path:
    # Pixel coordinates belong to the exact photo and mask. A file name is not
    # enough: a resized photo with the same name would reuse old coordinates.
    key = hashlib.sha256(json.dumps([first, second, content[first], content[second],
                                     settings.coarse_resolution,
                                     settings.refine_resolution, settings.lattice_step,
                                     settings.min_certainty, settings.max_verified_epipolar_px,
                                     settings.holdout_every]).encode()).hexdigest()[:24]
    return cache / f"{key}.npz"


def refine_cameras_with_matches(
        revision: str | Path, masks: str | Path, output: str | Path, *,
        settings: RefinementSettings | None = None, device: str = "auto", colmap: str = "colmap",
        exclude_cameras: list[str] | None = None, cache_dir: str | Path | None = None,
        predictor: Any = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Write a new camera revision whose cameras also fit dense matches."""
    import numpy as np

    # Load torch first when it is available. On Apple Silicon, torch then
    # crashes with a segmentation fault if pycolmap loaded the OpenMP runtime.
    if predictor is None and find_spec("torch") is not None:
        import torch  # noqa: F401
    import pycolmap
    from PIL import Image

    from .dense import _sha256
    from .registered_camera_recovery import _read_json

    settings = settings or RefinementSettings()
    source = Path(revision).resolve()
    mask_root = Path(masks).resolve()
    target = Path(output).resolve()
    if (Path(revision).is_symlink() or not source.is_dir() or Path(output).is_symlink() or
            target.exists() or target.is_relative_to(source) or source.is_relative_to(target) or
            target.is_relative_to(mask_root)):
        raise ReferenceError("Camera refinement needs a camera revision and an unused output "
                             "folder outside the revision and the masks")
    rectified = source / "undistorted"
    if not (rectified / "sparse").is_dir() or not (rectified / "images").is_dir():
        raise ReferenceError(f"Revision has no rectified model: {rectified}")
    source_report = _read_json(source / "reconstruction-report.json")
    model = pycolmap.Reconstruction(str(rectified / "sparse"))
    excluded = sorted(set(exclude_cameras or []))
    registered = {image.name: image for image in model.images.values() if image.has_pose}
    if set(excluded) - set(registered):
        raise ReferenceError("Excluded cameras are not registered: " +
                             ", ".join(sorted(set(excluded) - set(registered))))
    for name in excluded:
        model.deregister_frame(registered[name].frame_id)
    cameras = _camera_table(model)
    if not all((mask_root / (name + ".mask.png")).is_file() for name in cameras):
        raise ReferenceError(f"Masks do not cover every registered camera: {mask_root}")
    sizes = {name: (camera["width"], camera["height"]) for name, camera in cameras.items()}
    for name in sizes:
        with Image.open(rectified / "images" / name) as photo:
            if photo.size != sizes[name]:
                raise ReferenceError(f"Rectified photo does not fit its camera: {name}")
    pair_names = select_pairs(_shared_tracks(model), settings.pairs_per_camera)
    target.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema": SCHEMA, "status": "running", "source_revision": str(source),
        "mask_source": str(mask_root), "settings": asdict(settings), "excluded_cameras": excluded,
        "cameras": len(cameras), "pairs": len(pair_names),
        "approved_for_reconstruction": False, "distribution_allowed": False}

    def save() -> dict[str, Any]:
        (target / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    save()
    cache = Path(cache_dir).resolve() if cache_dir is not None else target / "match-cache"
    cache.mkdir(parents=True, exist_ok=True)
    content = {name: _sha256(rectified / "images" / name) + _sha256(mask_root / (name + ".mask.png"))
               for name in cameras}
    pairs = []
    for index, (first, second) in enumerate(pair_names, 1):
        path = _pair_cache_path(cache, first, second, settings, content)
        if path.is_file():
            with np.load(path) as stored:
                pair = {"first": first, "second": second, **{key: stored[key] for key in stored.files}}
        else:
            if predictor is None:
                from .roma_predictions import RoMaPredictor
                predictor = RoMaPredictor(device=device, coarse_resolution=settings.coarse_resolution,
                                          refine_resolution=settings.refine_resolution)
            pair = match_pair(predictor, rectified / "images", mask_root, first, second,
                              settings, sizes)
            np.savez_compressed(path, **{key: pair[key] for key in
                                         ("train_a", "train_b", "holdout_a", "holdout_b")})
        pairs.append(pair)
        if log:
            log(f"matched {index}/{len(pair_names)} {first} {second}: {len(pair['train_a'])} train, "
                f"{len(pair['holdout_a'])} held out")

    tracks = merge_tracks([(pair["first"], pair["second"], pair["train_a"], pair["train_b"])
                           for pair in pairs if len(pair["train_a"])], settings.track_cell_px)
    report.update(verified_pairs=sum(len(pair["train_a"]) > 0 for pair in pairs),
                  train_matches=sum(len(pair["train_a"]) for pair in pairs),
                  match_tracks=len(tracks))
    save()
    use_square_pixels(model)
    start_focals = _focal_lengths(model)
    add_match_tracks(model, cameras, tracks)
    report["refinement_stages"] = refine_model(model, settings, log)
    # Remove a failed camera, then settle the others without its observations.
    dropped: list[str] = []
    while unusable := find_unusable_focal_lengths(start_focals, _focal_lengths(model),
                                                  settings.max_focal_change):
        dropped += unusable
        for name in unusable:
            model.deregister_frame(model.find_image_with_name(name).frame_id)
        report["refinement_stages"] += refine_model(
            model, replace(settings, reprojection_schedule_px=settings.reprojection_schedule_px[-2:]),
            log)
    report["dropped_cameras"] = sorted(dropped)
    _align_to_source(model, cameras)
    refined_cameras = _camera_table(model)
    # Compare both sets of cameras on the same matches.
    pairs = [pair for pair in pairs if pair["first"] in refined_cameras
             and pair["second"] in refined_cameras]
    before = holdout_error(cameras, pairs)
    after = holdout_error(refined_cameras, pairs)
    report.update(holdout_before=before, holdout_after=after)
    moved = np.array([np.linalg.norm(refined_cameras[name]["center"] - cameras[name]["center"])
                      for name in refined_cameras])
    spread = np.linalg.norm(np.array([c["center"] for c in cameras.values()]) -
                            np.median([c["center"] for c in cameras.values()], axis=0), axis=1)
    report["camera_center_moved_relative_to_spread"] = {
        "median": round(float(np.median(moved) / np.median(spread)), 5),
        "max": round(float(moved.max() / np.median(spread)), 5)}
    if after["pair_mean_px"] >= before["pair_mean_px"]:
        report.update(status="no-refined-cameras", reason="held-out-matches-did-not-improve")
        return save()

    destination = target / "undistorted"
    destination.mkdir()
    shutil.copytree(rectified / "images", destination / "images")
    (destination / "sparse").mkdir()
    (destination / "text-model").mkdir()
    model.write(str(destination / "sparse"))
    model.write_text(str(destination / "text-model"))
    registered_names = sorted(refined_cameras)
    index = json.loads((source / "undistorted-reference-index.json").read_text(encoding="utf-8"))
    index["images"] = [item for item in index["images"] if item["filename"] in refined_cameras]
    (target / "undistorted-reference-index.json").write_text(json.dumps(index, indent=2) + "\n",
                                                             encoding="utf-8")
    for name in ("staged-reference-index.json",):
        shutil.copyfile(source / name, target / name)
    (target / "reconstruction-report.json").write_text(json.dumps({
        **source_report, "registered_images": len(registered_names),
        "registered_filenames": registered_names, "undistorted_images": len(registered_names),
        "status": "candidate / visual-review-required", "undistorted_model": "undistorted/text-model",
        "match_refined_from": str(source)}, indent=2) + "\n", encoding="utf-8")
    report.update(status=STAGED_STATUS, refined_revision=str(target),
                  limitations=("Cameras now depend on matches from a learned model. Held-out "
                               "matches come from the same model, so they show consistency, not "
                               "ground truth. Compare a new mesh with the photographs."))
    return save()
