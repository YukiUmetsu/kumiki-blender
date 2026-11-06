"""Fill gaps in a dense cloud with triangulated dense matches.

Patch-based stereo compares image brightness, so it fails between photos with
different light, resolution or date. A dense matcher can still connect those
photos. With calibrated cameras, its matches give real 3D points by
triangulation: nothing here predicts depth or invents a surface.

Each point must pass a fundamental-matrix check that does not use the cameras,
a reprojection check and a minimum ray angle. A point is kept only where the
dense cloud has a gap, and the masks of the other cameras can veto it. Patch
stereo results stay unchanged where they exist.

These points have less support than multi-view stereo: many come from two
photos. Their accuracy is lower, and a person must compare the mesh with the
photographs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

from .match_camera_refinement import _camera_table, _load_mask, _shared_tracks, select_pairs
from .references import ReferenceError


SCHEMA = "match-densified-dense-v1"
REPORT = "match-densification-report.json"
DENSE_CLOUD = "scene_dense.ply"
# A dense cloud record: xyz, rgb, normal, then the view and weight lists.
_RECORD_HEAD_BYTES = 27


@dataclass(frozen=True)
class DensificationSettings:
    """Limits for match densification. Pixel limits use a 2000 px long side."""

    pairs_per_camera: int = 6
    coarse_resolution: int = 560
    refine_resolution: int = 864
    lattice_step: int = 2
    min_certainty: float = .3
    max_verified_epipolar_px: float = 1.5
    max_reprojection_px: float = 3.0
    min_ray_angle_degrees: float = 3.0
    gap_radius_fraction: float = .003
    voxel_fraction: float = .001
    # A majority: reviewed masks can also miss part of a subject. At .3, holes
    # stayed on the Sphinx back. At .5, they closed, but stray fragments appeared.
    max_mask_veto_fraction: float = .5
    min_mask_votes: int = 3
    min_verified_matches: int = 200

    def __post_init__(self) -> None:
        for name in ("pairs_per_camera", "coarse_resolution", "refine_resolution", "lattice_step",
                     "min_mask_votes", "min_verified_matches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ReferenceError(f"Densification setting {name} must be a positive integer")
        if not 0 < self.min_certainty < 1 or not 0 < self.max_mask_veto_fraction < 1:
            raise ReferenceError("min_certainty and max_mask_veto_fraction must be between 0 and 1")
        if self.gap_radius_fraction <= 0 or self.voxel_fraction <= 0:
            raise ReferenceError("gap_radius_fraction and voxel_fraction must be positive")


def read_dense_cloud(path: Path) -> tuple[bytes, bytes, Any, list[int]]:
    """Return the header, body, xyz array and record offsets of an OpenMVS dense cloud."""
    import numpy as np

    data = path.read_bytes()
    end = data.find(b"end_header\n")
    if end < 0 or b"binary_little_endian" not in data[:end]:
        raise ReferenceError(f"Dense cloud is not a binary little-endian PLY: {path}")
    header = data[:end + 11]
    count = int(header.split(b"element vertex ")[1].split()[0])
    expected = [b"x", b"y", b"z", b"red", b"green", b"blue", b"nx", b"ny", b"nz",
                b"view_indices", b"view_weights"]
    names = [line.split()[-1] for line in header.split(b"\n") if line.startswith(b"property")]
    if names != expected:
        raise ReferenceError(f"Dense cloud has an unsupported layout: {path}")
    body = data[end + 11:]
    xyz, offsets = np.empty((count, 3), np.float32), []
    position = 0
    for index in range(count):
        offsets.append(position)
        xyz[index] = np.frombuffer(body, "<f4", 3, position)
        position += _RECORD_HEAD_BYTES
        position += 1 + 4 * body[position]
        position += 1 + 4 * body[position]
    return header, body, xyz.astype(float), offsets


def views_of_record(body: bytes, offset: int) -> list[int]:
    """Read the scene image indexes of one dense cloud record."""
    import numpy as np

    position = offset + _RECORD_HEAD_BYTES
    return np.frombuffer(body, "<u4", body[position], position + 1).tolist()


def encode_points(xyz: Any, rgb: Any, normals: Any, views: Any) -> bytes:
    """Encode points in the dense cloud layout, with two views of equal weight."""
    import numpy as np

    record = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3), ("normal", "<f4", 3),
                       ("view_count", "u1"), ("views", "<u4", 2),
                       ("weight_count", "u1"), ("weights", "<f4", 2)])
    rows = np.zeros(len(xyz), record)
    rows["xyz"], rows["rgb"], rows["normal"], rows["views"] = xyz, rgb, normals, views
    rows["view_count"], rows["weight_count"], rows["weights"] = 2, 2, 1.0
    return rows.tobytes()


def write_dense_cloud(path: Path, header: bytes, body: bytes, count: int, extra: bytes,
                      extra_count: int) -> None:
    """Write the existing records and the new records under an updated vertex count."""
    updated = header.replace(b"element vertex %d\n" % count,
                             b"element vertex %d\n" % (count + extra_count))
    if updated == header and extra_count:
        raise ReferenceError("Dense cloud header has no matching vertex count")
    path.write_bytes(updated + body + extra)


def scene_image_order(scene_bytes: bytes, names: list[str]) -> dict[str, int]:
    """Return the OpenMVS index of each photo from the file-name order in a scene."""
    offsets = {}
    for name in names:
        offset = scene_bytes.find(b"/" + name.encode())
        if offset < 0:
            raise ReferenceError(f"Dense scene does not contain photo {name}; "
                                 "use the dense revision of these cameras")
        offsets[name] = offset
    return {name: index for index, name in enumerate(sorted(offsets, key=offsets.get))}


def views_agree_with_cameras(xyz: Any, view_lists: list[list[int]],
                             cameras_by_index: dict[int, dict[str, Any]]) -> float:
    """Return the share of listed views that see their point in frame."""
    import numpy as np

    seen = total = 0
    for point, views in zip(xyz, view_lists):
        for view in views:
            camera = cameras_by_index[view]
            local = camera["R"] @ point + camera["t"]
            pixel = camera["K"] @ local
            total += 1
            if local[2] > 0:
                x, y = pixel[:2] / pixel[2]
                seen += 0 <= x < camera["width"] and 0 <= y < camera["height"]
    return float(seen / max(total, 1))


def triangulate_matches(first: dict[str, Any], second: dict[str, Any], points_a: Any, points_b: Any,
                        max_reprojection_px: float, min_ray_angle_degrees: float
                        ) -> tuple[Any, Any]:
    """Triangulate matches, and return the points and a mask of the points to keep."""
    import numpy as np

    def rays(camera: dict[str, Any], pixels: Any) -> Any:
        directions = (np.linalg.inv(camera["K"]) @ np.c_[pixels, np.ones(len(pixels))].T).T @ camera["R"]
        return directions / np.linalg.norm(directions, axis=1, keepdims=True)

    ray_a, ray_b = rays(first, points_a), rays(second, points_b)
    baseline = second["center"] - first["center"]
    cosine = (ray_a * ray_b).sum(axis=1)
    denominator = np.maximum(1 - cosine * cosine, 1e-12)
    along_a, along_b = ray_a @ baseline, ray_b @ baseline
    depth_a = (along_a - cosine * along_b) / denominator
    depth_b = (cosine * along_a - along_b) / denominator
    points = (first["center"] + ray_a * depth_a[:, None] +
              second["center"] + ray_b * depth_b[:, None]) / 2

    def error(camera: dict[str, Any], pixels: Any) -> Any:
        projected = (camera["K"] @ (camera["R"] @ points.T + camera["t"][:, None])).T
        return np.linalg.norm(projected[:, :2] / projected[:, 2:] - pixels, axis=1)

    scale = max(first["width"], first["height"], second["width"], second["height"]) / 2000
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    keep = ((depth_a > 0) & (depth_b > 0) & (angle >= min_ray_angle_degrees) &
            (np.maximum(error(first, points_a), error(second, points_b)) <= max_reprojection_px * scale))
    return points, keep


def mask_veto(points: Any, cameras: dict[str, dict[str, Any]], masks: dict[str, Any],
              settings: DensificationSettings) -> Any:
    """Return True for points that the masks of the cameras that see them contradict."""
    import numpy as np

    inside, outside = np.zeros(len(points), int), np.zeros(len(points), int)
    for name, camera in cameras.items():
        mask = masks[name]
        local = (camera["R"] @ points.T).T + camera["t"]
        front = local[:, 2] > 1e-9
        pixels = (camera["K"] @ local[front].T).T
        x, y = pixels[:, 0] / pixels[:, 2], pixels[:, 1] / pixels[:, 2]
        in_frame = (x >= 0) & (x < camera["width"]) & (y >= 0) & (y < camera["height"])
        indexes = np.flatnonzero(front)[in_frame]
        labels = mask[y[in_frame].astype(int), x[in_frame].astype(int)]
        inside[indexes] += labels
        outside[indexes] += ~labels
    votes = inside + outside
    return (votes >= settings.min_mask_votes) & (
        outside > settings.max_mask_veto_fraction * np.maximum(votes, 1))


def cameras_that_can_veto(cameras: dict[str, dict[str, Any]], unreviewed: list[str]
                          ) -> dict[str, dict[str, Any]]:
    """Leave out cameras with automatic masks.

    An automatic mask that misses part of the subject reads as a vote for
    background. On the Sphinx, 24 such masks raised the rejected points from 45%
    to 70%, and removed real surface on the paws and the back.
    """
    return {name: camera for name, camera in cameras.items() if name not in unreviewed}


def thin_to_voxels(points: Any, size: float) -> Any:
    """Return the indexes of one point per voxel, so a surface is not sampled twice."""
    import numpy as np

    cells = np.floor(points / size).astype(np.int64)
    return np.sort(np.unique(cells, axis=0, return_index=True)[1])


def _dense_pair_matches(predictor: Any, images: Path, first: str, second: str,
                        cameras: dict[str, dict[str, Any]], masks: dict[str, Any],
                        settings: DensificationSettings) -> tuple[Any, Any]:
    """Sample verified dense matches on the subject, for one pair of photos."""
    import cv2
    import numpy as np

    a, b = cameras[first], cameras[second]
    warp, certainty = predictor.dense_warp(images / first, images / second)
    step = settings.lattice_step
    sampled, score = warp[::step, ::step].reshape(-1, 4), certainty[::step, ::step].ravel()
    points_a = np.c_[(sampled[:, 0] + 1) / 2 * a["width"], (sampled[:, 1] + 1) / 2 * a["height"]]
    points_b = np.c_[(sampled[:, 2] + 1) / 2 * b["width"], (sampled[:, 3] + 1) / 2 * b["height"]]
    keep = (score >= settings.min_certainty) & (points_b >= 0).all(axis=1) & (
        points_b[:, 0] < b["width"]) & (points_b[:, 1] < b["height"]) & (
        points_a >= 0).all(axis=1) & (points_a[:, 0] < a["width"]) & (points_a[:, 1] < a["height"])
    keep[keep] &= masks[first][points_a[keep, 1].astype(int), points_a[keep, 0].astype(int)]
    keep[keep] &= masks[second][points_b[keep, 1].astype(int), points_b[keep, 0].astype(int)]
    empty = np.zeros((0, 2))
    if keep.sum() < settings.min_verified_matches:
        return empty, empty
    points_a, points_b = points_a[keep], points_b[keep]
    scale = max(a["width"], a["height"], b["width"], b["height"]) / 2000
    limit = settings.max_verified_epipolar_px * scale
    chosen = np.linspace(0, len(points_a) - 1, min(30000, len(points_a))).astype(int)
    fundamental, _ = cv2.findFundamentalMat(points_a[chosen], points_b[chosen], cv2.USAC_MAGSAC,
                                            limit, .9999, 20000)
    if fundamental is None or fundamental.shape != (3, 3):
        return empty, empty
    from .match_camera_refinement import epipolar_distance

    verified = epipolar_distance(fundamental, points_a, points_b) <= limit
    return points_a[verified], points_b[verified]


def densify_with_matches(
        reconstruction: str | Path, rectified_masks: str | Path, dense_revision: str | Path,
        output: str | Path, *, settings: DensificationSettings | None = None, device: str = "auto",
        predictor: Any = None, executables: dict[str, str] | None = None,
        log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Write a dense revision whose cloud also has triangulated matches in its gaps."""
    import numpy as np

    # Load torch first: see refine_cameras_with_matches.
    if predictor is None and find_spec("torch") is not None:
        import torch  # noqa: F401
    import pycolmap
    from PIL import Image
    from scipy.spatial import cKDTree

    from .dense import _json, rerun_dense_from_stage

    settings = settings or DensificationSettings()
    sparse = Path(reconstruction).resolve()
    masks_root = Path(rectified_masks).resolve()
    source = Path(dense_revision).resolve()
    target = Path(output).resolve()
    workspace = source / "workspace"
    if (Path(output).is_symlink() or target.exists() or target.is_relative_to(source) or
            target.is_relative_to(sparse) or source.is_relative_to(target)):
        raise ReferenceError("Match densification needs an unused output outside its inputs")
    for path in (workspace / DENSE_CLOUD, workspace / "scene.mvs", source / "dense-report.json"):
        if path.is_symlink() or not path.is_file():
            raise ReferenceError(f"Dense revision is incomplete; {path.name} is missing: {source}")
    dense_report = _json(source / "dense-report.json")
    if dense_report.get("source_revision") != str(sparse):
        raise ReferenceError("Dense revision belongs to another reconstruction")
    model = pycolmap.Reconstruction(str(sparse / "undistorted/sparse"))
    cameras = _camera_table(model)
    masks = {name: _load_mask(masks_root, name) for name in cameras}
    header, body, dense_xyz, offsets = read_dense_cloud(workspace / DENSE_CLOUD)
    order = scene_image_order((workspace / "scene.mvs").read_bytes(), sorted(cameras))
    by_index = {index: cameras[name] for name, index in order.items()}
    sample = np.linspace(0, len(offsets) - 1, min(2000, len(offsets))).astype(int)
    agreement = views_agree_with_cameras(dense_xyz[sample],
                                         [views_of_record(body, offsets[i]) for i in sample], by_index)
    if agreement < .95:
        raise ReferenceError(f"Scene image order does not fit the dense cloud ({agreement:.2f}); "
                             "new points would name the wrong cameras")
    diagonal = float(np.linalg.norm(np.percentile(dense_xyz, 99, axis=0) -
                                    np.percentile(dense_xyz, 1, axis=0)))
    tree = cKDTree(dense_xyz)
    pairs = select_pairs(_shared_tracks(model), settings.pairs_per_camera)
    if predictor is None:
        from .roma_predictions import RoMaPredictor
        predictor = RoMaPredictor(device=device, coarse_resolution=settings.coarse_resolution,
                                  refine_resolution=settings.refine_resolution)
    images = sparse / "undistorted/images"
    kept_points, kept_colors, kept_normals, kept_views = [], [], [], []
    triangulated = 0
    for number, (first, second) in enumerate(pairs, 1):
        points_a, points_b = _dense_pair_matches(predictor, images, first, second, cameras, masks,
                                                 settings)
        if not len(points_a):
            continue
        points, keep = triangulate_matches(cameras[first], cameras[second], points_a, points_b,
                                           settings.max_reprojection_px,
                                           settings.min_ray_angle_degrees)
        triangulated += int(keep.sum())
        keep[keep] &= tree.query(points[keep], workers=-1)[0] > settings.gap_radius_fraction * diagonal
        if not keep.any():
            continue
        points, pixels = points[keep], points_a[keep]
        with Image.open(images / first) as photo:
            color = np.asarray(photo.convert("RGB"))[pixels[:, 1].astype(int), pixels[:, 0].astype(int)]
        toward = (cameras[first]["center"] + cameras[second]["center"]) / 2 - points
        kept_points.append(points)
        kept_colors.append(color)
        kept_normals.append(toward / np.linalg.norm(toward, axis=1, keepdims=True))
        kept_views.append(np.tile([order[first], order[second]], (len(points), 1)))
        if log:
            log(f"pair {number}/{len(pairs)} {first} {second}: {len(points)} gap points")
    report: dict[str, Any] = {
        "schema": SCHEMA, "source_dense_revision": str(source), "settings": asdict(settings),
        "pairs": len(pairs), "triangulated_points": triangulated, "view_order_agreement": round(agreement, 4),
        "approved_for_reconstruction": False, "distribution_allowed": False}
    if not kept_points:
        raise ReferenceError("No triangulated match fell in a gap of the dense cloud; "
                             "this revision has nothing to add")
    points, colors = np.concatenate(kept_points), np.concatenate(kept_colors)
    normals, views = np.concatenate(kept_normals), np.concatenate(kept_views)
    gap_points = len(points)
    from .registered_camera_recovery import _read_json

    unreviewed = _read_json(sparse / "reconstruction-report.json").get("located_photos", [])
    vetoed = mask_veto(points, cameras_that_can_veto(cameras, unreviewed), masks, settings)
    points, colors, normals, views = points[~vetoed], colors[~vetoed], normals[~vetoed], views[~vetoed]
    chosen = thin_to_voxels(points, settings.voxel_fraction * diagonal)
    points, colors, normals, views = points[chosen], colors[chosen], normals[chosen], views[chosen]
    report.update(gap_points=gap_points, mask_vetoed_points=int(vetoed.sum()), added_points=len(points),
                  masks_without_a_vote=sorted(unreviewed),
                  existing_dense_points=len(offsets))

    # The new revision reruns meshing on the merged cloud.
    def replace_cloud(workspace_copy: Path) -> None:
        write_dense_cloud(workspace_copy / DENSE_CLOUD, header, body, len(offsets),
                          encode_points(points, colors, normals, views), len(points))
        (target / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rerun_dense_from_stage(sparse, masks_root, source, target, stage="mesh",
                           without=("scene_dense_mesh", DENSE_CLOUD), prepare=replace_cloud,
                           executables=executables)
    meshed = _json(target / "dense-report.json")
    report.update(status=meshed["status"], failed_stage=meshed.get("failed_stage"),
                  mesh=meshed.get("mesh"),
                  limitations=("Added points come from a learned matcher and most have two views. "
                               "They fill gaps in stereo, and are less accurate than stereo where "
                               "stereo works. Compare the mesh with the photographs."))
    (target / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
