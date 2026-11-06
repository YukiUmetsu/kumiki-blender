"""Conservative ground-plane candidates from a registered COLMAP reconstruction.

The plane is a geometric *hypothesis*, not semantic truth. A 3D point can
become a ground negative only when it fits a low, camera-up-aligned plane AND
independent photographs show it outside both SAM2 subject candidates. This
also lets an otherwise verified ground point veto a target mask which included
it: the target image itself never votes for its own negative label.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any


_MAX_MODEL_BYTES = 1_000_000_000


def _is_held_out(point_id: int) -> bool:
    # Identical to foreground track partition, deliberately not imported from
    # mask_track_guidance to avoid cyclic imports.
    return int.from_bytes(hashlib.sha256(str(point_id).encode()).digest()[:2],
                          "big") % 5 == 0


def _camera_geometry(model_dir: Path) -> tuple[list[list[float]], list[list[float]]]:
    """Read COLMAP camera-up and centers without loading 2D feature arrays."""
    ups, centers = [], []
    binary = model_dir / "images.bin"
    if not binary.is_file() or binary.is_symlink() or binary.stat().st_size > _MAX_MODEL_BYTES:
        return [], []
    try:
        with binary.open("rb") as source:
            header = source.read(8)
            if len(header) != 8:
                return [], []
            count, = struct.unpack("<Q", header)
            if count > 100_000:
                return [], []
            for _ in range(count):
                record = source.read(64)
                if len(record) != 64:
                    return [], []
                _, w, x, y, z, tx, ty, tz, _ = struct.unpack("<i7di", record)
                norm = math.sqrt(w*w + x*x + y*y + z*z)
                if norm < 1e-8:
                    return [], []
                w, x, y, z = (value / norm for value in (w, x, y, z))
                # COLMAP qvec rotates world to camera. Up is minus the second
                # row of R and camera center is -R.T @ t, both in world space.
                rotation = (
                    (1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
                    (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
                    (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)),
                )
                ups.append([-value for value in rotation[1]])
                translation = (tx, ty, tz)
                centers.append([-sum(rotation[i][j]*translation[i]
                                     for i in range(3)) for j in range(3)])
                for _char in range(4096):
                    if source.read(1) == b"\x00":
                        break
                else:
                    return [], []
                num_data = source.read(8)
                if len(num_data) != 8:
                    return [], []
                num_points, = struct.unpack("<Q", num_data)
                if num_points > 10_000_000:
                    return [], []
                source.seek(num_points * 24, 1)
    except (OSError, struct.error):
        return [], []
    return ups, centers


def _point_coordinates(model_dir: Path, needed: set[int]) -> dict[int, tuple[float, float, float]]:
    """Bounded streaming COLMAP points3D.bin reader; ignore unused tracks."""
    path = model_dir / "points3D.bin"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > _MAX_MODEL_BYTES:
        return {}
    points = {}
    try:
        with path.open("rb") as source:
            header = source.read(8)
            if len(header) != 8:
                return {}
            count, = struct.unpack("<Q", header)
            if count > 5_000_000:
                return {}
            for _ in range(count):
                record = source.read(51)
                if len(record) != 51:
                    return {}
                point_id, xx, yy, zz, _, _, _, _, length = struct.unpack(
                    "<QdddBBBdQ", record)
                if length > 1_000_000:
                    return {}
                if point_id in needed and all(map(math.isfinite, (xx, yy, zz))):
                    points[point_id] = (xx, yy, zz)
                source.seek(length * 8, 1)
    except (OSError, struct.error):
        return {}
    return points


def fit_ground_plane(model_dir: Path, observations: dict[int, list[dict[str, Any]]]
                     ) -> tuple[set[int], dict[str, Any]]:
    """Fit one low horizontal plane from multiview exterior tracks.

    No monocular depth guesses, source-mask labels, or detection boxes can by
    themselves certify ground. The candidate must form a spatially extended
    low plane in *triangulated* coordinates with consistent camera up vectors.
    """
    import numpy as np

    invalid = {"status": "unavailable / insufficient-verified-ground-plane",
               "candidate_tracks": 0, "inlier_tracks": 0}
    candidates = []
    for point_id, views in sorted(observations.items()):
        # One erroneous foreground vote is exactly the ground-leak scenario
        # we want to repair. Multiple independent foreground votes veto it.
        if len(views) < 4 or sum(view["supported"] for view in views) > 1:
            continue
        outside = sum(view["mask_exterior"] for view in views)
        lower = sum(view["lower_scene_exterior"] for view in views)
        # Two lower-scene views are needed to distinguish the floor from a
        # possible missed head/torso silhouette or the background horizon.
        if outside >= 3 and lower >= 2 and outside/len(views) >= .65:
            candidates.append(point_id)
    invalid["candidate_tracks"] = len(candidates)
    if len(candidates) < 24:
        return set(), invalid
    subject_ids = [point_id for point_id, views in observations.items()
                   if sum(view["supported"] for view in views) >= 3]
    all_xyz = _point_coordinates(model_dir, set(candidates) | set(subject_ids))
    xyz = {point_id: all_xyz[point_id] for point_id in candidates if point_id in all_xyz}
    if len(xyz) < 24:
        return set(), {**invalid, "reason": "missing-triangulated-3d-coordinates"}
    up_vectors, camera_positions = _camera_geometry(model_dir)
    camera_up = np.asarray(up_vectors, dtype=float)
    if len(camera_up) < 4:
        return set(), {**invalid, "reason": "missing-camera-orientations"}
    up = np.median(camera_up, axis=0)
    length = float(np.linalg.norm(up))
    if length < .5:
        return set(), {**invalid, "reason": "inconsistent-camera-up"}
    up /= length
    if np.mean(camera_up @ up > .72) < .75:
        return set(), {**invalid, "reason": "inconsistent-camera-up"}
    ids = np.array(sorted(xyz), dtype=np.int64)
    coords = np.array([xyz[int(i)] for i in ids], dtype=float)
    training = np.asarray([not _is_held_out(int(i)) for i in ids], dtype=bool)
    train_points = coords[training]
    if len(train_points) < 20:
        return set(), {**invalid, "reason": "insufficient-independent-plane-training-tracks"}
    low, high = np.quantile(train_points, [.05, .95], axis=0)
    scale = float(np.linalg.norm(high-low))
    if not math.isfinite(scale) or scale <= 1e-5:
        return set(), {**invalid, "reason": "degenerate-scene-scale"}
    heights = train_points @ up
    low_height = float(np.quantile(heights, .42))
    tolerance = max(1e-5, scale * .006)
    rng = np.random.default_rng(187)
    best = np.zeros(len(train_points), dtype=bool)
    best_normal = None
    best_offset = 0.0
    for _ in range(240):
        sample = train_points[rng.choice(len(train_points), 3, replace=False)]
        normal = np.cross(sample[1]-sample[0], sample[2]-sample[0])
        size = float(np.linalg.norm(normal))
        if size < scale*scale*1e-6:
            continue
        normal /= size
        if normal @ up < 0:
            normal = -normal
        if normal @ up < .82:
            continue
        offset = float(sample[0] @ normal)
        # Height must be measured at the sampled surface, not where an
        # inclined plane happens to intersect the arbitrary world origin.
        height = float(sample[0] @ up)
        if height > low_height + tolerance:
            continue
        inliers = np.abs(train_points @ normal - offset) <= tolerance
        if int(inliers.sum()) > int(best.sum()):
            best, best_normal, best_offset = inliers, normal, offset
    if best_normal is None or int(best.sum()) < max(18, min(40, int(.12*len(train_points)))):
        return set(), {**invalid, "reason": "no-strong-low-horizontal-plane"}
    # A line, small cluster, or one statue paw does not establish a scene floor.
    planar = train_points[best] - np.mean(train_points[best], axis=0)
    singular = np.linalg.svd(planar, compute_uv=False)
    if len(singular) < 2 or singular[1] < scale * .23:
        return set(), {**invalid, "reason": "ground-plane-not-spatially-extended"}
    centers = np.asarray(camera_positions, dtype=float)
    if (not np.isfinite(centers).all() or
            np.mean(centers @ best_normal - best_offset > 2*tolerance) < .75):
        return set(), {**invalid, "reason": "plane-not-below-registered-cameras"}
    subject_coords = np.asarray([all_xyz[point_id] for point_id in subject_ids
                                 if point_id in all_xyz], dtype=float)
    if len(subject_coords) >= 30:
        separation = subject_coords @ best_normal - best_offset
        if (np.mean(separation > 2*tolerance) < .55 or
                np.mean(separation < -2*tolerance) > .10):
            return set(), {**invalid, "reason": "ground-plane-conflicts-with-verified-statue-tracks"}
    # Held-out 3D tracks are checked against the frozen fitted plane; none of
    # them chose its normal, offset, threshold, or spatial extent.
    verified = np.abs(coords @ best_normal - best_offset) <= tolerance
    inliers = {int(point_id) for point_id in ids[verified]}
    return inliers, {
        "status": "candidate / low-camera-up-aligned-3d-plane",
        "candidate_tracks": len(candidates), "inlier_tracks": len(inliers),
        "training_inlier_tracks": int(best.sum()),
        "heldout_inlier_tracks": int(np.count_nonzero(verified & ~training)),
        "camera_above_plane_fraction": round(float(np.mean(
            centers @ best_normal - best_offset > 2*tolerance)), 5),
        "verified_statue_tracks_compared": len(subject_coords),
        "normal": [round(float(v), 7) for v in best_normal],
        "offset": round(best_offset, 7),
        "relative_inlier_tolerance": .006,
        "limitations": "Fitted planes are geometric candidates, not verified ground semantics."
    }


def evaluate_local_ground_spill(photo: Any, original: Any, proposed: Any,
                                ground_seeds: list[dict[str, Any]]) -> dict[str, Any]:
    """Segment only locally near measured ground seeds, never the whole scene.

    This is a *veto*, not proof that an arbitrary pixel is ground. Color alone
    never creates a background click or increases the allowed mask area. The
    independently held-out ground tracks are excluded from this calculation.
    """
    import numpy as np
    from PIL import Image

    seeds = [row for row in ground_seeds if row.get("source") ==
             "triangulated-low-ground-plane"]
    if len(seeds) < 3:
        return {"status": "abstained / insufficient-geometric-ground-seeds",
                "rejected": False, "ground_seeds": len(seeds)}
    scale = min(1.0, 256 / max(photo.size))
    dims = (max(1, round(photo.width*scale)), max(1, round(photo.height*scale)))
    rgb = np.asarray(photo.convert("RGB").resize(dims, Image.Resampling.BILINEAR),
                     dtype=np.float32)
    old = np.asarray(original.convert("L").resize(dims, Image.Resampling.NEAREST)) >= 128
    new = np.asarray(proposed.convert("L").resize(dims, Image.Resampling.NEAREST)) >= 128
    added = new & ~old
    if not np.any(added):
        return {"status": "candidate / no-new-foreground", "rejected": False,
                "ground_seeds": len(seeds), "ground_region_added_fraction": 0.0}
    height, width = rgb.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    region = np.zeros((height, width), dtype=bool)
    for row in seeds:
        u, v = row["normalized_xy"]
        sx, sy = min(width-1, round(u*(width-1))), min(height-1, round(v*(height-1)))
        # The very small color sample is anchored in a *triangulated plane*,
        # not an inferred image-wide sand color that might resemble limestone.
        patch = rgb[max(0, sy-2):sy+3, max(0, sx-2):sx+3]
        target = np.median(patch.reshape(-1, 3), axis=0)
        within = (((xx-sx)/max(1, width))**2 +
                  ((yy-sy)/max(1, height))**2) <= .06**2
        similarity = np.sum((rgb-target)**2, axis=2) <= 28**2
        region |= within & similarity
    added_region = int(np.count_nonzero(added & region))
    added_total = int(np.count_nonzero(added))
    fraction = added_region / max(1, added_total)
    area = added_region / (width*height)
    # Both a meaningful fraction of the new pixels and a nontrivial scene area
    # must resemble measured ground to avoid rejecting tiny boundary halos.
    rejected = fraction >= .16 and area >= .0015
    return {"status": "rejected / ground-plane-appearance-spill" if rejected else
            "candidate / no-material-ground-appearance-spill",
            "rejected": rejected, "ground_seeds": len(seeds),
            "ground_region_added_fraction": round(fraction, 6),
            "ground_region_added_image_fraction": round(area, 6),
            "limitations": "Local color agreement is a veto near measured planar seeds only, not a full semantic floor segmentation."}
