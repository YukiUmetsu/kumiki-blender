"""Verified 2D-track bridge and conservative Sim(3) for disjoint COLMAP models.

Only geometrically verified COLMAP match indices, triangulated 3D points and
source-photo confidence are eligible. Prediction-only geometry never enters.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

MAX_IMAGE_ID = 2_147_483_647


def _subject_lookup(path: Path | None):
    if path is None:
        return lambda _name, _xy: False
    import numpy as np
    from PIL import Image

    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or unsafe source confidence index")
    records = json.loads(path.read_text(encoding="utf-8"))
    if records.get("schema") != "subject-feature-index-v1":
        raise ValueError("Unrecognized source confidence index")
    arrays = {}
    for name, entry in records["images"].items():
        confidence = Path(entry["confidence_path"])
        if confidence.is_symlink() or not confidence.is_file():
            raise ValueError("Unsafe subject raster")
        with confidence.open("rb") as data:
            sha = hashlib.file_digest(data, "sha256").hexdigest()
        if sha != entry["confidence_sha256"]:
            raise ValueError("Subject confidence cache hash mismatch")
        with Image.open(confidence) as img:
            if img.mode != "L":
                raise ValueError("Invalid subject confidence mode")
            arrays[name] = np.asarray(img).copy()

    def verified(name, xy):
        data = arrays.get(name)
        if data is None:
            return False
        x, y = (int(round(float(coordinate) - 0.5)) for coordinate in xy)
        return 0 <= y < data.shape[0] and 0 <= x < data.shape[1] and data[y, x] == 255

    return verified


def extract_verified_3d_matches(database: Path, baseline, cohort, *,
                                qa_index: Path | None = None) -> list[dict]:
    """Join geometric 2D matches to real reconstructed 3D track identifiers.

    A verified two-view pair crosses from registered baseline image to a
    DIFFERENT registered cohort image. Collisions and many-to-one 3D tracks
    are dropped instead of choosing a geometrically convenient pairing.
    """
    import numpy as np

    if database.is_symlink() or not database.is_file():
        raise ValueError("Missing or unsafe COLMAP match database")
    baseline_ids = set(baseline.reg_image_ids())
    cohort_ids = set(cohort.reg_image_ids())
    if baseline_ids & cohort_ids:
        raise ValueError("Point-only alignment requires zero common registered camera IDs")
    subject = _subject_lookup(qa_index)
    matches = defaultdict(lambda: {"image_pairs": set(), "verified_subject_pairs": 0,
                                    "observations": 0, "sample": None})
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        for pair_id, count, columns, config, blob in db.execute(
                "SELECT pair_id, rows, cols, config, data FROM two_view_geometries WHERE rows>=12"):
            left, right = divmod(pair_id, MAX_IMAGE_ID)
            # Exclude planar-only, panoramic, watermark and degenerate
            # geometries: their inliers can describe background billboards.
            if config not in (2, 3, 4) or not ((left in baseline_ids and right in cohort_ids) or
                                         (right in baseline_ids and left in cohort_ids)):
                continue
            if columns != 2 or not isinstance(blob, bytes) or len(blob) != count * 8:
                continue
            import struct
            for left_idx, right_idx in struct.iter_unpack("<II", blob):
                base_id, co_id = ((left, right) if left in baseline_ids else (right, left))
                base_idx, co_idx = ((left_idx, right_idx) if left in baseline_ids else
                                    (right_idx, left_idx))
                base_img, co_img = baseline.images[base_id], cohort.images[co_id]
                if (base_idx >= len(base_img.points2D) or co_idx >= len(co_img.points2D)):
                    continue
                bp, cp = base_img.points2D[base_idx], co_img.points2D[co_idx]
                if not bp.has_point3D() or not cp.has_point3D():
                    continue
                bp_id, cp_id = int(bp.point3D_id), int(cp.point3D_id)
                if bp_id not in baseline.points3D or cp_id not in cohort.points3D:
                    continue
                bxyz = np.asarray(baseline.points3D[bp_id].xyz, dtype=float).reshape(3)
                cxyz = np.asarray(cohort.points3D[cp_id].xyz, dtype=float).reshape(3)
                if not np.isfinite(bxyz).all() or not np.isfinite(cxyz).all():
                    continue
                row = matches[(bp_id, cp_id)]
                row["image_pairs"].add((base_id, co_id))
                row["verified_subject_pairs"] += int(subject(base_img.name, bp.xy) and
                                                     subject(co_img.name, cp.xy))
                row["observations"] += 1
                row["sample"] = (bxyz, cxyz, base_id, co_id, base_idx, co_idx)
    by_base, by_cohort = defaultdict(set), defaultdict(set)
    for base_id, co_id in matches:
        by_base[base_id].add(co_id)
        by_cohort[co_id].add(base_id)
    result = []
    for (base_id, co_id), record in sorted(matches.items()):
        if len(by_base[base_id]) != 1 or len(by_cohort[co_id]) != 1:
            continue
        bxyz, cxyz, image_id, cohort_id, bidx, cidx = record["sample"]
        result.append({"baseline_point_id": base_id, "cohort_point_id": co_id,
                       "baseline_xyz": bxyz.tolist(), "cohort_xyz": cxyz.tolist(),
                       "image_pairs": sorted(record["image_pairs"]),
                       "observations": record["observations"],
                       "subject_verified": record["verified_subject_pairs"] > 0,
                       "sample": {"baseline_image_id": image_id, "cohort_image_id": cohort_id,
                                  "baseline_keypoint": bidx, "cohort_keypoint": cidx}})
    return result


def estimate_point_alignment(matches: list[dict], *, baseline_cloud=None, cohort_cloud=None,
                             minimum_inliers: int = 12, max_trials: int = 1200) -> dict:
    """Deterministic RANSAC Sim(3); reject thin/ambiguous/unverified support."""
    import numpy as np
    from .sfm_cohort_recovery import _similarity

    result = {"accepted": False, "reasons": [], "correspondences": len(matches), "inliers": 0}
    if len(matches) < minimum_inliers:
        result["reasons"].append("insufficient-cross-model-3d-correspondences")
        return result
    source = np.asarray([row["cohort_xyz"] for row in matches], dtype=float)
    target = np.asarray([row["baseline_xyz"] for row in matches], dtype=float)
    if source.shape != target.shape or not np.isfinite(source).all() or not np.isfinite(target).all():
        result["reasons"].append("non-finite-3d-correspondences")
        return result
    baseline_spread = float(np.sqrt(np.mean(np.sum((target - target.mean(axis=0)) ** 2, axis=1))))
    if baseline_spread < 1e-8:
        result["reasons"].append("degenerate-point-spread")
        return result
    for cloud in (source, target):
        singular = np.linalg.svd(cloud - cloud.mean(axis=0), compute_uv=False)
        if singular[0] < 1e-8 or singular[1] / singular[0] < 0.04:
            result["reasons"].append("degenerate-point-spread")
            return result
    threshold = max(1e-7, baseline_spread * 0.025)
    rng = np.random.default_rng(20261104)
    winner = None
    best = (-1, math.inf)
    for _ in range(max_trials):
        chosen = rng.choice(len(matches), 3, replace=False)
        try:
            fit = _similarity(source[chosen], target[chosen], min_spread_ratio=0.03)
        except (ValueError, np.linalg.LinAlgError):
            continue
        predicted = fit["scale"] * (source @ fit["rotation"].T) + fit["translation"]
        errors = np.linalg.norm(predicted - target, axis=1)
        inliers = np.flatnonzero(errors < threshold)
        score = (len(inliers), -float(np.median(errors[inliers])) if len(inliers) else -math.inf)
        if score > (best[0], -best[1]):
            winner = inliers
            best = (score[0], -score[1])
    if winner is None or len(winner) < minimum_inliers or len(winner) / len(matches) < 0.60:
        result["reasons"].append("ransac-insufficient-inlier-consensus")
        return result
    try:
        fit = _similarity(source[winner], target[winner], min_spread_ratio=0.04)
    except (ValueError, np.linalg.LinAlgError):
        result["reasons"].append("degenerate-point-spread")
        return result
    errors = np.linalg.norm(fit["scale"] * (source @ fit["rotation"].T) +
                            fit["translation"] - target, axis=1)
    inliers = np.flatnonzero(errors < threshold)
    if len(inliers) < minimum_inliers:
        result["reasons"].append("ransac-refinement-lost-support")
        return result
    actual_spread = float(np.sqrt(np.mean(np.sum((target[inliers] - target[inliers].mean(axis=0)) ** 2, axis=1))))
    def cloud_spread(values):
        points = np.asarray(values, dtype=float).reshape((-1, 3))
        center = np.median(points, axis=0)
        return float(np.percentile(np.linalg.norm(points - center, axis=1), 75))
    if (baseline_cloud is not None and actual_spread / max(cloud_spread(baseline_cloud), 1e-9) < 0.12) or (
            cohort_cloud is not None and
            (actual_spread / fit["scale"]) / max(cloud_spread(cohort_cloud), 1e-9) < 0.12):
        result["reasons"].append("object-surface-spatial-coverage-insufficient")
    unique_pairs = {tuple(pair) for index in inliers for pair in matches[index]["image_pairs"]}
    if (len(unique_pairs) < 3 or len({pair[0] for pair in unique_pairs}) < 2 or
            len({pair[1] for pair in unique_pairs}) < 2):
        result["reasons"].append("insufficient-independent-cross-view-pairs")
    subject_count = sum(bool(matches[index].get("subject_verified")) for index in inliers)
    if subject_count < max(8, len(inliers) * 0.60):
        result["reasons"].append("subject-surface-evidence-insufficient")
    if fit["spread_ratio"] < 0.04:
        result["reasons"].append("degenerate-point-spread")
    if float(np.percentile(errors[inliers], 95)) > threshold * 0.8:
        result["reasons"].append("large-3d-alignment-residual")
    result.update({"inliers": len(inliers), "inlier_fraction": round(len(inliers) / len(matches), 4),
                   "inlier_point_ids": [[matches[i]["baseline_point_id"], matches[i]["cohort_point_id"]]
                                        for i in inliers],
                   "cross_image_pairs": [list(pair) for pair in sorted(unique_pairs)],
                   "subject_verified_inliers": subject_count,
                   "median_residual_ratio": float(np.median(errors[inliers]) / baseline_spread),
                   "p95_residual_ratio": float(np.percentile(errors[inliers], 95) / baseline_spread),
                   "spatial_spread_ratio": fit["spread_ratio"],
                   "scale": fit["scale"], "rotation": fit["rotation"].tolist(),
                   "translation": fit["translation"].tolist()})
    result["accepted"] = not result["reasons"]
    return result


def validate_cross_reprojection(matches: list[dict], alignment: dict, baseline, cohort,
                                *, maximum_median_px: float = 3.0,
                                maximum_p95_px: float = 6.0) -> dict:
    """Project cross-model matched points in both real calibrated cameras."""
    import numpy as np

    if not alignment.get("accepted"):
        return {"accepted": False, "reason": "alignment-not-accepted"}
    rotation = np.asarray(alignment["rotation"])
    scale = alignment["scale"]
    translation = np.asarray(alignment["translation"])
    inliers = set(map(tuple, alignment["inlier_point_ids"]))
    errors = []
    for row in matches:
        if (row["baseline_point_id"], row["cohort_point_id"]) not in inliers:
            continue
        sample = row["sample"]
        a = baseline.images[sample["baseline_image_id"]]
        b = cohort.images[sample["cohort_image_id"]]
        original = np.asarray(row["baseline_xyz"])
        alternative = np.asarray(row["cohort_xyz"])
        mapped = scale * (rotation @ alternative) + translation
        back = rotation.T @ ((original - translation) / scale)
        for image, point, point_idx in ((a, mapped, sample["baseline_keypoint"]),
                                        (b, back, sample["cohort_keypoint"])):
            observed = np.asarray(image.points2D[point_idx].xy).reshape(2)
            projected = image.project_point(point)
            if projected is None:
                return {"accepted": False, "reason": "point-behind-camera"}
            projected = np.asarray(projected, dtype=float).reshape(2)
            error = float(np.linalg.norm(projected - observed))
            if not math.isfinite(error):
                return {"accepted": False, "reason": "non-finite-reprojection"}
            errors.append(error)
    if not errors:
        return {"accepted": False, "reason": "no-projectable-correspondences"}
    median = float(np.median(errors))
    p95 = float(np.percentile(errors, 95))
    return {"accepted": median <= maximum_median_px and p95 <= maximum_p95_px,
            "median_px": median, "p95_px": p95, "measurements": len(errors),
            "reason": "cross-model-reprojection-mismatch" if
            median > maximum_median_px or p95 > maximum_p95_px else None}


def fuse_disjoint_models(baseline, cohort, alignment: dict, *, pycolmap_module=None):
    """Safely combine genuinely disjoint PyCOLMAP frames and tracks after QA.

    Unlike COLMAP model_merger, this does not rely on common registered frames.
    Existing point observations are not silently welded; every track retains
    its original 2D source evidence and unique new 3D identifier.
    """
    import copy
    import numpy as np

    if not alignment.get("accepted"):
        raise ValueError("Cannot merge disconnected groups without accepted alignment")
    if pycolmap_module is None:
        import pycolmap as pycolmap_module
    ids = set(baseline.reg_image_ids())
    extra = set(cohort.reg_image_ids())
    if not extra or ids & extra or set(baseline.images) & extra:
        raise ValueError("Disconnected merge requires unique image IDs and nonempty camera gain")
    transform = np.column_stack((alignment["scale"] * np.asarray(alignment["rotation"], dtype=float),
                                  np.asarray(alignment["translation"], dtype=float).reshape(3)))
    if transform.shape != (3, 4) or not np.isfinite(transform).all() or alignment["scale"] <= 0:
        raise ValueError("Invalid finite Sim(3) transformation")
    # PyCOLMAP's copy constructor accepts a native Reconstruction on modern
    # builds; a deep copy avoids mutating any in-memory model or saved revision.
    from_candidates = copy.deepcopy(cohort)
    from_candidates.transform(pycolmap_module.Sim3d(transform))
    result = copy.deepcopy(baseline)
    for camera_id, camera in from_candidates.cameras.items():
        if camera_id in result.cameras:
            original = result.cameras[camera_id]
            if (original.model != camera.model or original.width != camera.width or
                    original.height != camera.height or
                    not np.allclose(original.params, camera.params, rtol=0, atol=1e-8)):
                raise ValueError("Conflicting COLMAP camera intrinsics for shared camera ID")
        else:
            result.add_camera(copy.deepcopy(camera))
    for rig_id, rig in from_candidates.rigs.items():
        if rig_id not in result.rigs:
            result.add_rig(copy.deepcopy(rig))
        elif result.rigs[rig_id].todict()["ref_sensor_id"] != rig.todict()["ref_sensor_id"]:
            raise ValueError("Conflicting rig ID between independently solved models")
    included_frames = {from_candidates.images[image_id].frame_id for image_id in extra}
    for frame_id, frame in from_candidates.frames.items():
        if frame_id not in included_frames:
            continue
        if frame_id in result.frames:
            raise ValueError("Conflicting COLMAP frame identifiers")
        prepared = copy.deepcopy(frame)
        prepared.reset_rig_ptr()
        result.add_frame(prepared)
    for image_id in sorted(extra):
        image = copy.deepcopy(from_candidates.images[image_id])
        image.reset_camera_ptr()
        image.reset_frame_ptr()
        for index, feature in enumerate(image.points2D):
            if feature.has_point3D():
                image.reset_point3D_for_point2D(index)
        result.add_image(image)
    for point in from_candidates.points3D.values():
        if not all(element.image_id in extra for element in point.track.elements):
            continue
        track = pycolmap_module.Track()
        for element in point.track.elements:
            track.add_element(element.image_id, element.point2D_idx)
        result.add_point3D(point.xyz, track, point.color)
    if (not result.is_valid() or set(result.reg_image_ids()) != ids | extra or
            result.num_points3D() < baseline.num_points3D()):
        raise ValueError("Combined COLMAP reconstruction fails structural validation")
    return result
