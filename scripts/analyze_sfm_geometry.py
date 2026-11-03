#!/usr/bin/env python3
"""Read-only, scale-invariant geometric diagnostics for COLMAP sparse models.

Reconstructed camera rays can reveal near-zero triangulation angles even when
feature inlier counts are high. These *estimated* angles cannot independently
validate the underlying camera solution or a monument's geometry.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

MAX_TRACK_VIEWS = 128
MAX_POINT_ROWS = 2_000_000


def _records(path: Path):
    if not path.is_file() or path.is_symlink():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith("#"):
                yield line.split()


def _camera_center(q: tuple[float, ...], t: tuple[float, ...]) -> tuple[float, float, float]:
    """COLMAP stores world-to-camera (qw,qx,qy,qz), t. C = -R.T @ t."""
    if len(q) != 4 or len(t) != 3 or not all(math.isfinite(v) for v in (*q, *t)):
        raise ValueError("Non-finite or malformed camera pose")
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-12:
        raise ValueError("Zero camera quaternion")
    w, x, y, z = (v / norm for v in q)
    rotation = (
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    )
    return tuple(-sum(rotation[row][col] * t[row] for row in range(3))
                 for col in range(3))


def _images(model: Path) -> dict[int, dict]:
    path = model / "images.txt"
    if not path.is_file() or path.is_symlink():
        return {}
    result = {}
    # images.txt alternates a pose row and a row of 2D keypoints; the latter
    # can contain only integers and must never be interpreted as a camera.
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10 or not parts[0].isdigit() or not parts[8].isdigit():
                raise ValueError(f"Malformed COLMAP pose row: {path}")
            camera_id = int(parts[8])
            image_id = int(parts[0])
            center = _camera_center(tuple(float(v) for v in parts[1:5]),
                                    tuple(float(v) for v in parts[5:8]))
            result[image_id] = {"name": parts[9], "camera_id": camera_id, "center": center}
            next(stream, None)
    return result


def _intrinsics(model: Path, images: dict[int, dict]) -> dict:
    used = {entry["camera_id"] for entry in images.values()}
    cameras = []
    for parts in _records(model / "cameras.txt"):
        if len(parts) < 5 or not parts[0].isdigit() or int(parts[0]) not in used:
            continue
        model_name, width, height = parts[1], int(parts[2]), int(parts[3])
        params = [float(value) for value in parts[4:]]
        if not params or width < 1 or height < 1 or not all(math.isfinite(p) for p in params):
            continue
        if model_name.startswith("SIMPLE_") or model_name in ("RADIAL", "RADIAL_FISHEYE"):
            focal = params[0]
        elif len(params) >= 2:
            focal = (params[0] + params[1]) / 2
        else:
            focal = None
        cameras.append({
            "camera_id": int(parts[0]), "model": model_name,
            "dimensions": [width, height],
            "focal_to_max_dimension": (round(focal / max(width, height), 4)
                                       if focal is not None else None),
            "images_using_camera": sum(entry["camera_id"] == int(parts[0])
                                       for entry in images.values()),
        })
    return {"camera_count": len(used), "camera_models": dict(sorted(Counter(
        item["model"] for item in cameras).items())), "cameras": cameras}


def _angle(a: tuple[float, ...], b: tuple[float, ...]) -> float | None:
    la = math.sqrt(sum(v*v for v in a))
    lb = math.sqrt(sum(v*v for v in b))
    if la < 1e-12 or lb < 1e-12:
        return None
    cosine = sum(x*y for x, y in zip(a, b)) / (la * lb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _triangulation(model: Path, images: dict[int, dict]) -> dict:
    points = model / "points3D.txt"
    if not points.is_file() or points.is_symlink():
        return {"status": "missing-points3D", "triangulated_points_with_two_cameras": 0}
    maxima, tracks, total = [], [], 0
    for fields in _records(points):
        total += 1
        if total > MAX_POINT_ROWS:
            raise ValueError("Sparse model exceeds diagnostic point limit")
        if len(fields) < 12 or (len(fields) - 8) % 2:
            continue
        try:
            xyz = tuple(float(value) for value in fields[1:4])
            if not all(math.isfinite(value) for value in xyz):
                continue
            image_ids = {int(fields[i]) for i in range(8, len(fields), 2)}
        except ValueError:
            continue
        if len(image_ids) < 2:
            continue
        # Per-point maximum triangulation angle is more informative than the
        # median of all pairwise angles for multi-view tracks.
        rays = [tuple(xyz[axis] - images[image_id]["center"][axis]
                      for axis in range(3))
                for image_id in sorted(image_ids) if image_id in images][:MAX_TRACK_VIEWS]
        if len(rays) < 2:
            continue
        angles = (_angle(a, b) for i, a in enumerate(rays) for b in rays[i+1:])
        valid = [angle for angle in angles if angle is not None]
        if valid:
            maxima.append(max(valid))
            tracks.append(len(rays))
    if not maxima:
        return {"status": "no-triangulable-tracks", "total_point_rows": total,
                "triangulated_points_with_two_cameras": 0}
    return {
        "status": "estimated-poses-only",
        "total_point_rows": total,
        "triangulated_points_with_two_cameras": len(maxima),
        "median_max_triangulation_angle_deg": round(statistics.median(maxima), 3),
        "p10_max_triangulation_angle_deg": round(sorted(maxima)[int(0.1 * (len(maxima)-1))], 3),
        "percent_points_below_1_deg": round(100 * sum(x < 1 for x in maxima) / len(maxima), 1),
        "percent_points_below_2_deg": round(100 * sum(x < 2 for x in maxima) / len(maxima), 1),
        "percent_points_below_5_deg": round(100 * sum(x < 5 for x in maxima) / len(maxima), 1),
        "median_track_length": statistics.median(tracks),
    }


def analyze(sparse: Path) -> dict:
    sparse = sparse.resolve()
    models_root = sparse / "text-models"
    if not models_root.is_dir():
        return {"status": "no-text-models", "models": []}
    models = []
    for folder in sorted(models_root.iterdir()):
        if not folder.is_dir() or folder.is_symlink():
            continue
        poses = _images(folder)
        if not poses:
            continue
        models.append({
            "model": folder.name,
            "registered_images": len(poses),
            "intrinsics": _intrinsics(folder, poses),
            "triangulation": _triangulation(folder, poses),
        })
    return {
        "status": "estimated-camera-geometry",
        "models": models,
        "interpretation": "Angles are derived from estimated COLMAP poses and tracks. "
                          "Small angles imply weak reconstructed depth baselines; "
                          "large angles do not prove correct camera geometry, "
                          "physical scale, or monument fidelity.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", type=Path)
    args = parser.parse_args()
    sparse = args.revision / "sparse-reconstruction"
    if not sparse.is_dir():
        sparse = args.revision
    try:
        result = analyze(sparse)
    except (OSError, ValueError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
