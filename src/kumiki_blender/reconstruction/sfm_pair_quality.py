"""Bounded, read-only geometric quality evidence for COLMAP image pairs.

Verified inlier counts alone favor duplicate screenshots. When a COLMAP
two-view pose and camera intrinsics are available, estimate angular separation
of matched camera rays. Also report match distribution and planar degeneracy.
These are seed-ranking hints, not proof of a valid 3D reconstruction.
"""
from __future__ import annotations

import math
import sqlite3
import struct
from collections import OrderedDict
from pathlib import Path

MAX_IMAGE_ID = 2_147_483_647
MAX_KEYPOINTS = 250_000
MAX_VERIFIED_MATCHES = 500_000
MAX_SAMPLE = 512


def _intrinsics(camera: tuple | None) -> tuple[float, float, float, float, bool] | None:
    if camera is None:
        return None
    model, width, height, blob = camera
    if (not isinstance(model, int) or not isinstance(width, int) or
            not isinstance(height, int) or width <= 0 or height <= 0 or
            not isinstance(blob, bytes) or len(blob) % 8 or not blob):
        return None
    p = struct.unpack(f"<{len(blob) // 8}d", blob)
    # COLMAP camera models: SIMPLE_PINHOLE=0, PINHOLE=1,
    # SIMPLE_RADIAL=2, RADIAL=3, OPENCV=4. Non-pinhole models remain unknown.
    if model in (0, 2, 3) and len(p) >= 3:
        fx = fy = p[0]
        cx, cy = p[1:3]
    elif model in (1, 4) and len(p) >= 4:
        fx, fy, cx, cy = p[:4]
    else:
        return None
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        return None
    # Distortion is not undone here; the result is a conservative angular
    # *proxy*, not calibrated ray triangulation for distorted models.
    return fx, fy, cx, cy, model not in (0, 1)


def _point(blob: bytes, columns: int, index: int) -> tuple[float, float]:
    return struct.unpack_from("<ff", blob, index * columns * 4)


def _grid(points: list[tuple[float, float]], dimensions: tuple[int, int]) -> float:
    width, height = dimensions
    if width <= 0 or height <= 0:
        return 0.0
    cells = {(min(3, int(4 * x / width)), min(3, int(4 * y / height)))
             for x, y in points if math.isfinite(x) and math.isfinite(y) and
             0 <= x < width and 0 <= y < height}
    return round(len(cells) / 16, 3)


def _pose_rotation(qvec: bytes | None, tvec: bytes | None) -> tuple[float, ...] | None:
    if qvec is None or tvec is None or len(qvec) != 32 or len(tvec) != 24:
        return None
    w, x, y, z = struct.unpack("<4d", qvec)
    translation = struct.unpack("<3d", tvec)
    qnorm = math.sqrt(w*w + x*x + y*y + z*z)
    if (not math.isfinite(qnorm) or not 0.8 < qnorm < 1.2 or
            not all(math.isfinite(t) for t in translation) or
            math.sqrt(sum(t*t for t in translation)) < 1e-6):
        return None
    w, x, y, z = (value / qnorm for value in (w, x, y, z))
    # Rotation from camera 1 into camera 2, COLMAP quaternion convention.
    return (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))


def _bearing(point: tuple[float, float], intr: tuple[float, float, float, float, bool]):
    fx, fy, cx, cy, _ = intr
    x, y = (point[0] - cx) / fx, (point[1] - cy) / fy
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    norm = math.sqrt(x*x + y*y + 1.0)
    return (x / norm, y / norm, 1.0 / norm)


def _angle(a, b, rotation) -> float:
    # Compare R12*a against b in camera-2 coordinates. Clamp roundoff.
    transformed = tuple(sum(rotation[row*3 + col] * a[col] for col in range(3))
                        for row in range(3))
    cos_angle = max(-1.0, min(1.0, sum(x*y for x, y in zip(transformed, b))))
    return math.degrees(math.acos(cos_angle))


def assess_verified_pairs(database: Path, edges: list[dict]) -> list[dict]:
    """Add optional geometric QA to edges using a bounded verified-inlier sample.

    Database schemas with no keypoint/pose blobs remain supported. Missing
    evidence is marked `unavailable`, never treated as true zero parallax.
    """
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        image_info = {name: (image_id, camera_id) for image_id, name, camera_id in
                      db.execute("SELECT image_id, name, camera_id FROM images")} if (
                          {"cameras", "keypoints"} <= tables and
                          "camera_id" in {col[1] for col in db.execute("PRAGMA table_info(images)")}
                      ) else {}
        cameras = {}
        if image_info:
            cameras = {row[0]: row[1:] for row in db.execute(
                "SELECT camera_id, model, width, height, params FROM cameras")}
        columns = {col[1] for col in db.execute("PRAGMA table_info(two_view_geometries)")}
        has_geometry_blobs = {"cols", "data", "qvec", "tvec"} <= columns
        cache = OrderedDict()

        def keypoints(image_id: int):
            if image_id not in cache:
                row = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?",
                                 (image_id,)).fetchone()
                if row is None:
                    cache[image_id] = None
                else:
                    count, cols, blob = row
                    if (not isinstance(count, int) or not 0 <= count <= MAX_KEYPOINTS or
                            not isinstance(cols, int) or not 2 <= cols <= 8 or
                            blob is None or len(blob) != count * cols * 4):
                        raise ValueError(f"Invalid COLMAP keypoints for image {image_id}")
                    cache[image_id] = (count, cols, blob)
            cache.move_to_end(image_id)
            if len(cache) > 24:
                cache.popitem(last=False)
            return cache[image_id]

        ranked = []
        for original in edges:
            entry = dict(original)
            quality = {"status": "unavailable / no-verified-pose-or-keypoints",
                       "coverage_a": None, "coverage_b": None,
                       "median_ray_angle_degrees": None,
                       "note": "No reliable triangulation angle from inlier count alone."}
            multiplier = 1.0
            config = original.get("geometry_config")
            if config in (4, 5, 6, 7):
                multiplier *= 0.15
                quality["configuration_warning"] = "planar-panoramic-or-watermark"
            if not has_geometry_blobs or not image_info:
                entry["quality"] = quality
                entry["seed_score"] = round(original["inliers"] * multiplier, 3)
                ranked.append(entry)
                continue
            a_info, b_info = image_info.get(original["a"]), image_info.get(original["b"])
            if a_info is None or b_info is None:
                raise ValueError("Verified pair images not present in COLMAP database")
            a_id, a_camera_id = a_info
            b_id, b_camera_id = b_info
            pair_id = min(a_id, b_id) * MAX_IMAGE_ID + max(a_id, b_id)
            row = db.execute("SELECT rows, cols, data, qvec, tvec FROM two_view_geometries "
                             "WHERE pair_id=?", (pair_id,)).fetchone()
            if row is None:
                raise ValueError(f"Verified pair disappeared from database: {pair_id}")
            n, cols, blob, qvec, tvec = row
            if (n != original["inliers"] or not 1 <= n <= MAX_VERIFIED_MATCHES or
                    cols != 2 or blob is None or len(blob) != n * 8):
                quality["status"] = "unavailable / invalid-inlier-data"
                entry["quality"] = quality
                entry["seed_score"] = round(original["inliers"] * multiplier, 3)
                ranked.append(entry)
                continue
            points_a, points_b = keypoints(a_id), keypoints(b_id)
            intr_a, intr_b = _intrinsics(cameras.get(a_camera_id)), _intrinsics(cameras.get(b_camera_id))
            if not points_a or not points_b or not intr_a or not intr_b:
                entry["quality"] = quality
                entry["seed_score"] = round(original["inliers"] * multiplier, 3)
                ranked.append(entry)
                continue
            widths = (int(cameras[a_camera_id][1]), int(cameras[b_camera_id][1]))
            heights = (int(cameras[a_camera_id][2]), int(cameras[b_camera_id][2]))
            indices = [min(n - 1, int((k + 0.5) * n / min(n, MAX_SAMPLE)))
                       for k in range(min(n, MAX_SAMPLE))]
            matched_a, matched_b = [], []
            for index in indices:
                kp_a, kp_b = struct.unpack_from("<II", blob, index * 8)
                if kp_a < points_a[0] and kp_b < points_b[0]:
                    matched_a.append(_point(points_a[2], points_a[1], kp_a))
                    matched_b.append(_point(points_b[2], points_b[1], kp_b))
            coverage_a = _grid(matched_a, (widths[0], heights[0]))
            coverage_b = _grid(matched_b, (widths[1], heights[1]))
            quality.update({"status": "inlier-distribution-only", "coverage_a": coverage_a,
                            "coverage_b": coverage_b, "sampled_inliers": len(matched_a)})
            # Spatially concentrated matches are weak scene-wide evidence.
            multiplier *= max(0.25, (coverage_a + coverage_b) / 2)
            rotation = _pose_rotation(qvec, tvec)
            if rotation is not None and len(matched_a) >= 15:
                angles = []
                for pos_a, pos_b in zip(matched_a, matched_b):
                    ray_a, ray_b = _bearing(pos_a, intr_a), _bearing(pos_b, intr_b)
                    if ray_a is not None and ray_b is not None:
                        angles.append(_angle(ray_a, ray_b, rotation))
                if len(angles) >= 15:
                    median = sorted(angles)[len(angles)//2]
                    quality["status"] = "relative-pose-bearing-angle-proxy"
                    quality["median_ray_angle_degrees"] = round(median, 3)
                    quality["approximate_distortion"] = intr_a[4] or intr_b[4]
                    multiplier *= max(0.1, min(2.5, median / 6.0))
                    if median < 1.5:
                        quality["configuration_warning"] = "near-zero-bearing-parallax"
            entry["quality"] = quality
            entry["seed_score"] = round(original["inliers"] * multiplier, 3)
            ranked.append(entry)
        return ranked
