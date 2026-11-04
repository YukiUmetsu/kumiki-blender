"""Conservative support from verified 2D matches to existing COLMAP 3D tracks."""
from __future__ import annotations

import sqlite3
import struct
from collections import defaultdict
from pathlib import Path

MAX_IMAGE_ID = 2_147_483_647
MAX_POINTS = 2_000_000
MAX_INLIERS = 500_000
MAX_TRACK_OBSERVATIONS = 1_000_000


def _tracks(path: Path) -> dict[tuple[int, int], int] | None:
    if path.is_symlink() or not path.is_file():
        return None
    found = {}
    observations = 0
    with path.open(encoding="utf-8") as source:
        count = 0
        for line in source:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            count += 1
            if count > MAX_POINTS:
                raise ValueError("COLMAP 3D track table exceeds safety limit")
            fields = line.split()
            if len(fields) < 10 or (len(fields) - 8) % 2:
                continue
            point_id = int(fields[0])
            for index in range(8, len(fields), 2):
                observations += 1
                if observations > MAX_TRACK_OBSERVATIONS:
                    raise ValueError("COLMAP source track observations exceed memory guard")
                image_id, kp_index = int(fields[index]), int(fields[index + 1])
                key = (image_id, kp_index)
                if key in found and found[key] != point_id:
                    raise ValueError("COLMAP source model assigns conflicting 3D tracks")
                found[key] = point_id
    return found


def audit_2d3d_support(sparse: Path, model: int, registered: set[str],
                       missing: set[str]) -> dict[str, dict]:
    """For each missing photo, count unambiguous one-to-one existing 3D tracks.

    This is only an association count. No new camera pose or PnP inliers are
    inferred, and a large count does not establish correct object geometry.
    """
    tracks = _tracks(sparse / "text-models" / str(model) / "points3D.txt")
    if tracks is None:
        return {name: {"status": "unavailable / no-text-points3D",
                       "conservative_one_to_one_2d3d": None} for name in sorted(missing)}
    with sqlite3.connect((sparse / "database.db").as_uri() + "?mode=ro", uri=True) as db:
        by_id = dict(db.execute("SELECT image_id, name FROM images"))
        if "cols" not in {col[1] for col in db.execute("PRAGMA table_info(two_view_geometries)")}:
            return {name: {"status": "unavailable / no-inlier-blobs",
                           "conservative_one_to_one_2d3d": None} for name in sorted(missing)}
        reg_ids = {image_id for image_id, name in by_id.items() if name in registered}
        missing_ids = {image_id for image_id, name in by_id.items() if name in missing}
        observations = defaultdict(lambda: defaultdict(set))
        bridge_count = defaultdict(int)
        for pair_id, rows, cols, blob, config in db.execute(
                "SELECT pair_id, rows, cols, data, config FROM two_view_geometries"):
            first, second = divmod(pair_id, MAX_IMAGE_ID)
            if (not isinstance(rows, int) or rows <= 0 or config in (0, 1) or
                    (first in reg_ids) == (second in reg_ids) or
                    not (first in missing_ids or second in missing_ids)):
                continue
            if rows > MAX_INLIERS or cols != 2 or blob is None or len(blob) != rows * 8:
                raise ValueError(f"Invalid verified 2D match blob: pair {pair_id}")
            reg, miss = (first, second) if first in reg_ids else (second, first)
            bridge_count[miss] += 1
            for left, right in struct.iter_unpack("<II", blob):
                kp_reg, kp_missing = (left, right) if reg == first else (right, left)
                point_id = tracks.get((reg, kp_reg))
                if point_id is not None:
                    observations[miss][kp_missing].add(point_id)
        report = {}
        for image_id in sorted(missing_ids):
            candidates = observations[image_id]
            unambiguous = {point: next(iter(ids)) for point, ids in candidates.items()
                           if len(ids) == 1}
            point_frequency = defaultdict(int)
            for point_id in unambiguous.values():
                point_frequency[point_id] += 1
            unique = sum(point_frequency[point_id] == 1 for point_id in unambiguous.values())
            report[by_id[image_id]] = {
                "status": "2d3d-candidate-tracks-only",
                "verified_bridge_pairs": bridge_count[image_id],
                "query_keypoints_touching_3d": len(candidates),
                "ambiguous_query_keypoints": sum(len(ids) > 1 for ids in candidates.values()),
                "conservative_one_to_one_2d3d": unique,
            }
    return report
