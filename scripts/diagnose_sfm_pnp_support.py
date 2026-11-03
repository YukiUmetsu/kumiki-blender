#!/usr/bin/env python3
"""Measure whether verified 2D matches reach existing 3D tracks in COLMAP models.

This is read-only candidate support for registering missing cameras, NOT a PnP
solution, geometric verification of 2D-3D associations, or a surface scan.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
from collections import Counter, defaultdict
from pathlib import Path

MAX_IMAGE_ID = 2_147_483_647
MAX_INLIERS = 500_000
MAX_POINTS = 2_000_000
MAX_KEYPOINTS = 200_000


def _model_tracks(model: Path) -> tuple[dict[int, str], dict[tuple[int, int], int]]:
    """COLMAP images.txt IDs and points3D.txt TRACK[image_id, point2D_idx]."""
    images_file = model / "images.txt"
    points_file = model / "points3D.txt"
    if not images_file.is_file() or not points_file.is_file():
        raise FileNotFoundError(f"Missing COLMAP text model in {model}")
    images = {}
    with images_file.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10 or not parts[0].isdigit():
                raise ValueError(f"Malformed image pose row in {images_file}")
            image_id = int(parts[0])
            images[image_id] = parts[9]
            next(stream, None)  # mandatory alternating POINTS2D[] row
    tracks = {}
    count = 0
    with points_file.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            count += 1
            if count > MAX_POINTS:
                raise ValueError("Sparse model exceeds point limit")
            fields = line.split()
            if len(fields) < 10 or (len(fields) - 8) % 2:
                continue
            point_id = int(fields[0])
            for i in range(8, len(fields), 2):
                image_id, keypoint_index = int(fields[i]), int(fields[i + 1])
                if image_id in images and keypoint_index >= 0:
                    key = (image_id, keypoint_index)
                    if key in tracks and tracks[key] != point_id:
                        raise ValueError("Conflicting 3D track assignments")
                    tracks[key] = point_id
    return images, tracks


def _verified_pairs(db: sqlite3.Connection, known: set[int]) -> list[tuple[int, int, int, list[tuple[int, int]]]]:
    pairs = []
    for pair_id, rows, cols, blob, config in db.execute(
        "SELECT pair_id, rows, cols, data, config FROM two_view_geometries"
    ):
        if not isinstance(rows, int) or rows < 1 or config in (0, 1):
            continue
        first, second = divmod(pair_id, MAX_IMAGE_ID)
        if first not in known or second not in known:
            continue
        if rows > MAX_INLIERS or cols != 2 or blob is None or len(blob) != rows * 8:
            raise ValueError(f"Invalid COLMAP verified-inlier BLOB for pair {pair_id}")
        pairs.append((first, second, config, list(struct.iter_unpack("<II", blob))))
    return pairs


def _grid(db: sqlite3.Connection, image_id: int, indices: list[int],
          size: tuple[int, int] | None) -> float | None:
    if size is None:
        return None
    width, height = size
    if width <= 0 or height <= 0:
        return None
    row = db.execute(
        "SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)
    ).fetchone()
    if row is None:
        return None
    count, cols, blob = row
    if (not isinstance(count, int) or count > MAX_KEYPOINTS or count < 0 or
            not isinstance(cols, int) or not 2 <= cols <= 8 or
            blob is None or len(blob) != count * cols * 4):
        raise ValueError(f"Invalid COLMAP keypoint BLOB for image {image_id}")
    cells = set()
    for index in indices:
        if index >= count or index < 0:
            continue
        x, y = struct.unpack_from("<ff", blob, index * cols * 4)
        if math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height:
            cells.add((min(3, int(4 * x / width)), min(3, int(4 * y / height))))
    return round(len(cells) / 16, 3)


def _source_names(sparse: Path, source_index: Path | None) -> dict[str, str]:
    index = sparse / "staged-reference-index.json"
    if not index.is_file():
        return {}
    staged = json.loads(index.read_text(encoding="utf-8"))["images"]
    names = {item["filename"]: item.get("original_filename") or item["filename"]
             for item in staged}
    if source_index is not None:
        source = json.loads(source_index.read_text(encoding="utf-8"))["images"]
        original_by_hash = {item["staged_sha256"]: item["filename"] for item in source}
        for item in staged:
            mapped = original_by_hash.get(item.get("original_sha256"))
            if mapped:
                names[item["filename"]] = mapped
    return names


def analyze(revision: Path, source_index: Path | None = None) -> dict:
    sparse = revision.resolve()
    if (sparse / "sparse-reconstruction").is_dir():
        sparse /= "sparse-reconstruction"
    db_path = sparse / "database.db"
    if not db_path.is_file() or db_path.is_symlink():
        raise FileNotFoundError(f"Missing or unsafe COLMAP database: {db_path}")
    models_dir = sparse / "text-models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Missing COLMAP text models: {models_dir}")
    original_names = _source_names(sparse, source_index)
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as db:
        info = {row[1] for row in db.execute("PRAGMA table_info(images)")}
        if "image_id" not in info or "name" not in info:
            raise ValueError("Invalid COLMAP images table")
        has_camera = "camera_id" in info
        image_rows = db.execute(
            "SELECT image_id, name, camera_id FROM images" if has_camera
            else "SELECT image_id, name, NULL FROM images"
        ).fetchall()
        names = {image_id: name for image_id, name, _ in image_rows}
        sizes = {}
        has_cameras = db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='cameras' AND type='table'"
        ).fetchone() is not None
        if has_cameras:
            camera_sizes = {
                camera_id: (width, height)
                for camera_id, width, height in db.execute(
                    "SELECT camera_id, width, height FROM cameras"
                )
            }
            sizes = {image_id: camera_sizes.get(camera_id)
                     for image_id, _, camera_id in image_rows}
        pairs = _verified_pairs(db, set(names))
        models = []
        for folder in sorted(models_dir.iterdir()):
            if not folder.is_dir() or folder.is_symlink():
                continue
            registered, tracks = _model_tracks(folder)
            if not registered:
                continue
            # The text model must refer to the same imported image IDs/names.
            for image_id, name in registered.items():
                if names.get(image_id) != name:
                    raise ValueError(f"COLMAP model/database image mismatch: {image_id}")
            model_ids = set(registered)
            support = defaultdict(lambda: defaultdict(set))
            inlier_totals = Counter()
            track_hits = Counter()
            neighbors = defaultdict(set)
            pair_count = Counter()
            geometry_types = defaultdict(Counter)
            for first, second, config, matches in pairs:
                if (first in model_ids) == (second in model_ids):
                    continue
                reg, missing = (first, second) if first in model_ids else (second, first)
                neighbors[missing].add(reg)
                pair_count[missing] += 1
                inlier_totals[missing] += len(matches)
                geometry_types[missing][str(config)] += 1
                for idx_first, idx_second in matches:
                    kp_reg, kp_missing = ((idx_first, idx_second) if reg == first
                                          else (idx_second, idx_first))
                    point_id = tracks.get((reg, kp_reg))
                    if point_id is not None:
                        track_hits[missing] += 1
                        support[missing][kp_missing].add(point_id)
            missing_images = []
            for image_id in sorted(set(names) - model_ids, key=lambda i: names[i]):
                candidate = support[image_id]
                conflicts = sum(len(points) > 1 for points in candidate.values())
                unambiguous = {idx: next(iter(points)) for idx, points in candidate.items()
                               if len(points) == 1}
                points_to_indices = defaultdict(list)
                for idx, point_id in unambiguous.items():
                    points_to_indices[point_id].append(idx)
                # Conservative: no repeated image feature and no repeated 3D point.
                one_to_one = [indices[0] for indices in points_to_indices.values()
                              if len(indices) == 1]
                missing_images.append({
                    "filename": names[image_id],
                    "original_filename": original_names.get(names[image_id], names[image_id]),
                    "registered_neighbors": len(neighbors[image_id]),
                    "verified_bridge_pairs": pair_count[image_id],
                    "verified_bridge_inliers": inlier_totals[image_id],
                    "inliers_touching_existing_3d_tracks": track_hits[image_id],
                    "candidate_query_keypoints": len(candidate),
                    "ambiguous_query_keypoints": conflicts,
                    "unambiguous_query_keypoints": len(unambiguous),
                    "distinct_existing_3d_points": len(points_to_indices),
                    "conservative_one_to_one_2d3d": len(one_to_one),
                    "one_to_one_4x4_image_coverage": _grid(
                        db, image_id, one_to_one, sizes.get(image_id)),
                    "verified_geometry_config_counts": dict(geometry_types[image_id]),
                })
            missing_images.sort(key=lambda item: (
                -item["conservative_one_to_one_2d3d"], -item["registered_neighbors"],
                item["filename"]))
            models.append({
                "model": folder.name, "registered": len(registered),
                "existing_3d_track_observations": len(tracks),
                "unregistered_images": missing_images,
            })
    return {
        "schema_version": 1,
        "status": "candidate-2d3d-support-only",
        "source_revision": str(revision.resolve()),
        "models": models,
        "interpretation": "Counts are candidate associations from COLMAP's verified 2D-2D "
                          "matches and existing reconstructed 3D tracks. They are NOT "
                          "PnP inliers or verified camera poses. Few candidates suggest "
                          "weak overlap with triangulated tracks; many candidates with "
                          "failed registration suggest 2D-3D inconsistency, calibration, "
                          "or mapper initialization problems. No 3D mesh is inferred.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", type=Path)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.revision, args.source_index)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            with args.json_output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
    except (OSError, ValueError, sqlite3.DatabaseError, KeyError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    for model in result["models"]:
        print(f"Model {model['model']}: {model['registered']} cameras, "
              f"{model['existing_3d_track_observations']} 3D track observations")
        for photo in model["unregistered_images"]:
            print(f"  {photo['filename']} (orig {photo['original_filename']}): "
                  f"{photo['verified_bridge_inliers']} bridge inliers; "
                  f"{photo['inliers_touching_existing_3d_tracks']} touch 3D tracks; "
                  f"{photo['conservative_one_to_one_2d3d']} one-to-one candidates; "
                  f"4x4 coverage={photo['one_to_one_4x4_image_coverage']}; "
                  f"ambiguous={photo['ambiguous_query_keypoints']}")
    if args.json_output:
        print(f"PnP support report: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
