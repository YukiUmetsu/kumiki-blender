#!/usr/bin/env python3
"""Visualize actual COLMAP verified 2D correspondences to diagnose bad SfM poses.

A large inlier count alone does not establish useful *subject* geometry.
This script plots inlier locations in both source photographs, emphasizing
registered-to-unregistered pairs and recording coverage. It does not modify
the COLMAP database or infer camera poses, depth, or a 3D mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sqlite3
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

MAX_IMAGE_ID = 2_147_483_647
MAX_KEYPOINTS = 200_000
MAX_INLIERS = 500_000


def _diagnostics():
    script = Path(__file__).with_name("diagnose_splat_registration.py")
    spec = importlib.util.spec_from_file_location("diagnose_splat_registration", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing registration diagnostic script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sparse_path(revision: Path) -> Path:
    revision = revision.resolve()
    return revision / "sparse-reconstruction" if (revision / "sparse-reconstruction").is_dir() else revision


def _read_keypoints(db: sqlite3.Connection, image_id: int) -> list[tuple[float, float]]:
    row = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,)).fetchone()
    if row is None:
        raise ValueError(f"Missing COLMAP keypoints for image ID {image_id}")
    count, columns, data = row
    if (not isinstance(count, int) or not 0 <= count <= MAX_KEYPOINTS or
            not isinstance(columns, int) or not 2 <= columns <= 8 or
            data is None or len(data) != count * columns * 4):
        raise ValueError(f"Invalid keypoint BLOB for image ID {image_id}")
    return [struct.unpack_from("<ff", data, i * columns * 4) for i in range(count)]


def _decode_inliers(count: int, columns: int, data: bytes | None) -> list[tuple[int, int]]:
    if (not isinstance(count, int) or not 0 <= count <= MAX_INLIERS or
            columns != 2 or data is None or len(data) != count * 8):
        raise ValueError("Invalid COLMAP verified-inlier BLOB")
    return list(struct.iter_unpack("<II", data))


def _grid_coverage(points: list[tuple[float, float]], dimensions: tuple[int, int]) -> float:
    width, height = dimensions
    if not points or width <= 0 or height <= 0:
        return 0.0
    occupied = {
        (min(3, max(0, int(x / width * 4))),
         min(3, max(0, int(y / height * 4))))
        for x, y in points if math.isfinite(x) and math.isfinite(y)
    }
    return round(len(occupied) / 16, 3)


def _photo(sparse: Path, name: str, box: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    # COLMAP image names must remain flat. Never permit path traversal.
    if Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Unsafe COLMAP photo name: {name}")
    path = sparse / "images" / name
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"COLMAP staged photo not found: {path}")
    with Image.open(path) as source:
        size = source.size
        thumb = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), box)
    return thumb, size


def _draw_pair(
    sparse: Path, pair: dict, keypoints_a: list[tuple[float, float]],
    keypoints_b: list[tuple[float, float]], indices: list[tuple[int, int]],
    *, max_draw: int,
) -> tuple[Image.Image, dict]:
    left, dims_a = _photo(sparse, pair["a"], (450, 350))
    right, dims_b = _photo(sparse, pair["b"], (450, 350))
    width, height = 1000, 440
    canvas = Image.new("RGB", (width, height), (26, 29, 34))
    offsets = ((20 + (450 - left.width) // 2, 45 + (350 - left.height) // 2),
               (530 + (450 - right.width) // 2, 45 + (350 - right.height) // 2))
    canvas.paste(left, offsets[0])
    canvas.paste(right, offsets[1])
    inlier_points_a, inlier_points_b = [], []
    valid = []
    for first, second in indices:
        if first >= len(keypoints_a) or second >= len(keypoints_b):
            continue
        a, b = keypoints_a[first], keypoints_b[second]
        if all(math.isfinite(value) for value in (*a, *b)):
            inlier_points_a.append(a)
            inlier_points_b.append(b)
            valid.append((a, b))
    if not valid:
        raise ValueError(f"No valid inlier coordinates: {pair['a']} / {pair['b']}")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rng = random.Random(pair["pair_id"])
    sample = rng.sample(range(len(valid)), min(len(valid), max_draw))
    for index in sample:
        a, b = valid[index]
        x1 = offsets[0][0] + a[0] / dims_a[0] * left.width
        y1 = offsets[0][1] + a[1] / dims_a[1] * left.height
        x2 = offsets[1][0] + b[0] / dims_b[0] * right.width
        y2 = offsets[1][1] + b[1] / dims_b[1] * right.height
        if not (0 <= a[0] < dims_a[0] and 0 <= a[1] < dims_a[1] and
                0 <= b[0] < dims_b[0] and 0 <= b[1] < dims_b[1]):
            continue
        draw.line(((x1, y1), (x2, y2)), fill=(64, 222, 235, 65), width=1)
        draw.ellipse((x1 - 2, y1 - 2, x1 + 2, y1 + 2), fill=(255, 210, 72, 220))
        draw.ellipse((x2 - 2, y2 - 2, x2 + 2, y2 + 2), fill=(255, 210, 72, 220))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), pair["a"] + (" [R]" if pair["registered_a"] else " [unregistered]"),
              fill=(236, 236, 236))
    draw.text((530, 12), pair["b"] + (" [R]" if pair["registered_b"] else " [unregistered]"),
              fill=(236, 236, 236))
    cover_a = _grid_coverage(inlier_points_a, dims_a)
    cover_b = _grid_coverage(inlier_points_b, dims_b)
    draw.text((20, 403),
              f"{pair['inliers']} verified inliers | 4x4 coverage {cover_a:.0%} / {cover_b:.0%}",
              fill=(255, 210, 72))
    return canvas, {"coverage_a": cover_a, "coverage_b": cover_b,
                    "valid_inliers": len(valid), "sampled_lines": len(sample),
                    "dimensions_a": dims_a, "dimensions_b": dims_b}


def _source_names(sparse: Path, source_index: Path | None) -> dict[str, str]:
    names = {}
    entries = []
    staged = sparse / "staged-reference-index.json"
    if staged.is_file():
        entries = json.loads(staged.read_text(encoding="utf-8"))["images"]
        names = {item["filename"]: item.get("original_filename", item["filename"])
                 for item in entries}
    if source_index:
        # Optional mapping from cohort-01/source-index.json to ORIGINAL staged IDs.
        provenance = json.loads(source_index.read_text(encoding="utf-8"))["images"]
        original_by_hash = {item["staged_sha256"]: item["filename"] for item in provenance}
        for entry in entries:
            original = original_by_hash.get(entry["original_sha256"])
            if original:
                names[entry["filename"]] = original
    return names


def audit(revision: Path, output: Path, *, max_pairs: int = 8,
          max_draw: int = 70, min_inliers: int = 15,
          source_index: Path | None = None) -> dict:
    if not 1 <= max_pairs <= 24 or not 1 <= max_draw <= 200 or min_inliers < 1:
        raise ValueError("max-pairs 1–24, max-draw 1–200, min-inliers >=1 required")
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"Choose a new QA directory: {output}")
    sparse = _sparse_path(revision)
    db_path = sparse / "database.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"Missing COLMAP match database: {db_path}")
    registered = _diagnostics().registered_images(sparse)
    original_names = _source_names(sparse, source_index)
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as db:
        images = dict(db.execute("SELECT image_id, name FROM images"))
        if not images:
            raise ValueError("No imported COLMAP images")
        pairs = []
        for pair_id, count, columns, data, config in db.execute(
            "SELECT pair_id, rows, cols, data, config FROM two_view_geometries"
        ):
            if count is None or count < min_inliers or config in (0, 1):
                continue
            a_id, b_id = divmod(pair_id, MAX_IMAGE_ID)
            if a_id not in images or b_id not in images:
                continue
            if columns != 2 or data is None or len(data) != count * 8:
                continue
            a, b = images[a_id], images[b_id]
            ra, rb = a in registered, b in registered
            category = "bridge" if ra != rb else "registered" if ra else "unregistered"
            pairs.append({"pair_id": pair_id, "a": a, "b": b,
                          "a_id": a_id, "b_id": b_id,
                          "inliers": count, "geometry_config": config,
                          "registered_a": ra, "registered_b": rb,
                          "category": category})
        if not pairs:
            raise ValueError("No verified inlier correspondences with readable BLOBs")
        # Prioritize missing-camera bridges, while retaining established
        # registered and unregistered examples for comparison.
        groups = {category: sorted(
            (pair for pair in pairs if pair["category"] == category),
            key=lambda pair: (-pair["inliers"], pair["a"], pair["b"])
        ) for category in ("bridge", "registered", "unregistered")}
        quotas = {"bridge": max(1, max_pairs - 2 * max(1, max_pairs // 4)),
                  "registered": max(1, max_pairs // 4),
                  "unregistered": max(1, max_pairs // 4)}
        selected, seen = [], set()
        for category in ("bridge", "registered", "unregistered"):
            for pair in groups[category][:quotas[category]]:
                if len(selected) >= max_pairs:
                    break
                selected.append(pair)
                seen.add(pair["pair_id"])
        if len(selected) < max_pairs:
            for pair in sorted(pairs, key=lambda p: (-p["inliers"], p["a"], p["b"])):
                if pair["pair_id"] not in seen:
                    selected.append(pair)
                    seen.add(pair["pair_id"])
                    if len(selected) == max_pairs:
                        break
        output.mkdir(parents=True, exist_ok=False)
        thumbnails, records = [], []
        keypoints = {}
        for pair in selected:
            try:
                for image_id in (pair["a_id"], pair["b_id"]):
                    if image_id not in keypoints:
                        keypoints[image_id] = _read_keypoints(db, image_id)
                row = db.execute(
                    "SELECT rows, cols, data FROM two_view_geometries WHERE pair_id=?",
                    (pair["pair_id"],)
                ).fetchone()
                indices = _decode_inliers(*row)
                thumb, coverage = _draw_pair(
                    sparse, pair, keypoints[pair["a_id"]], keypoints[pair["b_id"]],
                    indices, max_draw=max_draw,
                )
            except (OSError, ValueError, struct.error) as error:
                records.append({**pair, "status": "unrenderable", "error": str(error)[:240]})
                continue
            record = {**pair, **coverage, "status": "rendered",
                      "original_a": original_names.get(pair["a"], pair["a"]),
                      "original_b": original_names.get(pair["b"], pair["b"])}
            records.append(record)
            thumbnails.append(thumb)
        if not thumbnails:
            raise ValueError("No inlier pairs could be rendered")
        columns, tile_w, tile_h = 2, 1000, 440
        sheet = Image.new("RGB", (columns * tile_w,
                                math.ceil(len(thumbnails) / columns) * tile_h),
                          (26, 29, 34))
        for index, thumb in enumerate(thumbnails):
            sheet.paste(thumb, ((index % columns) * tile_w,
                                (index // columns) * tile_h))
        sheet.save(output / "match-contact-sheet.jpg", quality=90)
        report = {
            "schema_version": 1, "source_revision": str(revision.resolve()),
            "registered_in_largest_model": len(registered),
            "verified_pairs": len(pairs),
            "categories": {category: len(groups[category]) for category in groups},
            "pairs": records,
            "interpretation": "Inlier lines and 4x4 image coverage show 2D matching only. "
                              "Check whether features lie on the Sphinx rather than "
                              "pyramids, ground, sky or image cutout edges. "
                              "These metrics do NOT establish parallax, camera poses, "
                              "surface geometry, or monument fidelity.",
            "source_rights": "Private local photographs; distribution not verified.",
        }
        (output / "match-qa.json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-index", type=Path,
                        help="Optional cohort source-index.json for original photo IDs")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--max-draw", type=int, default=70)
    parser.add_argument("--min-inliers", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        report = audit(args.revision, args.output, max_pairs=args.max_pairs,
                       max_draw=args.max_draw, min_inliers=args.min_inliers,
                       source_index=args.source_index)
    except (OSError, ValueError, KeyError, sqlite3.DatabaseError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Verified pairs: {report['verified_pairs']} | "
          f"bridge pairs: {report['categories']['bridge']} | "
          f"rendered: {sum(p['status'] == 'rendered' for p in report['pairs'])}")
    print(f"Match QA: {args.output / 'match-contact-sheet.jpg'}")
    print(f"JSON: {args.output / 'match-qa.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
