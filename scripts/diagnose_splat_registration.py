#!/usr/bin/env python3
"""Explain failed COLMAP registration from a Kumiki splat revision (read-only).

Requires Python standard library only. No image data or model files are modified.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict, deque
from pathlib import Path

MAX_IMAGE_ID = 2_147_483_647


def registered_images(sparse: Path) -> set[str]:
    """Read COLMAP image poses, skipping their following 2D observations rows."""
    models = sparse / "text-models"
    if not models.is_dir():
        return set()
    best: set[str] = set()
    for model in models.iterdir():
        path = model / "images.txt"
        if not path.is_file():
            continue
        found: set[str] = set()
        lines = iter(path.read_text(encoding="utf-8").splitlines())
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 10 and fields[0].isdigit() and fields[8].isdigit():
                found.add(fields[9])
                next(lines, None)
        if len(found) > len(best):
            best = found
    return best


def _has_table(db: sqlite3.Connection, table: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def analyze(root: Path, min_inliers: int) -> dict:
    sparse = root / "sparse-reconstruction" if (root / "sparse-reconstruction").is_dir() else root
    db_path = sparse / "database.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"COLMAP database not found: {db_path}")
    registered = registered_images(sparse)
    staged_index = sparse / "staged-reference-index.json"
    staged = {}
    if staged_index.is_file():
        data = json.loads(staged_index.read_text(encoding="utf-8"))
        staged = {entry["filename"]: entry for entry in data["images"]}
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as db:
        names = dict(db.execute("SELECT image_id, name FROM images"))
        keypoints = dict(db.execute("SELECT image_id, rows FROM keypoints")) if _has_table(db, "keypoints") else {}
        if not _has_table(db, "two_view_geometries"):
            raise ValueError("COLMAP database has no two_view_geometries table")
        pairs = []
        neighbors = defaultdict(set)
        inlier_totals = defaultdict(int)
        for pair_id, rows, config in db.execute("SELECT pair_id, rows, config FROM two_view_geometries"):
            # Undefined and degenerate geometries do not establish usable overlap.
            if not rows or config in (0, 1):
                continue
            first, second = divmod(pair_id, MAX_IMAGE_ID)
            if first not in names or second not in names:
                continue
            pairs.append((rows, names[first], names[second], config))
            inlier_totals[first] += rows
            inlier_totals[second] += rows
            if rows >= min_inliers:
                neighbors[first].add(second)
                neighbors[second].add(first)
    remaining = set(names)
    components = []
    while remaining:
        start = remaining.pop()
        seen = {start}
        queue = deque([start])
        while queue:
            for neighbor in neighbors[queue.popleft()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    remaining.discard(neighbor)
                    queue.append(neighbor)
        components.append(sorted(names[image_id] for image_id in seen))
    components.sort(key=len, reverse=True)
    imported = set(names.values())
    omitted = []
    for filename in sorted(staged.keys() - imported):
        photo = sparse / "images" / filename
        omitted.append({
            "filename": filename, "extension": Path(filename).suffix.lower(),
            "staged_file_exists": photo.is_file(),
            "staged_bytes": photo.stat().st_size if photo.is_file() else None,
        })
    extraction_log = sparse / "logs" / "01-feature_extractor.log"
    warnings = []
    if extraction_log.is_file():
        for line in extraction_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(word in line.lower() for word in ("warn", "error", "fail", "skip", "cannot", "could not", "unsupported", "read image")):
                warnings.append(line.strip()[:400])
    images = [
        {"filename": name, "keypoints": keypoints.get(image_id, 0),
         "verified_inliers_across_pairs": inlier_totals[image_id],
         "strong_neighbors": len(neighbors[image_id]),
         "registered": name in registered}
        for image_id, name in sorted(names.items(), key=lambda item: item[1])
    ]
    return {
        "photos": len(names), "registered": len(registered),
        "staged_photos": len(staged) if staged else None,
        "missing_from_database": omitted,
        "feature_extraction_warnings": warnings[-30:],
        "min_inliers_for_diagnostic_edge": min_inliers,
        "strong_pair_count": sum(rows >= min_inliers for rows, *_ in pairs),
        "connected_component_sizes": [len(group) for group in components],
        "connected_components": components,
        "registered_filenames": sorted(registered),
        "weak_or_isolated": [image for image in images if image["strong_neighbors"] == 0],
        "images": images,
        "strongest_pairs": [
            {"inliers": rows, "a": first, "b": second, "geometry_config": config}
            for rows, first, second, config in sorted(pairs, reverse=True)[:12]
        ],
        "notes": "Feature matches are diagnostics, not proof of correct camera geometry. Background-only matches may be misleading.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", type=Path, help="Failed splat revision or sparse-reconstruction directory")
    parser.add_argument("--min-inliers", type=int, default=15, help="Diagnostic connectivity threshold (default 15)")
    parser.add_argument("--json", action="store_true", help="Print complete JSON diagnostics")
    args = parser.parse_args()
    if args.min_inliers < 1:
        parser.error("--min-inliers must be positive")
    try:
        result = analyze(args.revision.resolve(), args.min_inliers)
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"Registered: {result['registered']}/{result['photos']} database images")
    if result["staged_photos"] is not None:
        print(f"Staged photos: {result['staged_photos']}; missing from COLMAP database: {len(result['missing_from_database'])}")
        for item in result["missing_from_database"]:
            print(f"  NOT IMPORTED: {item['filename']} (exists={item['staged_file_exists']}, bytes={item['staged_bytes']})")
    if result["feature_extraction_warnings"]:
        print("Feature-extraction log warnings:")
        for warning in result["feature_extraction_warnings"]:
            print(f"  {warning}")
    print(f"Verified pairs with >= {args.min_inliers} inliers: {result['strong_pair_count']}")
    print("Connected component sizes:", result["connected_component_sizes"])
    print("Largest connected component:", ", ".join(result["connected_components"][0]))
    print("Registered:", ", ".join(result["registered_filenames"]) or "(none)")
    print("\nImages with no strong verified neighbor:")
    for item in result["weak_or_isolated"][:30]:
        print(f"  {item['filename']} (keypoints={item['keypoints']}, total-inliers={item['verified_inliers_across_pairs']})")
    if not result["weak_or_isolated"]:
        print("  (none)")
    print("\nStrongest verified image pairs:")
    for pair in result["strongest_pairs"]:
        print(f"  {pair['inliers']:5d}  {pair['a']} <-> {pair['b']}")
    print("\nInterpretation: connected feature matches are necessary, not sufficient, for a valid camera reconstruction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
