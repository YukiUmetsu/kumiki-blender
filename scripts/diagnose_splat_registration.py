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


def registered_models(sparse: Path) -> dict[str, set[str]]:
    """Read image names for every independent COLMAP sparse model."""
    models = sparse / "text-models"
    if not models.is_dir():
        return {}
    found_models: dict[str, set[str]] = {}
    for model in sorted(models.iterdir()):
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
        found_models[model.name] = found
    return found_models


def registered_images(sparse: Path) -> set[str]:
    """Return the largest registered model, not a misleading union of poses."""
    models = registered_models(sparse)
    return max(models.values(), key=len, default=set())

def _has_table(db: sqlite3.Connection, table: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def analyze(root: Path, min_inliers: int) -> dict:
    sparse = root / "sparse-reconstruction" if (root / "sparse-reconstruction").is_dir() else root
    db_path = sparse / "database.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"COLMAP database not found: {db_path}")
    models = registered_models(sparse)
    registered = max(models.values(), key=len, default=set())
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
        "sparse_models": [{"model": name, "registered": len(images), "filenames": sorted(images)}
                          for name, images in sorted(models.items())],
        "staged_photos": len(staged) if staged else None,
        "staged_filenames": sorted(staged) if staged else sorted(imported),
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


def make_contact_sheet(root: Path, result: dict, destination: Path) -> None:
    """Create a private visual triage sheet from the staged photographs."""
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as error:
        raise ValueError("Contact sheet needs Pillow: pip install Pillow") from error
    sparse = root / "sparse-reconstruction" if (root / "sparse-reconstruction").is_dir() else root
    if destination.is_symlink() or destination.exists():
        raise ValueError(f"Contact sheet already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    groups = {
        filename: index + 1
        for index, group in enumerate(result["connected_components"])
        for filename in group
    }
    imported = {item["filename"] for item in result["images"]}
    registered = set(result["registered_filenames"])
    filenames = result["staged_filenames"]
    columns, width, height = 4, 300, 248
    rows = (len(filenames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), (26, 29, 34))
    draw = ImageDraw.Draw(sheet)
    for index, filename in enumerate(filenames):
        left = (index % columns) * width
        top = (index // columns) * height
        if filename not in imported:
            status, color = "NOT IMPORTED", (225, 85, 85)
        elif filename in registered:
            status, color = "REGISTERED", (87, 207, 135)
        else:
            status, color = "UNREGISTERED", (232, 181, 75)
        photo = sparse / "images" / filename
        try:
            with Image.open(photo) as source:
                preview = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), (280, 198))
                sheet.paste(preview, (left + (width - preview.width) // 2, top + 4 + (198 - preview.height) // 2))
        except (OSError, ValueError):
            draw.text((left + 12, top + 90), "Image cannot be decoded", fill=(225, 85, 85))
        draw.rectangle((left + 4, top + 205, left + width - 4, top + height - 4), fill=(38, 42, 48))
        draw.text((left + 10, top + 209), filename[:37], fill=(245, 245, 245))
        draw.text((left + 10, top + 228), f"{status} | group {groups.get(filename, '-')}", fill=color)
    sheet.save(destination, quality=88)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", type=Path, help="Failed splat revision or sparse-reconstruction directory")
    parser.add_argument("--min-inliers", type=int, default=15, help="Diagnostic connectivity threshold (default 15)")
    parser.add_argument("--json", action="store_true", help="Print complete JSON diagnostics")
    parser.add_argument("--json-output", type=Path, help="Save diagnostics to a new JSON file")
    parser.add_argument("--contact-sheet", type=Path, help="Save private labeled photo montage (requires Pillow)")
    args = parser.parse_args()
    if args.min_inliers < 1:
        parser.error("--min-inliers must be positive")
    try:
        result = analyze(args.revision.resolve(), args.min_inliers)
    except (OSError, ValueError, sqlite3.DatabaseError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    if args.contact_sheet is not None:
        try:
            make_contact_sheet(args.revision.resolve(), result, args.contact_sheet.resolve())
        except (OSError, ValueError) as error:
            parser.exit(2, f"ERROR: {error}\\n")
        print(f"Contact sheet: {args.contact_sheet}", file=__import__("sys").stderr)
    if args.json_output is not None:
        destination = args.json_output
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\\n")
        except OSError as error:
            parser.exit(2, f"ERROR: Cannot save diagnostics: {error}\\n")
        print(f"Registration JSON: {destination}", file=__import__("sys").stderr)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"Registered: {result['registered']}/{result['photos']} database images")
    for model in result["sparse_models"]:
        print(f"  Sparse model {model['model']}: {model['registered']} cameras: {', '.join(model['filenames'])}")
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
