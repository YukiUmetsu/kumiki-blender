"""Memory-bounded SIFT-LightGlue matching against an existing COLMAP database.

Only selected keypoints are copied to the neural network. Their original row
indices are preserved in the raw COLMAP matches table for later verification.
This optional worker needs numpy, torch and the upstream cvg/LightGlue package.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import struct
import sys
from collections import defaultdict
from pathlib import Path

MAX_IMAGE_ID = 2147483647


def select_spatial_keypoints(points: list[tuple[float, float, float]],
                             width: int, height: int, limit: int,
                             grid_size: int = 8) -> list[int]:
    """Keep exact cap with deterministic coverage across the image.

    COLMAP's SIFT feature cap is not exact because alternate orientations can
    produce additional rows. We select *after* feature extraction and map
    matcher outputs back to original row indices.
    """
    if limit <= 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid feature limit or image dimensions")
    if len(points) <= limit:
        return list(range(len(points)))
    cells: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for index, (x, y, scale) in enumerate(points):
        if not all(math.isfinite(value) for value in (x, y, scale)):
            continue
        cell = (min(grid_size - 1, max(0, int(x * grid_size / width))),
                min(grid_size - 1, max(0, int(y * grid_size / height))))
        cells[cell].append((-scale, index))
    for cell in cells:
        cells[cell].sort()
    ordered_cells = sorted(cells)
    selected: list[int] = []
    row = 0
    while len(selected) < limit:
        consumed = False
        for cell in ordered_cells:
            if row < len(cells[cell]):
                selected.append(cells[cell][row][1])
                consumed = True
                if len(selected) == limit:
                    break
        if not consumed:
            break
        row += 1
    return sorted(selected)


def pair_id(first: int, second: int) -> int:
    if not 0 < first < MAX_IMAGE_ID or not 0 < second < MAX_IMAGE_ID or first == second:
        raise ValueError("Invalid COLMAP image identifiers")
    return min(first, second) * MAX_IMAGE_ID + max(first, second)


def read_pairs(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2 or fields[0] == fields[1]:
            raise ValueError(f"Invalid pair at line {number}")
        entries.append((fields[0], fields[1]))
    return entries


def _load_features(database: sqlite3.Connection, name: str, cap: int) -> dict:
    row = database.execute(
        "SELECT i.image_id, c.width, c.height, k.rows, k.cols, k.data, "
        "d.rows, d.cols, d.data "
        "FROM images i JOIN cameras c ON i.camera_id=c.camera_id "
        "JOIN keypoints k ON i.image_id=k.image_id "
        "JOIN descriptors d ON i.image_id=d.image_id WHERE i.name=?", (name,)).fetchone()
    if row is None:
        raise ValueError(f"Missing extracted COLMAP features for {name}")
    image_id, width, height, krows, kcols, kblob, drows, dcols, dblob = row
    if (kcols not in (4, 6) or dcols != 128 or krows != drows or
            len(kblob) != 4 * krows * kcols or len(dblob) != krows * dcols):
        raise ValueError(f"Unexpected SIFT feature dimensions for {name}")
    import numpy as np

    keypoints = np.frombuffer(kblob, dtype="<f4").reshape(krows, kcols)
    descriptors = np.frombuffer(dblob, dtype="u1").reshape(drows, dcols)
    if kcols != 4:
        raise ValueError("Affine SIFT keypoints are not supported by the MPS matcher")
    chosen = select_spatial_keypoints(
        [(float(k[0]), float(k[1]), float(k[2])) for k in keypoints],
        width, height, cap)
    selected = np.asarray(chosen, dtype=np.int64)
    # COLMAP uses pixel-corner (0.5, 0.5), LightGlue uses pixel-center (0, 0).
    kpts = keypoints[selected, :2].astype(np.float32, copy=True) - 0.5
    values = descriptors[selected].astype(np.float32)
    # SIFT descriptors are already RootSIFT-normalized by COLMAP. A second
    # square root would break compatibility with SIFT-LightGlue weights.
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    return {"image_id": image_id, "image_size": (width, height),
            "indices": selected, "keypoints": kpts, "descriptors": values,
            "scales": keypoints[selected, 2].astype(np.float32),
            "oris": keypoints[selected, 3].astype(np.float32),
            "original_count": krows}


def _match(matcher, torch, first: dict, second: dict, device: str):
    def feed(feature: dict) -> dict:
        return {key: torch.from_numpy(feature[key]).unsqueeze(0).to(device=device)
                for key in ("keypoints", "descriptors", "scales", "oris")} | {
                    "image_size": torch.tensor([feature["image_size"]],
                                               dtype=torch.float32, device=device)}
    with torch.inference_mode():
        result = matcher({"image0": feed(first), "image1": feed(second)})
    return result["matches"][0].detach().to("cpu").numpy()


def write_matches(database: sqlite3.Connection, first: dict, second: dict,
                  selected_matches) -> int:
    # COLMAP pair ids are canonical (ascending image id), and every stored
    # match index refers to the original database feature row.
    left_id, right_id = first["image_id"], second["image_id"]
    swapped = left_id > right_id
    pairs: list[tuple[int, int]] = []
    for a, b in selected_matches:
        a, b = int(a), int(b)
        if not 0 <= a < len(first["indices"]) or not 0 <= b < len(second["indices"]):
            raise ValueError("Matcher returned out-of-bounds feature index")
        pair = (int(first["indices"][a]), int(second["indices"][b]))
        pairs.append(pair[::-1] if swapped else pair)
    pairs = sorted(set(pairs))
    data = b"".join(struct.pack("<II", *pair) for pair in pairs)
    with database:
        database.execute(
            "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES(?, ?, 2, ?)",
            (pair_id(left_id, right_id), len(pairs), data))
    return len(pairs)


def _check_dependencies(device: str):
    """Validate the matcher environment without loading neural model weights."""
    try:
        import numpy  # noqa: F401  (required by feature loading)
        import torch
        from lightglue import LightGlue
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "MPS LightGlue requires working numpy, torch and cvg/LightGlue: "
            "pip install numpy torch && "
            "pip install git+https://github.com/cvg/LightGlue.git"
        ) from error
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable; check Apple Silicon and torch installation")
    return torch, LightGlue


def match_pair_file(database_path: Path, pairs_path: Path, max_keypoints: int,
                    device: str) -> None:
    # Import only here so every other CLI works without ML dependencies.
    torch, LightGlue = _check_dependencies(device)
    model = LightGlue(features="sift", mp=False).eval().to(device)
    names = read_pairs(pairs_path)
    with sqlite3.connect(database_path) as db:
        images = {name for pair in names for name in pair}
        features = {name: _load_features(db, name, max_keypoints) for name in sorted(images)}
        for first_name, second_name in names:
            first, second = features[first_name], features[second_name]
            if not len(first["indices"]) or not len(second["indices"]):
                matches = []
            else:
                matches = _match(model, torch, first, second, device)
            count = write_matches(db, first, second, matches)
            print(f"Matched {first_name} / {second_name}: {count} matches "
                  f"({len(first['indices'])}/{first['original_count']} and "
                  f"{len(second['indices'])}/{second['original_count']} keypoints)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded Apple MPS SIFT-LightGlue pair worker")
    parser.add_argument("command", choices=("match", "check"))
    parser.add_argument("--database", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--max-keypoints", type=int)
    parser.add_argument("--device", choices=("mps", "cpu"), required=True)
    args = parser.parse_args(argv)
    if args.command == "match":
        if args.database is None or args.pairs is None or args.max_keypoints is None:
            parser.error("match requires --database, --pairs, and --max-keypoints")
        if not 256 <= args.max_keypoints <= 32768:
            parser.error("max-keypoints must be 256–32768")
    try:
        if args.command == "check":
            _check_dependencies(args.device)
        else:
            match_pair_file(args.database, args.pairs, args.max_keypoints, args.device)
    except (OSError, sqlite3.Error, RuntimeError, ValueError) as error:
        print(f"MPS LightGlue worker failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
