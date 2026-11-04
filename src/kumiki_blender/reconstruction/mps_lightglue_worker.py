"""Memory-bounded SIFT-LightGlue matching against an existing COLMAP database.

Only selected keypoints are copied to the neural network. Their original row
indices are preserved in the raw COLMAP matches table for later verification.
This optional worker needs numpy, torch and the upstream cvg/LightGlue package.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sqlite3
import struct
import sys
from collections import defaultdict
from pathlib import Path

MAX_IMAGE_ID = 2147483647


def _load_lightglue_matcher():
    """Load the pure-PyTorch matcher without LightGlue's eager OpenCV imports.

    Upstream ``lightglue.__init__`` imports all extractors, including its SIFT
    extractor (and cv2). This worker uses *COLMAP's existing SIFT features*, so
    loading those extractors is unnecessary. On some macOS installations,
    importing cv2 after torch aborts the process with duplicate libomp (OMP
    Error #15), which cannot be caught by Python. Load the standalone upstream
    matcher module directly, before any package ``__init__`` is executed.

    This intentionally requires upstream's self-contained ``lightglue.py``
    layout. If the library changes, fail closed rather than importing OpenCV or
    enabling the unsafe KMP_DUPLICATE_LIB_OK workaround.
    """
    package = importlib.util.find_spec("lightglue")
    if package is None or not package.submodule_search_locations:
        raise ImportError("cvg/LightGlue package is not installed")
    for directory in package.submodule_search_locations:
        matcher_file = Path(directory) / "lightglue.py"
        if not matcher_file.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "_kumiki_lightglue_matcher", matcher_file)
        if spec is None or spec.loader is None:
            break
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        matcher = getattr(module, "LightGlue", None)
        if matcher is None:
            break
        return matcher
    raise ImportError(
        "cvg/LightGlue's standalone lightglue/lightglue.py matcher is missing "
        "or incompatible; install a supported upstream version"
    )


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
    if kcols == 4:
        scales, orientations = keypoints[:, 2], keypoints[:, 3]
    else:
        # COLMAP's six-column geometry encodes an affine shape as
        # (a11, a12, a21, a22), not (scale, orientation, ...). Use the
        # same scale/orientation projection as FeatureKeypoint::ComputeScale
        # and FeatureKeypoint::ComputeOrientation in COLMAP's feature/types.cc.
        a11, a12, a21, a22 = (keypoints[:, column] for column in range(2, 6))
        scales = (np.hypot(a11, a21) + np.hypot(a12, a22)) / 2.0
        orientations = np.arctan2(a21, a11)
    chosen = select_spatial_keypoints(
        [(float(k[0]), float(k[1]), float(scale)) for k, scale in zip(keypoints, scales)],
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
            "scales": scales[selected].astype(np.float32),
            "oris": orientations[selected].astype(np.float32),
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
        LightGlue = _load_lightglue_matcher()
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "MPS LightGlue requires working numpy, torch and cvg/LightGlue: "
            "pip install numpy torch && "
            "pip install git+https://github.com/cvg/LightGlue.git "
            f"(dependency detail: {error})"
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
