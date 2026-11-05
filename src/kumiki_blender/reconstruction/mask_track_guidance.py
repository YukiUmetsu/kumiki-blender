"""Conservative multi-view 3D track evidence for grounded SAM2 re-prompting.

Sparse COLMAP point tracks are measured geometry but do not carry semantic
labels. A point is only a *candidate* statue anchor when independently
predicted masks, the text-detector box and multiple photographs agree.
Conflicting/background tracks never turn into automatic foreground clicks.
Every suggested click retains the point ID and other supporting photographs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dense import _sha256
from .mask_consistency import (_MAX_MODEL_BYTES, _binary_image_tracks,
                               _text_image_tracks)
from .references import ReferenceError


SCHEMA = "multi-view-subject-track-evidence-v1"
MIN_SUPPORTING_CAMERAS = 3
MAX_SEEDS_PER_IMAGE = 12


def _safe_pinned_image(root: Path, relative: str, digest: str) -> Path:
    if (not isinstance(relative, str) or not relative or not isinstance(digest, str)
            or len(digest) != 64):
        raise ReferenceError("Missing candidate mask provenance")
    path = root / relative
    if (path.is_symlink() or not path.resolve().is_relative_to(root)
            or not path.is_file() or _sha256(path) != digest):
        raise ReferenceError("Mask candidate missing, modified, or outside source revision")
    return path


def _box_contains(guide: dict[str, Any], x: float, y: float) -> bool:
    boxes = guide.get("component_boxes") or {"subject": guide.get("box")}
    if not isinstance(boxes, dict):
        return False
    return any(isinstance(box, list) and len(box) == 4 and
               box[0] <= x <= box[2] and box[1] <= y <= box[3]
               for box in boxes.values())


def _inside(mask: Any, x: float, y: float) -> bool:
    return bool(mask[min(mask.shape[0]-1, round(y)),
                     min(mask.shape[1]-1, round(x))])


def _seed_class(point_id: int) -> str:
    # Independent held-out tracks cannot affect SAM2 prompts or guide ranking.
    return "holdout" if int.from_bytes(
        hashlib.sha256(str(point_id).encode()).digest()[:2], "big") % 5 == 0 else "seed"


def select_spatially_distributed(records: list[dict], *, max_count: int = MAX_SEEDS_PER_IMAGE
                                 ) -> list[dict]:
    """Strongest evidence first, at most one click per image grid cell."""
    selected = []
    occupied = set()
    for row in sorted(records, key=lambda item: (-item["supporting_cameras"],
                                                -item["confidence"], item["point3d_id"])):
        x, y = row["normalized_xy"]
        cell = (min(5, int(x * 6)), min(4, int(y * 5)))
        if cell in occupied:
            continue
        occupied.add(cell)
        selected.append(row)
        if len(selected) >= max_count:
            break
    return selected


def extract_track_guidance(reconstruction: Path, mask_revision: Path,
                           mask_report: dict[str, Any], *, minimum_support: int = 3
                           ) -> dict[str, Any]:
    """Propose reproducible seed/holdout points from a pinned registered model.

    Masks are downsampled *for anchor classification only*; SAM2 still reads
    the immutable full-resolution registered photos. No synthetic geometry.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    if not 2 <= minimum_support <= 12:
        raise ReferenceError("Track guidance needs 2–12 independent supporting cameras")
    root = mask_revision.resolve()
    source = reconstruction.resolve()
    if (mask_revision.is_symlink() or not root.is_dir() or
            mask_report.get("source_revision") != str(source)):
        raise ReferenceError("Mask report is not pinned to this registered camera revision")
    model_dir = source / "undistorted/sparse"
    model = model_dir / "images.bin"
    if not model.exists():
        model = model_dir / "images.txt"
    if (model.is_symlink() or not model.is_file() or
            model.stat().st_size > _MAX_MODEL_BYTES):
        raise ReferenceError("Missing or unsafe registered COLMAP point-track model")
    tracks = _binary_image_tracks(model) if model.suffix == ".bin" else _text_image_tracks(model)
    index = json.loads((source / "undistorted-reference-index.json").read_text(encoding="utf-8"))
    manifest = {item["filename"]: item for item in index["images"]}
    entries = {item["filename"]: item for item in mask_report["images"]}
    if not entries or set(entries) != set(manifest) or set(entries) != set(tracks):
        raise ReferenceError("Source cameras, image manifest and mask candidates differ")
    observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rejected = defaultdict(int)
    for name in sorted(entries):
        entry = entries[name]
        if entry.get("source_photo_sha256") != manifest[name]["sha256"]:
            raise ReferenceError(f"Rectified source photo changed: {name}")
        candidates = entry.get("candidate_masks", {})
        if not all(label in candidates for label in ("broad", "grounded")):
            raise ReferenceError(f"Missing two independent SAM2 candidates: {name}")
        selected = _safe_pinned_image(root, entry["mask"], entry["mask_sha256"])
        broad = _safe_pinned_image(root, candidates["broad"]["path"],
                                   candidates["broad"]["sha256"])
        grounded = _safe_pinned_image(root, candidates["grounded"]["path"],
                                      candidates["grounded"]["sha256"])
        photo_path = source / "undistorted/images" / name
        if (photo_path.is_symlink() or not photo_path.is_file() or
                _sha256(photo_path) != manifest[name]["sha256"]):
            raise ReferenceError(f"Registered photo evidence changed: {name}")
        with (Image.open(selected) as s, Image.open(broad) as b,
              Image.open(grounded) as g, Image.open(photo_path) as photo):
            dimensions = list(photo.size)
            if (dimensions != manifest[name]["dimensions"] or
                    any(image.mode != "L" or image.size != photo.size for image in (s, b, g))):
                raise ReferenceError("Mask/photo dimensions are inconsistent")
            photo_small = photo.convert("RGB")
            photo_small.thumbnail((1024, 1024))
            dims = photo_small.size
            arrays = []
            for binary in (s, b, g):
                small = binary.resize(dims, Image.Resampling.NEAREST)
                arrays.append(np.asarray(small) >= 128)
            selected_array, broad_array, grounded_array = arrays
            union_array = broad_array | grounded_array
            # Interior confidence excludes thin halos, boundaries and disconnected
            # mask pixels. This is still a *candidate*, never a statue label.
            broad_interior = np.asarray(
                b.resize(dims, Image.Resampling.NEAREST).filter(ImageFilter.MinFilter(7))) >= 128
            grounded_interior = np.asarray(
                g.resize(dims, Image.Resampling.NEAREST).filter(ImageFilter.MinFilter(7))) >= 128
            rgb = np.asarray(photo_small)
            seen = set()
            for point_id, x, y in tracks[name]:
                if point_id in seen:
                    rejected["duplicate_observation"] += 1
                    continue
                seen.add(point_id)
                if not (8 <= x < dimensions[0]-8 and 8 <= y < dimensions[1]-8):
                    rejected["edge_or_out_of_frame"] += 1
                    continue
                sx = min(dims[0]-1, round(x * (dims[0]-1) / max(1, dimensions[0]-1)))
                sy = min(dims[1]-1, round(y * (dims[1]-1) / max(1, dimensions[1]-1)))
                u, v = x / max(1, dimensions[0]-1), y / max(1, dimensions[1]-1)
                color = rgb[sy, sx]
                sky = bool(int(color[2]) > 120 and int(color[2]) > 1.15 * int(color[0])
                           and int(color[2]) > 1.06 * int(color[1]))
                detector = entry.get("detector_guide")
                corroborated = (not sky and
                                entry.get("guide_source") == "grounding-dino / candidate" and
                                isinstance(detector, dict) and
                                _box_contains(detector, u, v) and
                                _inside(broad_interior, sx, sy) and
                                _inside(grounded_interior, sx, sy))
                observations[point_id].append({
                    "name": name, "normalized_xy": [round(u, 6), round(v, 6)],
                    "supported": bool(corroborated), "sky": sky,
                    "selected": _inside(selected_array, sx, sy),
                    "union": _inside(union_array, sx, sy)})

    results: dict[str, dict[str, Any]] = {
        name: {"seed_candidates": [], "holdout_candidates": [],
               "supporting_3d_tracks": 0, "rejected_by_semantic_consensus": 0}
        for name in sorted(entries)}
    viable_tracks = 0
    for point_id, views in observations.items():
        if len(views) < minimum_support + 1 or any(view["sky"] for view in views):
            continue
        supporting = [view for view in views if view["supported"]]
        if len(supporting) < minimum_support:
            continue
        # Conservative threshold: detector box AND both masks agree in at
        # least 75% of this triangulated point's actual camera observations.
        if len(supporting) / len(views) < .75:
            continue
        viable_tracks += 1
        for view in views:
            if len(supporting) - int(view["supported"]) < minimum_support:
                continue
            result = results[view["name"]]
            result["supporting_3d_tracks"] += 1
            if not view["union"]:
                # Neither existing candidate sees the point. No independent
                # evidence that it is the subject: abstain.
                result["rejected_by_semantic_consensus"] += 1
                continue
            row = {"point3d_id": point_id, "normalized_xy": view["normalized_xy"],
                   "supporting_cameras": len(supporting) - int(view["supported"]),
                   "confidence": round(len(supporting) / len(views), 5),
                   "source_views": [v["name"] for v in supporting if v["name"] != view["name"]],
                   "previously_selected": view["selected"],
                   "ground_truth": False}
            result["holdout_candidates" if _seed_class(point_id) == "holdout"
                   else "seed_candidates"].append(row)
    for name, result in results.items():
        # Only missing, independently corroborated geometry should become a
        # positive click. Already-foreground tracks remain holdout context.
        result["seed_points"] = select_spatially_distributed([
            row for row in result.pop("seed_candidates") if not row["previously_selected"]])
        result["holdout_points"] = select_spatially_distributed(
            result.pop("holdout_candidates"), max_count=80)
        result["seed_grid_cells"] = len({(int(p["normalized_xy"][0] * 5),
                                         int(p["normalized_xy"][1] * 5))
                                        for p in result["seed_points"]})
    return {"schema": SCHEMA, "model_sha256": _sha256(model),
            "camera_revision": str(source), "mask_revision": str(root),
            "compared_track_ids": len(observations), "viable_track_ids": viable_tracks,
            "min_independent_camera_support": minimum_support,
            "rejected": dict(rejected), "images": results,
            "limitations": "Sparse 3D tracks have no verified semantic label; masks and detector agreement only produce candidate clicks. Holdout points were never used in prompting."}


def evaluate_track_resegmentation(initial: Any, proposed: Any,
                                   seeds: list[dict], holdouts: list[dict]
                                   ) -> dict[str, Any]:
    """Only adopt a revision with independent heldout improvement and no regression."""
    import numpy as np

    old = np.asarray(initial) >= 128
    new = np.asarray(proposed) >= 128
    if old.shape != new.shape or old.ndim != 2:
        raise ReferenceError("Track-guided SAM2 candidate has different dimensions")
    def hit(array, row):
        x, y = row["normalized_xy"]
        return bool(array[min(array.shape[0]-1, round(y * (array.shape[0]-1))),
                          min(array.shape[1]-1, round(x * (array.shape[1]-1)))])

    old_count, new_count = int(np.count_nonzero(old)), int(np.count_nonzero(new))
    recovered = sum(not hit(old, row) and hit(new, row) for row in holdouts)
    lost = sum(hit(old, row) and not hit(new, row) for row in holdouts)
    missed_seeds = sum(not hit(new, row) for row in seeds)
    missed_holdout_before = sum(not hit(old, row) for row in holdouts)
    reasons = []
    if len(seeds) < 3 or len(holdouts) < 5:
        reasons.append("insufficient-spatially-independent-seed-or-holdout-tracks")
    if old_count == 0 or new_count < .85 * old_count or new_count > 1.20 * old_count:
        reasons.append("excessive-foreground-area-change")
    if missed_seeds:
        reasons.append("new-sam2-candidate-violates-track-seed-prompts")
    if recovered < 2 or lost:
        reasons.append("heldout-tracks-do-not-show-safe-recovery")
    # Heldout tracks never certify the whole silhouette or an anatomy inventory.
    return {"status": "candidate / heldout-track-improvement" if not reasons
            else "rejected / insufficient-independent-subject-evidence",
            "accepted": not reasons, "reasons": reasons, "seed_count": len(seeds),
            "heldout_count": len(holdouts), "heldout_recovered": recovered,
            "heldout_lost": lost, "heldout_missing_before": missed_holdout_before,
            "foreground_area_ratio": round(new_count / max(1, old_count), 6),
            "limitations": "A passing track test is not verified monument anatomy or geometric completeness."}
