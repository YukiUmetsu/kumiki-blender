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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .dense import _sha256
from .mask_consistency import (_MAX_MODEL_BYTES, _binary_image_tracks,
                               _text_image_tracks)
from .references import ReferenceError


SCHEMA = "multi-view-subject-track-evidence-v5"
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
            # A negative requires clearance from *both* candidate silhouettes,
            # not merely a pixel that the currently selected mask omitted.
            # Dilation keeps uncertain contour and thin-anatomy pixels out.
            union_binary = Image.fromarray(np.where(union_array, 255, 0).astype("uint8"), "L")
            background_interior = np.asarray(
                union_binary.filter(ImageFilter.MaxFilter(9))) == 0
            rgb = np.asarray(photo_small)
            # The text detector's part boxes can cover the full image. Use
            # independent SAM2 silhouettes for *ground-plane fitting*; the
            # geometric plane will still need a low, camera-up-aligned fit.
            # A 3D track with multiple observations in the same image is
            # ambiguous. Keeping the first would make the evidence depend on
            # file order and could turn an incorrect location into a prompt.
            point_counts = Counter(point_id for point_id, _, _ in tracks[name])
            for point_id, x, y in tracks[name]:
                if point_counts[point_id] != 1:
                    rejected["duplicate_observation"] += 1
                    continue
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
                corroborated_background = (not sky and
                                           entry.get("guide_source") == "grounding-dino / candidate" and
                                           isinstance(detector, dict) and
                                           not _box_contains(detector, u, v) and
                                           _inside(background_interior, sx, sy))
                observations[point_id].append({
                    "name": name, "normalized_xy": [round(u, 6), round(v, 6)],
                    "supported": bool(corroborated), "sky": sky,
                    "background_supported": bool(corroborated_background),
                    "mask_exterior": bool(not sky and _inside(background_interior, sx, sy)),
                    "lower_scene_exterior": bool(not sky and v >= .48 and
                                                  _inside(background_interior, sx, sy)),
                    "selected": _inside(selected_array, sx, sy),
                    "union": _inside(union_array, sx, sy)})

    from .mask_ground_plane import fit_ground_plane
    ground_tracks, ground_plane = fit_ground_plane(model_dir, observations)
    points_model = model_dir / "points3D.bin"
    points_model_sha = (_sha256(points_model) if points_model.is_file() and
                        not points_model.is_symlink() and
                        points_model.stat().st_size <= _MAX_MODEL_BYTES else None)

    results: dict[str, dict[str, Any]] = {
        name: {"seed_candidates": [], "holdout_candidates": [],
               "background_seed_candidates": [], "background_holdout_candidates": [],
               "supporting_3d_tracks": 0, "rejected_by_semantic_consensus": 0}
        for name in sorted(entries)}
    viable_tracks = 0
    viable_background_tracks = 0
    for point_id, views in observations.items():
        if len(views) < minimum_support + 1 or any(view["sky"] for view in views):
            continue
        supporting = [view for view in views if view["supported"]]
        background_supporting = [view for view in views if view["background_supported"]]
        # Independent support may come from either (a) the strict detector
        # exterior or (b) a low *measured 3D plane* corroborated outside both
        # masks in several other cameras. Full-frame DINO part boxes no longer
        # suppress every true ground track. One disagreeing target view is
        # tolerated for a fitted plane; multiple statue votes veto that plane.
        planar_ground = point_id in ground_tracks
        if ((planar_ground and len(supporting) <= 1) or
                (not supporting and len(background_supporting) >= max(4, minimum_support)
                 and len(background_supporting) / len(views) >= .80)):
            viable_background_tracks += 1
            for view in views:
                other_exteriors = [v for v in views if v["name"] != view["name"] and
                                   (v["mask_exterior"] if planar_ground else
                                    v["background_supported"])]
                required = 3 if planar_ground else 4
                if view["sky"] or len(other_exteriors) < required or (
                        not planar_ground and view["union"]):
                    continue
                row = {"point3d_id": point_id, "normalized_xy": view["normalized_xy"],
                       "supporting_cameras": len(other_exteriors),
                       "confidence": round(len(other_exteriors) / max(1, len(views)-1), 5),
                       "source_views": [v["name"] for v in other_exteriors],
                       "source": "triangulated-low-ground-plane" if planar_ground
                       else "cross-view-mask-and-detector-exterior",
                       "ground_truth": False}
                results[view["name"]]["background_holdout_candidates" if
                                      _seed_class(point_id) == "holdout" else
                                      "background_seed_candidates"].append(row)
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
                # If both SAM2 masks miss a visible chest/torso point in the
                # target, don't permanently discard it. Its *independent*
                # other-camera 3D track may still establish a strong subject
                # candidate. Require stronger support than the normal path.
                if (view["sky"] or len(supporting)-int(view["supported"]) < 4 or
                        len(supporting)/len(views) < .80 or point_id in ground_tracks):
                    result["rejected_by_semantic_consensus"] += 1
                    continue
            row = {"point3d_id": point_id, "normalized_xy": view["normalized_xy"],
                   "supporting_cameras": len(supporting) - int(view["supported"]),
                   "confidence": round(len(supporting) / len(views), 5),
                   "source_views": [v["name"] for v in supporting if v["name"] != view["name"]],
                   "previously_selected": view["selected"],
                   "source": ("independent-cross-view-only-foreground" if not view["union"]
                              else "candidate-mask-and-cross-view-foreground"),
                   "ground_truth": False}
            result["holdout_candidates" if _seed_class(point_id) == "holdout"
                   else "seed_candidates"].append(row)
    for name, result in results.items():
        # Only missing, independently corroborated geometry should become a
        # positive click. Already-foreground tracks remain holdout context.
        # Stable already-selected geometry can re-prompt SAM2 to remove bad
        # ground when there is no missing subject geometry to recover.
        result["foreground_context_points"] = select_spatially_distributed([
            row for row in result["seed_candidates"] if row["previously_selected"]])
        result["seed_points"] = select_spatially_distributed([
            row for row in result.pop("seed_candidates") if not row["previously_selected"]])
        result["holdout_points"] = select_spatially_distributed(
            result.pop("holdout_candidates"), max_count=80)
        result["background_seed_points"] = select_spatially_distributed(
            result.pop("background_seed_candidates"), max_count=12)
        result["background_holdout_points"] = select_spatially_distributed(
            result.pop("background_holdout_candidates"), max_count=80)
        result["seed_grid_cells"] = len({(int(p["normalized_xy"][0] * 5),
                                         int(p["normalized_xy"][1] * 5))
                                        for p in result["seed_points"]})
    return {"schema": SCHEMA, "model_sha256": _sha256(model),
            "points3d_model_sha256": points_model_sha,
            "camera_revision": str(source), "mask_revision": str(root),
            "compared_track_ids": len(observations), "viable_track_ids": viable_tracks,
            "viable_background_track_ids": viable_background_tracks,
            "geometric_ground_plane": ground_plane,
            "min_independent_camera_support": minimum_support,
            "rejected": dict(rejected), "images": results,
            "limitations": "Sparse 3D tracks and low planar surfaces have no verified semantic label; cross-view mask agreement and a camera-up plane only produce candidate foreground/background clicks. Foreground and background holdout tracks were never used in prompting."}


def evaluate_track_resegmentation(initial: Any, proposed: Any,
                                   seeds: list[dict], holdouts: list[dict],
                                   background_seeds: list[dict] | None = None,
                                   background_holdouts: list[dict] | None = None,
                                   *, ground_repair: bool = False,
                                   foreground_context_count: int = 0
                                   ) -> dict[str, Any]:
    """Require recovered statue tracks and no independently held-out background.

    A larger mask can pass only with many spatially independent background
    counterexamples, recovered foreground holdouts and localized expansion.
    Without those checks, retain the original 20% growth guard.
    """
    import numpy as np

    background_seeds = background_seeds or []
    background_holdouts = background_holdouts or []
    evidence_rows = seeds + holdouts + background_seeds + background_holdouts
    point_ids = [row.get("point3d_id") for row in evidence_rows]
    missing_provenance = any(type(point_id) is not int or point_id < 0 for point_id in point_ids)
    duplicate_provenance = (not missing_provenance and len(set(point_ids)) != len(point_ids))
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
    background_violations = sum(hit(new, row) for row in background_seeds)
    # Negative holdouts are never provided to the SAM2 predictor. They are an
    # independent veto on new foreground, even when area statistics look good.
    background_holdout_gained = sum(not hit(old, row) and hit(new, row)
                                    for row in background_holdouts)
    background_holdout_included = sum(hit(new, row) for row in background_holdouts)
    background_holdout_repaired = sum(hit(old, row) and not hit(new, row)
                                      for row in background_holdouts)
    added = new & ~old
    step = max(1, max(old.shape) // 128)
    ys, xs = np.nonzero(added[::step, ::step])
    anchors = seeds + [row for row in holdouts if not hit(old, row) and hit(new, row)]
    supported_fraction = 0.0
    if len(xs) and anchors:
        xs = xs.astype(float) * step / max(1, old.shape[1]-1)
        ys = ys.astype(float) * step / max(1, old.shape[0]-1)
        locations = np.array([row["normalized_xy"] for row in anchors], dtype=float)
        near = np.zeros(len(xs), dtype=bool)
        # Bound intermediate arrays: never allocate (image pixels × tracks).
        for index in range(0, len(xs), 4096):
            distances = ((xs[index:index+4096, None]-locations[:, 0])**2 +
                         (ys[index:index+4096, None]-locations[:, 1])**2)
            near[index:index+4096] = np.min(distances, axis=1) <= .15**2
        supported_fraction = float(np.mean(near))
    area_ratio = new_count / max(1, old_count)
    growth_limit = 1.20
    if (len(seeds) >= 4 and len(background_seeds) >= 3 and
            len(background_holdouts) >= 8 and recovered >= 2 and not lost and
            not background_violations and not background_holdout_included and
            supported_fraction >= .75):
        growth_limit = 1.55
        if (len(background_holdouts) >= 20 and recovered >= 4 and
                len(seeds) >= 6 and supported_fraction >= .85):
            growth_limit = 1.90
    reasons = []
    if missing_provenance:
        reasons.append("missing-track-provenance")
    if duplicate_provenance:
        reasons.append("overlapping-or-duplicate-track-evidence")
    if len(seeds) < 3 or len(holdouts) < 5:
        reasons.append("insufficient-spatially-independent-seed-or-holdout-tracks")
    if old_count == 0 or area_ratio < .85 or area_ratio > growth_limit:
        reasons.append("excessive-foreground-area-change")
    if missed_seeds:
        reasons.append("new-sam2-candidate-violates-track-seed-prompts")
    if recovered < 2 or lost:
        reasons.append("heldout-tracks-do-not-show-safe-recovery")
    if background_violations or background_holdout_included:
        reasons.append("independent-background-tracks-became-foreground")
    # Ground removal is a different, independently validated improvement from
    # recovering a missed paw/torso feature. Verify that *held-out* ground was
    # removed while every held-out subject track remains, and keep the area
    # shrink modest. Ground prompts alone cannot certify their own success.
    verified_repair = (not missing_provenance and not duplicate_provenance and
                       ground_repair and foreground_context_count >= 3 and
                       len(background_seeds) >= 2 and len(background_holdouts) >= 2 and
                       background_holdout_repaired >= 1 and not lost and
                       not missed_seeds and not background_violations and
                       not background_holdout_included and
                       .85 <= area_ratio <= 1.02 and
                       int(np.count_nonzero(added)) <= .02*old_count and
                       len(holdouts) >= 5)
    if verified_repair:
        reasons = []
    # Heldout tracks never certify the whole silhouette or an anatomy inventory.
    return {"status": ("candidate / heldout-ground-repair" if verified_repair else
                      "candidate / heldout-track-improvement") if not reasons
            else "rejected / insufficient-independent-subject-evidence",
            "accepted": not reasons, "reasons": reasons, "seed_count": len(seeds),
            "heldout_count": len(holdouts), "heldout_recovered": recovered,
            "heldout_lost": lost, "heldout_missing_before": missed_holdout_before,
            "background_seed_count": len(background_seeds),
            "background_holdout_count": len(background_holdouts),
            "background_seed_violations": background_violations,
            "background_holdout_new_foreground": background_holdout_gained,
            "background_holdout_foreground": background_holdout_included,
            "background_holdout_repaired": background_holdout_repaired,
            "ground_repair_verified": verified_repair,
            "added_foreground_near_recovered_tracks_fraction": round(supported_fraction, 6),
            "adaptive_growth_limit": growth_limit,
            "foreground_area_ratio": round(area_ratio, 6),
            "limitations": "A passing track test is not verified monument anatomy or geometric completeness."}
