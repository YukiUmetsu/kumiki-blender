"""View-aligned evidence comparison, never a stand-in for 3D sculpt accuracy."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from .schema import AnnotationSet, ViewAnnotations


@dataclass(frozen=True)
class ViewPair:
    reference_image_id: str
    candidate_image_id: str
    alignment_id: str
    verified: bool


def _region(landmark_id: str) -> str:
    if landmark_id.startswith(("chin", "jaw")):
        return "jaw"
    if landmark_id.startswith("eye"):
        return "eyes"
    if landmark_id.startswith("brow"):
        return "brows"
    if landmark_id.startswith(("nose", "nostril")):
        return "nose"
    if landmark_id.startswith(("mouth", "upper-lip", "lower-lip")):
        return "mouth"
    if landmark_id.startswith(("hair",)):
        return "hair"
    if landmark_id.startswith("ear"):
        return "ears"
    if landmark_id.startswith("cheek"):
        return "cheeks"
    if landmark_id.startswith("neck"):
        return "neck"
    return "head-outline"


def _observed(view: ViewAnnotations) -> dict[str, tuple[float, float]]:
    verified = {}
    seen = set()
    for record in view.landmarks:
        if record.landmark_id in seen:
            raise ValueError(f"duplicate landmark: {record.landmark_id} in {view.image_id}")
        seen.add(record.landmark_id)
        if record.origin == "manual" and record.visibility == "visible" and record.xy_px is not None:
            if not all(math.isfinite(component) for component in record.xy_px):
                raise ValueError(f"Nonfinite landmark: {record.landmark_id}")
            verified[record.landmark_id] = record.xy_px
    return verified


def _scale(outline: tuple[tuple[float, float], ...] | None) -> float | None:
    if outline is None or len(outline) < 3:
        return None
    xs, ys = [point[0] for point in outline], [point[1] for point in outline]
    if not all(math.isfinite(value) for value in xs + ys):
        return None
    scale = max(max(xs) - min(xs), max(ys) - min(ys))
    return scale if scale >= 1 else None


def _silhouette_iou(a: tuple[tuple[float, float], ...] | None,
                    b: tuple[tuple[float, float], ...] | None) -> float | None:
    if not a or not b:
        return None
    from PIL import Image, ImageChops, ImageDraw  # optional QA extra

    coordinates = [coord for point in a + b for coord in point]
    if not all(math.isfinite(value) and 0 <= value <= 16384 for value in coordinates):
        return None
    max_x = max(p[0] for p in a + b)
    max_y = max(p[1] for p in a + b)
    factor = min(1.0, 2048.0 / max(1, max_x, max_y))
    size = (max(2, int(max_x * factor) + 2), max(2, int(max_y * factor) + 2))
    masks = []
    for polygon in (a, b):
        mask = Image.new("1", size, 0)
        ImageDraw.Draw(mask).polygon([(x * factor, y * factor) for x, y in polygon], fill=1)
        masks.append(mask)
    intersection = ImageChops.logical_and(masks[0], masks[1])
    union = ImageChops.logical_or(masks[0], masks[1])
    num_union = sum(count for count, value in union.getcolors(maxcolors=2) or [] if value)
    num_intersection = sum(count for count, value in intersection.getcolors(maxcolors=2) or [] if value)
    return num_intersection / num_union if num_union else None


def compare_views(reference: AnnotationSet, candidate: AnnotationSet,
                  pairs: Sequence[ViewPair]) -> dict[str, Any]:
    """Measure only pairs with declared matching-view pixel registration."""
    outputs = []
    regional: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_keys: set[tuple[str, str]] = set()
    for pair in pairs:
        key = (pair.reference_image_id, pair.candidate_image_id)
        if key in pair_keys:
            raise ValueError(f"duplicate view pair: {key}")
        pair_keys.add(key)
        if pair.reference_image_id not in reference.images or pair.candidate_image_id not in candidate.images:
            raise ValueError(f"Unknown view pair: {key}")
        ref, model = reference.images[pair.reference_image_id], candidate.images[pair.candidate_image_id]
        result = {"reference_image_id": ref.image_id, "candidate_image_id": model.image_id,
                  "alignment_id": pair.alignment_id, "view": ref.view, "status": "unmeasured",
                  "reason": None, "scale_px": None, "landmarks": {}, "silhouette_iou": None}
        ref_obs, model_obs = _observed(ref), _observed(model)
        scale = _scale(ref.outline_px)
        if not pair.verified or not pair.alignment_id.strip():
            result["reason"] = "unverified-registration"
        elif ref.view == "unknown" or ref.view != model.view:
            result["reason"] = "unmatched-view"
        elif scale is None:
            result["reason"] = "missing-reference-head-scale"
        else:
            result["scale_px"] = scale
            for name in sorted(ref_obs.keys() & model_obs.keys()):
                ax, ay = ref_obs[name]
                bx, by = model_obs[name]
                dx, dy = (bx - ax) / scale, (by - ay) / scale
                measured = {"region": _region(name), "error_normalized": math.hypot(dx, dy),
                            "delta_normalized": [dx, dy]}
                result["landmarks"][name] = measured
                regional[measured["region"]].append({"reference_image_id": ref.image_id,
                                                       "view": ref.view, **measured})
            result["silhouette_iou"] = _silhouette_iou(ref.outline_px, model.outline_px)
            if result["landmarks"] or result["silhouette_iou"] is not None:
                result["status"] = "measured"
            else:
                result["reason"] = "no-verified-correspondence"
        outputs.append(result)
    regions = {}
    for name, observations in sorted(regional.items()):
        count = len(observations)
        regions[name] = {
            "status": "measured", "landmark_count": count,
            "view_count": len({entry["reference_image_id"] for entry in observations}),
            "mean_error_normalized": sum(entry["error_normalized"] for entry in observations) / count,
            "mean_delta_normalized": [sum(entry["delta_normalized"][i] for entry in observations) / count
                                      for i in (0, 1)],
        }
    return {"schema_version": 1, "pairs": outputs, "regions": regions,
            "three_d_fidelity_verified": False,
            "limitations": "Only matching-camera projected 2D distances and outlines are measured. "
                           "Material, relief depth, unseen geometry and overall historical likeness are not verified."}


def summarize_observations(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Report signed image-plane symptoms; never prescribe unproven 3D edits."""
    observations = []
    for name, region in sorted(report.get("regions", {}).items()):
        dx, dy = region["mean_delta_normalized"]
        if abs(dy) >= abs(dx) and abs(dy) > 0.0001:
            direction = "higher" if dy < 0 else "lower"
        elif abs(dx) > 0.0001:
            direction = "further left" if dx < 0 else "further right"
        else:
            direction = "aligned"
        observations.append({"region": name, "status": "measured",
                             "mean_error_normalized": region["mean_error_normalized"],
                             "view_count": region["view_count"],
                             "observation": f"Candidate {name} appears {direction} in image coordinates; "
                                            "verify camera alignment before sculpt correction."})
    return observations
