"""Conservative *unverified* image-plane similarity registration suggestions.

Anchors are excluded from likeness diagnostics, so fitting an eye or mouth
cannot silently make that same feature appear more faithful. This is never
camera calibration or a substitute for registered 3D-to-2D reprojection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .compare import _observed, _scale
from .reports import _read_json
from .schema import AnnotationSet, LANDMARK_IDS


@dataclass(frozen=True)
class RegistrationPair:
    reference_image_id: str
    candidate_image_id: str
    anchor_landmark_ids: tuple[str, ...]


def load_registration_pairs(path: Path) -> tuple[RegistrationPair, ...]:
    """Read a bounded anchor manifest; no implicit or inferred matches."""
    raw = _read_json(path)
    if (not isinstance(raw, dict) or set(raw) != {"schema_version", "pairs"}
            or raw["schema_version"] != 1 or type(raw["schema_version"]) is not int):
        raise ValueError("Registration manifest requires schema_version=1 and pairs")
    if not isinstance(raw["pairs"], list) or not 1 <= len(raw["pairs"]) <= 256:
        raise ValueError("Registration manifest needs 1–256 pairs")
    pairs = []
    for row in raw["pairs"]:
        if (not isinstance(row, dict) or set(row) != {
                "reference_image_id", "candidate_image_id", "anchor_landmark_ids"}
                or any(not isinstance(row[key], str) or not 1 <= len(row[key]) <= 128
                       for key in ("reference_image_id", "candidate_image_id"))
                or not isinstance(row["anchor_landmark_ids"], list)):
            raise ValueError("Invalid registration pair")
        pairs.append(RegistrationPair(row["reference_image_id"], row["candidate_image_id"],
                                      tuple(row["anchor_landmark_ids"])))
    return tuple(pairs)


def _well_spread(points: list[tuple[float, float]]) -> bool:
    """Reject nearly collinear anchors, where a 2D similarity fit is unstable."""
    if len(points) < 3:
        return False
    xbar = sum(x for x, _ in points) / len(points)
    ybar = sum(y for _, y in points) / len(points)
    xx = sum((x - xbar) ** 2 for x, _ in points)
    yy = sum((y - ybar) ** 2 for _, y in points)
    xy = sum((x - xbar) * (y - ybar) for x, y in points)
    trace = xx + yy
    return trace > 1e-8 and (xx * yy - xy * xy) / (trace * trace) >= 1e-4


def _fit_similarity(reference: list[tuple[float, float]],
                    candidate: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    """Least-squares mapping candidate px → reference px, rotation + scale + translation."""
    count = len(reference)
    rx = sum(x for x, _ in reference) / count
    ry = sum(y for _, y in reference) / count
    cx = sum(x for x, _ in candidate) / count
    cy = sum(y for _, y in candidate) / count
    numerator_a = numerator_b = denominator = 0.0
    for (px, py), (qx, qy) in zip(reference, candidate):
        px, py, qx, qy = px - rx, py - ry, qx - cx, qy - cy
        numerator_a += px * qx + py * qy
        numerator_b += py * qx - px * qy
        denominator += qx * qx + qy * qy
    if denominator < 1e-8:
        return None
    a, b = numerator_a / denominator, numerator_b / denominator
    scale = math.hypot(a, b)
    if not 0.05 <= scale <= 20:
        return None
    return (a, b, rx - a * cx + b * cy, ry - b * cx - a * cy)


def _transform(xy: tuple[float, float], fit: tuple[float, float, float, float]
               ) -> tuple[float, float]:
    a, b, tx, ty = fit
    return (a * xy[0] - b * xy[1] + tx, b * xy[0] + a * xy[1] + ty)


def propose_registrations(reference: AnnotationSet, candidate: AnnotationSet,
                          pairs: Sequence[RegistrationPair]) -> dict[str, Any]:
    """Suggest 2D alignment from *manual visible* anchors, with held-out errors.

    No output from this function can set `verified=True` on the existing
    measured comparison path. Human view/camera review remains mandatory.
    """
    if not 1 <= len(pairs) <= 256:
        raise ValueError("Registration requires 1–256 pairs")
    results = []
    seen = set()
    for pair in pairs:
        key = (pair.reference_image_id, pair.candidate_image_id)
        if key in seen:
            raise ValueError(f"duplicate registration pair: {key}")
        seen.add(key)
        if pair.reference_image_id not in reference.images or pair.candidate_image_id not in candidate.images:
            raise ValueError(f"Unknown registration image pair: {key}")
        ids = pair.anchor_landmark_ids
        if not 3 <= len(ids) <= 32 or any(not isinstance(name, str) or name not in LANDMARK_IDS
                                        for name in ids):
            raise ValueError("Registration requires 3–32 known semantic anchor IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate registration anchor ID")
        ref, ren = reference.images[pair.reference_image_id], candidate.images[pair.candidate_image_id]
        result: dict[str, Any] = {
            "reference_image_id": ref.image_id, "candidate_image_id": ren.image_id,
            "anchor_landmark_ids": list(ids), "status": "unmeasured", "reason": None,
            "transform": None, "anchor_rms_normalized": None,
            "anchor_max_error_normalized": None, "heldout": {},
            "anchors_excluded_from_heldout": True,
            "camera_registration_verified": False,
            "reference_outline_px": None,
            "candidate_aligned_outline_px": None,
            "anchor_reference_px": {},
            "anchor_candidate_aligned_px": {},
        }
        ref_points, ren_points = _observed(ref), _observed(ren)
        scale = _scale(ref.outline_px)
        if ref.view == "unknown" or ref.view != ren.view:
            result["reason"] = "unknown-or-mismatched-view"
        elif scale is None:
            result["reason"] = "missing-reference-head-scale"
        elif not all(name in ref_points and name in ren_points for name in ids):
            result["reason"] = "missing-manually-reviewed-anchor"
        else:
            ref_anchors = [ref_points[name] for name in ids]
            ren_anchors = [ren_points[name] for name in ids]
            if not _well_spread(ref_anchors) or not _well_spread(ren_anchors):
                result["reason"] = "collinear-or-collapsed-anchors"
            else:
                fit = _fit_similarity(ref_anchors, ren_anchors)
                if fit is None:
                    result["reason"] = "degenerate-or-extreme-transform"
                else:
                    a, b, tx, ty = fit
                    residuals = [math.dist(point, _transform(other, fit)) / scale
                                 for point, other in zip(ref_anchors, ren_anchors)]
                    heldout = {}
                    for name in sorted((ref_points.keys() & ren_points.keys()) - set(ids)):
                        aligned = _transform(ren_points[name], fit)
                        dx = (aligned[0] - ref_points[name][0]) / scale
                        dy = (aligned[1] - ref_points[name][1]) / scale
                        heldout[name] = {
                            "reference_px": list(ref_points[name]),
                            "candidate_aligned_px": list(aligned),
                            "delta_normalized": [dx, dy],
                            "error_normalized": math.hypot(dx, dy),
                            "status": "unverified-projection",
                        }
                    result.update({
                        "status": "proposed-unverified",
                        "transform": {"candidate_to_reference": True, "scale": math.hypot(a, b),
                                      "rotation_degrees": math.degrees(math.atan2(b, a)),
                                      "translation_px": [tx, ty],
                                      "matrix_2x3": [[a, -b, tx], [b, a, ty]]},
                        "anchor_rms_normalized": math.sqrt(sum(x*x for x in residuals) / len(residuals)),
                        "anchor_max_error_normalized": max(residuals),
                        "heldout": heldout,
                        "reference_outline_px": [list(point) for point in ref.outline_px]
                                                if ref.outline_px else None,
                        "candidate_aligned_outline_px": [list(_transform(point, fit))
                                                         for point in ren.outline_px]
                                                        if ren.outline_px else None,
                        "anchor_reference_px": {name: list(ref_points[name]) for name in ids},
                        "anchor_candidate_aligned_px": {name: list(_transform(ren_points[name], fit))
                                                        for name in ids},
                        "reason": "review-camera-and-crop-before-scoring" if heldout
                                  else "no-held-out-points-to-test-likeness",
                    })
        results.append(result)
    return {
        "schema_version": 1,
        "status": "unverified-2d-registration-proposals",
        "pairs": results,
        "camera_registration_verified": False,
        "three_d_fidelity_verified": False,
        "limitations": "Similarity transforms can align 2D image landmarks, not calibrate 3D cameras. "
                       "Anchor features are excluded from held-out diagnostics. Review "
                       "view correspondence, crop and transformed overlays before any likeness claim.",
    }
