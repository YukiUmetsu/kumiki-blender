"""Conservative SAM2 candidate comparison; segmentation is never geometric proof.

Grounding DINO can localize a statue's head while missing most of its body.
Compare its SAM2 result with the independent broad-box SAM2 proposal before
choosing which *unverified* candidate to expose to reconstruction QA.
"""

from __future__ import annotations

from typing import Any


SCHEMA = "grounded-sam2-candidate-ensemble-v1"


def select_mask_candidate(broad: Any, grounded: Any, detection: dict[str, Any] | None
                          ) -> tuple[str, dict[str, Any]]:
    """Avoid losing anatomy to a narrow detector box; flag ambiguous masks.

    Area and agreement measure uncertainty, not foreground correctness. In
    particular, a large broad mask could include the ground or architecture:
    retaining it still requires human and multi-view review.
    """
    import numpy as np

    a = np.asarray(broad) >= 128
    b = np.asarray(grounded) >= 128
    if a.ndim != 2 or a.shape != b.shape or not a.size:
        raise ValueError("SAM2 mask candidate dimensions do not match")
    a_count, b_count = int(np.count_nonzero(a)), int(np.count_nonzero(b))
    intersection = int(np.count_nonzero(a & b))
    union = a_count + b_count - intersection
    overlap = round(intersection / union, 6) if union else 0.0
    area_ratio = round(b_count / a_count, 6) if a_count else 0.0
    reasons = []
    if not detection or not str(detection.get("status", "")).startswith("candidate"):
        reasons.append("no-confident-subject-detection")
    if a_count == 0 or b_count == 0:
        reasons.append("empty-candidate")
    if b_count < a_count * 0.70:
        reasons.append("grounded-candidate-loses-over-30-percent-of-broad-foreground")
    if b_count > a_count * 1.35:
        reasons.append("grounded-candidate-grows-over-35-percent-beyond-broad-foreground")
    if overlap < 0.55:
        reasons.append("low-broad-grounded-overlap")
    if b_count / a.size < 0.05:
        reasons.append("grounded-foreground-below-five-percent")
    choice = "broad" if reasons else "grounded"
    return choice, {
        "schema": SCHEMA, "selected": choice,
        "status": "uncertain / competing-candidates" if reasons else "candidate / similar-silhouettes",
        "reasons": reasons, "broad_foreground_fraction": round(a_count / a.size, 6),
        "grounded_foreground_fraction": round(b_count / a.size, 6),
        "grounded_to_broad_area_ratio": area_ratio,
        "intersection_over_union": overlap,
        "limitations": "Area/overlap are not verified subject boundaries; audit registered feature tracks and original photos.",
    }
