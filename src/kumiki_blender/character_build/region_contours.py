"""Reference-registered silhouette boundary metrics for small character parts.

Area IoU can stay high when a narrow feather, eyelid lip or muzzle tip is
deleted. Compare outline *location* in the same registered 400x300 source QA
grid, with no camera registration, rescaling or silhouette auto-alignment.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

_CROSS = np.array([[False, True, False],
                   [True, True, True],
                   [False, True, False]], dtype=bool)


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """One-pixel 4-connected internal outline, including holes and thin parts."""
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or min(value.shape) < 2:
        raise ValueError("Reference contour mask must be a 2D registered raster")
    return value & ~binary_erosion(value, structure=_CROSS, border_value=0)


def silhouette_boundary_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Symmetric signed-invariant boundary agreement in 400x300 QA pixels.

    F1(2 px) is a symmetric 2-pixel tolerance boundary precision/recall
    measure; the additional 1-pixel F1 exposes small details independently
    of the more robust 2-pixel gate. Chamfer is mean of the two directed
    nearest-boundary distances, and 95th percentile uses the worst direction.
    None means that one or both outlines were empty; it is *not* success.
    """
    ref = np.asarray(reference, dtype=bool)
    cand = np.asarray(candidate, dtype=bool)
    if ref.shape != cand.shape or ref.ndim != 2:
        raise ValueError("Reference and candidate masks must have equal 2D shapes")
    outline_ref = boundary_pixels(ref)
    outline_cand = boundary_pixels(cand)
    count_ref, count_cand = int(outline_ref.sum()), int(outline_cand.sum())
    result = {
        "boundary_reference_pixels": count_ref,
        "boundary_candidate_pixels": count_cand,
        "boundary_f1_1px": None,
        "boundary_f1_2px": None,
        "boundary_mean_distance_px": None,
        "boundary_p95_distance_px": None,
        "boundary_metric_grid": "registered 400x300 QA raster; 1px = 4 source-image pixels",
    }
    if not count_ref or not count_cand:
        return result
    ref_distance = distance_transform_edt(~outline_ref)[outline_cand]
    cand_distance = distance_transform_edt(~outline_cand)[outline_ref]
    for tolerance in (1, 2):
        precision = float(np.mean(ref_distance <= tolerance))
        recall = float(np.mean(cand_distance <= tolerance))
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        result[f"boundary_f1_{tolerance}px"] = round(f1, 6)
    result["boundary_mean_distance_px"] = round(
        (float(ref_distance.mean()) + float(cand_distance.mean())) / 2, 6)
    result["boundary_p95_distance_px"] = round(max(
        float(np.percentile(ref_distance, 95)),
        float(np.percentile(cand_distance, 95))), 6)
    return result
