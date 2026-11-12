"""Independent part-level shape, placement, base color, relief and detail diagnostics.

The intrinsic silhouette is fitted by ONE uniform scale with aspect ratio retained.
It never replaces registered size/position checks. Measurements cannot certify likeness.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y, x = np.nonzero(mask)
    return (int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)) if len(x) else None


def shape_only(reference: np.ndarray, sculpt: np.ndarray) -> float | None:
    """Uniformly fit each silhouette to a common canvas; preserve aspect ratio."""
    canvas = []
    for mask in (reference, sculpt):
        box = _bounds(mask)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        crop = mask[y0:y1, x0:x1].astype("uint8")
        scale = 120.0 / max(crop.shape)
        h, w = (max(1, round(crop.shape[0] * scale)), max(1, round(crop.shape[1] * scale)))
        small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        target = np.zeros((128, 128), dtype=bool)
        target[(128-h)//2:(128-h)//2+h, (128-w)//2:(128-w)//2+w] = small
        canvas.append(target)
    a, b = canvas
    return float(np.count_nonzero(a & b) / np.count_nonzero(a | b))


def shape(reference: np.ndarray, sculpt: np.ndarray, *, registered: bool) -> dict[str, Any]:
    """Report shape separately from absolute dimension, position and overlap."""
    reference = np.asarray(reference, bool)
    sculpt = np.asarray(sculpt, bool)
    if reference.ndim != 2 or sculpt.ndim != 2:
        raise ValueError("Shape masks must be 2D")
    result: dict[str, Any] = {"intrinsic_shape_iou": shape_only(reference, sculpt),
                               "registration_verified": registered}
    a, b = _bounds(reference), _bounds(sculpt)
    if a is None or b is None:
        return {**result, "status": "missing-part", "reference_area_px": int(reference.sum()),
                "sculpt_area_px": int(sculpt.sum())}
    result["status"] = "measured-intrinsic-only" if not registered else "measured"
    result["reference_aspect_ratio"] = round((a[2]-a[0]) / (a[3]-a[1]), 5)
    result["sculpt_aspect_ratio"] = round((b[2]-b[0]) / (b[3]-b[1]), 5)
    if not registered:
        return result
    if reference.shape != sculpt.shape:
        raise ValueError("Registered masks must share image-space dimensions")
    ref_xy = np.stack(np.nonzero(reference)[::-1], axis=-1).astype(float)
    mod_xy = np.stack(np.nonzero(sculpt)[::-1], axis=-1).astype(float)
    result.update({
        "reference_area_px": int(reference.sum()), "sculpt_area_px": int(sculpt.sum()),
        "area_ratio": round(float(sculpt.sum() / reference.sum()), 5),
        "reference_bbox_xyxy": list(a), "sculpt_bbox_xyxy": list(b),
        "width_ratio": round((b[2]-b[0]) / (a[2]-a[0]), 5),
        "height_ratio": round((b[3]-b[1]) / (a[3]-a[1]), 5),
        "centroid_offset_xy_px": np.round(mod_xy.mean(axis=0)-ref_xy.mean(axis=0), 3).tolist(),
        "registered_iou": round(float(np.count_nonzero(reference & sculpt) /
                                      np.count_nonzero(reference | sculpt)), 5),
    })
    kernel = np.ones((3, 3), dtype=np.uint8)
    ra = reference & ~cv2.erode(reference.astype("uint8"), kernel).astype(bool)
    rb = sculpt & ~cv2.erode(sculpt.astype("uint8"), kernel).astype(bool)
    dist_to_ra = distance_transform_edt(~ra)
    dist_to_rb = distance_transform_edt(~rb)
    nearest = np.concatenate((dist_to_ra[rb], dist_to_rb[ra]))
    result["symmetric_contour_mean_px"] = round(float(nearest.mean()), 3)
    result["symmetric_contour_p95_px"] = round(float(np.percentile(nearest, 95)), 3)
    for tol in (2, 5, 10):
        precision = float(np.mean(dist_to_ra[rb] <= tol))
        recall = float(np.mean(dist_to_rb[ra] <= tol))
        result[f"boundary_f1_{tol}px"] = round(2 * precision * recall / max(1e-12, precision + recall), 5)
    return result


def color(reference_rgb: np.ndarray, sculpt_rgb: np.ndarray,
          reference_mask: np.ndarray, sculpt_mask: np.ndarray) -> dict[str, Any]:
    """Unlit material-vs-art color summary, not a lit-pixel or texture match."""
    reference_rgb, sculpt_rgb = np.asarray(reference_rgb), np.asarray(sculpt_rgb)
    if reference_rgb.ndim != 3 or sculpt_rgb.ndim != 3:
        raise ValueError("RGB arrays required")
    if np.count_nonzero(reference_mask) < 16 or np.count_nonzero(sculpt_mask) < 16:
        return {"status": "insufficient-pixels"}
    samples = []
    for rgb, mask in ((reference_rgb, reference_mask), (sculpt_rgb, sculpt_mask)):
        pixels = rgb[:, :, :3][mask]
        indices = np.linspace(0, len(pixels)-1, min(len(pixels), 20000)).astype(int)
        pixels = pixels[indices].astype("uint8")
        median_rgb = np.median(pixels, axis=0)
        # OpenCV's 8-bit Lab encodes L in [0,255] and a/b with +128 offsets.
        lab = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
        median_lab = np.median(lab, axis=0)
        median_lab = np.array([median_lab[0] * 100 / 255, median_lab[1] - 128,
                               median_lab[2] - 128])
        samples.append((median_rgb, median_lab))
    return {"status": "diagnostic", "metric": "CIE76-median-Lab",
            "source_median_rgb": np.round(samples[0][0], 1).tolist(),
            "sculpt_median_rgb": np.round(samples[1][0], 1).tolist(),
            "delta_e76": round(float(np.linalg.norm(samples[0][1]-samples[1][1])), 3),
            "limitations": "Painted lighting, gradients and shadowing may shift the source palette"}


def surface_detail(rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Image-space detail proxy excluding silhouettes (not PBR texture accuracy)."""
    inner = cv2.erode(mask.astype("uint8"), np.ones((5, 5), np.uint8)).astype(bool)
    if np.count_nonzero(inner) < 32:
        return {"status": "insufficient-interior"}
    gray = cv2.cvtColor(rgb[:, :, :3].astype("uint8"), cv2.COLOR_RGB2GRAY)
    laplacian = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
    return {"status": "diagnostic", "mean_abs_laplacian": round(float(laplacian[inner].mean()), 3),
            "edge_fraction_threshold_12": round(float((laplacian[inner] > 12).mean()), 5)}


def depth(sculpt_depth: np.ndarray, sculpt_mask: np.ndarray, *,
          reference_depth: np.ndarray | None = None, reference_mask: np.ndarray | None = None,
          calibrated: bool = False) -> dict[str, Any]:
    """Describe projected depth; score truth only for calibrated measured reference depth."""
    values = np.asarray(sculpt_depth, dtype=float)
    if values.shape != sculpt_mask.shape:
        raise ValueError("Depth array and part mask must match")
    selected = sculpt_mask & np.isfinite(values) & (values > 0)
    if selected.sum() < 16:
        return {"status": "missing-visible-depth"}
    d = values[selected]
    report: dict[str, Any] = {"status": "sculpt-only / not-source-verified",
        "sculpt_visible_depth_m_p10_p50_p90": np.round(np.percentile(d, [10, 50, 90]), 5).tolist(),
        "visible_depth_span_m": round(float(np.percentile(d, 90)-np.percentile(d, 10)), 5)}
    if reference_depth is None or not calibrated:
        return report
    ref = np.asarray(reference_depth, dtype=float)
    if ref.shape != values.shape or reference_mask is None or reference_mask.shape != values.shape:
        raise ValueError("Calibrated source depth and masks must share render coordinates")
    valid = selected & reference_mask & np.isfinite(ref) & (ref > 0)
    if valid.sum() < 16:
        return {**report, "reason": "insufficient-depth-correspondence"}
    absolute = np.abs(ref[valid] - values[valid])
    return {**report, "status": "measured-calibrated-depth", "matched_pixels": int(valid.sum()),
        "depth_mae_m": round(float(absolute.mean()), 6),
        "depth_p95_abs_m": round(float(np.percentile(absolute, 95)), 6)}
