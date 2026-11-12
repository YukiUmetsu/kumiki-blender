"""Optional advanced evidence: contour geometry, perceptual color, relative depth.

Never combine these diagnostics into an artistic approval score. This module has
no remote dependencies and never loads or downloads AI weights.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr


def contour_geometry(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Compare intrinsic part silhouette, independently of camera placement.

    Normalize uniformly to a 256-square canvas (aspect ratio preserved). The
    Hausdorff P100 and P95 distances are normalized by the canvas diagonal.
    A curvature histogram captures narrow tips versus round masses, without
    assuming point-to-point contour correspondences or a common start vertex.
    """
    def normalize(mask):
        mask = np.asarray(mask, bool)
        yy, xx = np.nonzero(mask)
        if not len(xx):
            return None
        crop = mask[yy.min():yy.max()+1, xx.min():xx.max()+1].astype(np.uint8)
        scale = 240 / max(crop.shape)
        w, h = max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))
        small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((256, 256), np.uint8)
        canvas[(256-h)//2:(256-h)//2+h, (256-w)//2:(256-w)//2+w] = small
        return canvas

    a, b = normalize(reference), normalize(candidate)
    if a is None or b is None:
        return {"status": "missing-part"}

    def edge(mask):
        return (mask > 0) & ~cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)

    ea, eb = edge(a), edge(b)
    da, db = distance_transform_edt(~ea), distance_transform_edt(~eb)
    distances = np.concatenate((da[eb], db[ea])) / np.hypot(256, 256)

    def curvature_hist(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.zeros(8)
        curve = max(contours, key=cv2.contourArea)[:, 0, :].astype(float)
        if len(curve) < 12:
            return np.zeros(8)
        # Across six sample points on either side; limits pixel-grid noise.
        before = np.roll(curve, 6, axis=0) - curve
        after = np.roll(curve, -6, axis=0) - curve
        cross = before[:, 0]*after[:, 1] - before[:, 1]*after[:, 0]
        dot = (before*after).sum(axis=1)
        angles = np.arctan2(np.abs(cross), dot)
        hist, _ = np.histogram(angles, bins=8, range=(0, np.pi))
        return hist / max(1, hist.sum())

    ah, bh = curvature_hist(a), curvature_hist(b)
    return {"status": "diagnostic",
            "intrinsic_contour_distance_mean_normalized": round(float(distances.mean()), 6),
            "intrinsic_contour_distance_p95_normalized": round(float(np.percentile(distances, 95)), 6),
            "intrinsic_hausdorff_normalized": round(float(distances.max()), 6),
            "curvature_histogram_tvd": round(float(np.abs(ah-bh).sum()/2), 6),
            "limitations": "2D contour only; curvature distribution lacks spatial correspondence"}


def perceptual_color(reference_rgb: np.ndarray, candidate_rgb: np.ndarray,
                     reference_mask: np.ndarray, candidate_mask: np.ndarray,
                     *, registered: bool) -> dict:
    """CIEDE2000 for median palette and, only if registered, overlapping pixels."""
    from skimage.color import deltaE_ciede2000, rgb2lab

    if reference_mask.sum() < 16 or candidate_mask.sum() < 16:
        return {"status": "insufficient-pixels"}
    def lab(rgb):
        return rgb2lab(np.asarray(rgb[..., :3], dtype=float)/255.0)
    ref_lab, cand_lab = lab(reference_rgb), lab(candidate_rgb)
    ref_median = np.median(ref_lab[reference_mask], axis=0)
    cand_median = np.median(cand_lab[candidate_mask], axis=0)
    result = {"status": "diagnostic", "median_palette_delta_e2000": round(float(
        deltaE_ciede2000(ref_median, cand_median)), 4)}
    if registered and ref_lab.shape == cand_lab.shape:
        common = reference_mask & candidate_mask
        if common.sum() >= 16:
            deltas = deltaE_ciede2000(ref_lab[common], cand_lab[common])
            result["registered_overlap_delta_e2000_mean"] = round(float(deltas.mean()), 4)
            result["registered_overlap_delta_e2000_p95"] = round(float(np.percentile(deltas, 95)), 4)
            result["registered_overlap_pixels"] = int(common.sum())
    result["limitations"] = "Source illustration may contain artistic lighting; albedo comparisons are not material truth"
    return result


def relative_depth_hint(reference_hint: np.ndarray, sculpt_depth: np.ndarray,
                        reference_mask: np.ndarray, sculpt_mask: np.ndarray,
                        *, verified_alignment: bool, near_is_small: bool = True) -> dict:
    """Compare ordered depth only; never output metric source-depth errors.

    Used for AI-predicted/illustrated *relative* depth. A high correlation is
    weak evidence and must never be considered measured 3D accuracy.
    """
    if not verified_alignment:
        return {"status": "not-comparable", "reason": "unverified-registration"}
    if reference_hint.shape != sculpt_depth.shape or reference_hint.shape != reference_mask.shape or reference_hint.shape != sculpt_mask.shape:
        raise ValueError("Relative depth evidence requires matching registered image dimensions")
    valid = reference_mask & sculpt_mask & np.isfinite(reference_hint) & np.isfinite(sculpt_depth) & (sculpt_depth > 0)
    if valid.sum() < 32:
        return {"status": "insufficient-correspondence"}
    source = np.asarray(reference_hint[valid], float)
    target = np.asarray(sculpt_depth[valid], float)
    if not near_is_small:
        source = -source
    # Spearman invariant to monotonic shifts and scales in the estimation model.
    if np.ptp(source) <= 1e-7 or np.ptp(target) <= 1e-7:
        return {"status": "insufficient-depth-variation", "matched_pixels": int(valid.sum())}
    if valid.sum() > 50000:
        idx = np.linspace(0, valid.sum()-1, 50000, dtype=int)
        source, target = source[idx], target[idx]
    rho = float(spearmanr(source, target).statistic)
    return {"status": "unverified-relative-depth-hint", "rank_correlation_spearman": round(rho, 5),
            "matched_pixels": int(valid.sum()),
            "limitations": "Model-predicted source depth may be geometrically wrong, especially on paintings; no metric-depth claim"}
