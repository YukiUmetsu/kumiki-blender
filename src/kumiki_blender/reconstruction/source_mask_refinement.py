"""Conservative guidance from independently inferred foreground proposals.

RemBG/IS-Net is fallible on stone, fur and damaged surfaces. Its disagreement
with SAM2 is an uncertainty signal, not a license to erase object anatomy.
"""

from __future__ import annotations

import numpy as np


def _binary(array):
    image = np.asarray(array)
    if image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ValueError("Foreground proposal must be a finite grayscale image")
    return image if image.dtype.kind == "b" else image >= 128


def mask_agreement(primary, auxiliary) -> dict:
    """Tri-state subject/uncertain/background; never turn disagreement into background."""
    first, second = _binary(primary), _binary(auxiliary)
    if first.shape != second.shape:
        raise ValueError("Independent foreground proposals have different dimensions")
    primary_fraction = float(first.mean())
    auxiliary_fraction = float(second.mean())
    intersection = first & second
    union = first | second
    iou = float(intersection.sum() / max(1, union.sum()))
    # An IS-Net head-only or nearly-full-frame proposal offers no reliable
    # supervision for the torso or ground. Leave the primary classification.
    usable = (0.05 <= auxiliary_fraction <= 0.90 and
              intersection.sum() >= 0.35 * max(1, first.sum()) and iou >= 0.30)
    labels = np.full(first.shape, 128, dtype=np.uint8)
    if usable:
        labels[intersection] = 255
        labels[~union] = 0
    return {"usable": bool(usable), "labels": labels,
            "disagreement_fraction": round(float(np.not_equal(first, second).mean()), 4),
            "overlap_iou": round(iou, 4),
            "auxiliary_fraction": round(auxiliary_fraction, 4),
            "primary_fraction": round(primary_fraction, 4)}


def propose_refinement_guide(primary, auxiliary) -> tuple[dict | None, dict]:
    """Sample only *peripheral* disagreements as negative SAM2 click prompts.

    Interior IS-Net background may actually be the subject's body or paws.
    Foreground clicks require agreement, and outside peripheral pixels must
    belong to the primary SAM2 mask to target a plausible false-positive area.
    """
    confidence = mask_agreement(primary, auxiliary)
    report = {key: value for key, value in confidence.items() if key != "labels"}
    if not confidence["usable"]:
        return None, {**report, "status": "suspect / unusable-auxiliary-mask"}
    first, second = _binary(primary), _binary(auxiliary)
    h, w = first.shape
    positives, negatives = [], []
    for y in np.linspace(0.035, 0.965, 13):
        for x in np.linspace(0.035, 0.965, 17):
            xx, yy = min(w - 1, int(round(x * (w - 1)))), min(h - 1, int(round(y * (h - 1))))
            if first[yy, xx] and second[yy, xx]:
                positives.append([round(float(x), 5), round(float(y), 5)])
            if (first[yy, xx] and not second[yy, xx] and
                    (x < 0.16 or x > 0.84 or y < 0.12 or y > 0.88)):
                negatives.append([round(float(x), 5), round(float(y), 5)])
    if len(positives) < 3 or not negatives:
        return None, {**report, "status": "candidate / insufficient-safe-prompt-coverage"}

    def distributed(points: list[list[float]], limit: int) -> list[list[float]]:
        """Deterministic farthest-point sampling retains head *and* body coverage."""
        positions = np.asarray(points)
        chosen = [int(np.argmin(np.sum((positions - positions.mean(axis=0)) ** 2, axis=1)))]
        distance = np.sum((positions - positions[chosen[0]]) ** 2, axis=1)
        while len(chosen) < min(limit, len(points)):
            distance[chosen] = -1
            nxt = int(np.argmax(distance))
            chosen.append(nxt)
            distance = np.minimum(distance, np.sum((positions - positions[nxt]) ** 2, axis=1))
        return [points[index] for index in chosen]

    positives, negatives = distributed(positives, 12), distributed(negatives, 12)
    return {"foreground_points": positives, "background_points": negatives}, {
        **report, "status": "candidate / guided-refinement",
        "positive_prompts": len(positives), "negative_prompts": len(negatives)}


def select_refined_candidate(primary, refined, auxiliary) -> tuple[np.ndarray, dict]:
    """Accept only refinement that preserves agreed subject and reduces leakage."""
    first, second, other = _binary(primary), _binary(refined), _binary(auxiliary)
    if first.shape != second.shape or first.shape != other.shape:
        raise ValueError("Refined mask dimensions must match both original proposals")
    anchors = first & other
    retained = float(np.logical_and(second, anchors).sum() / max(1, anchors.sum()))
    before = int(first.sum())
    after = int(second.sum())
    primary_retained = float(np.logical_and(second, first).sum() / max(1, before))
    fraction = float(second.mean())
    # Candidate may remove background but cannot remove whole anatomical
    # regions where independent models agreed. Never fuse predicted geometry.
    accepted = (0.05 <= fraction <= 0.90 and retained >= 0.90 and
                primary_retained >= 0.75 and after <= before * 1.12)
    return np.where(second if accepted else first, 255, 0).astype(np.uint8), {
        "attempted": True, "accepted": bool(accepted),
        "consensus_subject_retained": round(retained, 4),
        "initial_subject_retained": round(primary_retained, 4),
        "initial_area": before, "refined_area": after,
        "reason": None if accepted else "unverified-subject-loss-or-background-expansion"}
