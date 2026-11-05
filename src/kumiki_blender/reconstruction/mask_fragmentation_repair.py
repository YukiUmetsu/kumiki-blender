"""Image-only fallback for strongly fragmented, nested SAM2 mask candidates.

Morphology can mend a recognizable connected silhouette, but cannot identify
limestone versus sand. Only the strict, stable nested-candidate case is
eligible; ground tracks veto additions and uncertainty remains in the report.
"""

from __future__ import annotations

from typing import Any


SCHEMA = "nested-fragmented-sam2-repair-v1"


def propose_fragmentation_repair(photo: Any, selected: Any, broad: Any,
                                 grounded: Any, *, background_seeds: list[dict],
                                 background_holdouts: list[dict],
                                 foreground_holdouts: list[dict]
                                 ) -> tuple[Any | None, dict[str, Any]]:
    """Return a conservative connected-body proposal, or abstain with reasons.

    This is specifically for photos with no independently usable 3D prompts.
    The broad proposal must be mostly nested in a larger grounded proposal.
    Local morphological agreement is a *stability* test, not a ground-truth
    score; the output always remains an unapproved candidate revision.
    """
    import numpy as np
    from PIL import Image

    details: dict[str, Any] = {"schema": SCHEMA, "accepted": False,
                               "status": "abstained / no-stable-nested-mask",
                               "reasons": []}

    def abstain(reason: str) -> tuple[None, dict[str, Any]]:
        details["reasons"].append(reason)
        return None, details

    if not (selected.size == broad.size == grounded.size == photo.size):
        return abstain("candidate-image-dimension-mismatch")
    old = np.asarray(selected.convert("L")) >= 128
    a = np.asarray(broad.convert("L")) >= 128
    b = np.asarray(grounded.convert("L")) >= 128
    a_count = int(np.count_nonzero(a))
    b_count = int(np.count_nonzero(b))
    old_count = int(np.count_nonzero(old))
    if not a_count or not b_count or not old_count:
        return abstain("empty-mask")
    containment = float(np.count_nonzero(a & b) / a_count)
    size_ratio = b_count / old_count
    details.update(broad_contained_by_grounded=round(containment, 6),
                   grounded_to_selected_area_ratio=round(size_ratio, 6))
    if (not np.array_equal(old, a) or containment < .85 or
            size_ratio < 1.70 or size_ratio > 7.0 or
            old_count / old.size > .30 or b_count / b.size > .62):
        return abstain("not-a-fragmented-nested-candidate")
    try:
        from scipy import ndimage as ndi  # type: ignore[import-not-found]
    except ImportError:
        return abstain("scipy-required-for-image-only-repair")

    # SAM2 inference is already capped in resolution. Work at 960 pixels so
    # a full-resolution photo does not cause quadratic memory/CPU work.
    scale = min(1., 960 / max(photo.size))
    dimensions = (max(1, round(photo.width * scale)),
                  max(1, round(photo.height * scale)))
    small = np.asarray(grounded.convert("L").resize(
        dimensions, Image.Resampling.NEAREST)) >= 128
    kernel = max(5, round(max(dimensions) * .011)) | 1

    def closed_body(size: int) -> tuple[Any, float]:
        closed = ndi.binary_closing(small, structure=np.ones((size, size), dtype=bool))
        labels, _ = ndi.label(closed)
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        main = labels == int(np.argmax(counts))
        coverage = int(np.count_nonzero(main)) / max(1, int(np.count_nonzero(closed)))
        return ndi.binary_fill_holes(main), coverage

    body, connected_fraction = closed_body(kernel)
    neighboring, _ = closed_body(kernel + 2)
    intersection = int(np.count_nonzero(body & neighboring))
    union = int(np.count_nonzero(body | neighboring))
    stability = intersection / max(1, union)
    repaired = np.asarray(Image.fromarray((body * 255).astype("uint8"), "L").resize(
        photo.size, Image.Resampling.NEAREST)) >= 128
    repaired_count = int(np.count_nonzero(repaired))
    details.update(kernel=kernel, analysis_dimensions=list(dimensions),
                   stable_scale_iou=round(stability, 6),
                   largest_closed_component_fraction=round(connected_fraction, 6),
                   repaired_foreground_fraction=round(repaired_count / old.size, 6))
    if (stability < .96 or connected_fraction < .85 or
            repaired_count < b_count * 1.20 or repaired_count > b_count * 1.90 or
            repaired_count / old.size > .68):
        return abstain("unstable-or-implausible-connected-body")

    # Independent ground tracks are stronger evidence than appearance or
    # candidate area. Never paint a known background point, even if it was
    # mistakenly present in an earlier candidate.
    for row in background_seeds + background_holdouts:
        x, y = row["normalized_xy"]
        pixel = (min(photo.width-1, round(x*(photo.width-1))),
                 min(photo.height-1, round(y*(photo.height-1))))
        if repaired[pixel[1], pixel[0]]:
            return abstain("verified-background-track-inside-proposal")
    for row in foreground_holdouts:
        x, y = row["normalized_xy"]
        pixel = (min(photo.width-1, round(x*(photo.width-1))),
                 min(photo.height-1, round(y*(photo.height-1))))
        if old[pixel[1], pixel[0]] and not repaired[pixel[1], pixel[0]]:
            return abstain("verified-foreground-track-lost")

    # Blue sky can be distinguished reliably enough for a veto. Stone and
    # ground cannot; do not classify sand with a global color threshold.
    from .mask_track_resegmentation import _mask_sky_leaks
    proposed = Image.fromarray((repaired * 255).astype("uint8"), "L")
    before_sky = _mask_sky_leaks(photo, selected)
    after_sky = _mask_sky_leaks(photo, proposed)
    details.update(sky_leaks_before=before_sky, sky_leaks_after=after_sky,
                   verified_ground_seeds=len(background_seeds),
                   verified_ground_holdouts=len(background_holdouts))
    # The 96x72 QA resize blends a narrow stone/sky boundary into blue pixels.
    # Tolerate at most half a percent of that fixed grid for edge aliasing.
    if after_sky > before_sky + round(.005 * 96 * 72):
        return abstain("new-blue-sky-in-proposal")
    from .mask_ground_plane import evaluate_local_ground_spill
    ground_qa = evaluate_local_ground_spill(photo, selected, proposed, background_seeds)
    details["ground_appearance_qa"] = ground_qa
    if ground_qa["rejected"]:
        return abstain("local-verified-ground-appearance-spill")
    details.update(accepted=True, status="candidate / automatic-fragmentation-repair",
                   limitations="Connected silhouette is not proof of stone-vs-ground; "
                               "independent ground tracks or optional user edits may improve it.")
    return proposed, details
