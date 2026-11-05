"""Conservative QA for SAM2 unions from automatically discovered object regions."""

from __future__ import annotations

from typing import Any


SCHEMA = "component-guided-mask-qa-v1"


def evaluate_component_recovery(photo: Any, current: Any, proposed: Any,
                                grounded: Any, *, guide: dict[str, Any],
                                foreground_holdouts: list[dict],
                                background_seeds: list[dict],
                                background_holdouts: list[dict]
                                ) -> tuple[Any | None, dict[str, Any]]:
    """Check connectedness, independent labels and source-mask proximity.

    This image-only path is intentionally not semantic proof: most new pixels
    must stay near pixels independently suggested by Grounding DINO + SAM2.
    Background tracks and sky pixels act as vetoes; no generated component is
    accepted merely because a model predicts a high score.
    """
    import numpy as np
    from PIL import Image

    receipt: dict[str, Any] = {"schema": SCHEMA, "accepted": False,
                               "status": "rejected / unverified-component-proposal",
                               "reasons": []}

    def reject(reason: str) -> tuple[None, dict[str, Any]]:
        receipt["reasons"].append(reason)
        return None, receipt

    if photo.size != current.size or photo.size != proposed.size or photo.size != grounded.size:
        return reject("image-mask-size-mismatch")
    old = np.asarray(current.convert("L")) >= 128
    try:
        from scipy import ndimage as ndi  # type: ignore[import-not-found]
    except ImportError:
        return reject("scipy-required")
    small_dims = (max(8, round(photo.width*min(1., 512/max(photo.size)))),
                  max(8, round(photo.height*min(1., 512/max(photo.size)))))
    small = np.asarray(proposed.convert("L").resize(
        small_dims, Image.Resampling.NEAREST)) >= 128
    kernel = max(3, round(max(small_dims)*.008)) | 1
    closed = ndi.binary_closing(small, structure=np.ones((kernel, kernel), dtype=bool))
    labels, groups = ndi.label(closed)
    if not groups:
        return reject("empty-connected-proposal")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    main = labels == int(np.argmax(counts))
    connected_fraction = np.count_nonzero(main) / max(1, np.count_nonzero(closed))
    body = ndi.binary_fill_holes(main)
    repaired = np.asarray(Image.fromarray((body*255).astype("uint8"), "L").resize(
        photo.size, Image.Resampling.NEAREST)) >= 128
    old_count = np.count_nonzero(old)
    new_count = np.count_nonzero(repaired)
    retained = np.count_nonzero(repaired & old) / max(1, old_count)
    growth = new_count / max(1, old_count)
    receipt.update(before_fraction=round(old_count/old.size, 6),
                   after_fraction=round(new_count/old.size, 6),
                   retained_previous_fraction=round(retained, 6),
                   area_growth=round(growth, 6),
                   connected_component_fraction=round(connected_fraction, 6))
    if (old_count == 0 or growth < 1.04 or growth > 1.45 or
            new_count/old.size > .72 or retained < .88 or
            connected_fraction < .88):
        return reject("implausible-area-or-lost-original-silhouette")

    # A component mask may close gaps in the earlier grounded prediction, but
    # must not roam across unrelated walls, ground or other objects.
    candidate_small = np.asarray(Image.fromarray((repaired*255).astype("uint8"), "L").resize(
        small_dims, Image.Resampling.NEAREST)) >= 128
    prior_small = np.asarray(current.convert("L").resize(
        small_dims, Image.Resampling.NEAREST)) >= 128
    supported_small = np.asarray(grounded.convert("L").resize(
        small_dims, Image.Resampling.NEAREST)) >= 128
    envelope = ndi.binary_dilation(supported_small,
                                   iterations=max(2, round(max(small_dims)*.015)))
    added = candidate_small & ~prior_small
    unsupported = np.count_nonzero(added & ~envelope) / max(1, np.count_nonzero(added))
    receipt["added_outside_grounded_envelope_fraction"] = round(unsupported, 6)
    if unsupported > .12:
        return reject("expansion-unsupported-by-independent-grounded-candidate")

    def hits(rows: list[dict], mask: Any) -> int:
        return sum(bool(mask[min(photo.height-1, round(row["normalized_xy"][1] * (photo.height-1))),
                             min(photo.width-1, round(row["normalized_xy"][0] * (photo.width-1)))])
                   for row in rows)

    negatives = background_seeds + background_holdouts
    receipt["verified_background_hits"] = hits(negatives, repaired)
    if receipt["verified_background_hits"]:
        return reject("verified-background-point-in-component-union")
    lost_foreground = sum(
        bool(old[min(photo.height-1, round(row["normalized_xy"][1]*(photo.height-1))),
                 min(photo.width-1, round(row["normalized_xy"][0]*(photo.width-1)))]) and
        not bool(repaired[min(photo.height-1, round(row["normalized_xy"][1]*(photo.height-1))),
                          min(photo.width-1, round(row["normalized_xy"][0]*(photo.width-1)))])
        for row in foreground_holdouts)
    receipt["lost_foreground_holdouts"] = lost_foreground
    if lost_foreground > 0:
        return reject("lost-independent-foreground-holdout")
    for points in guide.get("part_points", {}).values():
        if any(not repaired[min(photo.height-1, round(y*(photo.height-1))),
                            min(photo.width-1, round(x*(photo.width-1)))] for x, y in points):
            return reject("discovered-region-positive-point-missing")

    from .mask_track_resegmentation import _mask_sky_leaks
    before_sky = _mask_sky_leaks(photo, current)
    after_sky = _mask_sky_leaks(photo, Image.fromarray((repaired*255).astype("uint8"), "L"))
    receipt.update(sky_leaks_before=before_sky, sky_leaks_after=after_sky)
    if after_sky > before_sky:
        return reject("new-blue-sky-foreground")
    from .mask_ground_plane import evaluate_local_ground_spill
    proposal = Image.fromarray((repaired*255).astype("uint8"), "L")
    ground_qa = evaluate_local_ground_spill(photo, current, proposal, background_seeds)
    receipt["ground_appearance_qa"] = ground_qa
    if ground_qa["rejected"]:
        return reject("verified-local-ground-appearance-spill")
    receipt.update(accepted=True, status="candidate / auto-discovered-component-recovery",
                   limitations="No independent track evidence means component semantics and ground remain uncertain.")
    return proposal, receipt
