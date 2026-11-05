"""Conservative QA for SAM2 unions from automatically discovered object regions."""

from __future__ import annotations

from typing import Any


SCHEMA = "component-guided-mask-qa-v4"


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
    # Closing with the default zero-valued boundary erases real object pixels\n    # whenever a cropped head, paw, or torso reaches the image edge. Extend\n    # observed boundary values before cleanup, then return to original size.\n    border = kernel // 2\n    padded = np.pad(small, border, mode="edge")\n    closed = ndi.binary_closing(\n        padded, structure=np.ones((kernel, kernel), dtype=bool))[border:-border, border:-border]
    labels, groups = ndi.label(closed)
    if not groups:
        return reject("empty-connected-proposal")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    primary_label = int(np.argmax(counts))
    main = labels == primary_label
    # An occluder can split one real object into separate image components.
    # Never retain arbitrary SAM2 islands, but allow a bounded secondary
    # component containing a *cross-view foreground seed* and also supported
    # by the independently generated grounded mask. The later negative/holdout
    # checks still veto leaked ground, scaffolding and other objects.
    grounded_small = np.asarray(grounded.convert("L").resize(
        small_dims, Image.Resampling.NEAREST)) >= 128
    retained_islands: list[dict[str, Any]] = []
    for name, points in guide.get("part_points", {}).items():
        if not name.startswith("tracked_"):
            continue
        for x, y in points:
            px = min(small_dims[0]-1, max(0, round(x*(small_dims[0]-1))))
            py = min(small_dims[1]-1, max(0, round(y*(small_dims[1]-1))))
            label = int(labels[py, px])
            if label == 0 or label == primary_label or any(
                    item["label"] == label for item in retained_islands):
                continue
            island = labels == label
            size = int(counts[label])
            independently_supported = float(np.count_nonzero(island & grounded_small) /
                                            max(1, size))
            if (size < .005*closed.size or size > .25*counts[primary_label] or
                    independently_supported < .15):
                continue
            main |= island
            retained_islands.append({"label": label, "source": name,
                                     "fraction": round(size/closed.size, 6),
                                     "grounded_support_fraction": round(independently_supported, 6)})
    receipt["retained_tracked_islands"] = retained_islands
    connected_fraction = np.count_nonzero(main) / max(1, np.count_nonzero(closed))
    # Fill holes per component. Joining unrelated components before filling can
    # accidentally bridge a large, unobserved background region.
    selected_labels = {primary_label, *(item["label"] for item in retained_islands)}
    body = np.zeros_like(main)
    for label in selected_labels:
        body |= ndi.binary_fill_holes(labels == label)
    repaired = np.asarray(Image.fromarray((body*255).astype("uint8"), "L").resize(
        photo.size, Image.Resampling.NEAREST)) >= 128
    old_count = np.count_nonzero(old)
    new_count = np.count_nonzero(repaired)
    retained = float(np.count_nonzero(repaired & old) / max(1, old_count))
    growth = float(new_count / max(1, old_count))
    def hits(rows: list[dict], mask: Any) -> int:
        return sum(bool(mask[min(photo.height-1, round(row["normalized_xy"][1] * (photo.height-1))),
                             min(photo.width-1, round(row["normalized_xy"][0] * (photo.width-1)))])
                   for row in rows)

    # Shrinking a mask is permitted only when separately triangulated ground
    # seeds are actually removed. No color-only guess may erase torso pixels.
    ground_seed_hits_before = hits(background_seeds, old)
    ground_seed_hits_after = hits(background_seeds, repaired)
    trimming_ground = growth < .98
    receipt.update(operation="ground-trim" if trimming_ground else "body-recovery",
                   verified_background_seed_hits_before=ground_seed_hits_before,
                   verified_background_seed_hits_after=ground_seed_hits_after)
    receipt.update(before_fraction=round(float(old_count/old.size), 6),
                   after_fraction=round(float(new_count/old.size), 6),
                   retained_previous_fraction=round(retained, 6),
                   area_growth=round(growth, 6),
                   connected_component_fraction=round(connected_fraction, 6))
    # A verified, separately visible section of the same object can account
    # for additional growth, but an ordinary single-component proposal cannot.
    maximum_growth = 1.60 if retained_islands else 1.45
    if (old_count == 0 or new_count/old.size > .72 or connected_fraction < .88 or
            (trimming_ground and (growth < .68 or retained < .68 or
                                  ground_seed_hits_before < 2 or ground_seed_hits_after)) or
            (not trimming_ground and (growth < 1.04 or growth > maximum_growth or retained < .88))):
        return reject("implausible-area-or-lost-original-silhouette")
    if trimming_ground and len(foreground_holdouts) < 3:
        return reject("ground-trim-needs-independent-foreground-holdouts")

    # A component mask may close gaps in the earlier grounded prediction, but
    # must not roam across unrelated walls, ground or other objects.
    candidate_small = np.asarray(Image.fromarray((repaired*255).astype("uint8"), "L").resize(
        small_dims, Image.Resampling.NEAREST)) >= 128
    prior_small = np.asarray(current.convert("L").resize(
        small_dims, Image.Resampling.NEAREST)) >= 128
    envelope = ndi.binary_dilation(grounded_small,
                                   iterations=max(2, round(max(small_dims)*.015)))
    added = candidate_small & ~prior_small
    unsupported = np.count_nonzero(added & ~envelope) / max(1, np.count_nonzero(added))
    receipt["added_outside_grounded_envelope_fraction"] = round(unsupported, 6)
    if unsupported > .12:
        return reject("expansion-unsupported-by-independent-grounded-candidate")

    negatives = background_seeds + background_holdouts
    receipt["verified_background_hits"] = hits(negatives, repaired)
    if receipt["verified_background_hits"]:
        return reject("verified-background-point-in-component-union")
    prompt_negatives = [{"normalized_xy": p} for p in guide.get("background_points", [])]
    receipt["sam2_negative_prompt_hits"] = hits(prompt_negatives, repaired)
    if receipt["sam2_negative_prompt_hits"]:
        return reject("sam2-negative-prompt-violated")
    lost_foreground = sum(
        bool(old[min(photo.height-1, round(row["normalized_xy"][1]*(photo.height-1))),
                 min(photo.width-1, round(row["normalized_xy"][0]*(photo.width-1)))]) and
        not bool(repaired[min(photo.height-1, round(row["normalized_xy"][1]*(photo.height-1))),
                          min(photo.width-1, round(row["normalized_xy"][0]*(photo.width-1)))])
        for row in foreground_holdouts)
    receipt["lost_foreground_holdouts"] = lost_foreground
    if lost_foreground > 0:
        return reject("lost-independent-foreground-holdout")
    region_points = [point for points in guide.get("part_points", {}).values()
                     for point in points]
    present = sum(bool(repaired[min(photo.height-1, round(y*(photo.height-1))),
                                min(photo.width-1, round(x*(photo.width-1)))])
                  for x, y in region_points)
    receipt["retained_region_points"] = present
    if (not region_points or
            (trimming_ground and present < max(2, round(.6*len(region_points)))) or
            (not trimming_ground and present != len(region_points))):
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
    receipt.update(accepted=True, status=("candidate / verified-ground-seed-trim"
                                          if trimming_ground else
                                          "candidate / auto-discovered-component-recovery"),
                   limitations=("Background seeds corroborated across views are still not semantic ground truth."
                                if trimming_ground else
                                "Without independent track evidence, component semantics and ground remain uncertain."))
    return proposal, receipt
