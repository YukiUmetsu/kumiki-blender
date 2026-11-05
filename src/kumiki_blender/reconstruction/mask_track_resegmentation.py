"""Immutable, selectively retried SAM2 masks from independent COLMAP tracks.

This never edits the original masks or camera reconstruction, and abstains
when candidate subject labels cannot be corroborated in other photographs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .dense import _sha256
from .mask_track_guidance import (SCHEMA as TRACK_SCHEMA,
                                  evaluate_track_resegmentation,
                                  extract_track_guidance)
from .references import ReferenceError


SCHEMA = "track-guided-mask-revision-v2"
# Inference results depend on the actual guide and checkpoint, not the QA
# algorithm version. Keep the original prompt-cache namespace so unchanged
# guides reuse prior MPS work; new negative clicks alter the cache key.
PROMPT_CACHE_SCHEMA = "track-guided-mask-revision-v1"


def _mask_sky_leaks(photo: Any, mask: Any) -> int:
    """Check confidently blue sky on a fixed pixel grid; never label stone."""
    import numpy as np
    from PIL import Image

    tiny = photo.convert("RGB").resize((96, 72), Image.Resampling.BILINEAR)
    cutout = mask.convert("L").resize((96, 72), Image.Resampling.NEAREST)
    rgb = np.asarray(tiny).astype("int16")
    subject = np.asarray(cutout) >= 128
    blue = ((rgb[:, :, 2] > 120) & (rgb[:, :, 2] > 1.15 * rgb[:, :, 0]) &
            (rgb[:, :, 2] > 1.06 * rgb[:, :, 1]))
    return int(np.count_nonzero(blue & subject))


def refine_masks_from_tracks(reconstruction: str | Path, mask_revision: str | Path,
                             output: str | Path, *, sam2_checkpoint: str | Path,
                             sam2_size: str = "small", sam2_device: str = "auto",
                             cache_dir: str | Path | None = None,
                             max_side: int = 1280, resume: bool = False,
                             progress: Callable[[int, int, str], None] | None = None
                             ) -> dict[str, Any]:
    """Build a new candidate mask revision using strict held-out feature QA."""
    from PIL import Image
    from .masking import (_DEFAULT_BOX, _auto_sam2_guide, _cached_mask, _guide,
                          _load_sam2_predictor, _sam2_mask, _save_mask_cache)

    source = Path(reconstruction).resolve()
    original = Path(mask_revision).resolve()
    target = Path(output).resolve()
    checkpoint = Path(sam2_checkpoint)
    if (Path(output).is_symlink() or Path(mask_revision).is_symlink() or
            target == original or target.is_relative_to(original) or
            target == source or target.is_relative_to(source)):
        raise ReferenceError("Track-guided output must be a new, separate immutable revision")
    if not checkpoint.is_file() or checkpoint.is_symlink() or sam2_size not in (
            "tiny", "small", "base_plus", "large") or sam2_device not in ("auto", "mps", "cpu"):
        raise ReferenceError("Track guidance needs existing SAM2 weights and a valid model/device")
    if not 256 <= max_side <= 4096:
        raise ReferenceError("Track guidance max_side must be 256–4096")
    if cache_dir is not None:
        cache_root = Path(cache_dir).resolve()
        if (Path(cache_dir).is_symlink() or cache_root.is_relative_to(target) or
                cache_root.is_relative_to(original) or cache_root.is_relative_to(source)):
            raise ReferenceError("Mask cache must be separate from all reconstruction revisions")
    else:
        cache_root = None
    report_path = original / "mask-report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise ReferenceError("Original mask revision has no trustworthy mask report")
    prior = json.loads(report_path.read_text(encoding="utf-8"))
    if not str(prior.get("status", "")).startswith("candidate"):
        raise ReferenceError("Track guidance needs a completed candidate mask revision")
    evidence = extract_track_guidance(source, original, prior)
    model_sha = _sha256(checkpoint)
    config = hashlib.sha256(json.dumps([
        SCHEMA, _sha256(report_path), evidence, model_sha, sam2_size, sam2_device,
        max_side, [item["mask_sha256"] for item in prior["images"]],
        [[i["candidate_masks"][key]["sha256"] for key in ("broad", "grounded")]
         for i in prior["images"]]], sort_keys=True).encode()).hexdigest()
    old = None
    if resume:
        if not target.is_dir() or not (target / "mask-report.json").is_file():
            raise ReferenceError("Cannot resume absent track-guided revision")
        old = json.loads((target / "mask-report.json").read_text(encoding="utf-8"))
        if (old.get("schema") != SCHEMA or old.get("settings_sha256") != config or
                old.get("status") != "incomplete / track-refinement-review-required"):
            raise ReferenceError("Track guidance resume requires identical settings and unfinished output")
    else:
        target.mkdir(parents=True, exist_ok=False)
    masks_dir = target / "masks"
    review_dir = target / "review"
    candidate_dir = target / "candidates"
    alternatives_dir = target / "alternatives"
    for folder in (masks_dir, review_dir, candidate_dir, alternatives_dir):
        if folder.is_symlink():
            raise ReferenceError("Track-refinement output cannot contain symlinks")
        folder.mkdir(exist_ok=resume)
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    (target / "track-evidence-report.json").write_text(json.dumps(
        evidence, indent=2) + "\n", encoding="utf-8")
    images = sorted(prior["images"], key=lambda entry: entry["filename"])
    # Future QA and comparisons must be self-contained: no dangling paths into
    # mutable output folders or dependence on the original candidate revision.
    for item in images:
        for variant in item["candidate_masks"].values():
            relative = Path(variant["path"])
            evidence_file = original / relative
            if (evidence_file.is_symlink() or not evidence_file.resolve().is_relative_to(original)
                    or _sha256(evidence_file) != variant["sha256"]):
                raise ReferenceError("Original alternative mask changed during refinement")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                raise ReferenceError("Alternative mask destination is a symlink")
            if destination.exists() and (not destination.is_file() or
                                         _sha256(destination) != variant["sha256"]):
                raise ReferenceError("Previously copied candidate mask changed during resume")
            if not destination.exists():
                shutil.copyfile(evidence_file, destination)
            if variant.get("preview"):
                preview_relative = Path(variant["preview"])
                original_preview = original / preview_relative
                output_preview = target / preview_relative
                if (original_preview.is_symlink() or not original_preview.resolve().is_relative_to(original)
                        or not original_preview.is_file() or output_preview.is_symlink()):
                    raise ReferenceError("Unsafe candidate preview path")
                output_preview.parent.mkdir(parents=True, exist_ok=True)
                if not output_preview.exists():
                    shutil.copyfile(original_preview, output_preview)
    completed = {entry["filename"]: entry for entry in old.get("images", [])} if old else {}
    result: dict[str, Any] = {
        "schema": SCHEMA, "schema_version": 1, "source_revision": str(source),
        "parent_mask_revision": str(original), "settings_sha256": config,
        "source_mask_report_sha256": _sha256(report_path),
        "track_evidence_sha256": _sha256(target / "track-evidence-report.json"),
        "segmentation_model_sha256": model_sha, "sam2_device": sam2_device,
        "status": "incomplete / track-refinement-review-required",
        "approved_for_reconstruction": False, "distribution_allowed": False,
        "requires_anatomy_inventory": True,
        "selection_mode": "automatic", "optional_manual_corrections": True,
        "images": [], "summary": {"accepted": 0, "rejected": 0, "abstained": 0,
                                 "cache_hits": 0, "inference_count": 0},
        "limitations": "COLMAP tracks and text grounding do not establish statue anatomy; no reduction of camera, mask, or mesh quality gates."}

    def checkpoint_report() -> None:
        # The resume receipt must never become truncated on a worker crash.
        transient = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target,
                                             prefix=".mask-report-", suffix=".json",
                                             delete=False) as stream:
                transient = Path(stream.name)
                json.dump(result, stream, indent=2)
                stream.write("\n")
            os.replace(transient, target / "mask-report.json")
            transient = None
        finally:
            if transient is not None:
                transient.unlink(missing_ok=True)

    predictor = None
    checkpoint_report()
    for position, item in enumerate(images, 1):
        name = item["filename"]
        if Path(name).name != name or name in ("", ".", ".."):
            raise ReferenceError(f"Unsafe registered image filename: {name!r}")
        source_photo = source / "undistorted/images" / name
        if source_photo.is_symlink() or _sha256(source_photo) != item["source_photo_sha256"]:
            raise ReferenceError(f"Registered image changed during mask refinement: {name}")
        selected_file = original / item["mask"]
        output_mask = masks_dir / (name + ".mask.png")
        if name in completed:
            entry = completed[name]
            if (output_mask.is_symlink() or not output_mask.is_file() or
                    _sha256(output_mask) != entry.get("mask_sha256")):
                raise ReferenceError(f"Completed mask changed during resume: {name}")
            result["images"].append(entry)
            result["summary"]["cache_hits"] += 1
            if progress:
                progress(position, len(images), "resumed")
            checkpoint_report()
            continue
        if selected_file.is_symlink() or _sha256(selected_file) != item["mask_sha256"]:
            raise ReferenceError(f"Original selected mask changed: {name}")
        shutil.copyfile(selected_file, output_mask)
        seeds = evidence["images"][name]["seed_points"]
        holdout = evidence["images"][name]["holdout_points"]
        background_seeds = evidence["images"][name]["background_seed_points"]
        background_holdouts = evidence["images"][name]["background_holdout_points"]
        receipt = {"status": "abstained / insufficient-triangulated-subject-evidence",
                   "accepted": False, "seed_count": len(seeds),
                   "heldout_count": len(holdout), "reasons": [],
                   "background_seed_count": len(background_seeds),
                   "background_holdout_count": len(background_holdouts),
                   "source_evidence": "track-evidence-report.json"}
        candidate_file = None
        if len(seeds) >= 3 and len(holdout) >= 5:
            with Image.open(source_photo) as original_photo, Image.open(selected_file) as base_mask:
                photo = original_photo.convert("RGB")
                original_binary = base_mask.copy()
                guide_base = _auto_sam2_guide(photo, _DEFAULT_BOX)
                points = [row["normalized_xy"] for row in seeds]
                if len({(int(x*5), int(y*5)) for x, y in points}) < 2:
                    receipt["reasons"].append("subject-seed-points-are-spatially-degenerate")
                else:
                    # Broad box encloses the original default box and evidence
                    # clicks; negative prompts come only from clear sky.
                    left = max(0.0, min(.12, *(x-.04 for x, _ in points)))
                    top = max(0.0, min(.04, *(y-.04 for _, y in points)))
                    right = min(1.0, max(.88, *(x+.04 for x, _ in points)))
                    bottom = min(1.0, max(.96, *(y+.04 for _, y in points)))
                    negatives = [point for point in guide_base["background_points"]
                                 if all(math.dist(point, seed) >= .05 for seed in points)]
                    used_background = []
                    for row in background_seeds:
                        point = row["normalized_xy"]
                        if (all(math.dist(point, seed) >= .08 for seed in points)
                                and all(math.dist(point, existing) >= .035 for existing in negatives)):
                            negatives.append(point)
                            used_background.append(row)
                    from .masking import _MASK_CACHE_SCHEMA
                    guide = _guide({"box": [left, top, right, bottom],
                                    "foreground_points": points,
                                    "background_points": negatives}, default_box=_DEFAULT_BOX)
                    cache_key = hashlib.sha256(json.dumps([
                        _MASK_CACHE_SCHEMA, PROMPT_CACHE_SCHEMA, item["source_photo_sha256"],
                        model_sha, evidence["model_sha256"], item["mask_sha256"],
                        sam2_size, sam2_device, max_side, guide],
                        sort_keys=True).encode()).hexdigest()
                    cache = _cached_mask(cache_root, cache_key, list(photo.size))
                    if cache is not None:
                        with Image.open(cache[0]) as cached_image:
                            refined = cached_image.copy()
                        result["summary"]["cache_hits"] += 1
                    else:
                        if predictor is None:
                            predictor, actual_device = _load_sam2_predictor(
                                checkpoint.resolve(), sam2_size, sam2_device)
                            result["effective_sam2_device"] = actual_device
                        binary, details = _sam2_mask(photo, guide, predictor, max_side)
                        refined = Image.fromarray(binary, "L")
                        result["summary"]["inference_count"] += 1
                    candidate_file = candidate_dir / (name + ".track-guided.png")
                    refined.save(candidate_file, format="PNG", optimize=True)
                    if cache is None:
                        _save_mask_cache(cache_root, cache_key, candidate_file,
                                         list(photo.size), details)
                    receipt = evaluate_track_resegmentation(
                        original_binary, refined, seeds, holdout,
                        background_seeds=used_background,
                        background_holdouts=background_holdouts)
                    receipt.update(cache_key=cache_key, cache_hit=cache is not None,
                                   proposed_mask="candidates/" + candidate_file.name,
                                   proposed_mask_sha256=_sha256(candidate_file),
                                   sky_leaks_before=_mask_sky_leaks(photo, original_binary),
                                   sky_leaks_after=_mask_sky_leaks(photo, refined),
                                   guide=guide)
                    if receipt["sky_leaks_after"] > receipt["sky_leaks_before"]:
                        receipt["accepted"] = False
                        receipt["status"] = "rejected / new-blue-sky-foreground"
                        receipt["reasons"].append("new-blue-sky-foreground")
                    if receipt["accepted"]:
                        shutil.copyfile(candidate_file, output_mask)
        result["summary"]["accepted" if receipt["accepted"] else
                          "abstained" if receipt["status"].startswith("abstained") else
                          "rejected"] += 1
        with Image.open(source_photo) as photo, Image.open(output_mask) as chosen:
            histogram = chosen.histogram()
            new_fraction = round(sum(histogram[128:]) / (chosen.width * chosen.height), 6)
            preview = photo.convert("RGB")
            preview.thumbnail((900, 900))
            mask_small = chosen.convert("L").resize(preview.size, Image.Resampling.NEAREST)
            alpha = mask_small.point(lambda value: round(value * .28))
            merged = Image.composite(Image.new("RGB", preview.size, (238, 58, 35)),
                                     preview, alpha)
            preview_path = review_dir / (name + ".jpg")
            merged.save(preview_path, "JPEG", quality=86)
        entry = {**item, "mask": "masks/" + output_mask.name,
                 "mask_sha256": _sha256(output_mask), "preview": "review/" + preview_path.name,
                 "foreground_fraction": new_fraction,
                 "track_guidance": receipt,
                 "track_guided_replaced": receipt["accepted"]}
        if receipt["accepted"]:
            entry["original_foreground_fraction"] = item.get("foreground_fraction")
            entry["original_candidate_comparison"] = entry.get("candidate_comparison")
            entry["candidate_comparison"] = {
                "status": "candidate / independent-track-recovery",
                "selected": "track-guided",
                "holdout_recovered": receipt["heldout_recovered"],
                "holdout_lost": receipt["heldout_lost"],
                "limitations": "Sparse tracks do not verify statue anatomy; normal mask and camera QA still apply"}
        result["images"].append(entry)
        checkpoint_report()
        if progress:
            progress(position, len(images), receipt["status"])
    from PIL import ImageDraw
    columns, width, height = 4, 288, 238
    contact = Image.new("RGB", (columns * width,
                               ((len(images)+columns-1)//columns)*height), (28, 31, 34))
    draw = ImageDraw.Draw(contact)
    for i, item in enumerate(result["images"]):
        x, y = i % columns * width, i // columns * height
        with Image.open(target / item["preview"]) as preview:
            preview.thumbnail((width-8, height-30))
            contact.paste(preview, (x+4, y+4))
        draw.text((x+5, y+height-23), item["filename"][:42], fill=(234, 236, 238))
    contact.save(target / "mask-contact-sheet.jpg", "JPEG", quality=85)
    result["contact_sheet"] = "mask-contact-sheet.jpg"
    result["status"] = "candidate / automatic-track-guided-qa"
    checkpoint_report()
    return result
