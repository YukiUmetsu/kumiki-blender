"""Review-only SAM2 retries for a few broken masks in an immutable SfM revision.

The source masks and registered camera revision are never modified. Candidate
outputs are *not* dense-ready masks and require photographic QA before use.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from .dense import DenseSettings, _json, _sha256, plan_dense_reconstruction
from .mask_shape_qa import inspect_mask_shape
from .mask_track_resegmentation import _mask_sky_leaks
from .references import ReferenceError


def _atomic_report(path: Path, report: dict) -> None:
    transient = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".sam2-review-", suffix=".json",
                                         delete=False) as stream:
            transient = Path(stream.name)
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(transient, path)
    finally:
        if transient is not None:
            transient.unlink(missing_ok=True)


def _prompt_qa(mask, guide: dict) -> dict:
    """Measure prompt compliance without treating clicks as geometry proof."""
    binary = mask.convert("L")
    positives = list(guide["foreground_points"])
    for rows in guide["part_points"].values():
        positives.extend(rows)
    negatives = guide["background_points"]

    def marked(point: list[float]) -> bool:
        x, y = point
        return binary.getpixel((round(x * (binary.width - 1)),
                                round(y * (binary.height - 1)))) >= 128

    parts = {part: {"points": len(rows),
                    "missed": sum(not marked(point) for point in rows)}
             for part, rows in guide["part_points"].items()}
    return {"positive_points": len(positives),
            "missing_positive_points": sum(not marked(point) for point in positives),
            "negative_points": len(negatives),
            "violated_negative_points": sum(marked(point) for point in negatives),
            "anatomical_parts": parts}


def _pixel_change_qa(original, proposed) -> dict:
    """Compare masks on a bounded grid; pixel changes are not quality proof."""
    import numpy as np
    from PIL import Image

    reference = original.convert("L")
    candidate = proposed.convert("L")
    reference.thumbnail((384, 384), Image.Resampling.NEAREST)
    candidate.thumbnail((384, 384), Image.Resampling.NEAREST)
    old = np.asarray(reference) >= 128
    new = np.asarray(candidate) >= 128
    if old.shape != new.shape:
        raise ReferenceError("SAM2 mask comparisons require identical dimensions")
    total = old.size
    intersection = int(np.count_nonzero(old & new))
    union = int(np.count_nonzero(old | new))
    return {"sampled_dimensions": list(reference.size),
            "original_foreground_fraction": round(float(np.count_nonzero(old) / total), 6),
            "candidate_foreground_fraction": round(float(np.count_nonzero(new) / total), 6),
            "newly_included_fraction": round(float(np.count_nonzero(new & ~old) / total), 6),
            "newly_excluded_fraction": round(float(np.count_nonzero(old & ~new) / total), 6),
            "intersection_over_union": round(intersection / max(1, union), 6),
            "warning": "These are area-change measurements, not a silhouette correctness score."}


def retry_sam2_masks(reconstruction: str | Path, source_masks: str | Path,
                     guides: str | Path, output: str | Path, *,
                     targets: list[str], sam2_checkpoint: str | Path,
                     sam2_size: str = "small", sam2_device: str = "auto",
                     max_side: int = 1280, predictor: Any = None) -> dict:
    """Generate separate, labeled SAM2 candidates for selected registered photos.

    Strong head/body/paw clicks must be supplied explicitly. No predicted
    matches, mask-vote majorities, or detector boxes are silently trusted as
    subject pixels. The command cannot replace production masks.
    """
    from PIL import Image
    from .masking import (_DEFAULT_BOX, SAM2_SELECTION_VERSION, _load_guides,
                          _load_sam2_predictor, _sam2_mask)

    if (not isinstance(targets, list) or not 1 <= len(targets) <= 12 or
            len(set(targets)) != len(targets) or
            any(not isinstance(name, str) or Path(name).name != name for name in targets)):
        raise ReferenceError("Retry requires 1–12 distinct, safe registered photo filenames")
    if type(max_side) is not int or not 256 <= max_side <= 4096:
        raise ReferenceError("SAM2 retry max_side must be 256–4096")
    if sam2_size not in ("tiny", "small", "base_plus", "large") or sam2_device not in ("auto", "mps", "cpu"):
        raise ReferenceError("Unknown SAM2 retry model size or device")
    checkpoint = Path(sam2_checkpoint)
    if (checkpoint.is_symlink() or not checkpoint.is_file() or
            not 0 < checkpoint.stat().st_size < 2_000_000_000):
        raise ReferenceError("SAM2 retry requires a regular local checkpoint")
    original_masks = Path(source_masks).resolve()
    raw_output, destination = Path(output), Path(output).resolve()
    source = Path(reconstruction).resolve()
    if (raw_output.is_symlink() or destination.exists() or
            destination == source or destination.is_relative_to(source) or
            destination == original_masks or destination.is_relative_to(original_masks) or
            original_masks.is_relative_to(destination) or source.is_relative_to(destination)):
        raise ReferenceError("Retry output must be a new directory separate from masks and cameras")
    guide_path = Path(guides)
    if guide_path.is_symlink() or not guide_path.is_file():
        raise ReferenceError("SAM2 retry guide must be a regular JSON file")
    guide_doc = _json(guide_path)
    if (not isinstance(guide_doc, dict) or not isinstance(guide_doc.get("images"), dict)
            or not set(targets) <= set(guide_doc["images"])):
        raise ReferenceError("Every retry target needs an explicit anatomical guide")
    # A no-write dense preflight verifies all rectified pixels and original
    # mask hashes before any candidate output directory is created.
    plan = plan_dense_reconstruction(
        reconstruction, destination, DenseSettings(min_registered_images=3,
                                                   min_registered_fraction=0.25,
                                                   refine=False, texture=False),
        rectified_masks=original_masks, executables={})
    if Path(plan["source_revision"]) != source:
        raise ReferenceError("Dense preflight returned an inconsistent camera revision")
    if not set(targets) <= set(plan["undistorted_images"]):
        raise ReferenceError("SAM2 retry target is not a registered image")
    guide_map, guide_sha = _load_guides(guide_path, plan["undistorted_images"], _DEFAULT_BOX)
    for name in targets:
        guide = guide_map[name]
        positives = guide["foreground_points"] + [point for part in guide["part_points"].values()
                                                    for point in part]
        if (len(positives) < 2 or not guide["background_points"] or
                not any(math.dist(a, b) >= 0.12 for a in positives for b in positives)):
            raise ReferenceError(f"{name} needs spatially distinct anatomical positive clicks and background clicks")
        if any(math.dist(a, b) < 0.035 for a in positives for b in guide["background_points"]):
            raise ReferenceError(f"{name} has conflicting positive/negative prompts")
    masks_index = {item["filename"]: item["sha256"] for item in plan["rectified_masks"]}
    photos_index = {item["filename"]: item["sha256"] for item in
                    _json(source / "undistorted-reference-index.json")["images"]}
    guide_bytes = guide_path.read_bytes()
    if hashlib.sha256(guide_bytes).hexdigest() != guide_sha:
        raise ReferenceError("SAM2 guide changed during preflight; retry with a stable guide")
    checkpoint_sha = _sha256(checkpoint)
    effective_device = None
    if predictor is None:
        predictor, effective_device = _load_sam2_predictor(
            checkpoint.resolve(), sam2_size, sam2_device)
        if _sha256(checkpoint) != checkpoint_sha:
            raise ReferenceError("SAM2 checkpoint changed during model loading; retry with a stable checkpoint")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "guides.json").write_bytes(guide_bytes)
    (destination / "candidates").mkdir()
    (destination / "review").mkdir()
    report = {"status": "incomplete / human-photo-review-required",
              "source_revision": str(source), "source_mask_directory": str(original_masks),
              "guides_sha256": guide_sha, "guides_copy": "guides.json",
              "sam2_selection_version": SAM2_SELECTION_VERSION,
              "sam2_checkpoint_sha256": checkpoint_sha,
              "sam2_size": sam2_size, "sam2_device": sam2_device, "max_side": max_side,
              "requested_targets": targets, "images": [],
              "approved_for_reconstruction": False, "distribution_allowed": False,
              "limitations": ("Part-click agreement cannot prove a statue silhouette. "
                              "Candidates cannot be passed directly as a complete dense mask revision; "
                              "inspect photographic boundaries, sand, scaffolding and full anatomy first.")}
    if effective_device is not None:
        report["effective_sam2_device"] = effective_device
    receipt_path = destination / "targeted-sam2-report.json"
    _atomic_report(receipt_path, report)
    for name in targets:
        image_file = source / "undistorted/images" / name
        mask_file = original_masks / (name + ".mask.png")
        if (image_file.is_symlink() or mask_file.is_symlink() or
                _sha256(image_file) != photos_index[name] or
                _sha256(mask_file) != masks_index[name]):
            raise ReferenceError(f"Photo or original mask changed during retry: {name}")
        with Image.open(image_file) as raw_photo, Image.open(mask_file) as raw_mask:
            photo = raw_photo.convert("RGB")
            original = raw_mask.convert("L")
        if original.size != photo.size:
            raise ReferenceError(f"Mismatched photo/mask dimensions for {name}")
        output_pixels, details = _sam2_mask(photo, guide_map[name], predictor, max_side)
        proposed = Image.fromarray(output_pixels, mode="L")
        if proposed.size != photo.size:
            raise ReferenceError(f"SAM2 returned incorrect dimensions for {name}")
        colors = proposed.getcolors(maxcolors=3)
        if colors is None or {pixel for _, pixel in colors} != {0, 255}:
            raise ReferenceError(f"SAM2 returned nonbinary or empty/full mask for {name}")
        compliance = _prompt_qa(proposed, guide_map[name])
        if (_sha256(image_file) != photos_index[name] or
                _sha256(mask_file) != masks_index[name] or
                _sha256(checkpoint) != report["sam2_checkpoint_sha256"]):
            raise ReferenceError(f"A pinned source input changed during SAM2 inference: {name}")
        candidate = destination / "candidates" / (name + ".mask.png")
        proposed.save(candidate, "PNG", optimize=True)
        small = photo.copy()
        small.thumbnail((1200, 1200))
        alpha = proposed.resize(small.size, Image.Resampling.NEAREST).point(lambda x: round(x * .25))
        overlay = Image.composite(Image.new("RGB", small.size, (238, 58, 35)), small, alpha)
        preview = destination / "review" / (name + ".jpg")
        overlay.save(preview, "JPEG", quality=88)
        status = ("candidate / human-photo-review-required" if
                  compliance["missing_positive_points"] == 0 and
                  compliance["violated_negative_points"] == 0 else
                  "prompt-violations / reject-and-review")
        report["images"].append({"filename": name, "status": status,
                                  "source_photo_sha256": photos_index[name],
                                  "original_mask_sha256": masks_index[name],
                                  "candidate": "candidates/" + candidate.name,
                                  "candidate_sha256": _sha256(candidate),
                                  "preview": "review/" + preview.name,
                                  "original_shape": inspect_mask_shape(original),
                                  "candidate_shape": inspect_mask_shape(proposed),
                                  "pixel_change_qa": _pixel_change_qa(original, proposed),
                                  "blue_sky_foreground_proxy": {
                                      "sample_grid": [96, 72],
                                      "original": _mask_sky_leaks(photo, original),
                                      "candidate": _mask_sky_leaks(photo, proposed),
                                      "limitations": "Only confidently blue sky; zero does not rule out scenery leakage."},
                                  "original_prompt_qa": _prompt_qa(original, guide_map[name]),
                                  "prompt_qa": compliance, "sam2_details": details})
        _atomic_report(receipt_path, report)
    report["status"] = "review-only-sam2-candidates / manual-photo-qa-required"
    _atomic_report(receipt_path, report)
    return report
