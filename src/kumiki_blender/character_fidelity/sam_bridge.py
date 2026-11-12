"""Compare unapproved SAM2 masks with visibility-aware Blender component capture.

The supplied SAM image hash must match the reference image. We stage masks in
a temporary manifest and reuse shape, placement, color, texture and relative
depth QA. All source assets, .blend files and checkpoint files stay unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .schema import asset, load_manifest


def _observations(item: dict[str, Any], conflict: bool) -> list[dict[str, str]]:
    observations: list[dict[str, str]] = []
    shape = item.get("shape", {})
    if shape.get("status") == "missing-part":
        observations.append({"channel": "mapping", "severity": "critical",
            "message": "Part missing from either source mask or visible sculpt. Check object selection and occlusion first."})
    elif shape.get("intrinsic_shape_iou") is not None and shape["intrinsic_shape_iou"] < .70:
        observations.append({"channel": "shape", "severity": "high",
            "message": "Contour differs even after uniform size normalization; inspect outline, taper and tips before changing position."})
    if shape.get("registration_verified"):
        delta = shape.get("centroid_offset_xy_px")
        if delta is not None and max(abs(value) for value in delta) >= 10:
            observations.append({"channel": "position", "severity": "medium",
                "message": f"Part centroid is offset {delta} registered pixels; verify camera landmarks and attachment."})
    else:
        observations.append({"channel": "registration", "severity": "review",
            "message": "Camera or pose not verified. Position, pixelwise color and metric depth agreement cannot be claimed."})
    de = item.get("perceptual_color", {}).get("median_palette_delta_e2000")
    if de is not None and de > 12:
        observations.append({"channel": "color", "severity": "medium",
            "message": "Unlit sculpt base-color palette differs from painted reference; lighting style may contribute."})
    if conflict:
        observations.append({"channel": "segmentation", "severity": "high",
            "message": "SAM part overlaps another source mask; review original source layers or re-prompt segmentation."})
    return observations


def _unregistered_previews(report, stage, capture, view_id, source_size, output):
    """Write a clearly labeled, unregistered intrinsic-shape overlay per part."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from .evaluate import _mask, _visible_labels

    metadata = json.loads(asset(capture, "capture.json").read_text(encoding="utf-8"))
    labels = np.asarray(Image.open(asset(capture, metadata["views"][view_id]["labels"])).convert("RGBA"))
    part_masks = _visible_labels(labels, metadata["palette"])

    def normalize(mask):
        y, x = np.nonzero(mask)
        canvas = np.zeros((256, 256), dtype=bool)
        if not len(x):
            return canvas
        crop = mask[y.min():y.max() + 1, x.min():x.max() + 1].astype(np.uint8)
        scale = 240 / max(crop.shape)
        h, w = max(1, round(crop.shape[0] * scale)), max(1, round(crop.shape[1] * scale))
        canvas[(256-h)//2:(256-h)//2+h, (256-w)//2:(256-w)//2+w] = cv2.resize(
            crop, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return canvas

    for item in report["parts"]:
        if item.get("comparison") or item.get("shape", {}).get("registration_verified"):
            continue
        part = item["part"]
        original = normalize(_mask(asset(stage, f"mask-{part}.png"), source_size))
        sculpt = normalize(part_masks[part])
        pixels = np.full((256, 256, 3), 247, dtype=np.uint8)
        pixels[original & ~sculpt] = (225, 65, 65)
        pixels[sculpt & ~original] = (60, 95, 222)
        pixels[original & sculpt] = (90, 165, 104)
        image = Image.new("RGB", (256, 302), "#e9edf2")
        text = ImageDraw.Draw(image)
        text.text((7, 5), "SHAPE ONLY / NOT REGISTERED", fill="#162334")
        text.text((7, 23), "RED source / BLUE sculpt / GREEN both", fill="#162334")
        image.paste(Image.fromarray(pixels), (0, 46))
        relative = f"comparisons/{view_id}/{part}.png"
        destination = Path(output) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        item["comparison"] = relative


def compare_sam_to_capture(manifest: str | Path, capture: str | Path,
                           segmentation: str | Path, output: str | Path, *,
                           view_id: str = "front", depth_hints: str | Path | None = None,
                           mapping: dict[str, str] | None = None) -> dict[str, Any]:
    """Reconcile unapproved SAM proposals with an existing Blender capture.

    Optional mapping maps manifest part IDs to SAM proposal IDs. When SAM
    has no matching part, retain the existing manifest mask with explicit
    provenance; never infer a missing mask from a different camera view.
    """
    manifest, capture, segmentation, output = map(Path, (manifest, capture, segmentation, output))
    config = load_manifest(manifest)
    view = next((v for v in config["views"] if v["id"] == view_id), None)
    if view is None:
        raise ValueError(f"Unknown reference view: {view_id}")
    if output.is_symlink() or output.exists():
        raise ValueError("QA output must be a new directory")
    if any(output.resolve().is_relative_to(path.resolve()) for path in
           (manifest.parent, capture, segmentation)):
        raise ValueError("QA output must be outside reference, segmentation and capture directories")
    source = asset(manifest.resolve().parent, view["reference_image"])
    report = json.loads(asset(segmentation, "report.json").read_text(encoding="utf-8"))
    if report.get("provider") != "sam2.1-local" or report.get("schema_version") != 1:
        raise ValueError("Expected SAM 2.1 proposal report")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if report.get("image_sha256") != digest:
        raise ValueError("SAM proposal image SHA-256 does not match reference artwork")
    entries = report.get("parts")
    if (not isinstance(entries, list) or
            any(not isinstance(item, dict) or not isinstance(item.get("part"), str) for item in entries)):
        raise ValueError("Malformed SAM part proposal report")
    sam = {item["part"]: item for item in entries}
    if len(sam) != len(entries):
        raise ValueError("Duplicate SAM part identifiers")
    known_ids = {part["id"] for part in config["parts"]}
    if mapping is not None and (not isinstance(mapping, dict) or any(
            part_id not in known_ids or sam_id not in sam for part_id, sam_id in mapping.items())):
        raise ValueError("Mapping contains unknown Blender or SAM parts")
    mapping = mapping or {}
    conflicts = {frozenset((item.get("first"), item.get("second")))
                 for item in report.get("semantic_overlap_warnings", []) if isinstance(item, dict)}
    provenance = {}
    with tempfile.TemporaryDirectory(prefix="kumiki-sam-evidence-") as folder:
        temp = Path(folder)
        staged = copy.deepcopy(config)
        staged["views"] = [next(v for v in staged["views"] if v["id"] == view_id)]
        current_view = staged["views"][0]
        shutil.copyfile(source, temp / "source.png")
        current_view["reference_image"] = "source.png"
        for field, filename in (("reference_depth_hint", "saved_source_hint.npz"),
                                ("reference_depth", "saved_metric_source_depth.npz")):
            if field in current_view:
                declared = current_view[field]
                shutil.copyfile(asset(manifest.resolve().parent, declared["path"]), temp / filename)
                declared["path"] = filename
        if depth_hints is not None:
            hint = asset(Path(depth_hints), "depth_hint.npz")
            shutil.copyfile(hint, temp / "depth_hint.npz")
            current_view["reference_depth_hint"] = {
                "path": "depth_hint.npz", "provenance": "estimated", "near_is_small": True}
        # Estimated depth hints never replace a separate calibrated source depth.
        for part in staged["parts"]:
            part["unmatched_pose_views"] = [v for v in part.get("unmatched_pose_views", []) if v == view_id]
            pid, sid = part["id"], mapping.get(part["id"], part["id"])
            if sid in sam:
                proposal = sam[sid]
                if proposal.get("category") is not None and proposal["category"] != part["category"]:
                    raise ValueError(f"SAM part category conflicts with {pid}")
                mask = asset(segmentation, proposal.get("proposed_mask"))
                provenance[pid] = {"source": "sam-proposal-unapproved", "sam_part": sid,
                    "predicted_iou_not_ground_truth": proposal.get("predicted_iou"),
                    "mask_conflict": any(sid in overlap for overlap in conflicts),
                    "mask_review": proposal.get("mask_review", {})}
            elif view_id in part["reference_masks"]:
                mask = asset(manifest.resolve().parent, part["reference_masks"][view_id])
                provenance[pid] = {"source": "existing-manifest-mask / review-required",
                                   "mask_conflict": False}
            else:
                raise ValueError(f"No SAM or existing mask for {pid} in {view_id}")
            destination = temp / f"mask-{pid}.png"
            shutil.copyfile(mask, destination)
            part["reference_masks"] = {view_id: destination.name}
        (temp / "manifest.json").write_text(json.dumps(staged), encoding="utf-8")
        from .evaluate import evaluate_character
        result = evaluate_character(temp / "manifest.json", capture, output)
        from PIL import Image
        with Image.open(source) as original_image:
            source_size = original_image.size
        _unregistered_previews(result, temp, capture, view_id, source_size, output)
    for item in result["parts"]:
        p = item["part"]
        item["source_mask_provenance"] = provenance[p]
        incomplete = bool(provenance[p].get("mask_review", {}).get("incomplete_shape_evidence"))
        item["source_mask_incomplete"] = incomplete
        item["shape"]["source_evidence_status"] = "partial / review-required" if incomplete else "unapproved / review-required"
        observations = _observations(item, provenance[p]["mask_conflict"])
        if incomplete:
            # Keep numeric diagnostics but never prescribe geometry edits on
            # source boundaries missing pixels due unresolved part ownership.
            observations = [note for note in observations if note["channel"] not in ("shape", "position", "color")]
            observations.insert(0, {"channel": "mask-evidence", "severity": "critical",
                "message": "Source mask excludes unresolved ambiguous pixels. Shape and appearance numbers are partial evidence; resolve mask ownership before sculpt edits."})
        item["correction_observations"] = observations
    result["status"] = "segmentation-review-required / no-automatic-approval"
    result["semantic_mask_conflicts"] = report.get("semantic_overlap_warnings", [])
    result["source_mask_review"] = report.get("mask_review")
    result["source_image_sha256"] = digest
    result["limitations"] += [
        "SAM predicted IoU is not artist-verified mask overlap.",
        "Unapproved SAM masks can conflict and should not be sculpt targets without review.",
        "Model depth is measured; MoGe depth hints are not metric source ground truth.",
        "No combined likeness score or automatic geometry correction."]
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ["# Character fidelity review", "",
          "Unapproved SAM source masks versus a read-only Blender component capture.", "",
          "| Part | Category | Intrinsic shape overlap | Median color DeltaE00 | Mask |",
          "|---|---|---:|---:|---|"]
    for row in result["parts"]:
        pid = row["part"]
        iou = row.get("shape", {}).get("intrinsic_shape_iou")
        delta = row.get("perceptual_color", {}).get("median_palette_delta_e2000")
        md.append("| " + " | ".join((
            pid, row["category"], f"{iou:.3f}" if iou is not None else "n/a",
            f"{delta:.1f}" if delta is not None else "n/a",
            "incomplete" if row["source_mask_incomplete"] else "conflict" if provenance[pid]["mask_conflict"] else "unapproved")) + " |")
    for row in result["parts"]:
        md.extend(("", "## " + row["category"] + " / " + row["part"]))
        for finding in row["correction_observations"]:
            md.append("- **" + finding["channel"] + "**: " + finding["message"])
    md.extend(("", "No geometry was modified; confirm source masks and view registration first.", ""))
    (output / "review.md").write_text("\n".join(md), encoding="utf-8")
    return result
