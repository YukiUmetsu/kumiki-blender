#!/usr/bin/env python3
"""Read-only gate for a reviewed, registered-photo foreground splat run.

The semantic/GrabCut result and the multi-view track audit are *candidates*.
This validates their exact provenance; an operator must still inspect every
mask and explicitly acknowledge review before training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kumiki_blender.reconstruction.dense import _check_regular, _json, _sha256
from kumiki_blender.reconstruction.mask_selection import _mask_source
from kumiki_blender.reconstruction.references import ReferenceError


def validate_foreground_pipeline(reconstruction: Path, mask_revision: Path,
                                 audit_revision: Path, output: Path, *,
                                 require_quality: bool = True,
                                 min_compared_tracks: int = 25) -> dict:
    """Check provenance and anatomy/track quality before any trainer is invoked.

    ``require_quality=False`` is for a *diagnostic report only*, not a
    training bypass. Neither branch writes or approves model artifacts.
    """
    if type(min_compared_tracks) is not int or min_compared_tracks < 3:
        raise ReferenceError("min_compared_tracks must be at least 3")
    if (reconstruction.is_symlink() or mask_revision.is_symlink() or
            audit_revision.is_symlink()):
        raise ReferenceError("Foreground source revisions must not be symlinks")
    reconstruction = reconstruction.resolve(strict=True)
    mask_revision = mask_revision.resolve(strict=True)
    audit_revision = audit_revision.resolve(strict=True)
    if (Path(output).is_symlink() or output.exists()):
        raise ReferenceError("Training output must be a new immutable revision")
    target = output.resolve()
    index = _json(reconstruction / "undistorted-reference-index.json")
    filenames = [entry["filename"] for entry in index["images"]]
    masks = _mask_source(reconstruction, target, mask_revision, filenames)
    audit_file = audit_revision / "mask-consistency-report.json"
    _check_regular(audit_file, audit_revision)
    # A 26+ camera audit can contain thousands of bounded per-feature review
    # markers and exceed the normal 1 MiB reconstruction metadata limit.
    # Keep that default for all other evidence; only this audit gets 16 MiB.
    audit = _json(audit_file, max_bytes=16 * 1024 * 1024)
    if (audit.get("status") != "candidate / multi-view-mask-review-required" or
            audit.get("source_revision") != str(reconstruction) or
            audit.get("rectified_mask_source") != str(mask_revision / "masks") or
            audit.get("approved_for_reconstruction") is not False):
        raise ReferenceError("Cross-view audit is missing or belongs to different masks/cameras")
    model_path = reconstruction / audit.get("model_file", "")
    if not model_path.resolve().is_relative_to(reconstruction / "undistorted/sparse"):
        raise ReferenceError("Cross-view audit references an unsafe COLMAP model")
    _check_regular(model_path, reconstruction)
    if _sha256(model_path) != audit.get("model_sha256"):
        raise ReferenceError("COLMAP model changed after mask consistency audit")
    reviewed_entries = audit.get("images")
    if (not isinstance(reviewed_entries, list) or len(reviewed_entries) != len(filenames)
            or {entry.get("filename") for entry in reviewed_entries
                if isinstance(entry, dict)} != set(filenames)):
        raise ReferenceError("Cross-view audit is missing registered photo coverage")
    for entry in reviewed_entries:
        if (entry.get("mask_sha256") != masks["masks"][entry["filename"]]["sha256"] or
                entry.get("source_photo_sha256") != masks["entries"][entry["filename"]]["source_photo_sha256"]):
            raise ReferenceError(f"Mask or photo changed since cross-view audit: {entry['filename']}")
    high = [entry["filename"] for entry in reviewed_entries
            if entry.get("review_priority") == "high"]
    warnings = [entry["filename"] for entry in masks["entries"].values()
                if entry.get("warnings")]
    from PIL import Image, ImageChops, ImageDraw

    issues: list[str] = []
    recommendations: dict[str, list[str]] = {}
    camera_quality: list[dict[str, Any]] = []
    image_audit = {entry["filename"]: entry for entry in reviewed_entries}
    for name in filenames:
        current = masks["entries"][name]
        audit_entry = image_audit[name]
        recommendations[name] = []
        with Image.open(mask_revision / current["mask"]) as image:
            matte = image.convert("L")
            width, height = matte.size
            fraction = sum(matte.histogram()[128:]) / (width*height)
            pixels = matte.load()
            if fraction < 0.05 or fraction > 0.90:
                issues.append(f"{name}: foreground {fraction:.1%} outside 5–90% range")
                recommendations[name].append("Re-segment the complete silhouette; check head, torso, paws and tail")
            guide = current.get("guide")
            boxes = guide.get("component_boxes") if isinstance(guide, dict) else None
            outside_fraction = 0.0
            if isinstance(boxes, dict) and boxes:
                allowed = Image.new("L", (width, height), 0)
                drawer = ImageDraw.Draw(allowed)
                for part, bounds in boxes.items():
                    if (not isinstance(bounds, list) or len(bounds) != 4
                            or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in bounds)
                            or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
                        issues.append(f"{name}: invalid {part} component box")
                        continue
                    left, top, right, bottom = bounds
                    drawer.rectangle((round(left*(width-1)), round(top*(height-1)),
                                      round(right*(width-1)), round(bottom*(height-1))), fill=255)
                leaked = ImageChops.subtract(matte, allowed)
                outside_fraction = sum(leaked.histogram()[128:]) / max(1, sum(matte.histogram()[128:]))
                if outside_fraction > 0.05:
                    issues.append(f"{name}: {outside_fraction:.1%} mask foreground outside component boxes (>5%)")
                    recommendations[name].append("Inspect sky, sand or buildings; add negative prompts or expand boxes around real anatomy")
            parts = guide.get("part_points") if isinstance(guide, dict) else None
            if not isinstance(parts, dict) or not parts:
                issues.append(f"{name}: no verified visible-part landmarks")
                recommendations[name].append("Mark at least one visible part; mark every visible head/headdress/neck/torso/paws/tail part")
                parts = {}
            missing_parts = []
            for part, landmarks in parts.items():
                if not isinstance(landmarks, list) or not landmarks:
                    issues.append(f"{name}: invalid {part} landmarks")
                    continue
                for point in landmarks:
                    if (not isinstance(point, list) or len(point) != 2
                            or any(type(v) not in (float, int) or not 0 <= v <= 1 for v in point)):
                        issues.append(f"{name}: invalid {part} landmark coordinates")
                        break
                    x, y = point
                    if not pixels[round(x*(width-1)), round(y*(height-1))]:
                        missing_parts.append(part)
                        break
            if missing_parts:
                issues.append(f"{name}: missing visible anatomy: {', '.join(sorted(set(missing_parts)))}")
                recommendations[name].append("Add the missing part with another component box/positive point or include polygon")
            positives = guide.get("foreground_points", []) if isinstance(guide, dict) else []
            positive_missing = 0
            for point in positives if isinstance(positives, list) else []:
                if (isinstance(point, list) and len(point) == 2
                        and all(type(v) in (int, float) and 0 <= v <= 1 for v in point)):
                    x, y = point
                    positive_missing += int(not pixels[round(x*(width-1)), round(y*(height-1))])
                else:
                    issues.append(f"{name}: invalid foreground point")
            if positive_missing:
                issues.append(f"{name}: {positive_missing} explicit foreground points not retained")
                recommendations[name].append("Restore missing positive-point anatomy with a tighter box or include polygon")
            negatives = guide.get("background_points", []) if isinstance(guide, dict) else []
            negative_hits = 0
            for point in negatives if isinstance(negatives, list) else []:
                if (isinstance(point, list) and len(point) == 2
                        and all(type(v) in (int, float) and 0 <= v <= 1 for v in point)):
                    x, y = point
                    negative_hits += int(bool(pixels[round(x*(width-1)), round(y*(height-1))]))
                else:
                    issues.append(f"{name}: invalid background point")
            if negative_hits:
                issues.append(f"{name}: {negative_hits} explicit background points remain foreground")
                recommendations[name].append("Exclude scenery by negative point or polygon; do not erase thin anatomy")
        compared = audit_entry.get("compared_observations", 0)
        outlier_fraction = audit_entry.get("consensus_outlier_fraction", 0)
        if type(compared) is not int or compared < 1:
            issues.append(f"{name}: no cross-view feature-track support")
        if (type(outlier_fraction) not in (float, int) or not 0 <= outlier_fraction <= 1):
            issues.append(f"{name}: invalid track-outlier fraction")
        elif outlier_fraction > 0.20:
            issues.append(f"{name}: {outlier_fraction:.1%} track majority outliers (>20%)")
            recommendations[name].append("Inspect cyan missing-subject and red background-leak markers; never accept majority automatically")
        camera_quality.append({"filename": name, "foreground_fraction": round(fraction, 5),
                               "foreground_outside_component_boxes": round(outside_fraction, 5),
                               "parts_checked": sorted(parts), "missing_parts": sorted(set(missing_parts)),
                               "positive_seed_violations": positive_missing,
                               "negative_seed_violations": negative_hits,
                               "compared_observations": compared,
                               "consensus_outlier_fraction": outlier_fraction,
                               "corrections": recommendations[name]})
    compared_tracks = audit.get("compared_3d_tracks")
    mixed_tracks = audit.get("mixed_label_tracks")
    if type(compared_tracks) is not int or compared_tracks < min_compared_tracks:
        issues.append(f"Only {compared_tracks} compared tracks; need {min_compared_tracks} to assess consistency")
    if (type(mixed_tracks) is not int or type(compared_tracks) is not int
            or not 0 <= mixed_tracks <= compared_tracks or compared_tracks == 0):
        issues.append("Invalid aggregate multi-view feature counts")
        mixed_fraction = None
    else:
        mixed_fraction = round(mixed_tracks / compared_tracks, 5)
        if mixed_fraction > 0.15:
            issues.append(f"{mixed_fraction:.1%} shared tracks have mixed labels (>15%)")
    result = {"status": ("blocked / mask-quality-gates-failed" if issues else
                         "verified / human-review-still-required"),
            "registered_photos": len(filenames),
            "mask_revision": str(mask_revision),
            "mask_report_sha256": masks["report_sha256"],
            "audit_revision": str(audit_revision),
            "audit_report_sha256": _sha256(audit_file),
            "high_priority_mask_reviews": sorted(high),
            "segmentation_warning_photos": sorted(warnings),
            "comparison_tracks": audit.get("compared_3d_tracks"),
            "mixed_label_fraction": mixed_fraction,
            "quality_passed": not issues,
            "quality_failures": issues,
            "images": camera_quality,
            "limitations": "Sparse feature agreement cannot prove silhouette completeness; visually verify every visible statue part",
            "output": str(target)}
    if issues and require_quality:
        raise ReferenceError("Mask quality gates failed; no training permitted:\n- " +
                             "\n- ".join(issues[:12]) +
                             (f"\n... {len(issues)-12} more; use --allow-failures for full report" if len(issues) > 12 else ""))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("mask_revision", type=Path)
    parser.add_argument("audit_revision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-failures", action="store_true",
                        help="Generate diagnostics even when quality fails; never usable as training permission")
    parser.add_argument("--report-output", type=Path,
                        help="Unused file path for immutable JSON quality report")
    args = parser.parse_args()
    try:
        report = validate_foreground_pipeline(
            args.reconstruction, args.mask_revision, args.audit_revision, args.output,
            require_quality=not args.allow_failures)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    if args.report_output:
        if args.report_output.is_symlink() or args.report_output.exists() or not args.report_output.parent.is_dir():
            parser.exit(2, "ERROR: Quality report needs an unused path in an existing directory\n")
        args.report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
