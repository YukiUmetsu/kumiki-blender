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

from kumiki_blender.reconstruction.dense import _check_regular, _json, _sha256
from kumiki_blender.reconstruction.mask_selection import _mask_source
from kumiki_blender.reconstruction.references import ReferenceError


def validate_foreground_pipeline(reconstruction: Path, mask_revision: Path,
                                 audit_revision: Path, output: Path) -> dict:
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
    return {"status": "verified / human-review-still-required",
            "registered_photos": len(filenames),
            "mask_revision": str(mask_revision),
            "mask_report_sha256": masks["report_sha256"],
            "audit_revision": str(audit_revision),
            "audit_report_sha256": _sha256(audit_file),
            "high_priority_mask_reviews": sorted(high),
            "segmentation_warning_photos": sorted(warnings),
            "comparison_tracks": audit.get("compared_3d_tracks"),
            "output": str(target)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("mask_revision", type=Path)
    parser.add_argument("audit_revision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_foreground_pipeline(
            args.reconstruction, args.mask_revision, args.audit_revision, args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
