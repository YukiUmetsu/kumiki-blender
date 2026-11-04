"""Read-only registration comparison for immutable COLMAP sparse revisions.

An increased camera count is evidence of more overlap, not of correct poses or
an accurate monument surface. Compare matching source photographs by their
original content hashes so renamed input files cannot fabricate improvements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .colmap import _registered_image_names
from .references import ReferenceError


def _load_revision(path: str | Path) -> dict[str, Any]:
    requested = Path(path)
    source = requested.resolve()
    if requested.is_symlink() or not source.is_dir():
        raise ReferenceError(f"Sparse camera revision is missing or a symlink: {path}")
    if (source / "sparse" / "reconstruction-report.json").is_file():
        source = source / "sparse"  # A complete photo-model revision.
    try:
        report = json.loads((source / "reconstruction-report.json").read_text(encoding="utf-8"))
        index = json.loads((source / "staged-reference-index.json").read_text(encoding="utf-8"))
        entries = index["images"]
        if (not isinstance(entries, list) or not 3 <= len(entries) <= 256 or
                not isinstance(report, dict)):
            raise ValueError("Malformed sparse registration evidence")
        by_name: dict[str, str] = {}
        for entry in entries:
            name, digest = entry["filename"], entry.get("original_sha256", entry.get("sha256"))
            if (not isinstance(name, str) or not isinstance(digest, str)
                    or not len(digest) == 64 or name in by_name):
                raise ValueError("Invalid or duplicate staged image evidence")
            by_name[name] = digest
        registered = report.get("registered_filenames")
        if registered is None:
            # Backward compatible with sparse revisions built before this feature.
            model = report.get("source_sparse_model")
            if model is None and report.get("models"):
                model = max(report["models"], key=lambda item: item["registered_images"])["model"]
            if not isinstance(model, str) or not model.isdigit():
                raise ValueError("No usable camera model in revision")
            pose_file = source / "text-models" / model / "images.txt"
            if pose_file.is_symlink() or not pose_file.is_file():
                raise ValueError(f"Missing legacy camera pose evidence: {pose_file}")
            registered = _registered_image_names(pose_file.parent)
            selected_model = next((entry for entry in report.get("models", [])
                                   if entry.get("model") == model), None)
            reported_count = (selected_model.get("registered_images") if selected_model else
                              report.get("registered_images"))
            if type(reported_count) is int and len(registered) != reported_count:
                raise ValueError("Legacy camera count differs from recorded pose evidence")
        if (not isinstance(registered, list) or any(
                not isinstance(name, str) or name not in by_name for name in registered)):
            raise ValueError("Registered images are not present in staged source evidence")
        if len(registered) != len(set(registered)):
            raise ValueError("Duplicate registered image names")
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ReferenceError(f"Cannot compare camera revision {source}: {error}") from error
    return {"path": str(source), "report": report, "all": by_name,
            "registered": {by_name[name]: name for name in registered}}


def compare_camera_registrations(baseline: str | Path, candidate: str | Path) -> dict[str, Any]:
    """Contrast two camera reconstructions, showing gains/losses by filename."""
    before, after = _load_revision(baseline), _load_revision(candidate)
    before_all, after_all = set(before["all"].values()), set(after["all"].values())
    shared = before_all & after_all
    if not shared:
        raise ReferenceError("Camera revisions contain no common source photographs")
    before_registered = set(before["registered"]) & shared
    after_registered = set(after["registered"]) & shared
    gained = sorted(after["registered"][digest] for digest in after_registered - before_registered)
    lost = sorted(before["registered"][digest] for digest in before_registered - after_registered)
    still_missing = sorted(name for name, digest in after["all"].items()
                           if digest in shared and digest not in after_registered)
    return {
        "schema_version": 1, "status": ("mixed-coverage" if gained and lost else
                                         "coverage-improved" if gained else
                                         "coverage-regressed" if lost else "unchanged"),
        "baseline": before["path"], "candidate": after["path"],
        "same_source_set": before_all == after_all,
        "shared_source_images": len(shared),
        "baseline_registered_shared": len(before_registered),
        "candidate_registered_shared": len(after_registered),
        "newly_registered": gained, "lost_registrations": lost,
        "still_unregistered": still_missing,
        "baseline_matching_preset": before["report"].get("matching_preset", "sift"),
        "candidate_matching_preset": after["report"].get("matching_preset", "sift"),
        "baseline_sparse_feature_evidence": before["report"].get("sparse_feature_evidence"),
        "candidate_sparse_feature_evidence": after["report"].get("sparse_feature_evidence"),
        "historical_fidelity_verified": False,
        "limitations": "Registration count and reprojection errors are not proof of correct camera poses, a complete surface, or historical likeness. Inspect 3D views before meshing.",
    }
