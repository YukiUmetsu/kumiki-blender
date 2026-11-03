"""Convert explicitly accepted COLMAP mask disagreements into fresh SAM2 guides.

No sparse-track majority is treated as ground truth or painted into a mask.
Each suggestion needs a per-photo, per-point human decision and a new
segmentation/QA run on immutable registered photographs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dense import _check_regular, _json, _sha256
from .mask_selection import _mask_source
from .masking import _DEFAULT_BOX, _guide, _normalized_pair
from .references import ReferenceError


def accept_mask_point_suggestions(reconstruction: str | Path,
                                  mask_revision: str | Path,
                                  audit_revision: str | Path,
                                  output: str | Path, *,
                                  accept: list[str]) -> dict[str, Any]:
    """Export manually selected cyan/red track prompts without modifying masks.

    Each ``accept`` entry is ``<registered-filename>:<point3d-id>``.
    Ambiguous two-view ties are intentionally ineligible.
    """
    if (not isinstance(accept, list) or not accept or len(accept) > 500
            or any(not isinstance(item, str) for item in accept)
            or len(set(accept)) != len(accept)):
        raise ReferenceError("Accept 1–500 distinct photo:point3d-id decisions")
    input_paths = [Path(reconstruction), Path(mask_revision), Path(audit_revision)]
    if any(path.is_symlink() or not path.is_dir() for path in input_paths):
        raise ReferenceError("All input revisions must be regular directories, not symlinks")
    sparse, masks, audit_root = [path.resolve() for path in input_paths]
    target = Path(output).resolve()
    if (Path(output).is_symlink() or target.exists()
            or any(target == source or target.is_relative_to(source)
                   for source in (sparse, masks, audit_root))):
        raise ReferenceError("Guide output already exists or overlaps a source revision")

    index = _json(sparse / "undistorted-reference-index.json")
    names = [item["filename"] for item in index["images"]]
    source = _mask_source(sparse, target, masks, names)
    audit_file = audit_root / "mask-consistency-report.json"
    _check_regular(audit_file, audit_root)
    evidence = _json(audit_file, max_bytes=16 * 1024 * 1024)
    if (evidence.get("source_revision") != str(sparse)
            or evidence.get("rectified_mask_source") != str(masks / "masks")
            or evidence.get("status") != "candidate / multi-view-mask-review-required"):
        raise ReferenceError("Cross-view audit is stale or belongs to other masks/cameras")
    model_path = sparse / evidence.get("model_file", "")
    if not model_path.resolve().is_relative_to(sparse / "undistorted/sparse"):
        raise ReferenceError("Unsafe COLMAP model path in audit")
    _check_regular(model_path, sparse)
    if _sha256(model_path) != evidence.get("model_sha256"):
        raise ReferenceError("COLMAP model changed after audit")
    audited = evidence.get("images")
    if (not isinstance(audited, list) or len(audited) != len(names)
            or {item.get("filename") for item in audited if isinstance(item, dict)} != set(names)):
        raise ReferenceError("Cross-view audit is missing per-photo evidence")
    by_name = {item["filename"]: item for item in audited}
    for name in names:
        entry = by_name[name]
        if (entry.get("mask_sha256") != source["masks"][name]["sha256"]
                or entry.get("source_photo_sha256") != source["entries"][name]["source_photo_sha256"]):
            raise ReferenceError(f"Cross-view evidence changed since segmentation: {name}")

    guides = {name: _guide(source["entries"][name].get("guide") or {},
                           default_box=_DEFAULT_BOX) for name in names}
    decisions = []
    seen_points: set[tuple[str, int]] = set()
    for decision in accept:
        name, separator, point_text = decision.rpartition(":")
        if not separator or name not in by_name or not point_text.isdecimal():
            raise ReferenceError(f"Unknown registered photo or invalid point ID: {decision}")
        point_id = int(point_text)
        key = (name, point_id)
        if key in seen_points:
            raise ReferenceError(f"Duplicate point decision: {decision}")
        seen_points.add(key)
        proposal = next((item for item in by_name[name].get("correction_recommendations", [])
                         if item.get("point3d_id") == point_id), None)
        if proposal is None or proposal.get("kind") not in (
                "possible_missing_subject", "possible_background_leak"):
            raise ReferenceError(f"Not an eligible non-tied correction suggestion: {decision}")
        positives = proposal.get("foreground_votes")
        negatives = proposal.get("background_votes")
        if (type(positives) is not int or type(negatives) is not int
                or positives + negatives < 3 or
                (proposal["kind"] == "possible_missing_subject" and positives <= negatives) or
                (proposal["kind"] == "possible_background_leak" and negatives <= positives)):
            raise ReferenceError(f"Correction suggestion has no valid majority: {decision}")
        xy = list(_normalized_pair(proposal.get("normalized_xy"), "correction point"))
        prompt_type = ("foreground_points" if proposal["kind"] == "possible_missing_subject"
                       else "background_points")
        if xy not in guides[name][prompt_type]:
            guides[name][prompt_type].append(xy)
        # Validate *after* each addition: a suggested positive outside a
        # component box needs manual box repair, not silent expansion.
        guides[name] = _guide(guides[name], default_box=_DEFAULT_BOX)
        decisions.append({"filename": name, "point3d_id": point_id,
                          "prompt_type": prompt_type, "normalized_xy": xy,
                          "votes": {"foreground": positives, "background": negatives}})

    target.mkdir(parents=True, exist_ok=False)
    guide_file = target / "mask-guides.json"
    guide_file.write_text(json.dumps({"images": guides}, indent=2) + "\n", encoding="utf-8")
    result = {"schema_version": 1, "status": "candidate / reviewed-mask-prompts",
              "source_revision": str(sparse), "source_mask_revision": str(masks),
              "source_mask_report_sha256": source["report_sha256"],
              "source_audit_revision": str(audit_root),
              "source_audit_report_sha256": _sha256(audit_file),
              "guide_file": "mask-guides.json", "guide_sha256": _sha256(guide_file),
              "accepted_count": len(decisions), "accepted": decisions,
              "approved_for_training": False,
              "next_action": "Review these prompts on the original rectified photos, rerun segmentation, then audit new masks; never reuse the old audit"}
    (target / "mask-guide-decision-report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
