"""Private, source-pinned before/after multi-view foreground-mask comparison.

Do not promote a revised mask because an automatic metric improved. Sparse
COLMAP tracks cannot see untextured anatomy or consistently included walls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dense import DenseSettings, _sha256, plan_dense_reconstruction
from .mask_consistency import audit_multiview_masks
from .references import ReferenceError


def compare_multiview_masks(reconstruction: str | Path,
                            baseline_masks: str | Path,
                            candidate_masks: str | Path,
                            output: str | Path, *,
                            max_overlay_points: int = 200) -> dict[str, Any]:
    """Compare two rectified mask revisions without editing either revision."""
    baseline = Path(baseline_masks).resolve()
    candidate = Path(candidate_masks).resolve()
    target = Path(output).resolve()
    if (type(max_overlay_points) is not int or not 1 <= max_overlay_points <= 2000):
        raise ReferenceError("max_overlay_points must be an integer in [1, 2000]")
    if (Path(baseline_masks).is_symlink() or Path(candidate_masks).is_symlink()
            or baseline == candidate or Path(output).is_symlink() or target.exists()
            or target.is_relative_to(baseline) or target.is_relative_to(candidate)):
        raise ReferenceError("Mask comparison needs distinct source masks and unused separate output")
    settings = DenseSettings(min_registered_images=3, min_registered_fraction=0.25,
                             refine=False, texture=False)
    # Validate both sides *before* producing a partially written comparison.
    plans = [plan_dense_reconstruction(reconstruction, target / name, settings,
                                       rectified_masks=mask_root, executables={})
             for name, mask_root in (("baseline", baseline), ("candidate", candidate))]
    if plans[0]["source_revision"] != plans[1]["source_revision"]:
        raise ReferenceError("Mask comparison must use one registered camera revision")
    target.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_revision": plans[0]["source_revision"],
        "baseline_mask_source": str(baseline),
        "candidate_mask_source": str(candidate),
        "approved_for_reconstruction": False, "distribution_allowed": False,
        "status": "incomplete / mask-comparison-review-required",
    }

    def checkpoint() -> None:
        (target / "mask-comparison-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    try:
        previous = audit_multiview_masks(reconstruction, baseline,
                                         target / "baseline",
                                         max_overlay_points=max_overlay_points)
        revised = audit_multiview_masks(reconstruction, candidate,
                                        target / "candidate",
                                        max_overlay_points=max_overlay_points)
    except Exception as error:
        report["error"] = str(error)
        checkpoint()
        raise
    if (previous["model_sha256"] != revised["model_sha256"]
            or previous["compared_3d_tracks"] != revised["compared_3d_tracks"]):
        raise ReferenceError("Sparse camera model changed between comparison runs")
    before_images = {entry["filename"]: entry for entry in previous["images"]}
    after_images = {entry["filename"]: entry for entry in revised["images"]}
    if set(before_images) != set(after_images):
        raise ReferenceError("Baseline and candidate camera views differ")
    fields = ("consensus_outliers", "possible_background_leaks",
              "possible_missing_subject", "ambiguous_observations")
    per_image = []
    for filename in sorted(before_images):
        before, after = before_images[filename], after_images[filename]
        if before["source_photo_sha256"] != after["source_photo_sha256"]:
            raise ReferenceError("Baseline and candidate photo source hashes differ")
        changes = {field + "_delta": after[field] - before[field] for field in fields}
        per_image.append({"filename": filename,
                          "baseline_mask_sha256": before["mask_sha256"],
                          "candidate_mask_sha256": after["mask_sha256"],
                          "baseline_review": "baseline/" + before["review"],
                          "candidate_review": "candidate/" + after["review"],
                          **changes})
    per_image.sort(key=lambda item: (-item["possible_background_leaks_delta"],
                                     -item["consensus_outliers_delta"], item["filename"]))
    change = {
        "mixed_label_tracks_delta": revised["mixed_label_tracks"] - previous["mixed_label_tracks"],
        "ambiguous_label_tracks_delta": revised["ambiguous_label_tracks"] - previous["ambiguous_label_tracks"],
        **{field + "_delta": sum(item[field + "_delta"] for item in per_image)
           for field in fields},
    }
    report.update({
        "colmap_model_sha256": previous["model_sha256"],
        "compared_3d_tracks": previous["compared_3d_tracks"],
        "baseline": {"audit": "baseline/mask-consistency-report.json",
                     "sha256": _sha256(target / "baseline/mask-consistency-report.json"),
                     "mixed_label_tracks": previous["mixed_label_tracks"],
                     "ambiguous_label_tracks": previous["ambiguous_label_tracks"]},
        "candidate": {"audit": "candidate/mask-consistency-report.json",
                      "sha256": _sha256(target / "candidate/mask-consistency-report.json"),
                      "mixed_label_tracks": revised["mixed_label_tracks"],
                      "ambiguous_label_tracks": revised["ambiguous_label_tracks"]},
        "delta": change,
        "consistency_regressed": (change["mixed_label_tracks_delta"] > 0
                                  or change["possible_background_leaks_delta"] > 0),
        "images": per_image,
        "interpretation": ("Positive deltas mean more sparse-track disagreement. "
                           "Negative deltas are not proof of accurate segmentation; "
                           "compare both photo overlays and detail crops manually."),
        "status": "candidate / manual-mask-comparison-required",
    })
    checkpoint()
    return report
