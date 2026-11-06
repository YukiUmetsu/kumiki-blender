"""Carry reviewed masks to a camera revision that adds or changes photos.

Camera recovery writes a new rectified revision. A reviewed mask stays valid
there only if its rectified photo is byte-identical. This module keeps those
masks, takes the other masks from a revision generated for the new cameras,
and records the origin of each mask. It never edits a mask or approves one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .dense import _sha256
from .mask_track_guidance import _safe_pinned_image
from .mask_track_selection import _verified_binary
from .references import ReferenceError


SCHEMA = "carried-mask-revision-v1"
REVIEWED = "reviewed-mask-of-identical-photo"
GENERATED = "generated-for-new-or-changed-photo"


def choose_mask_origins(photos: list[dict[str, Any]], reviewed: dict[str, dict[str, Any]],
                        generated: dict[str, dict[str, Any]]
                        ) -> tuple[dict[str, str], list[str]]:
    """Choose a mask origin for each photo, and list the photos without a mask."""
    origins: dict[str, str] = {}
    missing: list[str] = []
    for photo in photos:
        name = photo["filename"]
        if reviewed.get(name, {}).get("source_photo_sha256") == photo["sha256"]:
            origins[name] = REVIEWED
        elif generated.get(name, {}).get("source_photo_sha256") == photo["sha256"]:
            origins[name] = GENERATED
        else:
            missing.append(name)
    return origins, missing


def _mask_entries(revision: Path, label: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report_path = revision / "mask-report.json"
    if revision.is_symlink() or report_path.is_symlink() or not report_path.is_file():
        raise ReferenceError(f"The {label} mask revision has no mask-report.json: {revision}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (not str(report.get("status", "")).startswith("candidate") or
            not isinstance(report.get("images"), list)):
        raise ReferenceError(f"The {label} mask revision is incomplete: {revision}")
    return {entry["filename"]: entry for entry in report["images"]}, report


def carry_masks_forward(reconstruction: str | Path, reviewed_masks: str | Path,
                        output: str | Path, *,
                        generated_masks: str | Path | None = None) -> dict[str, Any]:
    """Write a mask revision for `reconstruction` that keeps every still-valid reviewed mask."""
    sparse = Path(reconstruction).resolve()
    reviewed_root = Path(reviewed_masks).resolve()
    generated_root = Path(generated_masks).resolve() if generated_masks is not None else None
    target = Path(output).resolve()
    inputs = [sparse, reviewed_root] + ([generated_root] if generated_root else [])
    if (Path(output).is_symlink() or target.exists() or
            any(target == folder or target.is_relative_to(folder) or folder.is_relative_to(target)
                for folder in inputs)):
        raise ReferenceError("Carried masks need an unused output folder outside every input")
    index_path = sparse / "undistorted-reference-index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ReferenceError(f"Reconstruction has no rectified photo index: {index_path}")
    photos = json.loads(index_path.read_text(encoding="utf-8"))["images"]
    reviewed, reviewed_report = _mask_entries(reviewed_root, "reviewed")
    generated: dict[str, dict[str, Any]] = {}
    generated_report: dict[str, Any] | None = None
    if generated_root is not None:
        generated, generated_report = _mask_entries(generated_root, "generated")
        if generated_report.get("source_revision") != str(sparse):
            raise ReferenceError("Generated masks belong to a different reconstruction; "
                                 "run prepare-dense-masks on this reconstruction")

    origins, missing = choose_mask_origins(photos, reviewed, generated)
    if missing:
        raise ReferenceError(
            f"{len(missing)} rectified photos are new or changed and have no mask: "
            f"{', '.join(missing[:8])}. Run prepare-dense-masks on this reconstruction "
            "and pass that revision as --generated-masks")
    sources: dict[str, Path] = {}
    for photo in photos:
        name = photo["filename"]
        root, entry = ((reviewed_root, reviewed[name]) if origins[name] == REVIEWED
                       else (generated_root, generated[name]))
        sources[name] = _safe_pinned_image(root, entry["mask"], entry["mask_sha256"])
        _verified_binary(sources[name], photo["dimensions"])

    (target / "masks").mkdir(parents=True)
    images = []
    for photo in sorted(photos, key=lambda item: item["filename"]):
        name = photo["filename"]
        destination = target / "masks" / (name + ".mask.png")
        shutil.copyfile(sources[name], destination)
        images.append({"filename": name, "source_photo_sha256": photo["sha256"],
                       "mask": "masks/" + destination.name,
                       "mask_sha256": _sha256(destination), "origin": origins[name]})
    report = {
        "schema": SCHEMA, "source_revision": str(sparse),
        "reviewed_mask_revision": str(reviewed_root),
        "reviewed_mask_report_sha256": _sha256(reviewed_root / "mask-report.json"),
        "generated_mask_revision": str(generated_root) if generated_root else None,
        "generated_mask_report_sha256": (_sha256(generated_root / "mask-report.json")
                                         if generated_root else None),
        "status": "candidate / carried-forward-masks",
        "approved_for_reconstruction": False, "distribution_allowed": False,
        "summary": {"reviewed_masks_kept": sum(origin == REVIEWED for origin in origins.values()),
                    "generated_masks_used": sum(origin == GENERATED for origin in origins.values()),
                    "reviewed_masks_not_reusable": sorted(
                        name for name in reviewed if origins.get(name) != REVIEWED)},
        "images": images,
        "limitations": ("Generated masks are automatic and unreviewed. Inspect them before "
                        "dense reconstruction. A kept mask is as correct as its review, "
                        "no more."),
    }
    (target / "mask-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
