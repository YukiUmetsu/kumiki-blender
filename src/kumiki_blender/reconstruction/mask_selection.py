"""Compose an immutable, photo-by-photo selection from two reviewed mask revisions.

Neither segmentation method is assumed correct.  The selected masks retain
their original hashes and receive fresh preview overlays for human inspection.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .dense import (DenseSettings, _check_regular, _json, _sha256,
                    plan_dense_reconstruction)
from .references import ReferenceError


def _mask_source(reconstruction: Path, output: Path, revision: str | Path,
                 filenames: list[str]) -> dict[str, Any]:
    source = Path(revision).resolve()
    if (Path(revision).is_symlink() or not source.is_dir()
            or (source / "masks").is_symlink() or
            output == source or output.is_relative_to(source) or
            source.is_relative_to(output)):
        raise ReferenceError("Mask source must be a separate regular revision")
    report_path = source / "mask-report.json"
    _check_regular(report_path, source)
    report = _json(report_path)
    if (not str(report.get("status", "")).startswith("candidate / ")
            or report.get("source_revision") != str(reconstruction)
            or not isinstance(report.get("images"), list)):
        raise ReferenceError("Mask source is incomplete or belongs to another reconstruction")

    # Re-run the existing photo/mask QA to verify exact dimensions, binary
    # pixels, camera-registration provenance and all source photo checksums.
    plan = plan_dense_reconstruction(
        reconstruction, output,
        DenseSettings(min_registered_images=3, min_registered_fraction=0.25,
                      refine=False, texture=False),
        rectified_masks=source / "masks", executables={})
    if plan["undistorted_images"] != filenames:
        raise ReferenceError("Mask revision image coverage differs from reconstruction")
    entries: dict[str, dict] = {}
    for entry in report["images"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("filename"), str)
                or entry["filename"] in entries):
            raise ReferenceError("Mask report contains invalid or duplicate photo entries")
        entries[entry["filename"]] = entry
    if set(entries) != set(filenames):
        raise ReferenceError("Mask report does not cover every rectified image")
    images = _json(reconstruction / "undistorted-reference-index.json")["images"]
    photos = {item["filename"]: item["sha256"] for item in images}
    for item in plan["rectified_masks"]:
        name = item["filename"]
        expected = "masks/" + name + ".mask.png"
        if (entries[name].get("mask") != expected
                or entries[name].get("mask_sha256") != item["sha256"]
                or entries[name].get("source_photo_sha256") != photos[name]):
            raise ReferenceError(f"Source mask/photo provenance changed: {name}")
    return {"path": source, "report_sha256": _sha256(report_path),
            "requires_anatomy_inventory": report.get("requires_anatomy_inventory") is True
            or report.get("segmenter") == "sam2",
            "entries": entries, "masks": {item["filename"]: item for item in plan["rectified_masks"]}}


def compose_rectified_masks(reconstruction: str | Path, primary_mask_revision: str | Path,
                            alternate_mask_revision: str | Path, output: str | Path, *,
                            alternate_images: list[str]) -> dict[str, Any]:
    """Combine pinned rembg/GrabCut candidates, explicitly selecting each override.

    All two-source validation happens *before* the output directory is made.
    The resulting candidate is still not approved for reconstruction.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise ReferenceError("Hybrid masks require Pillow: pip install -e '.[masks]'") from error
    reconstruction_path = Path(reconstruction).resolve()
    target = Path(output).resolve()
    if (Path(output).is_symlink() or target.exists()
            or target == reconstruction_path or target.is_relative_to(reconstruction_path)):
        raise ReferenceError("Hybrid masks require a new output outside the reconstruction")
    if (not isinstance(alternate_images, list) or not alternate_images
            or any(not isinstance(name, str) for name in alternate_images)
            or len(set(alternate_images)) != len(alternate_images)):
        raise ReferenceError("Select one or more unique alternate photo filenames")

    initial = plan_dense_reconstruction(
        reconstruction, output,
        DenseSettings(min_registered_images=3, min_registered_fraction=0.25,
                      refine=False, texture=False), executables={})
    filenames = initial["undistorted_images"]
    unknown = set(alternate_images) - set(filenames)
    if unknown:
        raise ReferenceError("Unknown registered alternate photo: " + ", ".join(sorted(unknown)))
    primary = _mask_source(reconstruction_path, target, primary_mask_revision, filenames)
    alternate = _mask_source(reconstruction_path, target, alternate_mask_revision, filenames)
    if primary["path"] == alternate["path"]:
        raise ReferenceError("Primary and alternate masks must be distinct revisions")

    target.mkdir(parents=True, exist_ok=False)
    (target / "masks").mkdir()
    (target / "review").mkdir()
    report: dict[str, Any] = {
        "schema_version": 1, "source_revision": str(reconstruction_path),
        "source_mask_revisions": {"primary": str(primary["path"]),
                                  "alternate": str(alternate["path"])},
        "source_mask_reports_sha256": {"primary": primary["report_sha256"],
                                       "alternate": alternate["report_sha256"]},
        "alternate_images": sorted(alternate_images),
        "requires_anatomy_inventory": (primary["requires_anatomy_inventory"]
                                       or alternate["requires_anatomy_inventory"]),
        "approved_for_reconstruction": False,
        "status": "incomplete / manual-mask-review-required", "images": [],
    }

    def checkpoint() -> None:
        (target / "mask-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")

    checkpoint()
    index = _json(reconstruction_path / "undistorted-reference-index.json")
    photo_hashes = {entry["filename"]: entry["sha256"] for entry in index["images"]}
    for name in filenames:
        selected = "alternate" if name in alternate_images else "primary"
        source = alternate if selected == "alternate" else primary
        entry = source["entries"][name]
        source_mask = Path(source["masks"][name]["path"])
        original_hash = source["masks"][name]["sha256"]
        # Never trust a preflight check alone if a source changed mid-copy.
        _check_regular(source_mask, source["path"])
        if _sha256(source_mask) != original_hash:
            raise ReferenceError(f"Mask changed after preflight: {name}")
        destination = target / "masks" / (name + ".mask.png")
        shutil.copyfile(source_mask, destination)
        if _sha256(destination) != original_hash:
            raise ReferenceError(f"Mask changed during hybrid copy: {name}")
        with Image.open(destination) as mask:
            mask.load()
            foreground = sum(mask.convert("L").histogram()[128:])
            occupancy = round(foreground / (mask.width * mask.height), 5)
            with Image.open(reconstruction_path / "undistorted/images" / name) as image:
                image.load()
                if _sha256(reconstruction_path / "undistorted/images" / name) != photo_hashes[name]:
                    raise ReferenceError(f"Rectified photo changed after preflight: {name}")
                preview = image.convert("RGB")
                preview.thumbnail((700, 700))
                red = Image.new("RGB", preview.size, (240, 64, 42))
                coverage = mask.resize(preview.size, Image.Resampling.NEAREST)
                overlay = Image.composite(red, preview, coverage.point(lambda x: round(x * .3)))
                preview_path = target / "review" / (name + ".jpg")
                overlay.save(preview_path, "JPEG", quality=86)
        warnings = list(entry.get("warnings", [])) if isinstance(entry.get("warnings"), list) else []
        if occupancy < 0.05 or occupancy > 0.90:
            warnings.append("Suspicious foreground fraction; inspect missing subject/background")
        if selected == "alternate":
            warnings.append("Chosen from alternate segmentation; inspect scenery leakage")
        report["images"].append({
            "filename": name, "source_photo_sha256": photo_hashes[name],
            "selected_source": selected, "source_mask_sha256": original_hash,
            "mask": str(destination.relative_to(target)), "mask_sha256": original_hash,
            "preview": str(preview_path.relative_to(target)),
            "foreground_fraction": occupancy, "warnings": warnings,
            "guide": entry.get("guide"),
        })
        checkpoint()
    columns, width, height = 4, 288, 238
    sheet = Image.new("RGB", (columns * width,
                               ((len(filenames) + columns - 1) // columns) * height),
                      (28, 31, 34))
    draw = ImageDraw.Draw(sheet)
    for i, item in enumerate(report["images"]):
        x, y = (i % columns) * width, (i // columns) * height
        with Image.open(target / item["preview"]) as overlay:
            overlay.thumbnail((width - 8, height - 30))
            sheet.paste(overlay, (x + 4, y + 4))
        draw.text((x + 6, y + height - 23),
                  f"{item['filename'][:30]} ({item['selected_source']})", fill=(234, 236, 238))
    sheet.save(target / "mask-contact-sheet.jpg", "JPEG", quality=85)
    report["contact_sheet"] = "mask-contact-sheet.jpg"
    report["status"] = "candidate / hybrid-mask-review-required"
    checkpoint()
    return report
