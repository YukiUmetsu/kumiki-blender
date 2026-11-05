"""Optional, explicitly requested correction of an automatic mask revision.

The normal track-guided pipeline is automatic. This operation exists only for
people who want to promote a withheld proposal or paint small normalized
include/exclude polygons without modifying any existing mask revision.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .dense import _sha256
from .mask_refinement import _load_edits
from .mask_track_guidance import _safe_pinned_image
from .mask_track_resegmentation import SCHEMA as TRACK_SCHEMA
from .references import ReferenceError


SCHEMA = "optional-track-mask-corrections-v1"


def _verified_binary(path: Path, dimensions: list[int]) -> None:
    """Reject tampered, wrong-sized, or non-binary masks before output exists."""
    from PIL import Image

    with Image.open(path) as mask:
        mask.load()
        if (mask.format != "PNG" or mask.mode != "L" or
                list(mask.size) != dimensions):
            raise ReferenceError(f"Mask is not a rectified grayscale PNG: {path.name}")
        colors = mask.getcolors(maxcolors=3)
        if colors is None or {value for _, value in colors} != {0, 255}:
            raise ReferenceError(f"Mask must contain both binary foreground and background: {path.name}")


def select_track_mask_candidates(reconstruction: str | Path, mask_revision: str | Path,
                                 output: str | Path, *, use_proposals: list[str],
                                 edits: str | Path | None = None) -> dict[str, Any]:
    """Copy chosen proposals, optionally apply user polygons, and re-run normal QA.

    This is never invoked from the default automatic pipeline. Explicit user
    choices are recorded as such and never passed off as model verification.
    """
    from PIL import Image, ImageDraw

    sparse = Path(reconstruction).resolve()
    source = Path(mask_revision).resolve()
    target = Path(output).resolve()
    if (Path(mask_revision).is_symlink() or not source.is_dir()
            or Path(output).is_symlink() or target.exists()
            or target == source or target.is_relative_to(source)
            or target == sparse or target.is_relative_to(sparse)
            or source.is_relative_to(target)):
        raise ReferenceError("Mask correction requires an existing revision and unused separate output")
    if (not isinstance(use_proposals, list) or not use_proposals or
            any(not isinstance(name, str) for name in use_proposals) or
            len(set(use_proposals)) != len(use_proposals)):
        raise ReferenceError("Choose at least one distinct candidate photo filename")
    report_path = source / "mask-report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise ReferenceError("Missing source mask report")
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    if (previous.get("schema") not in (TRACK_SCHEMA, "track-guided-mask-revision-v1",
                                         "track-guided-mask-revision-v2") or
            previous.get("source_revision") != str(sparse) or
            not str(previous.get("status", "")).startswith("candidate / ") or
            not isinstance(previous.get("images"), list)):
        raise ReferenceError("Only a completed track-guided mask revision can supply proposals")
    manifest_path = sparse / "undistorted-reference-index.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ReferenceError("Registered camera manifest is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    photos = {item["filename"]: item for item in manifest["images"]}
    entries = {item["filename"]: item for item in previous["images"]}
    if (len(photos) != len(manifest["images"]) or
            len(entries) != len(previous["images"]) or not entries or
            set(entries) != set(photos) or set(use_proposals) - set(entries)):
        raise ReferenceError("Requested proposals or mask/photo coverage do not match registered cameras")
    choices = set(use_proposals)
    edits_map, edits_sha = _load_edits(edits, list(entries))
    source_files: dict[str, Path] = {}
    for name, entry in entries.items():
        if (not name or Path(name).name != name or
                entry.get("source_photo_sha256") != photos[name]["sha256"]):
            raise ReferenceError(f"Invalid registered photo provenance: {name}")
        original = _safe_pinned_image(source, entry["mask"], entry["mask_sha256"])
        _verified_binary(original, photos[name]["dimensions"])
        photograph = sparse / "undistorted/images" / name
        if (photograph.is_symlink() or not photograph.is_file() or
                _sha256(photograph) != photos[name]["sha256"]):
            raise ReferenceError(f"Source photograph changed: {name}")
        candidate = original
        if name in choices:
            receipt = entry.get("track_guidance", {})
            expected = f"candidates/{name}.track-guided.png"
            if (receipt.get("proposed_mask") != expected or
                    not isinstance(receipt.get("proposed_mask_sha256"), str)):
                raise ReferenceError(f"No pinned SAM2 proposal to promote for {name}")
            candidate = _safe_pinned_image(source, expected, receipt["proposed_mask_sha256"])
            _verified_binary(candidate, photos[name]["dimensions"])
        source_files[name] = candidate

    target.mkdir(parents=True, exist_ok=False)
    (target / "masks").mkdir()
    (target / "review").mkdir()
    report: dict[str, Any] = {
        "schema": SCHEMA, "schema_version": 1,
        "source_revision": str(sparse), "source_mask_revision": str(source),
        "source_mask_report_sha256": _sha256(report_path), "edits_sha256": edits_sha,
        "use_proposals": sorted(choices),
        "status": "incomplete / optional-mask-correction", "images": [],
        "approved_for_reconstruction": False, "distribution_allowed": False,
        "requires_anatomy_inventory": True,
        "limitations": "User-selected proposals and polygons are not independent ground truth. Existing coverage and cross-view QA still apply.",
    }

    def checkpoint() -> None:
        (target / "mask-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8")

    checkpoint()
    for name in sorted(entries):
        entry = entries[name]
        chosen = source_files[name]
        if _sha256(chosen) != (entry["track_guidance"]["proposed_mask_sha256"]
                                 if name in choices else entry["mask_sha256"]):
            raise ReferenceError(f"Source mask changed during correction: {name}")
        destination = target / "masks" / (name + ".mask.png")
        changed = bool(edits_map.get(name) and any(edits_map[name].values()))
        if changed:
            with Image.open(chosen) as mask:
                corrected = mask.copy()
            draw = ImageDraw.Draw(corrected)
            for label, fill in (("include_polygons", 255), ("exclude_polygons", 0)):
                for polygon in edits_map[name][label]:
                    draw.polygon([(round(x*(corrected.width-1)),
                                   round(y*(corrected.height-1))) for x, y in polygon],
                                 fill=fill)
            corrected.save(destination, format="PNG", optimize=True)
            _verified_binary(destination, photos[name]["dimensions"])
        else:
            shutil.copyfile(chosen, destination)
        with Image.open(destination) as binary, Image.open(
                sparse / "undistorted/images" / name) as original_photo:
            fraction = round(sum(binary.histogram()[128:]) / (binary.width * binary.height), 6)
            preview = original_photo.convert("RGB")
            preview.thumbnail((900, 900))
            coverage = binary.resize(preview.size, Image.Resampling.NEAREST)
            overlay = Image.composite(Image.new("RGB", preview.size, (238, 58, 35)),
                                      preview, coverage.point(lambda value: round(value * .28)))
            preview_path = target / "review" / (name + ".jpg")
            overlay.save(preview_path, "JPEG", quality=86)
        # Alternative and proposed-mask paths belonged to the parent revision;
        # keep their provenance via the pinned parent report, not dangling
        # relative paths that falsely appear to exist in this new revision.
        portable = {key: value for key, value in entry.items()
                    if key not in ("candidate_masks", "track_guidance")}
        report["images"].append({
            **portable, "mask": "masks/" + destination.name,
            "mask_sha256": _sha256(destination), "preview": "review/" + preview_path.name,
            "foreground_fraction": fraction,
            "selection": "explicit-user-proposal" if name in choices else "automatic-base",
            "user_polygon_edit": changed,
            "source_proposal_sha256": (entry["track_guidance"]["proposed_mask_sha256"]
                                       if name in choices else None),
            "original_mask_sha256": entry["mask_sha256"],
            "original_candidate_comparison": entry.get("candidate_comparison"),
            "candidate_comparison": ({"status": "candidate / explicit-user-review",
                                       "selected": "track-guided-proposal"}
                                     if name in choices or changed else entry.get("candidate_comparison")),
        })
        checkpoint()
    columns, width, height = 4, 288, 238
    contact = Image.new("RGB", (columns*width,
                               ((len(entries)+columns-1)//columns)*height), (28, 31, 34))
    draw = ImageDraw.Draw(contact)
    for i, entry in enumerate(report["images"]):
        x, y = i % columns*width, i // columns*height
        with Image.open(target / entry["preview"]) as preview:
            preview.thumbnail((width-8, height-30))
            contact.paste(preview, (x+4, y+4))
        label = "user" if entry["selection"] == "explicit-user-proposal" else "auto"
        draw.text((x+5, y+height-23), f'{entry["filename"][:32]} ({label})',
                  fill=(234, 236, 238))
    contact.save(target / "mask-contact-sheet.jpg", "JPEG", quality=85)
    report["contact_sheet"] = "mask-contact-sheet.jpg"
    report["status"] = "candidate / optional-user-corrections"
    checkpoint()
    return report
