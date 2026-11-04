"""Content-addressed, private per-photo normalization and foreground previews.

COLMAP still receives the full normalized RGB photograph. Foreground masks are
*candidates* for visual review, never fabricated photographic measurements.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .references import ReferenceError, ReferenceImage, ReferenceSet

SCHEMA = "source-photo-preprocess-v1"
FOREGROUND_SCHEMA = "source-foreground-sam2-v1"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _folder(cache: Path, sha: str) -> Path:
    if len(sha) != 64 or not all(char in "0123456789abcdef" for char in sha):
        raise ReferenceError("Invalid per-photo content hash")
    result = cache / sha[:2] / sha
    if any(path.is_symlink() for path in (cache, result.parent, result)):
        raise ReferenceError("Per-photo cache must not contain directory symlinks")
    return result


def _read_entry(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (ValueError, OSError, UnicodeError):
        return None


def _atomic_image(image, path: Path, *, fmt: str = "PNG") -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".photo-", suffix=".png",
                                         delete=False) as stream:
            temporary = Path(stream.name)
        image.save(temporary, format=fmt)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".receipt-", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_normalized_photo(cache: Path, photo: ReferenceImage) -> tuple[Path, dict] | None:
    """Reuse only a byte-verified normalization of precisely these source bytes."""
    folder = _folder(cache, photo.sha256)
    image = folder / "normalized.png"
    receipt = _read_entry(folder / "normalized.json")
    if not receipt or image.is_symlink() or not image.is_file():
        return None
    if (receipt.get("schema") != SCHEMA or receipt.get("source_sha256") != photo.sha256
            or receipt.get("normalized_sha256") != _digest(image)):
        return None
    try:
        from PIL import Image
        with Image.open(image) as decoded:
            decoded.verify()
            if decoded.mode != "RGB" or list(decoded.size) != receipt.get("dimensions"):
                return None
        return image, receipt
    except (OSError, ValueError):
        return None


def normalize_cached_photo(photo: ReferenceImage, cache: Path) -> tuple[Path, dict, bool]:
    """Normalize one photo once, irrespective of photo order or asset revision."""
    import io
    from PIL import Image, ImageOps

    hit = read_normalized_photo(cache, photo)
    if hit is not None:
        return *hit, True
    folder = _folder(cache, photo.sha256)
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ("normalized.png", "normalized.json"):
        if (folder / filename).is_symlink():
            raise ReferenceError("Cache file is a symlink")
    data = photo.path.read_bytes()
    if hashlib.sha256(data).hexdigest() != photo.sha256:
        raise ReferenceError(f"Source photo changed during preparation: {photo.path.name}")
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        original_format, original_mode = source.format, source.mode
        orientation = source.getexif().get(274, 1)
        orientation = orientation if orientation in range(1, 9) else 1
        upright = ImageOps.exif_transpose(source)
        alpha = "A" in upright.getbands() or "transparency" in upright.info
        if alpha:
            rgba = upright.convert("RGBA")
            normalized = Image.new("RGB", rgba.size, (255, 255, 255))
            normalized.paste(rgba, mask=rgba.getchannel("A"))
        else:
            normalized = upright.convert("RGB")
        _atomic_image(normalized, folder / "normalized.png")
        receipt = {"schema": SCHEMA, "source_sha256": photo.sha256,
                   "normalized_sha256": _digest(folder / "normalized.png"),
                   "dimensions": list(normalized.size),
                   "original_dimensions": list(photo.dimensions),
                   "source_format": original_format, "source_mode": original_mode,
                   "exif_orientation_applied": orientation,
                   "alpha_composited_on_white": bool(alpha),
                   "background_removed": False,
                   "interpretation": "Exact RGB normalization; full evidence retained for SfM."}
    _atomic_json(receipt, folder / "normalized.json")
    return folder / "normalized.png", receipt, False


def _foreground_key(normalized_sha: str, checkpoint_sha: str, size: str) -> str:
    fields = json.dumps([FOREGROUND_SCHEMA, normalized_sha, checkpoint_sha, size, 1280])
    return hashlib.sha256(fields.encode()).hexdigest()


def _foreground_preview(normalized: Path, cache: Path, source_sha: str,
                        checkpoint_sha: str, checkpoint: Path, size: str, device: str,
                        predictor_loader: Callable[..., Any], state: dict) -> tuple[dict, bool]:
    from PIL import Image
    import numpy as np
    from .masking import _auto_sam2_guide, _sam2_mask, _DEFAULT_BOX

    key = _foreground_key(_digest(normalized), checkpoint_sha, size)
    folder = _folder(cache, source_sha)
    mask = folder / f"foreground-{key}.png"
    preview = folder / f"isolated-{key}.png"
    receipt_file = folder / f"foreground-{key}.json"
    record = _read_entry(receipt_file)
    if (record is not None and record.get("schema") == FOREGROUND_SCHEMA
            and record.get("key") == key and mask.is_file() and preview.is_file()
            and not mask.is_symlink() and not preview.is_symlink()
            and record.get("mask_sha256") == _digest(mask)
            and record.get("isolated_sha256") == _digest(preview)):
        try:
            with Image.open(normalized) as rgb, Image.open(mask) as old_mask, Image.open(preview) as rgba:
                if (old_mask.mode == "L" and rgba.mode == "RGBA" and
                        rgb.size == old_mask.size == rgba.size):
                    old_mask.verify()
                    rgba.verify()
                    return record, True
        except (OSError, ValueError):
            pass
    if any(path.is_symlink() for path in (mask, preview, receipt_file)):
        raise ReferenceError("Foreground cache contains symlink")
    if "predictor" not in state:
        state["predictor"], state["device"] = predictor_loader(checkpoint, size, device)
    with Image.open(normalized) as source:
        photo = source.convert("RGB")
    guide = _auto_sam2_guide(photo, _DEFAULT_BOX)
    mask_array, details = _sam2_mask(photo, guide, state["predictor"])
    binary = Image.fromarray(np.asarray(mask_array, dtype=np.uint8), mode="L")
    _atomic_image(binary, mask)
    rgba = photo.convert("RGBA")
    rgba.putalpha(binary)
    _atomic_image(rgba, preview)
    occupied = float(np.count_nonzero(mask_array)) / float(mask_array.size)
    record = {"schema": FOREGROUND_SCHEMA, "key": key,
              "source_sha256": source_sha, "normalized_sha256": _digest(normalized),
              "model_sha256": checkpoint_sha, "sam2_size": size,
              "device": state["device"], "mask_sha256": _digest(mask),
              "isolated_sha256": _digest(preview),
              "foreground_fraction": round(occupied, 6),
              "status": "candidate / manual-review-required" if 0.05 <= occupied <= 0.90
              else "suspect / manual-review-required",
              "details": details,
              "limitations": "Broad-prompt foreground preview, not verified statue segmentation or camera evidence."}
    _atomic_json(record, receipt_file)
    return record, False


def prepare_statue_photo_cache(references: ReferenceSet, cache: Path, output: Path, *,
                               checkpoint: Path | None = None, sam2_size: str = "small",
                               sam2_device: str = "mps", foreground: bool = True,
                               predictor_loader: Callable[..., Any] | None = None,
                               progress: Callable[[int, int, bool], None] | None = None,
                               resume: bool = False) -> dict:
    """Process all new photos, reuse unchanged cached work, write per-run index."""
    from .masking import _load_sam2_predictor

    if (cache.is_symlink() or output.is_symlink() or (output.exists() and not resume)
            or output.resolve().is_relative_to(cache.resolve())
            or cache.resolve().is_relative_to(output.resolve())):
        raise ReferenceError("Photo preprocessing output and cache must be separate real directories")
    if foreground and (checkpoint is None or checkpoint.is_symlink() or not checkpoint.is_file()):
        raise ReferenceError("SAM2 foreground preparation needs an existing checkpoint")
    if sam2_size not in ("tiny", "small", "base_plus", "large"):
        raise ReferenceError("Invalid SAM2 checkpoint size")
    cache.mkdir(parents=True, exist_ok=True)
    sha = _digest(checkpoint) if foreground and checkpoint is not None else None
    if resume:
        prior = _read_entry(output / "photo-preprocess-report.json")
        if (not output.is_dir() or prior is None or
                prior.get("source_manifest_sha256") != references.manifest_sha256 or
                prior.get("foreground_model_sha256") != sha):
            raise ReferenceError("Cannot resume preprocessing with changed photos or model")
    else:
        output.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {}
    report: dict[str, Any] = {"schema_version": 1, "asset_id": references.asset_id,
                              "status": "running", "cache_root": str(cache.resolve()),
                              "source_manifest_sha256": references.manifest_sha256,
                              "foreground_model_sha256": sha, "source_images": len(references.images),
                              "images": [], "normalized_hits": 0, "normalized_misses": 0,
                              "foreground_hits": 0, "foreground_misses": 0,
                              "qa_hits": 0, "qa_misses": 0,
                              "background_removal": "SAM2 candidate previews; never used as original evidence"
                              if foreground else "disabled"}

    def checkpoint_report():
        _atomic_json(report, output / "photo-preprocess-report.json")

    checkpoint_report()
    for position, photo in enumerate(references.images, 1):
        normalized, receipt, hit = normalize_cached_photo(photo, cache)
        report["normalized_hits" if hit else "normalized_misses"] += 1
        entry = {"original_filename": photo.path.name, "source_sha256": photo.sha256,
                 "normalized_sha256": receipt["normalized_sha256"],
                 "normalized_cache_path": str(normalized),
                 "dimensions": receipt["dimensions"], "normalized_cache_hit": hit}
        if foreground:
            mask, mask_hit = _foreground_preview(
                normalized, cache, photo.sha256, sha, checkpoint, sam2_size, sam2_device,
                predictor_loader or _load_sam2_predictor, state)
            report["foreground_hits" if mask_hit else "foreground_misses"] += 1
            entry.update(foreground_cache_hit=mask_hit, foreground_status=mask["status"],
                         foreground_fraction=mask["foreground_fraction"],
                         foreground_key=mask["key"],
                         foreground_mask_path=str(_folder(cache, photo.sha256) /
                                                  f"foreground-{mask['key']}.png"),
                         isolated_preview_path=str(_folder(cache, photo.sha256) /
                                                  f"isolated-{mask['key']}.png"))
        from .photo_early_qa import analyze_cached_photo
        qa, qa_hit = analyze_cached_photo(
            normalized, photo.sha256, cache,
            mask_path=Path(entry["foreground_mask_path"]) if foreground else None,
            model_sha=sha)
        report["qa_hits" if qa_hit else "qa_misses"] += 1
        entry.update(qa_cache_hit=qa_hit, qa_key=qa["key"],
                     qa_confidence_path=qa["confidence_path"],
                     qa_confidence_sha256=qa["confidence_sha256"],
                     qa_warnings=qa["warnings"], qa_status=qa["status"],
                     qa_brightness=qa["brightness"], qa_blur_score=qa["blur_score"],
                     qa_average_hash=qa["average_hash"])
        report["images"].append(entry)
        checkpoint_report()
        if progress is not None:
            progress(position, len(references.images), hit)
    # Duplicate warnings are advisory; identical photographic appearances
    # are not independent camera baselines, even when file bytes differ.
    for pos, entry in enumerate(report["images"]):
        for earlier in report["images"][:pos]:
            hamming = sum(a != b for a, b in zip(entry["qa_average_hash"],
                                                  earlier["qa_average_hash"]))
            if hamming <= 3:
                entry["qa_warnings"].append(
                    f"possible-near-duplicate:{earlier['original_filename']}")
                break
    from .photo_early_qa import write_qa_contact_sheet
    write_qa_contact_sheet(report["images"], output / "photo-qa-contact-sheet.jpg")
    report["qa_contact_sheet"] = "photo-qa-contact-sheet.jpg"
    report["qa_warning_count"] = sum(bool(row["qa_warnings"]) for row in report["images"])
    report["status"] = "candidate / cached-photo-preparation"
    checkpoint_report()
    return report
