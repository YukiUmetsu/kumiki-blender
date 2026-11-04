"""Conservative, content-addressed source-photo quality and subject confidence.

The three-class output is an *advisory* feature-priority map, NOT a binary
segmentation or photographic measurement. A suspect mask makes pixels uncertain
so an incorrect SAM2 silhouette cannot erase real anatomy from SfM.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .references import ReferenceError

QA_VERSION = "source-photo-qa-v1"
UNKNOWN, SUBJECT, BACKGROUND = 128, 255, 0


def analyze_cached_photo(normalized: Path, source_sha: str, cache: Path, *,
                         mask_path: Path | None = None, model_sha: str | None = None,
                         minimum_side: int = 800) -> tuple[dict, bool]:
    """Return verified cached quality receipt and 0/128/255 confidence raster."""
    import numpy as np
    from PIL import Image, ImageFilter, ImageStat
    from .photo_preprocess_cache import _atomic_image, _atomic_json, _digest, _folder, _read_entry

    if minimum_side < 32:
        raise ValueError("Minimum source-photo side must be at least 32 pixels")
    if normalized.is_symlink() or not normalized.is_file():
        raise ReferenceError("Normalized photo must be a real file")
    if mask_path is not None and (mask_path.is_symlink() or not mask_path.is_file()):
        raise ReferenceError("QA mask is missing or a symlink")
    normalized_sha = _digest(normalized)
    mask_sha = _digest(mask_path) if mask_path is not None else None
    key = hashlib.sha256(json.dumps([QA_VERSION, normalized_sha, mask_sha,
                                    model_sha, minimum_side]).encode()).hexdigest()
    folder = _folder(cache, source_sha)
    folder.mkdir(parents=True, exist_ok=True)
    confidence_file = folder / f"qa-confidence-{key}.png"
    receipt_file = folder / f"qa-{key}.json"
    if confidence_file.is_symlink() or receipt_file.is_symlink():
        raise ReferenceError("QA cache cannot contain symbolic links")
    old = _read_entry(receipt_file)
    if (old and old.get("schema") == QA_VERSION and old.get("key") == key
            and old.get("confidence_sha256") and confidence_file.is_file()
            and old["confidence_sha256"] == _digest(confidence_file)):
        try:
            with Image.open(normalized) as source, Image.open(confidence_file) as confidence:
                if confidence.mode == "L" and confidence.size == source.size:
                    confidence.verify()
                    return old, True
        except (OSError, ValueError):
            pass

    with Image.open(normalized) as input_image:
        rgb = input_image.convert("RGB")
    width, height = rgb.size
    if width < 32 or height < 32:
        raise ReferenceError("Photo too small to assess reliable features")
    preview = rgb.copy()
    preview.thumbnail((512, 512))
    grayscale = np.asarray(preview.convert("L"), dtype=np.float32)
    # Multi-directional second differences; the statistic is diagnostic, not
    # a calibrated optical measurement of blur.
    dx = grayscale[:, 2:] - 2 * grayscale[:, 1:-1] + grayscale[:, :-2]
    dy = grayscale[2:, :] - 2 * grayscale[1:-1, :] + grayscale[:-2, :]
    blur_score = float((np.var(dx) + np.var(dy)) / 2)
    brightness = float(ImageStat.Stat(preview.convert("L")).mean[0])
    small = np.asarray(preview.convert("L").resize((8, 8)), dtype=np.float32)
    ahash = "".join("1" if x >= small.mean() else "0" for x in small.flat)
    warnings = []
    if min(width, height) < minimum_side:
        warnings.append("low-resolution")
    if brightness < 35:
        warnings.append("underexposed")
    if brightness > 225:
        warnings.append("overexposed")
    if blur_score < 8:
        warnings.append("low-detail-or-blurred")

    confidence = Image.new("L", rgb.size, UNKNOWN)
    fraction = None
    if mask_path is None:
        warnings.append("no-foreground-candidate")
    else:
        with Image.open(mask_path) as image:
            if image.mode != "L" or image.size != rgb.size:
                raise ReferenceError("QA mask has incompatible pixel grid or mode")
            mask = image.copy().point(lambda value: 255 if value >= 128 else 0)
        # Downsample only for the diagnostics; feature confidence remains in
        # exactly the original normalized-photo pixel grid.
        mask_small = np.asarray(mask.resize(preview.size, Image.Resampling.NEAREST)) >= 128
        fraction = float(np.asarray(mask_small, dtype=np.float32).mean())
        if fraction < 0.05:
            warnings.append("foreground-too-small")
        if fraction > 0.90:
            warnings.append("foreground-too-large")
        boundary = np.concatenate((mask_small[0], mask_small[-1],
                                   mask_small[:, 0], mask_small[:, -1]))
        if float(boundary.mean()) > 0.20:
            warnings.append("foreground-touches-frame")
        colors = np.asarray(preview, dtype=np.float32)
        sky_like = ((colors[:, :, 2] > colors[:, :, 0] * 1.3) &
                    (colors[:, :, 2] > colors[:, :, 1] * 1.09) &
                    (colors[:, :, 2] > 100))
        if int(mask_small.sum()) and float((sky_like & mask_small).sum() / mask_small.sum()) > 0.20:
            warnings.append("possible-blue-background-leakage")
        suspect = any(label in warnings for label in (
            "foreground-too-small", "foreground-too-large", "foreground-touches-frame",
            "possible-blue-background-leakage"))
        if not suspect:
            # Keep a broad uncertain band on both sides of the SAM2 boundary;
            # no feature is *removed* on the basis of segmentation alone.
            erosion = mask.filter(ImageFilter.MinFilter(9))
            dilation = mask.filter(ImageFilter.MaxFilter(15))
            confidence.paste(SUBJECT, mask=erosion)
            far_background = dilation.point(lambda value: 255 if value == 0 else 0)
            confidence.paste(BACKGROUND, mask=far_background)

    _atomic_image(confidence, confidence_file)
    receipt = {"schema": QA_VERSION, "key": key, "source_sha256": source_sha,
               "normalized_sha256": normalized_sha, "foreground_sha256": mask_sha,
               "foreground_model_sha256": model_sha, "dimensions": [width, height],
               "confidence_path": str(confidence_file.resolve()),
               "confidence_sha256": _digest(confidence_file),
               "foreground_fraction": round(fraction, 6) if fraction is not None else None,
               "brightness": round(brightness, 2), "blur_score": round(blur_score, 2),
               "average_hash": ahash, "warnings": warnings,
               "status": "suspect / mask-uncertain" if any(x.startswith("foreground-") or
                           "leakage" in x for x in warnings) else "candidate / review-required",
               "interpretation": "Uncertain pixels retain real features. No verified anatomical labels."}
    _atomic_json(receipt, receipt_file)
    return receipt, False


def write_qa_contact_sheet(entries: list[dict], output: Path) -> None:
    """Show source and confidence overlay, including warning annotations."""
    from PIL import Image, ImageDraw

    columns, cell_width, cell_height = 3, 320, 260
    sheet = Image.new("RGB", (columns * cell_width,
                              ((len(entries) + columns - 1) // columns) * cell_height), "#19212b")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(entries):
        with Image.open(row["normalized_cache_path"]) as image, \
                Image.open(row["qa_confidence_path"]) as confidence:
            source = image.convert("RGB")
            overlay = Image.new("RGB", source.size, (250, 175, 40))
            # Gold = uncertain, green = prioritized subject, blue = background.
            values = confidence.convert("RGB").point(lambda x: x)
            import numpy as np
            labels = np.asarray(confidence)
            false_color = np.zeros((*labels.shape, 3), dtype=np.uint8)
            false_color[labels == 0] = (65, 115, 230)
            false_color[labels == 128] = (245, 177, 50)
            false_color[labels == 255] = (70, 205, 130)
            overlay = Image.fromarray(false_color, "RGB")
            source = Image.blend(source, overlay, 0.25)
            source.thumbnail((cell_width - 12, 190))
            x, y = (index % columns) * cell_width + 6, (index // columns) * cell_height + 6
            sheet.paste(source, (x, y))
            draw.text((x, y + 194), str(row["original_filename"])[:44], fill="white")
            draw.text((x, y + 212), ", ".join(row["qa_warnings"])[:48] or "No automatic warnings",
                      fill="#ffcc88" if row["qa_warnings"] else "#9ddabe")
    sheet.save(output, quality=86)
