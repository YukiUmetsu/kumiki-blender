"""Automated inspection contact sheets and optional image-regression checks.

Image regression is NOT a photographic-fidelity score: it detects changes
between comparable camera/light/material revisions of the *same* asset.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .revision_audit import RevisionAuditError, audit_revision


class VisualQAError(ValueError):
    """A visual inspection request cannot safely compare its inputs."""


def _revision_views(root: Path) -> tuple[str, list[str]]:
    try:
        verification = audit_revision(root)
    except RevisionAuditError as error:
        raise VisualQAError(f"Cannot inspect unverified revision {root}: {error}") from error
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return verification["asset_id"], manifest["files"]["quality_views"]
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise VisualQAError(f"Revision has no valid inspection views: {root}") from error


def _brightness(image: Any) -> dict[str, float]:
    histogram = image.convert("L").histogram()
    pixels = image.width * image.height
    return {
        "mean_luma": round(sum(index * count for index, count in enumerate(histogram)) / (pixels * 255), 5),
        "near_black_fraction": round(sum(histogram[:8]) / pixels, 5),
        "near_white_fraction": round(sum(histogram[248:]) / pixels, 5),
    }


def _contact_sheet(images: list[tuple[str, Any]], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    width, height, margin, label_height = 640, 360, 16, 34
    cols = 2 if len(images) > 1 else 1
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (width + margin) + margin,
                              rows * (height + label_height + margin) + margin), "#15191E")
    draw = ImageDraw.Draw(sheet)
    for index, (label, frame) in enumerate(images):
        left = margin + (index % cols) * (width + margin)
        top = margin + (index // cols) * (height + label_height + margin)
        draw.text((left + 6, top + 7), label, fill="#EEF2F5")
        picture = ImageOps.contain(frame, (width, height), method=Image.Resampling.LANCZOS)
        x = left + (width - picture.width) // 2
        y = top + label_height + (height - picture.height) // 2
        sheet.paste(picture, (x, y))
    sheet.save(output, format="PNG")


def _compare_images(reference: Any, candidate: Any, *,
                    pixel_threshold: int, max_rms: float, max_changed: float,
                    difference_path: Path) -> dict[str, Any]:
    from PIL import ImageChops, ImageStat

    if reference.size != candidate.size:
        return {"status": "review-required", "reason": "image dimensions differ",
                "reference_size": list(reference.size), "candidate_size": list(candidate.size)}
    difference = ImageChops.difference(reference, candidate)
    rms_channels = ImageStat.Stat(difference).rms
    rms = math.sqrt(sum(value * value for value in rms_channels) / len(rms_channels)) / 255.0
    # Mark a pixel changed when *any* color channel exceeds the threshold.
    masks = [channel.point(lambda value: 255 if value > pixel_threshold else 0)
             for channel in difference.split()]
    changed = ImageChops.lighter(masks[0], ImageChops.lighter(masks[1], masks[2]))
    changed_fraction = changed.histogram()[255] / (reference.width * reference.height)
    status = "pass" if rms <= max_rms and changed_fraction <= max_changed else "review-required"
    if status != "pass":
        difference.point(lambda value: min(255, value * 4)).save(difference_path, format="PNG")
    return {"status": status, "rms_normalized": round(rms, 6),
            "changed_fraction": round(changed_fraction, 6),
            "diff_image": str(difference_path.name) if status != "pass" else None}


def inspect_visual_revision(
    revision: str | Path,
    output: str | Path,
    *,
    baseline: str | Path | None = None,
    max_rms: float = 0.02,
    max_changed: float = 0.05,
    pixel_threshold: int = 16,
    use_idiff: bool = False,
) -> dict[str, Any]:
    """Write an immutable contact sheet/report and optional same-camera diffs.

    A trusted, complete revision is required. Output never replaces revision
    artifacts or silently approves visual craftsmanship. OpenImageIO's `idiff`
    is optional and runs as an external tool, never inside Blender Python.
    """
    if any(not isinstance(value, (float, int)) or isinstance(value, bool) or
           not math.isfinite(value) or not 0 <= value <= 1
           for value in (max_rms, max_changed)):
        raise VisualQAError("max_rms and max_changed must be finite fractions in [0, 1]")
    if type(pixel_threshold) is not int or not 0 <= pixel_threshold <= 255:
        raise VisualQAError("pixel_threshold must be an integer in [0, 255]")
    if use_idiff and baseline is None:
        raise VisualQAError("--idiff requires --baseline")
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise VisualQAError("Visual QA requires: pip install 'kumiki-blender[qa]'") from error

    requested_target = Path(output)
    if requested_target.is_symlink():
        raise VisualQAError(f"Visual QA output must not be a symbolic link: {requested_target}")
    target = requested_target.resolve()
    if target.exists() or target.is_symlink():
        raise VisualQAError(f"Visual QA output already exists: {target}")
    current = Path(revision).resolve()
    asset_id, views = _revision_views(current)
    before = Path(baseline).resolve() if baseline is not None else None
    if target.is_relative_to(current) or (before is not None and target.is_relative_to(before)):
        raise VisualQAError("Visual QA output must be outside the immutable revision directories")
    if before is not None:
        before_asset, previous_views = _revision_views(before)
        if before_asset != asset_id or previous_views != views:
            raise VisualQAError("Baseline must have the same asset ID and ordered camera views")
    idiff = shutil.which("idiff") if use_idiff else None
    if use_idiff and idiff is None:
        raise VisualQAError("OpenImageIO idiff is unavailable; install it or omit --idiff")

    image_pairs: list[tuple[str, Any, Any | None]] = []
    try:
        for view in views:
            with Image.open(current / view) as file:
                if file.width * file.height > 16_777_216:
                    raise VisualQAError(f"Inspection image exceeds 16 megapixels: {view}")
                frame = file.convert("RGB")
            reference = None
            if before is not None:
                with Image.open(before / view) as file:
                    if file.width * file.height > 16_777_216:
                        raise VisualQAError(f"Baseline image exceeds 16 megapixels: {view}")
                    reference = file.convert("RGB")
            image_pairs.append((view, frame, reference))
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise VisualQAError(f"Cannot inspect rendered PNG: {error}") from error

    target.mkdir(parents=True, exist_ok=False)
    _contact_sheet([(Path(name).stem, frame) for name, frame, _ in image_pairs],
                   target / "contact-sheet.png")
    results = []
    for index, (name, frame, reference) in enumerate(image_pairs):
        result: dict[str, Any] = {"view": name, "dimensions": [frame.width, frame.height],
                                  "exposure": _brightness(frame)}
        if reference is not None:
            diff_name = f"diff-{index:02d}-{Path(name).stem}.png"
            result["regression"] = _compare_images(reference, frame,
                pixel_threshold=pixel_threshold, max_rms=max_rms,
                max_changed=max_changed, difference_path=target / diff_name)
            if use_idiff:
                completed = subprocess.run([
                    idiff, "-p", "-fail", str(max_rms), "-failpercent", str(max_changed*100),
                    str(before / name), str(current / name),
                ], text=True, capture_output=True, check=False)
                result["openimageio"] = {
                    "status": ("pass" if completed.returncode == 0 else
                               "review-required" if completed.returncode in (1, 2) else "error"),
                    "exit_code": completed.returncode,
                    "summary": (completed.stdout + completed.stderr)[-1500:],
                }
        results.append(result)
    failed = any(result.get("regression", {}).get("status") == "review-required" or
                 result.get("openimageio", {}).get("status") in ("review-required", "error")
                 for result in results)
    report = {
        "schema_version": 1, "asset_id": asset_id,
        "mode": "regression" if before is not None else "inspection",
        "status": ("review-required" if failed else "pass" if before is not None
                   else "inspection-only / artistic-review-required"),
        "thresholds": {"max_rms": max_rms, "max_changed": max_changed,
                       "pixel_threshold": pixel_threshold},
        "contact_sheet": "contact-sheet.png",
        "views": results,
        "limitations": "Image regression detects visual changes, not photographic or historical accuracy",
    }
    (target / "visual-qa.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
