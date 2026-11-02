"""Reviewable foreground masks for registered, *rectified* statue photographs.

GrabCut supplies a CPU-only starting point, not proof of subject isolation.
Source photos remain immutable. Every generated mask requires visual review.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .dense import (DenseSettings, _json, _sha256,
                    plan_dense_reconstruction)
from .references import ReferenceError

_DEFAULT_BOX = (0.12, 0.04, 0.88, 0.96)


def _normalized_pair(value: Any, label: str) -> tuple[float, float]:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(x) not in (float, int) or not math.isfinite(x)
                   or x < 0 or x > 1 for x in value)):
        raise ReferenceError(f"{label} requires normalized [x,y] values in [0,1]")
    return float(value[0]), float(value[1])


def _box(value: Any) -> tuple[float, float, float, float]:
    if (not isinstance(value, (tuple, list)) or len(value) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   or not 0 <= x <= 1 for x in value)):
        raise ReferenceError("Subject box needs four normalized [left,top,right,bottom] numbers")
    left, top, right, bottom = (float(x) for x in value)
    if right-left < 0.05 or bottom-top < 0.05:
        raise ReferenceError("Subject box must have positive width and height (at least 5%)")
    return left, top, right, bottom


def _guide(value: Any, *, default_box: tuple[float, ...]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - {"box", "foreground_points", "background_points"}:
        raise ReferenceError("Mask guide may contain only box, foreground_points, background_points")
    box = _box(value.get("box", default_box))
    points: dict[str, list[list[float]]] = {}
    for key in ("foreground_points", "background_points"):
        raw = value.get(key, [])
        if not isinstance(raw, list) or len(raw) > 100:
            raise ReferenceError(f"{key} requires at most 100 normalized points")
        points[key] = [list(_normalized_pair(item, key)) for item in raw]
    return {"box": list(box), **points}


def _load_guides(path: str | Path | None, filenames: list[str],
                 default_box: tuple[float, ...]) -> tuple[dict[str, dict[str, Any]], str | None]:
    if path is None:
        return {name: _guide({}, default_box=default_box) for name in filenames}, None
    source = Path(path).resolve()
    if Path(path).is_symlink():
        raise ReferenceError("Mask guides cannot be a symlink")
    document = _json(source)
    if set(document) - {"default", "images"}:
        raise ReferenceError("Mask guide JSON may contain only default and images")
    image_guides = document.get("images", {})
    if not isinstance(image_guides, dict) or set(image_guides) - set(filenames):
        raise ReferenceError("Mask guide filenames must exactly identify registered rectified photos")
    default = _guide(document.get("default", {}), default_box=default_box)
    guides = {}
    for name in filenames:
        specific = image_guides.get(name, {})
        if not isinstance(specific, dict):
            raise ReferenceError("Per-photo mask guide must be an object")
        guides[name] = _guide({**default, **specific}, default_box=default_box)
    return guides, _sha256(source)


def _grabcut(photo: Any, guide: dict[str, Any], max_side: int,
             iterations: int) -> Any:
    """Run GrabCut at bounded resolution, then nearest-neighbor upsample binary labels."""
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    width, height = photo.size
    scale = min(1.0, max_side / max(width, height))
    small_w, small_h = max(8, round(width*scale)), max(8, round(height*scale))
    rgb = np.asarray(photo.convert("RGB"))
    small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    left, top, right, bottom = guide["box"]
    x0, y0 = round(left*small_w), round(top*small_h)
    x1, y1 = round(right*small_w), round(bottom*small_h)
    x0, y0 = min(small_w-2, max(1, x0)), min(small_h-2, max(1, y0))
    x1, y1 = max(x0+1, min(small_w-1, x1)), max(y0+1, min(small_h-1, y1))
    mask = np.zeros((small_h, small_w), dtype=np.uint8)
    bg_model, fg_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask,
                    (x0, y0, x1-x0, y1-y0), bg_model, fg_model,
                    iterations, cv2.GC_INIT_WITH_RECT)
        if guide["foreground_points"] or guide["background_points"]:
            radius = max(2, round(min(small_w, small_h)*0.012))
            for key, label in (("foreground_points", cv2.GC_FGD),
                               ("background_points", cv2.GC_BGD)):
                for x, y in guide[key]:
                    cv2.circle(mask, (min(small_w-1, round(x*(small_w-1))),
                                      min(small_h-1, round(y*(small_h-1)))),
                               radius, label, thickness=-1)
            cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, None,
                        bg_model, fg_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error as error:
        raise ReferenceError("GrabCut could not segment this photograph; review its box/points") from error
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                      255, 0).astype(np.uint8)
    if not binary.any() or np.all(binary == 255):
        raise ReferenceError("GrabCut produced an empty or entirely foreground mask")
    if (small_w, small_h) != (width, height):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    return binary


def generate_rectified_masks(reconstruction: str | Path, output: str | Path, *,
                             guides: str | Path | None = None,
                             center_box: tuple[float, ...] = _DEFAULT_BOX,
                             max_side: int = 1280,
                             iterations: int = 5) -> dict[str, Any]:
    """Make immutable mask candidates, overlays and QA metrics for every photo."""
    if type(max_side) is not int or not 256 <= max_side <= 4096:
        raise ReferenceError("Mask max_side must be 256–4096")
    if type(iterations) is not int or not 1 <= iterations <= 12:
        raise ReferenceError("GrabCut iterations must be 1–12")
    default_box = _box(center_box)
    try:
        import cv2  # noqa: F401  # type: ignore[import-not-found]
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise ReferenceError("Mask generation requires pip install -e '.[masks]'") from error
    plan = plan_dense_reconstruction(reconstruction, output,
                                     DenseSettings(min_registered_images=3,
                                                   min_registered_fraction=0.25,
                                                   refine=False, texture=False),
                                     executables={})
    source = Path(plan["source_revision"])
    index = _json(source / "undistorted-reference-index.json")
    images = {item["filename"]: item for item in index["images"]}
    guide_map, guide_sha = _load_guides(guides, plan["undistorted_images"], default_box)
    target = Path(plan["output"])
    target.mkdir(parents=True, exist_ok=False)
    mask_dir, preview_dir = target / "masks", target / "review"
    mask_dir.mkdir()
    preview_dir.mkdir()
    report: dict[str, Any] = {
        "schema_version": 1, "source_revision": str(source),
        "guide_sha256": guide_sha, "max_side": max_side, "iterations": iterations,
        "status": "incomplete / manual-mask-review-required", "images": [],
        "approved_for_reconstruction": False,
    }

    def checkpoint() -> None:
        (target / "mask-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")

    checkpoint()
    for filename in plan["undistorted_images"]:
        image_path = source / "undistorted/images" / filename
        try:
            with Image.open(image_path) as original:
                original.load()
                expected = images[filename]["dimensions"]
                if list(original.size) != expected:
                    raise ReferenceError(f"Rectified photo size changed: {filename}")
                binary = _grabcut(original, guide_map[filename], max_side, iterations)
                mask = Image.fromarray(binary, mode="L")
                occupancy = round(sum(mask.histogram()[128:])/ (mask.width*mask.height), 5)
                warnings = []
                if occupancy < 0.05 or occupancy > 0.90:
                    warnings.append("Suspicious foreground fraction; check seed box and background")
                edges = (mask.crop((0, 0, mask.width, 1)),
                         mask.crop((0, mask.height-1, mask.width, mask.height)),
                         mask.crop((0, 0, 1, mask.height)),
                         mask.crop((mask.width-1, 0, mask.width, mask.height)))
                if any(edge.getbbox() is not None for edge in edges):
                    warnings.append("Foreground touches photo border; inspect cropped statue")
                mask_path = mask_dir / (filename + ".mask.png")
                mask.save(mask_path, format="PNG", optimize=True)
                original.thumbnail((900, 900))
                base = original.convert("RGB")
                cover = Image.new("RGB", base.size, (238, 58, 35))
                visible = mask.resize(base.size, Image.Resampling.NEAREST).point(
                    lambda intensity: round(intensity * 0.28))
                overlay = Image.composite(cover, base, visible)
                preview = preview_dir / (filename + ".jpg")
                overlay.save(preview, "JPEG", quality=86)
                report["images"].append({
                    "filename": filename, "source_photo_sha256": images[filename]["sha256"],
                    "mask": str(mask_path.relative_to(target)), "mask_sha256": _sha256(mask_path),
                    "preview": str(preview.relative_to(target)),
                    "foreground_fraction": occupancy,
                    "guide": guide_map[filename], "warnings": warnings,
                })
        except Exception as error:
            report.update(failed_image=filename, error=str(error))
            checkpoint()
            raise
        checkpoint()
    columns, cell_width, cell_height = 4, 288, 238
    rows = (len(report["images"]) + columns - 1) // columns
    contact = Image.new("RGB", (columns * cell_width, rows * cell_height), (28, 31, 34))
    draw = ImageDraw.Draw(contact)
    for index, item in enumerate(report["images"]):
        x = index % columns * cell_width
        y = index // columns * cell_height
        with Image.open(target / item["preview"]) as preview:
            preview.thumbnail((cell_width-8, cell_height-30))
            contact.paste(preview, (x+4, y+4))
        draw.text((x+6, y+cell_height-23), item["filename"][:42], fill=(234, 236, 238))
    contact.save(target / "mask-contact-sheet.jpg", "JPEG", quality=85)
    report["contact_sheet"] = "mask-contact-sheet.jpg"
    report["status"] = "candidate / inspect-every-mask-before-dense-reconstruction"
    checkpoint()
    return report
