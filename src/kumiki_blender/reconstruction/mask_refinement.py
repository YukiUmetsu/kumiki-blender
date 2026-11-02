"""Non-destructive foreground-mask cleanup for rectified multi-view statues.

Automated operations only remove tiny connected islands or fill bounded holes;
normalized per-view polygon edits supply the missing artist-directed control.
No original photographs or prior mask revisions are overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dense import (DenseSettings, _check_regular, _json, _sha256,
                    plan_dense_reconstruction)
from .masking import _normalized_pair
from .references import ReferenceError


def _polygon_list(value: Any, label: str) -> list[list[list[float]]]:
    if not isinstance(value, list) or len(value) > 32:
        raise ReferenceError(f"{label} requires at most 32 polygons")
    result = []
    for polygon in value:
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= 128:
            raise ReferenceError(f"{label} polygons require 3–128 normalized points")
        points = [list(_normalized_pair(point, label)) for point in polygon]
        if len({tuple(point) for point in points}) < 3:
            raise ReferenceError(f"{label} polygons need three distinct vertices")
        result.append(points)
    return result


def _load_edits(path: str | Path | None, filenames: list[str]) -> tuple[dict[str, Any], str | None]:
    if path is None:
        return {}, None
    source = Path(path).resolve()
    if Path(path).is_symlink():
        raise ReferenceError("Mask edits cannot be a symlink")
    document = _json(source)
    if set(document) != {"images"} or not isinstance(document["images"], dict):
        raise ReferenceError("Mask edits must contain an images mapping")
    if set(document["images"]) - set(filenames):
        raise ReferenceError("Mask edits contain unknown rectified photo filenames")
    edits = {}
    for filename, item in document["images"].items():
        if not isinstance(item, dict) or set(item) - {"include_polygons", "exclude_polygons"}:
            raise ReferenceError("Per-photo mask edits allow only include/exclude_polygons")
        edits[filename] = {key: _polygon_list(item.get(key, []), key)
                           for key in ("include_polygons", "exclude_polygons")}
    return edits, _sha256(source)


def _clean_mask(binary: Any, *, min_island_pixels: int, max_hole_pixels: int) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    result = binary.copy()
    if min_island_pixels:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < min_island_pixels:
                result[labels == label] = 0
    if max_hole_pixels:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - result, connectivity=8)
        height, width = result.shape
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if (area <= max_hole_pixels and x > 0 and y > 0
                    and x+w < width and y+h < height):
                result[labels == label] = 255
    return np.where(result > 0, 255, 0).astype(np.uint8)


def _paint_polygons(binary: Any, edits: dict[str, Any]) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    result = binary.copy()
    height, width = result.shape
    # Exclusions happen last, so artists can explicitly cut away the base or
    # surrounding ground even when the original segmenter included them.
    for key, color in (("include_polygons", 255), ("exclude_polygons", 0)):
        for polygon in edits.get(key, []):
            points = np.array([[round(x*(width-1)), round(y*(height-1))]
                               for x, y in polygon], dtype=np.int32)
            cv2.fillPoly(result, [points], color)
    return result


def _inset_mask(binary: Any, pixels: int) -> Any:
    """Optional conservative contour inset to avoid reconstructing edge halos."""
    if not pixels:
        return binary
    import cv2  # type: ignore[import-not-found]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2*pixels+1, 2*pixels+1))
    return cv2.erode(binary, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT,
                     borderValue=0)


def _color_refine_boundary(binary: Any, photo: Any, *, band_pixels: int,
                           max_side: int, iterations: int,
                           trim_only: bool = False) -> Any:
    """Reclassify only a narrow original-mask boundary band via color GrabCut.

    Hard foreground/background interiors never change. This is an optional
    artist-review *candidate*: textured walls and similarly colored surfaces
    can still fool GrabCut, especially around thin fingers and hair.
    """
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    height, width = binary.shape
    if photo.size != (width, height):
        raise ReferenceError("Color boundary refinement requires an exactly rectified photo/mask pair")
    scale = min(1.0, max_side / max(width, height))
    small_w, small_h = max(8, round(width * scale)), max(8, round(height * scale))
    small_binary = cv2.resize(binary, (small_w, small_h),
                              interpolation=cv2.INTER_NEAREST)
    radius = max(1, round(band_pixels * scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*radius+1, 2*radius+1))
    fixed_fg = cv2.erode(small_binary, kernel) > 0
    fixed_bg = cv2.dilate(small_binary, kernel) == 0
    if not np.any(fixed_fg) or not np.any(fixed_bg):
        raise ReferenceError("Color boundary refinement needs foreground and background seed interiors; reduce band")
    labels = np.where(small_binary > 0, cv2.GC_PR_FGD,
                      cv2.GC_PR_BGD).astype(np.uint8)
    labels[fixed_fg] = cv2.GC_FGD
    labels[fixed_bg] = cv2.GC_BGD
    rgb = np.asarray(photo.convert("RGB"))
    small_photo = cv2.resize(rgb, (small_w, small_h),
                             interpolation=cv2.INTER_AREA)
    bg_model, fg_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(small_photo, cv2.COLOR_RGB2BGR), labels,
                    None, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error as error:
        raise ReferenceError("Color boundary GrabCut could not refine mask; try a smaller band") from error
    result_small = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD),
                            255, 0).astype(np.uint8)
    # Interpolate before binary thresholding to avoid obvious block-shaped
    # halos when the photo is much larger than the CPU GrabCut working grid.
    resized = cv2.resize(result_small, (width, height),
                         interpolation=cv2.INTER_LINEAR)
    proposal = np.where(resized >= 128, 255, 0).astype(np.uint8)
    full_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2*band_pixels+1, 2*band_pixels+1))
    hard_fg = cv2.erode(binary, full_kernel) > 0
    hard_bg = cv2.dilate(binary, full_kernel) == 0
    proposal[hard_fg] = 255
    proposal[hard_bg] = 0
    if trim_only:
        # In cluttered outdoor scenes color similarity can make GrabCut grow
        # a semantic silhouette into architecture, pavement or foliage.
        # Opt-in conservative mode only removes candidate background pixels.
        proposal = np.minimum(proposal, binary)
    return proposal


def refine_rectified_masks(reconstruction: str | Path, mask_revision: str | Path,
                           output: str | Path, *, edits: str | Path | None = None,
                           min_island_pixels: int = 0,
                           max_hole_pixels: int = 0,
                           edge_inset_pixels: int = 0,
                           color_band_pixels: int = 0,
                           color_max_side: int = 1280,
                           color_iterations: int = 3,
                           color_trim_only: bool = False) -> dict[str, Any]:
    """Create a new hash-pinned mask revision and before/after QA contact sheet."""
    for key, value in (("min_island_pixels", min_island_pixels),
                       ("max_hole_pixels", max_hole_pixels)):
        if type(value) is not int or not 0 <= value <= 100_000:
            raise ReferenceError(f"{key} must be an integer from 0 to 100000")
    if type(edge_inset_pixels) is not int or not 0 <= edge_inset_pixels <= 8:
        raise ReferenceError("edge_inset_pixels must be an integer from 0 to 8")
    if type(color_band_pixels) is not int or not 0 <= color_band_pixels <= 32:
        raise ReferenceError("color_band_pixels must be an integer from 0 to 32")
    if type(color_max_side) is not int or not 256 <= color_max_side <= 4096:
        raise ReferenceError("color_max_side must be an integer from 256 to 4096")
    if type(color_iterations) is not int or not 1 <= color_iterations <= 8:
        raise ReferenceError("color_iterations must be an integer from 1 to 8")
    if type(color_trim_only) is not bool:
        raise ReferenceError("color_trim_only must be a boolean")
    if color_trim_only and not color_band_pixels:
        raise ReferenceError("color_trim_only needs positive color_band_pixels")
    if edits is None and not (min_island_pixels or max_hole_pixels or edge_inset_pixels
                              or color_band_pixels):
        raise ReferenceError("Mask refinement needs edits or a positive cleanup threshold")
    try:
        import cv2  # noqa: F401  # type: ignore[import-not-found]
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise ReferenceError("Mask refinement requires pip install -e '.[masks]'") from error

    source = Path(mask_revision).resolve()
    target = Path(output).resolve()
    sparse = Path(reconstruction).resolve()
    if (Path(mask_revision).is_symlink() or not source.is_dir()
            or Path(output).is_symlink() or target.exists()
            or target == source or target.is_relative_to(source)
            or target.is_relative_to(sparse)):
        raise ReferenceError("Refinement requires an existing mask revision and unused separate output")
    plan = plan_dense_reconstruction(reconstruction, output,
                                     DenseSettings(min_registered_images=3,
                                                   min_registered_fraction=0.25,
                                                   refine=False, texture=False),
                                     executables={})
    previous = _json(source / "mask-report.json")
    if (not str(previous.get("status", "")).startswith("candidate / ")
            or previous.get("source_revision") != str(sparse)
            or not isinstance(previous.get("images"), list)):
        raise ReferenceError("Prior mask revision is incomplete or for another reconstruction")
    entries = {}
    for entry in previous["images"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("filename"), str)
                or entry["filename"] in entries):
            raise ReferenceError("Prior mask report contains invalid or duplicate image")
        entries[entry["filename"]] = entry
    if set(entries) != set(plan["undistorted_images"]):
        raise ReferenceError("Prior masks do not match current rectified photo coverage")
    edits_map, edits_sha = _load_edits(edits, plan["undistorted_images"])
    index = _json(sparse / "undistorted-reference-index.json")
    images = {item["filename"]: item for item in index["images"]}

    # Verify every source mask before writing anything; reject changed masks
    # even if a user edits the report to point to a different directory.
    for name, entry in entries.items():
        relative = "masks/" + name + ".mask.png"
        path = source / relative
        _check_regular(path, source)
        if (entry.get("mask") != relative or entry.get("mask_sha256") != _sha256(path)
                or entry.get("source_photo_sha256") != images[name]["sha256"]):
            raise ReferenceError(f"Prior mask/photo provenance changed: {name}")
        with Image.open(path) as image:
            image.load()
            if (image.format != "PNG" or image.mode not in ("L", "1")
                    or list(image.size) != images[name]["dimensions"]
                    or {v for _, v in image.convert("L").getcolors(maxcolors=3) or []} != {0, 255}):
                raise ReferenceError(f"Prior mask is not full-size binary PNG: {name}")

    target.mkdir(parents=True, exist_ok=False)
    (target / "masks").mkdir()
    (target / "review").mkdir()
    report = {
        "schema_version": 1, "source_revision": str(sparse),
        "source_mask_revision": str(source),
        "source_mask_report_sha256": _sha256(source / "mask-report.json"),
        "edits_sha256": edits_sha,
        "min_island_pixels": min_island_pixels,
        "max_hole_pixels": max_hole_pixels,
        "edge_inset_pixels": edge_inset_pixels,
        "color_band_pixels": color_band_pixels,
        "color_max_side": color_max_side,
        "color_iterations": color_iterations,
        "color_trim_only": color_trim_only,
        "approved_for_reconstruction": False,
        "status": "incomplete / manual-mask-review-required",
        "images": [],
    }

    def checkpoint() -> None:
        (target / "mask-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")

    checkpoint()
    for name in plan["undistorted_images"]:
        with Image.open(source / ("masks/" + name + ".mask.png")) as original:
            original.load()
            before = np.asarray(original.convert("L"))
            refined = _clean_mask(before, min_island_pixels=min_island_pixels,
                                  max_hole_pixels=max_hole_pixels)
            color_changed = 0
            if color_band_pixels:
                with Image.open(sparse / "undistorted/images" / name) as photo:
                    photo.load()
                    color_refined = _color_refine_boundary(
                        refined, photo, band_pixels=color_band_pixels,
                        max_side=color_max_side, iterations=color_iterations,
                        trim_only=color_trim_only)
                color_changed = int(np.count_nonzero(refined != color_refined))
                refined = color_refined
            refined = _inset_mask(refined, edge_inset_pixels)
            # Inclusions happen after the inset, so a thin hair/arm can be
            # selectively restored rather than permanently eroded.
            refined = _paint_polygons(refined, edits_map.get(name, {}))
            if not refined.any() or np.all(refined == 255):
                report.update(failed_image=name, error="Refinement erased subject/background")
                checkpoint()
                raise ReferenceError("Refinement produced an all-white or all-black mask")
            mask_file = target / "masks" / (name + ".mask.png")
            Image.fromarray(refined, mode="L").save(mask_file, "PNG", optimize=True)
            before_pixels = before > 0
            after_pixels = refined > 0
            added = int(np.count_nonzero(after_pixels & ~before_pixels))
            removed = int(np.count_nonzero(before_pixels & ~after_pixels))
            warnings = []
            area = float(refined.size)
            fraction = round(float(np.count_nonzero(after_pixels)) / area, 5)
            if fraction < 0.05 or fraction > 0.9:
                warnings.append("Suspicious subject coverage after refinement")
            if (added+removed)/area > 0.20:
                warnings.append("Large mask edit; inspect contour and missing anatomy")
            if color_band_pixels:
                warnings.append("Color-guided boundary candidate: inspect thin anatomy and similar-colored backgrounds")
            with Image.open(sparse / "undistorted/images" / name) as photo:
                preview = photo.convert("RGB")
                preview.thumbnail((700, 700))
                thumbnail_mask = Image.fromarray(refined, mode="L").resize(
                    preview.size, Image.Resampling.NEAREST)
                tint = Image.new("RGB", preview.size, (240, 64, 42))
                overlay = Image.composite(tint, preview, thumbnail_mask.point(lambda x: round(x*.3)))
                preview_file = target / "review" / (name + ".jpg")
                overlay.save(preview_file, "JPEG", quality=84)
            report["images"].append({
                "filename": name, "source_photo_sha256": images[name]["sha256"],
                "source_mask_sha256": entries[name]["mask_sha256"],
                "mask": str(mask_file.relative_to(target)), "mask_sha256": _sha256(mask_file),
                "preview": str(preview_file.relative_to(target)),
                "foreground_fraction": fraction, "pixels_added": added,
                "pixels_removed": removed, "color_boundary_pixels_changed": color_changed,
                "warnings": warnings,
            })
        checkpoint()
    columns, width, height = 4, 288, 238
    contact = Image.new("RGB", (columns*width,
                                 ((len(report["images"])+columns-1)//columns)*height),
                        (28, 31, 34))
    painter = ImageDraw.Draw(contact)
    for index, item in enumerate(report["images"]):
        x, y = index % columns * width, index // columns * height
        with Image.open(target / item["preview"]) as preview:
            preview.thumbnail((width-8, height-30))
            contact.paste(preview, (x+4, y+4))
        painter.text((x+6, y+height-23), item["filename"][:42], fill=(234, 236, 238))
    contact.save(target / "mask-contact-sheet.jpg", "JPEG", quality=85)
    report["contact_sheet"] = "mask-contact-sheet.jpg"
    report["status"] = "candidate / refined-mask-review-required"
    checkpoint()
    return report
