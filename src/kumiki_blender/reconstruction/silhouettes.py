"""Calibrated, mask-only comparison of reference and rendered statue silhouettes.

This scores geometry placement in known cameras, never image color, artistic
likeness, heritage authenticity, or appearance of unseen parts of a statue.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .blender_reference import load_camera_plan
from .references import ReferenceError


def _load_review_regions(path: str | Path | None, camera_ids: set[str]) -> dict[str, list[dict]]:
    """Validate hand-annotated image-local regions, never infer anatomy from sparse points."""
    if path is None:
        return {}
    try:
        source = Path(path)
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReferenceError(f"Cannot read silhouette review regions: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ReferenceError("Silhouette review regions must have schema_version 1")
    views = document.get("views")
    if not isinstance(views, dict) or not views or not set(views).issubset(camera_ids):
        raise ReferenceError("Review regions must reference known camera image IDs")
    result = {}
    for image_id, regions in views.items():
        if not isinstance(regions, list) or not regions:
            raise ReferenceError(f"Review regions for {image_id} must be a nonempty list")
        seen = set()
        result[image_id] = []
        for region in regions:
            if not isinstance(region, dict):
                raise ReferenceError("Each review region must be an object")
            name, box = region.get("label"), region.get("box_norm")
            if not isinstance(name, str) or not (1 <= len(name) <= 80) or name in seen:
                raise ReferenceError(f"Review region labels for {image_id} must be distinct nonempty strings")
            seen.add(name)
            if (not isinstance(box, list) or len(box) != 4 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in box) or
                not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                raise ReferenceError(f"Invalid normalized review region for {image_id}: {name}")
            result[image_id].append({"label": name, "box_norm": box})
    return result


def compare_silhouettes(plan_file: str | Path, render_directory: str | Path,
                        output: str | Path, *, min_iou: float = 0.8,
                        regions_file: str | Path | None = None) -> dict[str, Any]:
    """Write original mask-only difference overlays and an auditable IoU report."""
    if type(min_iou) not in (float, int) or not math.isfinite(min_iou) or not 0 < min_iou <= 1:
        raise ReferenceError("min_iou must be a finite fraction in (0, 1]")
    try:
        from PIL import Image, ImageChops
    except ImportError as error:
        raise ReferenceError("Silhouette QA requires: pip install 'kumiki-blender[qa]'") from error
    plan = load_camera_plan(plan_file)
    review_regions = _load_review_regions(regions_file,
                                          {camera["image_id"] for camera in plan["cameras"]})
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Silhouette QA output already exists: {target}")
    render_root = Path(render_directory).resolve()
    if not render_root.is_dir() or target.is_relative_to(render_root):
        raise ReferenceError("Silhouette QA output must be outside the existing render directory")
    observations: list[tuple[dict[str, Any], Any | None]] = []
    for camera in plan["cameras"]:
        result: dict[str, Any] = {"image_id": camera["image_id"]}
        regions = review_regions.get(camera["image_id"], [])
        if regions:
            result["regions"] = []
        mask_path = camera["source_mask_path"]
        render_path = render_root / (camera["image_id"] + ".png")
        if mask_path is None:
            result.update({"status": "unscored", "reason": "No manually checked subject mask"})
            observations.append((result, None))
            continue
        if not render_path.is_file() or render_path.is_symlink():
            result.update({"status": "unscored", "reason": "No matched-camera clay RGBA render"})
            observations.append((result, None))
            continue
        try:
            with Image.open(mask_path) as ref, Image.open(render_path) as scene:
                size = (camera["intrinsics"]["width"], camera["intrinsics"]["height"])
                if ref.size != size or scene.size != size:
                    raise ReferenceError(f"Silhouette dimensions differ from camera for {camera['image_id']}")
                if scene.mode != "RGBA":
                    raise ReferenceError(f"Clay render must have transparent RGBA for {camera['image_id']}")
                reference = ref.convert("L").point(lambda v: 255 if v >= 128 else 0)
                candidate = scene.getchannel("A").point(lambda v: 255 if v >= 128 else 0)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            if isinstance(error, ReferenceError):
                raise
            raise ReferenceError(f"Could not read silhouette for {camera['image_id']}: {error}") from error
        reference_count = reference.histogram()[255]
        candidate_count = candidate.histogram()[255]
        total = size[0]*size[1]
        foreground = reference_count/total
        if not 0.02 <= foreground <= 0.95:
            result.update({"status": "unscored", "reason": "Reference mask is nearly empty or all white",
                           "reference_foreground_fraction": round(foreground, 5)})
            observations.append((result, None))
            continue
        common = ImageChops.multiply(reference, candidate)
        intersection = common.histogram()[255]
        union = reference_count + candidate_count - intersection
        iou = intersection / union if union else 0.0
        result.update({
            "status": "threshold-met / review-required" if iou >= min_iou else "review-required",
            "iou": round(iou, 6),
            "reference_foreground_fraction": round(foreground, 5),
            "render_foreground_fraction": round(candidate_count/total, 5),
            "precision": round(intersection/candidate_count, 6) if candidate_count else 0,
            "recall": round(intersection/reference_count, 6),
            "mask_overlay": "overlay-" + camera["image_id"] + ".png",
        })
        # Red = mesh-only, blue = reference-only, gray = intersection; no source pixels.
        red = ImageChops.subtract(candidate, reference)
        blue = ImageChops.subtract(reference, candidate)
        overlay = Image.merge("RGB", (ImageChops.lighter(red, common.point(lambda v: v//2)),
                                      common.point(lambda v: v//2),
                                      ImageChops.lighter(blue, common.point(lambda v: v//2))))
        for region_index, region in enumerate(regions, 1):
            x0, y0, x1, y1 = region["box_norm"]
            box = (math.floor(x0 * size[0]), math.floor(y0 * size[1]),
                   math.ceil(x1 * size[0]), math.ceil(y1 * size[1]))
            ref_part, render_part = reference.crop(box), candidate.crop(box)
            reference_pixels = ref_part.histogram()[255]
            rendered_pixels = render_part.histogram()[255]
            item: dict[str, Any] = {"label": region["label"],
                                    "box_norm": region["box_norm"],
                                    "box_px": list(box),
                                    "reference_pixels": reference_pixels,
                                    "rendered_pixels": rendered_pixels}
            if reference_pixels == 0:
                item.update({"status": "unscored", "reason": "Reference mask has no foreground in this region"})
            else:
                intersection = ImageChops.multiply(ref_part, render_part).histogram()[255]
                iou_region = intersection / (reference_pixels + rendered_pixels - intersection)
                item.update({"status": ("threshold-met / review-required" if iou_region >= min_iou
                                         else "review-required"),
                             "iou": round(iou_region, 6),
                             "missing_reference_fraction": round((reference_pixels-intersection)/reference_pixels, 6),
                             "extra_render_fraction": round((rendered_pixels-intersection)/rendered_pixels, 6)
                             if rendered_pixels else 0})
            item["overlay"] = f"region-overlay-{camera['image_id']}-{region_index:02}.png"
            item["overlay_image"] = overlay.crop(box)
            result["regions"].append(item)
        observations.append((result, overlay))

    scored = [item for item, _ in observations if "iou" in item]
    expected_regions = sum(len(items) for items in review_regions.values())
    scored_regions = [region for item, _ in observations for region in item.get("regions", [])
                      if "iou" in region]
    regions_pass = (not review_regions or
                    (len(scored_regions) == expected_regions and
                     all(region["iou"] >= min_iou for region in scored_regions)))
    target.mkdir(parents=True, exist_ok=False)
    for item, overlay in observations:
        if overlay is not None:
            overlay.save(target / item["mask_overlay"], format="PNG")
        for region in item.get("regions", []):
            region.pop("overlay_image").save(target / region["overlay"], format="PNG")
    result = {
        "schema_version": 1, "asset_id": plan["asset_id"],
        "status": ("mask-threshold-candidate / artistic-review-required"
                   if len(scored) >= 3 and len(scored) == len(plan["cameras"])
                   and all(item["iou"] >= min_iou for item in scored) and regions_pass
                   else "review-required"),
        "registered_cameras": len(plan["cameras"]), "scored_cameras": len(scored),
        "review_regions_requested": expected_regions,
        "review_regions_scored": len(scored_regions),
        "minimum_iou": min_iou,
        "views": [item for item, _ in observations],
        "historical_fidelity_verified": False,
        "limitations": ("Mask IoU only tests projection silhouette placement. Hand-annotated regions can expose "
                        "missing paws, face or rear-body outlines hidden by a good global score, but they do "
                        "not establish anatomical fidelity. Reference masks must delineate the actual statue. "
                        "Hidden surfaces, sculptural details and photographic color remain unverified."),
    }
    (target / "silhouette-qa.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
