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


def compare_silhouettes(plan_file: str | Path, render_directory: str | Path,
                        output: str | Path, *, min_iou: float = 0.8) -> dict[str, Any]:
    """Write original mask-only difference overlays and an auditable IoU report."""
    if type(min_iou) not in (float, int) or not math.isfinite(min_iou) or not 0 < min_iou <= 1:
        raise ReferenceError("min_iou must be a finite fraction in (0, 1]")
    try:
        from PIL import Image, ImageChops
    except ImportError as error:
        raise ReferenceError("Silhouette QA requires: pip install 'kumiki-blender[qa]'") from error
    plan = load_camera_plan(plan_file)
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Silhouette QA output already exists: {target}")
    render_root = Path(render_directory).resolve()
    if not render_root.is_dir() or target.is_relative_to(render_root):
        raise ReferenceError("Silhouette QA output must be outside the existing render directory")
    observations: list[tuple[dict[str, Any], Any | None]] = []
    for camera in plan["cameras"]:
        result: dict[str, Any] = {"image_id": camera["image_id"]}
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
        observations.append((result, overlay))

    scored = [item for item, _ in observations if "iou" in item]
    target.mkdir(parents=True, exist_ok=False)
    for item, overlay in observations:
        if overlay is not None:
            overlay.save(target / item["mask_overlay"], format="PNG")
    result = {
        "schema_version": 1, "asset_id": plan["asset_id"],
        "status": ("mask-threshold-candidate / artistic-review-required"
                   if len(scored) >= 3 and len(scored) == len(plan["cameras"])
                   and all(item["iou"] >= min_iou for item in scored) else "review-required"),
        "registered_cameras": len(plan["cameras"]), "scored_cameras": len(scored),
        "minimum_iou": min_iou,
        "views": [item for item, _ in observations],
        "historical_fidelity_verified": False,
        "limitations": "Mask IoU only tests projection silhouette placement. Masks must delineate the statue, not simply select a background. It does not measure face anatomy, drapery, hidden geometry or photographic color.",
    }
    (target / "silhouette-qa.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
