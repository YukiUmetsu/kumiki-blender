"""Compare registered artwork masks against visibility-aware Blender render passes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .metrics import color, depth, shape, surface_detail
from .advanced_metrics import contour_geometry, perceptual_color, relative_depth_hint
from .schema import asset, load_manifest


def _mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f"Mask must be in full reference-image coordinates: {path}")
        alpha = np.asarray(image.getchannel("A")) if "A" in image.getbands() else None
        if alpha is not None and alpha.min() != alpha.max():
            return alpha > 127
        return np.asarray(image.convert("L")) > 127


def _warp(image: np.ndarray, transform: list[list[float]], size: tuple[int, int], *,
          nearest: bool) -> np.ndarray:
    return cv2.warpAffine(image, np.array(transform, dtype=np.float64), size,
                          flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _visible_labels(png: np.ndarray, palette: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Classify emissive ID colors by nearest calibrated hue (antialias-safe interior)."""
    if png.ndim != 3 or png.shape[2] != 4:
        raise ValueError("Capture labels must be RGBA")
    hsv = cv2.cvtColor(png[:, :, :3], cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(float)
    names = list(palette)
    if not names:
        raise ValueError("Missing semantic render palette")
    candidates = np.array([[int(round(value * 255)) for value in palette[name]] for name in names], dtype=np.uint8)
    palette_h = cv2.cvtColor(candidates.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0, :, 0].astype(float)
    distances = np.abs(hue[:, :, None] - palette_h[None, None, :])
    distances = np.minimum(distances, 180-distances)
    assignment = np.argmin(distances, axis=2)
    valid = (png[:, :, 3] > 160) & (hsv[:, :, 1] > 70) & (np.min(distances, axis=2) < 7)
    return {name: (assignment == index) & valid for index, name in enumerate(names)}


def _comparison(path: Path, source: np.ndarray, sculpt: np.ndarray,
                reference_mask: np.ndarray, sculpt_mask: np.ndarray, title: str) -> None:
    """Create matching-scale source, captured albedo and red/blue shape discrepancy panes."""
    yy, xx = np.nonzero(reference_mask | sculpt_mask)
    if not len(xx):
        return
    x0, y0 = max(0, int(xx.min())-16), max(0, int(yy.min())-16)
    x1, y1 = min(source.shape[1], int(xx.max())+17), min(source.shape[0], int(yy.max())+17)
    ref = source[y0:y1, x0:x1].copy()
    mod = sculpt[y0:y1, x0:x1].copy()
    rm = reference_mask[y0:y1, x0:x1]
    sm = sculpt_mask[y0:y1, x0:x1]
    ref[~rm] = 245
    mod[~sm] = 245
    diff = np.full_like(ref, 245)
    diff[rm & ~sm] = (240, 60, 60)
    diff[sm & ~rm] = (50, 95, 235)
    diff[sm & rm] = (95, 165, 105)
    h, w = ref.shape[:2]
    # Preserve exactly the same scale in all three panes, limit huge images only as a group.
    cell_scale = min(1.0, 580/max(w, h))
    cw, ch = max(1, round(w*cell_scale)), max(1, round(h*cell_scale))
    result = Image.new("RGB", (3*cw + 32, ch+48), "#e8ebef")
    draw = ImageDraw.Draw(result)
    for index, (label, frame) in enumerate((("SOURCE", ref), ("BLENDER ALBEDO", mod),
                                            ("RED missing / BLUE extra", diff))):
        picture = Image.fromarray(frame).resize((cw, ch), Image.Resampling.LANCZOS)
        result.paste(picture, (8+index*(cw+8), 34))
        draw.text((8+index*(cw+8), 6), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)


def evaluate_character(manifest: str | Path, capture: str | Path, output: str | Path) -> dict[str, Any]:
    """Write a new read-only assessment revision. Never change the Blender source scene."""
    manifest_path = Path(manifest).resolve()
    config = load_manifest(manifest_path)
    capture = Path(capture).resolve()
    target = Path(output)
    if target.is_symlink() or target.exists():
        raise ValueError("QA output must be a new directory")
    target = target.resolve()
    if target.is_relative_to(manifest_path.parent) or target.is_relative_to(capture):
        raise ValueError("QA output must be outside source and capture directories")
    meta_path = asset(capture, "capture.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or set(metadata.get("palette", {})) != {p["id"] for p in config["parts"]}:
        raise ValueError("Captured part names do not match manifest")
    results: list[dict[str, Any]] = []
    # Validate all input evidence before creating an output directory.
    for view in config["views"]:
        vid, size = view["id"], (view["width"], view["height"])
        view_meta = metadata["views"][vid]
        labels = np.array(Image.open(asset(capture, view_meta["labels"])).convert("RGBA"))
        flat = np.array(Image.open(asset(capture, view_meta["albedo"])).convert("RGB"))
        source_file = asset(manifest_path.parent, view["reference_image"])
        source_image = Image.open(source_file).convert("RGB")
        if labels.shape[:2] != (size[1], size[0]) or flat.shape[:2] != labels.shape[:2]:
            raise ValueError(f"Blender render dimensions do not match manifest view {vid}")
        labels_by_part = _visible_labels(labels, metadata["palette"])
        verified = view["alignment"]["verified"]
        transform = view["alignment"]["source_to_render"]
        source_rgb = np.array(source_image)
        aligned_rgb = _warp(source_rgb, transform, size, nearest=False) if verified else None
        depth_data = np.load(asset(capture, view_meta["depth_npz"]), allow_pickle=False)
        candidate_depth = depth_data["depth_m"]
        ref_hint = None
        if view.get("reference_depth_hint") and verified:
            hint_npz = np.load(asset(manifest_path.parent, view["reference_depth_hint"]["path"]), allow_pickle=False)
            if "depth_relative" not in hint_npz:
                raise ValueError("Relative depth hint NPZ must contain depth_relative")
            raw_hint = hint_npz["depth_relative"]
            if raw_hint.shape != source_rgb.shape[:2]:
                raise ValueError("Relative depth hint must match source illustration dimensions")
            ref_hint = _warp(raw_hint.astype(np.float32), transform, size, nearest=False)
        ref_depth = None
        if view.get("reference_depth") and verified:
            source_depth = np.load(asset(manifest_path.parent, view["reference_depth"]["path"]),
                                   allow_pickle=False)
            source_depth = source_depth["depth_m"] if isinstance(source_depth, np.lib.npyio.NpzFile) else source_depth
            if source_depth.shape != source_rgb.shape[:2]:
                raise ValueError("Reference depth array must match its source illustration dimensions")
            ref_depth = _warp(source_depth.astype(np.float32), transform, size, nearest=True)
        for part in config["parts"]:
            pid = part["id"]
            if vid not in part["reference_masks"]:
                results.append({"view": vid, "part": pid, "category": part["category"],
                                "status": "no-reference-mask"})
                continue
            source_mask = _mask(asset(manifest_path.parent, part["reference_masks"][vid]),
                                source_image.size)
            mod_mask = labels_by_part[pid]
            pose = vid not in part.get("unmatched_pose_views", [])
            score = verified and pose
            ref_mask = _warp(source_mask.astype(np.uint8), transform, size, nearest=True).astype(bool) if score else source_mask
            result: dict[str, Any] = {"view": vid, "part": pid, "category": part["category"],
                "pose_matched": pose, "shape": shape(ref_mask, mod_mask, registered=score)}
            result["contour_geometry"] = contour_geometry(ref_mask, mod_mask)
            # Palette ΔE00 is intrinsic; per-pixel ΔE00 requires trusted registration.
            result["perceptual_color"] = perceptual_color(
                aligned_rgb if score else source_rgb, flat, ref_mask, mod_mask, registered=score)
            # Color distributions require no pixel correspondence; shape does.
            result["color"] = color(source_rgb, flat, source_mask, mod_mask)
            result["detail_proxy"] = {"source": surface_detail(source_rgb, source_mask),
                                       "sculpt": surface_detail(flat, mod_mask)}
            result["depth"] = depth(candidate_depth, mod_mask, reference_depth=ref_depth if score else None,
                                     reference_mask=ref_mask if score else None, calibrated=score and ref_depth is not None)
            if ref_hint is not None and score:
                result["depth_hint"] = relative_depth_hint(ref_hint, candidate_depth, ref_mask, mod_mask,
                    verified_alignment=True, near_is_small=view["reference_depth_hint"]["near_is_small"])
            result["camera_depth_extent_m"] = view_meta.get("part_depth_extent_m", {}).get(pid)
            if score:
                result["comparison"] = f"comparisons/{vid}/{pid}.png"
                result["_comparison_data"] = (aligned_rgb, flat, ref_mask, mod_mask)
            results.append(result)
    target.mkdir(parents=True, exist_ok=False)
    for result in results:
        if "_comparison_data" in result:
            _comparison(target / result["comparison"], *result.pop("_comparison_data"),
                        f"{result['view']} / {result['part']}")
    report = {"schema_version": 1, "status": "artistic-review-required", "views": [v["id"] for v in config["views"]],
        "parts": results,
        "by_category": {category: [f"{row['view']}:{row['part']}" for row in results if row["category"] == category]
                        for category in ("anatomy", "costume", "accessory", "prop")},
        "capture_assignments": metadata.get("assignments", {}),
        "limitations": ["No global likeness score or automatic artistic approval.",
            "Shape and placement use explicit verified view alignment; mismatched poses are intrinsic-only.",
            "Unlit color and 2D detail are diagnostics, not texture/material fidelity certification.",
            "Single artwork cannot establish actual depth; numeric depth error requires measured same-camera metric ground truth.",
            "AI relative-depth hints report rank ordering only and are not validated source depth."]}
    (target / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
