"""Conservative fusion of independent UV-reprojected photographic views.

This estimates a less shadow-biased *candidate* color, not ground-truth
reflectance. Source photographs must already be registered and projected into
the same atlas UV coordinates; camera-space JPGs are never silently treated
as UV-aligned images. Per-view confidence masks explicitly gate occlusions.
"""

from __future__ import annotations

import math
from typing import Any


def _linear(rgb: Any, np: Any) -> Any:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.where(values <= .04045, values / 12.92,
                    ((values + .055) / 1.055) ** 2.4)


def _srgb(linear: Any, np: Any) -> Any:
    result = np.where(linear <= .0031308, linear * 12.92,
                      1.055 * np.maximum(linear, 0) ** (1 / 2.4) - .055)
    return np.uint8(np.clip(result * 255 + .5, 0, 255))


def fuse_uv_photo_observations(original: Any, regions: tuple[Any, ...],
                               observations: tuple[Any, ...], *,
                               strength: float, max_ev: float,
                               protect_black: int = 12) -> tuple[Any, dict[str, Any]]:
    """Estimate albedo by confidence-weighted median of >=3 UV-aligned views.

    Normalize each observation *per material* with bounded global linear RGB
    gains before taking a per-pixel luminance median. The original atlas is
    left unchanged without three independent valid observations. Cropped
    processing bounds peak temporary tensor memory at high atlas resolutions.
    """
    from PIL import Image
    import numpy as np

    if not 0 <= strength <= 1 or not 0 <= max_ev <= 2:
        raise ValueError("Multi-view fusion strength/max_ev out of range")
    rgb = original.convert("RGB")
    raw_photos = [entry.photo.open_verified() for entry in observations]
    photos = [image.convert("RGB") for image in raw_photos]
    photo_alpha = [image.getchannel("A") if "A" in image.getbands() else None
                   for image in raw_photos]
    confidence = [entry.confidence.open_verified() for entry in observations]
    masks = [region.mask.open_verified() for region in regions]
    if len(photos) < 3 or any(photo.size != rgb.size for photo in photos):
        raise ValueError("Multi-view observations need at least three matching UV atlas photos")
    alpha = (original.convert("RGBA").getchannel("A") if
             "A" in original.getbands() or "transparency" in original.info else None)

    # Global per-material illumination/exposure normalization is evaluated
    # from small, overlapping thumbnails rather than one scalar across stone,
    # bronze, foliage, and unrelated UV islands.
    size = (max(1, round(rgb.width * min(1, 512 / max(rgb.size)))),
            max(1, round(rgb.height * min(1, 512 / max(rgb.size)))))
    base_sample = np.asarray(rgb.resize(size, Image.Resampling.BILINEAR))
    base_lin = _linear(base_sample, np)
    base_valid = np.max(base_sample, axis=2) > max(16, protect_black)
    if alpha is not None:
        base_valid &= np.asarray(alpha.resize(size, Image.Resampling.NEAREST)) >= 128
    view_sample = [np.asarray(photo.resize(size, Image.Resampling.BILINEAR))
                   for photo in photos]
    view_lin = [_linear(sample, np) for sample in view_sample]
    view_valid = []
    for mask, sample, view_alpha in zip(confidence, view_sample, photo_alpha):
        valid = ((np.asarray(mask.resize(size, Image.Resampling.NEAREST)) >= 128)
                 & (np.max(sample, axis=2) > 16))
        if view_alpha is not None:
            valid &= np.asarray(view_alpha.resize(size, Image.Resampling.NEAREST)) >= 128
        view_valid.append(valid)
    gains: list[list[Any]] = [[] for _ in regions]
    diagnostics = []
    max_factor = 2 ** max_ev
    for region_index, region in enumerate(regions):
        region_sample = np.asarray(masks[region_index].resize(
            size, Image.Resampling.NEAREST)) >= 128
        page = []
        for view_index, observation in enumerate(observations):
            valid = region_sample & base_valid & view_valid[view_index]
            count = int(np.count_nonzero(valid))
            if count >= 64:
                # Log ratios reduce the influence of bright outliers. Channel
                # normalization helps correct colored view illumination too.
                ratios = np.log2((base_lin[valid] + .01) /
                                 (view_lin[view_index][valid] + .01))
                offset = np.clip(np.median(ratios, axis=0), -max_ev, max_ev)
                gain = np.power(2.0, offset).astype(np.float32)
            else:
                gain = np.ones(3, dtype=np.float32)
            gains[region_index].append(gain)
            page.append({"view_id": observation.id, "calibration_overlap_pixels": count,
                         "linear_rgb_gain": [round(float(value), 5) for value in gain],
                         "calibrated": count >= 64})
        diagnostics.append({"region": region.id, "views": page})

    output = Image.new("RGB", rgb.size)
    changed_pixels = covered_pixels = 0
    for top in range(0, rgb.height, 128):
        box = (0, top, rgb.width, min(rgb.height, top + 128))
        base_rgb = np.asarray(rgb.crop(box))
        source = _linear(base_rgb, np)
        current = source.copy()
        protect = np.max(base_rgb, axis=2) > max(16, protect_black)
        if alpha is not None:
            protect &= np.asarray(alpha.crop(box)) >= 128
        stacked = np.stack([_linear(np.asarray(image.crop(box)), np)
                            for image in photos], axis=0)
        weights = np.stack([np.asarray(image.crop(box), dtype=np.float32) / 255
                            for image in confidence], axis=0)
        for view_index, view_alpha in enumerate(photo_alpha):
            if view_alpha is not None:
                weights[view_index] *= (np.asarray(view_alpha.crop(box),
                                                   dtype=np.float32) / 255)
        weights *= np.max(stacked, axis=3) > .005
        for region_index, mask in enumerate(masks):
            region_weight = np.asarray(mask.crop(box), dtype=np.float32) / 255
            effective = weights * (region_weight[None] >= .5) * protect[None]
            eligible = np.sum(effective >= .5, axis=0) >= 3
            if not np.any(eligible):
                continue
            aligned = stacked * np.stack(gains[region_index], axis=0)[:, None, None, :]
            luminance = (aligned[..., 0] * .2126 + aligned[..., 1] * .7152
                         + aligned[..., 2] * .0722)
            order = np.argsort(luminance, axis=0)
            ordered_weights = np.take_along_axis(effective, order, axis=0)
            cumulative = np.cumsum(ordered_weights, axis=0)
            midpoint = np.sum(ordered_weights, axis=0) * .5
            selected = np.argmax(cumulative >= midpoint[None], axis=0)
            chosen = np.take_along_axis(order, selected[None], axis=0)[0]
            recovered = np.take_along_axis(
                aligned, np.broadcast_to(chosen[None, ..., None],
                                         (1, *chosen.shape, 3)), axis=0)[0]
            recovered = np.clip(recovered, source / max_factor, source * max_factor)
            blend = (region_weight * strength * eligible)[..., None]
            current = current * (1 - blend) + recovered * blend
            covered_pixels += int(np.count_nonzero(eligible))
        new_pixels = _srgb(current, np)
        changed_pixels += int(np.count_nonzero(
            np.any(np.abs(new_pixels.astype(np.int16) - base_rgb.astype(np.int16)) >= 2,
                   axis=2)))
        output.paste(Image.fromarray(new_pixels, "RGB"), (0, top))
    if alpha is not None:
        output.putalpha(alpha)
    total = rgb.width * rgb.height
    return output, {
        "method": "UV-registered-photo-weighted-median / not-calibrated-reflectance",
        "view_count": len(observations), "strength": strength,
        "max_correction_ev": max_ev,
        "independently_covered_fraction": round(covered_pixels / total, 6),
        "changed_fraction": round(changed_pixels / total, 6),
        "per_material_exposure": diagnostics,
        "input_sha256": [{"id": entry.id, "photo": entry.photo.sha256,
                          "confidence": entry.confidence.sha256} for entry in observations],
        "limitations": ("Only pixels visible in >=3 independently masked UV-reprojected photographs "
                        "are fused. Shared shadows, occlusions not removed by confidence masks, "
                        "and unknown BRDF/illumination remain unresolved."),
    }
