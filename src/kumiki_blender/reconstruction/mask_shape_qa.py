"""Bounded silhouette diagnostics independent of sparse COLMAP track votes.

Enclosed background can be a missing body patch *or* a legitimate gap between
limbs. These measurements are review signals and never authorize mask edits.
"""
from __future__ import annotations

from collections import deque


def inspect_mask_shape(mask, *, max_side: int = 384) -> dict:
    """Inspect connected binary mask regions at bounded resolution.

    The input PIL image must already have passed the dense mask preflight.
    Four-connected regions are used to avoid merging tiny diagonal leaks.
    """
    from PIL import Image

    width, height = mask.size
    if not (1 <= max_side <= 1024 and width > 0 and height > 0):
        raise ValueError("Invalid mask dimensions or silhouette QA budget")
    scale = min(1.0, max_side / max(width, height))
    sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
    small = mask.convert("L").resize((sw, sh), Image.Resampling.NEAREST)
    foreground = bytearray(1 if value >= 128 else 0 for value in small.tobytes())
    pixels = sw * sh
    foreground_pixels = sum(foreground)
    visited = bytearray(pixels)
    interior_background: list[int] = []
    foreground_regions: list[int] = []

    def flood(start: int, color: int) -> tuple[int, bool]:
        queue = deque([start])
        visited[start] = 1
        size = 0
        touches_border = False
        while queue:
            pos = queue.popleft()
            size += 1
            x, y = pos % sw, pos // sw
            if x == 0 or x == sw-1 or y == 0 or y == sh-1:
                touches_border = True
            for adjacent in ((pos-1 if x else -1),
                             (pos+1 if x+1 < sw else -1),
                             (pos-sw if y else -1),
                             (pos+sw if y+1 < sh else -1)):
                if (adjacent >= 0 and not visited[adjacent]
                        and foreground[adjacent] == color):
                    visited[adjacent] = 1
                    queue.append(adjacent)
        return size, touches_border

    for index in range(pixels):
        if visited[index]:
            continue
        color = foreground[index]
        size, border = flood(index, color)
        if color:
            foreground_regions.append(size)
        elif not border:
            interior_background.append(size)

    # Suppress 1–2 sampled pixels of JPEG-like aliasing. Report actual hole
    # area separately from very small speckles; never automatically fill it.
    minimum_region = max(3, round(pixels * 0.00005))
    significant_holes = [size for size in interior_background
                         if size >= minimum_region]
    significant_components = [size for size in foreground_regions
                              if size >= minimum_region]
    largest_component = max(foreground_regions, default=0)
    lower_band = min(sh, max(1, round(sh * 0.05)))
    lower_fraction = sum(foreground[-sw * lower_band:]) / (sw * lower_band)
    metrics = {
        "sampled_dimensions": [sw, sh],
        "foreground_fraction": round(foreground_pixels / pixels, 6),
        "interior_background_regions": len(significant_holes),
        "interior_background_fraction": round(sum(significant_holes) / pixels, 6),
        "largest_interior_background_fraction": round(max(significant_holes, default=0) / pixels, 6),
        "significant_foreground_components": len(significant_components),
        "detached_foreground_fraction": round((foreground_pixels - largest_component) / pixels, 6),
        "bottom_five_percent_foreground_fraction": round(lower_fraction, 6),
    }
    flags = []
    if metrics["interior_background_fraction"] >= 0.002:
        flags.append("interior-background-review")
    if metrics["detached_foreground_fraction"] >= 0.005:
        flags.append("detached-foreground-review")
    if lower_fraction >= 0.15:
        flags.append("bottom-border-foreground-review")
    metrics["review_flags"] = flags
    metrics["limitations"] = (
        "Enclosed black pixels can be true limb gaps; bottom contact can be an intentionally cropped statue. "
        "Neither proves ground leakage or authorizes morphology changes.")
    return metrics
