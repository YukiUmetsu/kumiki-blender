"""Opt-in UV-island edge padding for private dense-scan texture atlases.

Only *unmapped*, dark/transparent gutter texels may be extended from nearby
valid mapped texels. Black/transparent pixels inside real UV faces are never
filled: a missing photographed surface cannot be recovered by texture padding.
"""

from __future__ import annotations

from typing import Any

from .references import ReferenceError


_MAX_ATLAS_PIXELS = 100_000_000
_TILE_SIZE = 1024
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _shift(photo: Any, dx: int, dy: int) -> Any:
    from PIL import Image

    shifted = Image.new(photo.mode, photo.size, 0)
    shifted.paste(photo, (dx, dy))
    return shifted


def _pad_tile(corrected: Any, original: Any, coverage: Any, radius: int,
              protect_black: int) -> tuple[Any, Any]:
    """Fill only eligible gutter pixels using nearest Manhattan-distance color.

    A simultaneous four-neighbor frontier avoids blur/color mixing across
    adjacent islands; source values for each propagation step come from the
    previous frontier, not from arbitrary atlas pixels.
    """
    from PIL import Image, ImageChops

    gray = original.convert("RGB").convert("L")
    valid = gray.point(lambda value: 255 if value >= max(16, protect_black) else 0)
    if "A" in original.getbands() or "transparency" in original.info:
        alpha = original.convert("RGBA").getchannel("A")
        valid = ImageChops.multiply(
            valid, alpha.point(lambda value: 255 if value >= 128 else 0))
    source = ImageChops.multiply(valid, coverage)
    eligibility = ImageChops.multiply(ImageChops.invert(coverage),
                                      ImageChops.invert(valid))
    padded = Image.new("L", coverage.size, 0)
    frontier = source
    if source.getbbox() and eligibility.getbbox():
        for _step in range(radius):
            next_frontier = Image.new("L", coverage.size, 0)
            available = ImageChops.subtract(eligibility, padded)
            for dx, dy in _NEIGHBORS:
                reachable = _shift(frontier, dx, dy)
                paint = ImageChops.multiply(
                    reachable, ImageChops.subtract(available, next_frontier))
                if paint.getbbox():
                    corrected.paste(_shift(corrected, dx, dy), (0, 0), paint)
                    next_frontier = ImageChops.lighter(next_frontier, paint)
            if not next_frontier.getbbox():
                break
            padded = ImageChops.lighter(padded, next_frontier)
            frontier = next_frontier
    if corrected.mode == "RGBA" and padded.getbbox():
        alpha = corrected.getchannel("A")
        alpha.paste(255, mask=padded)
        corrected.putalpha(alpha)
    return corrected, padded


def pad_uv_gutters(corrected: Any, original: Any, coverage: Any, *,
                   radius: int, protect_black: int) -> dict[str, Any]:
    """Pad a full-resolution atlas in bounded tiles with a UV-safe halo.

    The halo is at least the propagation radius, so processing a giant atlas
    in tiles produces the same nearest-edge padding as processing it at once.
    ``corrected`` changes in place; ``original`` and ``coverage`` are read-only.
    """
    if corrected.size != original.size or corrected.size != coverage.size:
        raise ReferenceError("UV gutter mask must match the native atlas dimensions")
    if corrected.mode not in ("RGB", "RGBA") or coverage.mode != "L":
        raise ReferenceError("UV gutter padding needs an RGB(A) atlas and L coverage mask")
    if type(radius) is not int or not 1 <= radius <= 16:
        raise ReferenceError("UV gutter radius must be an integer between 1 and 16")
    width, height = corrected.size
    if width * height > _MAX_ATLAS_PIXELS:
        raise ReferenceError("UV gutter atlas exceeds the 100-megapixel safety budget")
    padded_pixels = 0
    dark_mapped = 0
    valid_mapped = 0
    from PIL import ImageChops

    for top in range(0, height, _TILE_SIZE):
        for left in range(0, width, _TILE_SIZE):
            right = min(width, left+_TILE_SIZE)
            bottom = min(height, top+_TILE_SIZE)
            region = (max(0, left-radius), max(0, top-radius),
                      min(width, right+radius), min(height, bottom+radius))
            occupancy = coverage.crop(region)
            if not occupancy.getbbox():
                continue
            tile, painted = _pad_tile(
                corrected.crop(region), original.crop(region), occupancy,
                radius, protect_black)
            inner = (left-region[0], top-region[1], right-region[0], bottom-region[1])
            inner_occupancy = occupancy.crop(inner)
            inner_original = original.crop((left, top, right, bottom))
            gray = inner_original.convert("RGB").convert("L")
            valid = gray.point(lambda value: 255 if value >= max(16, protect_black) else 0)
            if "A" in inner_original.getbands() or "transparency" in inner_original.info:
                valid = ImageChops.multiply(
                    valid, inner_original.convert("RGBA").getchannel("A").point(
                        lambda value: 255 if value >= 128 else 0))
            good = ImageChops.multiply(valid, inner_occupancy)
            valid_mapped += good.histogram()[255]
            dark_mapped += ImageChops.subtract(inner_occupancy, good).histogram()[255]
            applied = painted.crop(inner)
            count = applied.histogram()[255]
            if count:
                corrected.paste(tile.crop(inner), (left, top))
                padded_pixels += count
    warnings = []
    if dark_mapped:
        warnings.append("Dark/transparent texels on actual UV faces remain unfilled")
    if not padded_pixels:
        warnings.append("No eligible dark/transparent UV gutters were found near mapped texels")
    return {"enabled": True, "radius_pixels": radius, "padded_pixels": padded_pixels,
            "valid_mapped_texels": valid_mapped,
            "invalid_mapped_texels_preserved": dark_mapped,
            "warnings": warnings}
