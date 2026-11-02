"""Guided, non-destructive surface finishing for private photographic UV atlases.

Photographs are not reflectance measurements. Illumination must be supplied or
estimated heuristically, roughness is a material prior, and height is *never*
inferred from diffuse photographs. All inputs are local, dimension-checked and
hash-pinned; outputs require artist review.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .references import ReferenceError


_IDENTIFIER = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_MAX_DIM = 16384
_MAX_FILE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class PinnedSurfaceImage:
    path: Path
    sha256: str
    size: tuple[int, int]
    mode: str

    def open_verified(self) -> Any:
        from PIL import Image

        if (self.path.is_symlink() or not self.path.is_file()
                or self.path.stat().st_size > _MAX_FILE_BYTES
                or hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256):
            raise ReferenceError(f"Surface source changed after validation: {self.path.name}")
        with Image.open(self.path) as photo:
            photo.load()
            if photo.size != self.size or photo.mode != self.mode:
                raise ReferenceError(f"Surface source dimensions/mode changed: {self.path.name}")
            return photo.copy()


@dataclass(frozen=True)
class MaterialRegion:
    id: str
    mask: PinnedSurfaceImage
    roughness: float
    roughness_map: PinnedSurfaceImage | None = None


@dataclass(frozen=True)
class AtlasSurfaceSpec:
    regions: tuple[MaterialRegion, ...]
    illumination_map: PinnedSurfaceImage | None = None
    damage_mask: PinnedSurfaceImage | None = None
    height_map: PinnedSurfaceImage | None = None
    displacement_scale_m: float | None = None


@dataclass(frozen=True)
class SurfaceOptions:
    delight_strength: float = 0.0
    max_delight_ev: float = 1.0
    low_frequency_radius: int = 96
    cleanup_radius: int = 0

    def validate(self) -> "SurfaceOptions":
        for name, low, high in (("delight_strength", 0, 1),
                                ("max_delight_ev", 0, 2)):
            number = getattr(self, name)
            if type(number) not in (float, int) or not math.isfinite(number) or not low <= number <= high:
                raise ReferenceError(f"{name} must be finite in [{low}, {high}]")
        if (type(self.low_frequency_radius) is not int
                or not 16 <= self.low_frequency_radius <= 1024):
            raise ReferenceError("low_frequency_radius must be an integer in [16, 1024]")
        if type(self.cleanup_radius) is not int or not 0 <= self.cleanup_radius <= 16:
            raise ReferenceError("cleanup_radius must be an integer in [0, 16]")
        return self


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceError(f"Duplicate surface recipe property: {key}")
        result[key] = value
    return result


def _pinned_image(directory: Path, relative: str, size: tuple[int, int],
                  *, grayscale: bool = True, height: bool = False) -> PinnedSurfaceImage:
    from PIL import Image

    if (not isinstance(relative, str) or not relative or len(relative) > 240
            or "\\" in relative or Path(relative).is_absolute()
            or any(part in (".", "..") for part in Path(relative).parts)):
        raise ReferenceError("Surface image path must be relative to the recipe and cannot traverse directories")
    path = directory
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ReferenceError("Surface image may not follow a symlink")
    try:
        resolved = path.resolve(strict=True)
        if (not resolved.is_relative_to(directory) or not resolved.is_file()
                or resolved.stat().st_size > _MAX_FILE_BYTES):
            raise ReferenceError("Surface image is missing, outside recipe directory or oversized")
        with Image.open(resolved) as image:
            if image.format != "PNG" or image.size != size:
                raise ReferenceError("Surface map must be PNG with dimensions matching the source atlas")
            if height and image.mode != "I;16":
                raise ReferenceError("Height map must be a genuine 16-bit grayscale PNG")
            if grayscale and not height and image.mode != "L":
                raise ReferenceError("Surface mask/illumination must be 8-bit grayscale PNG")
            image.verify()
            mode = image.mode
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        return PinnedSurfaceImage(resolved, digest, size, mode)
    except (OSError, ValueError) as error:
        if isinstance(error, ReferenceError):
            raise
        raise ReferenceError(f"Cannot read surface map {relative}: {error}") from error


def load_surface_recipe(path: str | Path,
                        atlases: dict[str, Path]) -> dict[str, AtlasSurfaceSpec]:
    """Preflight exact per-atlas material masks before a revision is created."""
    from PIL import Image, ImageChops

    recipe = Path(path)
    if recipe.is_symlink() or not recipe.is_file() or recipe.stat().st_size > 65536:
        raise ReferenceError("Surface recipe must be an existing unlinked JSON file <= 64 KiB")
    directory = recipe.resolve().parent
    try:
        data = json.loads(recipe.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError) as error:
        if isinstance(error, ReferenceError):
            raise
        raise ReferenceError(f"Invalid surface recipe: {error}") from error
    if (not isinstance(data, dict) or set(data) != {"schema_version", "atlases"}
            or type(data["schema_version"]) is not int or data["schema_version"] != 1
            or not isinstance(data["atlases"], dict) or not data["atlases"]
            or set(data["atlases"]) != set(atlases) or len(atlases) > 64):
        raise ReferenceError("Surface recipe must specify schema_version 1 and exactly the pinned atlas names")
    result: dict[str, AtlasSurfaceSpec] = {}
    for name, entry in data["atlases"].items():
        if not isinstance(entry, dict) or not {"regions"} <= set(entry) or set(entry) - {
                "regions", "illumination_map", "damage_mask", "height_map", "displacement_scale_m"}:
            raise ReferenceError(f"Invalid material regions/maps for atlas {name}")
        atlas = atlases[name]
        with Image.open(atlas) as original:
            size = original.size
        if not all(0 < axis <= _MAX_DIM for axis in size):
            raise ReferenceError("Source atlas dimensions exceed the material-finishing limit")
        raw_regions = entry["regions"]
        if not isinstance(raw_regions, list) or not 1 <= len(raw_regions) <= 16:
            raise ReferenceError("Each atlas needs 1–16 material regions")
        regions: list[MaterialRegion] = []
        occupied = Image.new("L", size, 0)
        ids: set[str] = set()
        for raw in raw_regions:
            if (not isinstance(raw, dict) or not {"id", "mask", "roughness"} <= set(raw)
                    or set(raw) - {"id", "mask", "roughness", "roughness_map"}):
                raise ReferenceError("Material region requires id, mask and roughness, optional roughness_map")
            identity, rough = raw["id"], raw["roughness"]
            if (not isinstance(identity, str) or not _IDENTIFIER.fullmatch(identity)
                    or identity in ids):
                raise ReferenceError("Material region id must be unique and hyphenated")
            if type(rough) not in (float, int) or not math.isfinite(rough) or not .02 <= rough <= 1:
                raise ReferenceError("Material roughness prior must be finite in [0.02, 1]")
            mask = _pinned_image(directory, raw["mask"], size)
            with Image.open(mask.path) as image:
                if ImageChops.multiply(occupied, image).getbbox():
                    raise ReferenceError("Material region masks overlap; separate the surfaces first")
                occupied = ImageChops.lighter(occupied, image)
            rough_map = (_pinned_image(directory, raw["roughness_map"], size)
                         if "roughness_map" in raw else None)
            regions.append(MaterialRegion(identity, mask, float(rough), rough_map))
            ids.add(identity)
        illumination = (_pinned_image(directory, entry["illumination_map"], size)
                        if "illumination_map" in entry else None)
        damage = (_pinned_image(directory, entry["damage_mask"], size)
                  if "damage_mask" in entry else None)
        height = (_pinned_image(directory, entry["height_map"], size, height=True)
                  if "height_map" in entry else None)
        scale = entry.get("displacement_scale_m")
        if ((height is None) != (scale is None) or
                (scale is not None and (type(scale) not in (float, int)
                                        or not math.isfinite(scale) or not .00001 <= scale <= .5))):
            raise ReferenceError("height_map requires a displacement_scale_m in [0.00001, 0.5]")
        result[name] = AtlasSurfaceSpec(tuple(regions), illumination, damage, height,
                                        float(scale) if scale is not None else None)
    return result


def _auto_illumination(photo: Any, mask: Any, radius: int) -> Any:
    """Bounded LOW-FREQUENCY *heuristic*, not physically recovered lighting."""
    from PIL import Image, ImageChops, ImageFilter
    import numpy as np

    small = photo.convert("RGB")
    small.thumbnail((2048, 2048), Image.Resampling.BILINEAR)
    region = mask.resize(small.size, Image.Resampling.BILINEAR)
    samples = np.asarray(small, dtype=np.float32) / 255
    linear = np.where(samples <= .04045, samples / 12.92,
                      ((samples + .055) / 1.055) ** 2.4)
    linear_luma = (linear[..., 0] * .2126 + linear[..., 1] * .7152
                   + linear[..., 2] * .0722)
    luma = Image.fromarray(np.uint8(np.clip(linear_luma * 255, 0, 255)), "L")
    valid = ImageChops.multiply(region, small.convert("L").point(
        lambda value: 255 if value >= 16 else 0))
    if not valid.getbbox():
        return Image.new("L", photo.size, 128)
    sigma = max(4, radius * small.width / max(1, photo.width))
    weighted = ImageChops.multiply(luma, valid).filter(ImageFilter.GaussianBlur(sigma))
    normalizer = valid.filter(ImageFilter.GaussianBlur(sigma))
    weights = np.asarray(normalizer, dtype=np.float32)
    field = np.asarray(weighted, dtype=np.float32) / np.maximum(weights / 255, .02)
    pixels = np.asarray(luma, dtype=np.float32)
    region_pixels = np.asarray(valid) > 128
    median = float(np.median(pixels[region_pixels]))
    relative = np.clip(field / max(1.0, median), .25, 1.99)
    result = Image.fromarray(np.uint8(np.clip(relative * 128, 1, 255)), "L")
    return result.resize(photo.size, Image.Resampling.BILINEAR)


def _cleanup_damage(photo: Any, masks: list[Any], damage: Any, radius: int) -> tuple[Any, int]:
    from PIL import Image, ImageChops, ImageFilter
    import numpy as np

    if radius <= 0:
        raise ReferenceError("A damage_mask requires cleanup_radius > 0")
    if damage.getbbox() is None:
        return photo.copy(), 0
    total = photo.width * photo.height
    if sum(damage.histogram()[1:]) > total * .05:
        raise ReferenceError("Damage mask exceeds 5% of atlas; use manual reconstruction")
    allowed = Image.new("L", photo.size, 0)
    for region in masks:
        allowed = ImageChops.lighter(allowed, region)
    if ImageChops.subtract(damage, allowed).getbbox():
        raise ReferenceError("Damage mask contains pixels outside assigned material regions")
    original = photo.convert("RGB")
    output = original.copy()
    for region in masks:
        area = ImageChops.multiply(region, damage)
        if not area.getbbox():
            continue
        donors = ImageChops.multiply(region, ImageChops.invert(damage))
        denom = donors.filter(ImageFilter.GaussianBlur(radius))
        insufficient = denom.point(lambda value: 255 if value < 8 else 0)
        if ImageChops.multiply(area, insufficient).getbbox():
            raise ReferenceError("Damage contains unsupported holes beyond cleanup radius; review manually")
        layers = []
        for channel in original.split():
            weighted = ImageChops.multiply(channel, donors).filter(ImageFilter.GaussianBlur(radius))
            restored = Image.new("L", original.size)
            for y in range(0, original.height, 256):
                h = min(256, original.height - y)
                numerator = np.asarray(weighted.crop((0, y, original.width, y + h)), dtype=np.float32)
                denominator = np.asarray(denom.crop((0, y, original.width, y + h)), dtype=np.float32)
                restored.paste(Image.fromarray(np.uint8(np.clip(
                    numerator * 255 / np.maximum(denominator, 1), 0, 255))), (0, y))
            layers.append(restored)
        rebuilt = Image.merge("RGB", tuple(layers))
        output = Image.composite(rebuilt, output, area)
    if "A" in photo.getbands():
        output.putalpha(photo.getchannel("A"))
    return output, sum(damage.histogram()[1:])


def finish_surface_atlas(photo: Any, spec: AtlasSurfaceSpec,
                         options: SurfaceOptions, *, protect_black: int = 12) -> tuple[Any, Any, dict[str, Any]]:
    """Return base color, roughness PNG image, and honest process diagnostics."""
    from PIL import Image
    import numpy as np

    options.validate()
    if not 0 <= protect_black <= 32 or type(protect_black) is not int:
        raise ReferenceError("protect_black must be in [0, 32]")
    if any(item.mask.size != photo.size for item in spec.regions):
        raise ReferenceError("Material region dimensions differ from source atlas")
    masks = [region.mask.open_verified() for region in spec.regions]
    rough_maps = [region.roughness_map.open_verified() if region.roughness_map else None
                  for region in spec.regions]
    has_alpha = "A" in photo.getbands() or "transparency" in photo.info
    original = photo.convert("RGBA" if has_alpha else "RGB")
    working = original
    cleanup_count = 0
    if spec.damage_mask is not None:
        working, cleanup_count = _cleanup_damage(
            working, masks, spec.damage_mask.open_verified(), options.cleanup_radius)
    guides: list[Any] = []
    if options.delight_strength > 0:
        if spec.illumination_map is not None:
            guidance = spec.illumination_map.open_verified()
            guides = [guidance] * len(masks)
            method = "guided-illumination"
        else:
            guides = [_auto_illumination(working, mask, options.low_frequency_radius)
                      for mask in masks]
            method = "low-frequency-heuristic"
    else:
        method = "disabled"
    corrected = Image.new("RGB", original.size)
    roughness = Image.new("L", original.size, 217)  # Unlabeled regions: conservative matte.
    alpha = original.getchannel("A") if "A" in original.getbands() else None
    working_rgb = working.convert("RGB")
    original_luma = original.convert("L")
    for y in range(0, original.height, 256):
        height = min(256, original.height - y)
        box = (0, y, original.width, y + height)
        rgb = np.asarray(working_rgb.crop(box), dtype=np.float32) / 255
        lin = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        out = lin.copy()
        rough = np.full(rgb.shape[:2], 217, dtype=np.uint8)
        dark = np.asarray(original_luma.crop(box)) <= protect_black
        for index, (region, mask) in enumerate(zip(spec.regions, masks)):
            mix = np.asarray(mask.crop(box), dtype=np.float32) / 255
            mix[dark] = 0
            if alpha is not None:
                mix *= np.asarray(alpha.crop(box), dtype=np.float32) / 255
            if rough_maps[index] is not None:
                values = np.asarray(rough_maps[index].crop(box), dtype=np.uint8)
                rough[mix > 0] = values[mix > 0]
            else:
                rough[mix > 0] = round(region.roughness * 255)
            if guides:
                field = np.asarray(guides[index].crop(box), dtype=np.float32) / 128
                # Zero is unknown/invalid guidance, not an infinitely deep shadow.
                gain = np.clip(1 / np.maximum(field, 1 / 255),
                               2 ** -options.max_delight_ev, 2 ** options.max_delight_ev)
                gain[field < (16 / 128)] = 1
                gain = gain ** options.delight_strength
                out = out * (1 - mix[..., None]) + lin * gain[..., None] * mix[..., None]
        srgb = np.where(out <= .0031308, out * 12.92,
                        1.055 * np.maximum(out, 0) ** (1 / 2.4) - .055)
        strip = Image.fromarray(np.uint8(np.clip(srgb * 255 + .5, 0, 255)), "RGB")
        corrected.paste(strip, (0, y))
        roughness.paste(Image.fromarray(rough, "L"), (0, y))
    if alpha is not None:
        corrected.putalpha(alpha)
    return corrected, roughness, {
        "delighting_method": method,
        "delight_strength": options.delight_strength,
        "max_delight_ev": options.max_delight_ev,
        "cleanup_pixels": cleanup_count,
        "roughness_source": ("material-priors-and-authored-maps-not-measured"
                             if any(rough_maps) else "material-priors-not-measured"),
        "regions": [{"id": item.id, "roughness": item.roughness,
                     "mask_sha256": item.mask.sha256,
                     "roughness_map_sha256": (item.roughness_map.sha256
                                              if item.roughness_map else None)}
                    for item in spec.regions],
        "limitations": ("Illumination estimates are not uniquely determined from photographs; "
                        "roughness values are priors, and real height requires a measured map."),
    }
