"""Strict, Blender-independent asset recipe parsing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RecipeError(ValueError):
    """Input cannot safely or predictably describe a supported asset."""


@dataclass(frozen=True)
class TowerRecipe:
    asset_id: str
    height_m: float
    base_span_m: float
    top_span_m: float
    levels: int
    leg_radius_m: float
    brace_radius_m: float
    steel_color: str
    roughness: float
    preview_width: int
    preview_height: int
    samples: int
    recipe_sha256: str

    @property
    def preview_size(self) -> tuple[int, int]:
        return self.preview_width, self.preview_height


def _fields(value: Any, required: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecipeError(f"{label} must be an object")
    unknown = set(value) - required
    missing = required - set(value)
    if unknown:
        raise RecipeError(f"{label}: unsupported properties: {', '.join(sorted(unknown))}")
    if missing:
        raise RecipeError(f"{label}: missing properties: {', '.join(sorted(missing))}")
    return value


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise RecipeError(f"{label} must be finite")
    if not minimum <= number <= maximum:
        raise RecipeError(f"{label} must be between {minimum} and {maximum}")
    return number


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise RecipeError(f"{label} must be an integer between {minimum} and {maximum}")
    return value


def _distinct_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecipeError(f"Duplicate JSON property: {key}")
        result[key] = value
    return result


def _nonfinite(token: str) -> None:
    raise RecipeError(f"JSON must contain only finite numbers: {token}")


def load_recipe(path: str | Path) -> TowerRecipe:
    """Load a v1 lattice-tower recipe, rejecting unknown and hazardous inputs."""
    source = Path(path)
    try:
        payload = source.read_bytes()
        raw = json.loads(payload, object_pairs_hook=_distinct_pairs, parse_constant=_nonfinite)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecipeError(f"Cannot load recipe {source}: {error}") from error

    root = _fields(raw, {"schema_version", "asset_id", "generator", "geometry", "material", "preview"}, "recipe")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise RecipeError("schema_version must be 1")
    if root["generator"] != "lattice-tower":
        raise RecipeError("generator must be lattice-tower")
    asset_id = root["asset_id"]
    if not isinstance(asset_id, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", asset_id) or len(asset_id) > 64:
        raise RecipeError("asset_id must be a lowercase, hyphenated name (max 64 characters)")

    geo = _fields(root["geometry"], {"height_m", "base_span_m", "top_span_m", "levels", "leg_radius_m", "brace_radius_m"}, "geometry")
    height = _number(geo["height_m"], "height_m", 10, 1000)
    base_span = _number(geo["base_span_m"], "base_span_m", 2, 300)
    top_span = _number(geo["top_span_m"], "top_span_m", 1, 300)
    if top_span >= base_span:
        raise RecipeError("top_span_m must be smaller than base_span_m")
    if base_span > height:
        raise RecipeError("base_span_m must not exceed height_m")
    levels = _integer(geo["levels"], "levels", 4, 64)
    leg_radius = _number(geo["leg_radius_m"], "leg_radius_m", 0.02, 10)
    brace_radius = _number(geo["brace_radius_m"], "brace_radius_m", 0.01, 5)
    if leg_radius >= top_span / 4 or brace_radius >= leg_radius:
        raise RecipeError("Structural radii are incompatible with tower spans")

    mat = _fields(root["material"], {"steel_color", "roughness"}, "material")
    color = mat["steel_color"]
    if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise RecipeError("steel_color must be a six-digit hexadecimal sRGB color")
    roughness = _number(mat["roughness"], "roughness", 0.1, 1.0)

    preview = _fields(root["preview"], {"width", "height", "samples"}, "preview")
    width = _integer(preview["width"], "width", 320, 4096)
    preview_height = _integer(preview["height"], "height", 320, 4096)
    if width * preview_height > 8_388_608:
        raise RecipeError("preview resolution exceeds the 8-megapixel safety limit")
    samples = _integer(preview["samples"], "samples", 1, 256)

    return TowerRecipe(
        asset_id=asset_id, height_m=height, base_span_m=base_span,
        top_span_m=top_span, levels=levels, leg_radius_m=leg_radius,
        brace_radius_m=brace_radius, steel_color=color.upper(),
        roughness=roughness, preview_width=width, preview_height=preview_height,
        samples=samples, recipe_sha256=hashlib.sha256(payload).hexdigest(),
    )
