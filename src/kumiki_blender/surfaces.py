"""Domain-neutral physically based material definitions for Blender/GLB.

All parameters map to core glTF PBR or well-known extensions. Image/normal
textures are intentionally deferred until there is a tested bake pipeline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace


class SurfaceError(ValueError):
    """A material preset or override is unsupported or invalid."""


@dataclass(frozen=True)
class PBRSurface:
    name: str
    color: str
    metallic: float
    roughness: float
    transmission: float = 0.0
    ior: float = 1.5


_SURFACES = {
    "painted-steel": PBRSurface("Painted structural steel", "#9F3D33", 0.04, 0.43),
    "galvanized-steel": PBRSurface("Satin galvanized metal", "#ADB5BB", 0.88, 0.33),
    "cast-concrete": PBRSurface("Cast architectural concrete", "#787B79", 0.0, 0.85),
    "natural-stone": PBRSurface("Architectural stone", "#BBB3A5", 0.0, 0.7),
    "timber": PBRSurface("Seasoned timber", "#714831", 0.0, 0.58),
    "optical-glass": PBRSurface("Optical glass", "#D6E0E5", 0.0, 0.06, 0.95, 1.48),
}


def get_surface(kind: str, *, color: str | None = None, roughness: float | None = None) -> PBRSurface:
    """Return an independent immutable PBR recipe with validated overrides."""
    if kind not in _SURFACES:
        raise SurfaceError(f"Unknown surface kind: {kind}")
    surface = _SURFACES[kind]
    if color is not None:
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
            raise SurfaceError("Surface color must be six-digit hexadecimal sRGB")
        surface = replace(surface, color=color.upper())
    if roughness is not None:
        if isinstance(roughness, bool) or not isinstance(roughness, (int, float)) or not math.isfinite(roughness) or not 0.02 <= roughness <= 1.0:
            raise SurfaceError("Surface roughness must be a finite number between 0.02 and 1")
        surface = replace(surface, roughness=float(roughness))
    return surface
