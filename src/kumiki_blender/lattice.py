"""Deterministic, Blender-independent structural geometry for lattice towers.

This is an architectural *construction scaffold*, not an authenticated replica.
Its named parts can be refined, replaced or inspected by a downstream renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .recipe import TowerRecipe


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Beam:
    name: str
    start: Vec3
    end: Vec3
    radius: float
    group: str


@dataclass(frozen=True)
class Deck:
    name: str
    z: float
    span_m: float
    thickness: float


@dataclass(frozen=True)
class TowerPlan:
    beams: tuple[Beam, ...]
    decks: tuple[Deck, ...]
    bounds_min: Vec3
    bounds_max: Vec3


_CORNERS = ((-1, -1), (1, -1), (1, 1), (-1, 1))


def make_lattice_tower(recipe: TowerRecipe) -> TowerPlan:
    """Create a four-legged tapered lattice with rings, cross braces and decks.

    Beams use explicit endpoints to remain testable without Blender. The tower's
    local origin is the center of its base; native units are meters, +Z is up.
    """
    beams: list[Beam] = []
    h = recipe.height_m
    n = recipe.levels

    def span_at(z: float) -> float:
        return recipe.base_span_m + (recipe.top_span_m - recipe.base_span_m) * (z / h)

    def corner(index: int, z: float) -> Vec3:
        sx, sy = _CORNERS[index]
        span = span_at(z)
        return (sx * span / 2, sy * span / 2, z)

    for level in range(n):
        z0, z1 = h * level / n, h * (level + 1) / n
        for side in range(4):
            nxt = (side + 1) % 4
            beams.append(Beam(f"leg-{side}-{level:02d}", corner(side, z0), corner(side, z1), recipe.leg_radius_m, "leg"))
            beams.append(Beam(f"brace-{level:02d}-{side}-a", corner(side, z0), corner(nxt, z1), recipe.brace_radius_m, "brace"))
            beams.append(Beam(f"brace-{level:02d}-{side}-b", corner(nxt, z0), corner(side, z1), recipe.brace_radius_m, "brace"))

    for level in range(n + 1):
        z = h * level / n
        for side in range(4):
            beams.append(Beam(f"ring-{level:02d}-{side}", corner(side, z), corner((side + 1) % 4, z), recipe.brace_radius_m * 1.45, "ring"))

    decks = tuple(
        Deck(f"observation-deck-{index}", h * fraction, span_at(h * fraction) * scale, max(0.3, h * 0.004))
        for index, (fraction, scale) in enumerate(((0.68, 1.35), (0.81, 1.25)), start=1)
    )

    for deck in decks:
        rail_z = deck.z + 1.35
        half = deck.span_m / 2
        corners = [(x * half, y * half) for x, y in _CORNERS]
        for side in range(4):
            x0, y0 = corners[side]
            x1, y1 = corners[(side + 1) % 4]
            beams.append(Beam(f"{deck.name}-post-{side}", (x0, y0, deck.z), (x0, y0, rail_z), 0.055, "railing"))
            beams.append(Beam(f"{deck.name}-rail-{side}", (x0, y0, rail_z), (x1, y1, rail_z), 0.045, "railing"))

    beams.append(Beam("antenna-core", (0, 0, h), (0, 0, h * 1.14), recipe.leg_radius_m * 0.55, "antenna"))

    extent = max(recipe.base_span_m / 2, *(deck.span_m / 2 for deck in decks))
    return TowerPlan(
        beams=tuple(beams), decks=decks,
        bounds_min=(-extent, -extent, 0.0),
        bounds_max=(extent, extent, h * 1.14),
    )
