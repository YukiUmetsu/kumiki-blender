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
class GussetPlate:
    name: str
    center: Vec3
    axis_u: Vec3
    axis_v: Vec3
    normal: Vec3
    width: float
    height: float
    thickness: float


@dataclass(frozen=True)
class TowerPlan:
    beams: tuple[Beam, ...]
    decks: tuple[Deck, ...]
    gussets: tuple[GussetPlate, ...]
    fasteners: tuple[Beam, ...]
    bounds_min: Vec3
    bounds_max: Vec3


_CORNERS = ((-1, -1), (1, -1), (1, 1), (-1, 1))


def make_lattice_tower(recipe: TowerRecipe) -> TowerPlan:
    """Create a four-legged tapered lattice with rings, cross braces and decks.

    Beams use explicit endpoints to remain testable without Blender. The tower's
    local origin is the center of its base; native units are meters, +Z is up.
    """
    beams: list[Beam] = []
    gussets: list[GussetPlate] = []
    fasteners: list[Beam] = []
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

            # A physical gusset bridges the crossing diagonals in each face.
            # Its long axis follows the bay's edge, and its normal points out.
            zmid = (z0 + z1) / 2
            p0, p1 = corner(side, zmid), corner(nxt, zmid)
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length = (dx * dx + dy * dy) ** 0.5
            u: Vec3 = (dx / length, dy / length, 0.0)
            v: Vec3 = (0.0, 0.0, 1.0)
            normal: Vec3 = (u[1], -u[0], 0.0)
            center: Vec3 = ((p0[0] + p1[0]) / 2 + normal[0] * recipe.brace_radius_m * 0.7,
                            (p0[1] + p1[1]) / 2 + normal[1] * recipe.brace_radius_m * 0.7,
                            zmid)
            plate_size = min(length * 0.16, (z1 - z0) * 0.45)
            plate = GussetPlate(
                f"bay-{level:02d}-face-{side}-gusset", center, u, v, normal,
                width=plate_size, height=plate_size, thickness=max(0.05, recipe.brace_radius_m * 0.34),
            )
            gussets.append(plate)
            for bolt_index, (su, sv) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
                bolt_origin = tuple(
                    center[i] + su * u[i] * plate.width * 0.31 + sv * v[i] * plate.height * 0.31
                    + normal[i] * plate.thickness / 2
                    for i in range(3)
                )
                bolt_length = max(0.035, plate.thickness * 0.8)
                bolt_end = tuple(bolt_origin[i] + normal[i] * bolt_length for i in range(3))
                fasteners.append(Beam(f"{plate.name}-bolt-{bolt_index}", bolt_origin, bolt_end,
                                      max(0.017, recipe.brace_radius_m * 0.15), "fastener"))

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

    # Use true conservative extents, including outward-facing plates, bolt
    # heads and radial member thickness. Camera framing must not crop details.
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3

    def include(center: Vec3, radius: Vec3) -> None:
        for axis in range(3):
            minimum[axis] = min(minimum[axis], center[axis] - radius[axis])
            maximum[axis] = max(maximum[axis], center[axis] + radius[axis])

    for beam in (*beams, *fasteners):
        radius = (beam.radius,) * 3
        include(beam.start, radius)
        include(beam.end, radius)
    for deck in decks:
        include((0, 0, deck.z - deck.thickness / 2),
                (deck.span_m / 2, deck.span_m / 2, deck.thickness / 2))
    for plate in gussets:
        plate_extent = tuple(sum(abs(direction[axis]) * half for direction, half in (
            (plate.axis_u, plate.width / 2), (plate.axis_v, plate.height / 2),
            (plate.normal, plate.thickness / 2))) for axis in range(3))
        include(plate.center, plate_extent)
    return TowerPlan(
        beams=tuple(beams), decks=decks,
        gussets=tuple(gussets), fasteners=tuple(fasteners),
        bounds_min=tuple(minimum),
        bounds_max=tuple(maximum),
    )
