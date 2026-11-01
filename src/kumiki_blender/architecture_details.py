"""Composable, physically closed building details independent of Blender.

All sizes are meters. X runs horizontally, Z vertically, and the facade faces
-Y. Returned parts are separate named meshes with explicit material roles, so
landmark recipes can replace any window, masonry block or molding independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from .architecture_geometry import (
    GeometryError, Mesh, MeshInstance, _finite_number, extrude_profile,
    place_mesh, revolve_profile,
)


@dataclass(frozen=True)
class Opening:
    left: float
    right: float
    bottom: float
    top: float


@lru_cache(maxsize=4096)
def box_mesh(width: float, depth: float, height: float) -> Mesh:
    """Watertight axis-aligned box centered at the origin, with outward faces."""
    x = _finite_number(width, "width", 0.001, 1000) / 2
    y = _finite_number(depth, "depth", 0.001, 1000) / 2
    z = _finite_number(height, "height", 0.001, 1000) / 2
    return Mesh(((-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
                 (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)),
                ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
                 (2, 3, 7, 6), (3, 0, 4, 7), (1, 2, 6, 5)))


def _subtract(rect: Opening, hole: Opening) -> tuple[Opening, ...]:
    """Subtract an axis-aligned opening; pieces are disjoint rectangles."""
    lx, rx = max(rect.left, hole.left), min(rect.right, hole.right)
    bz, tz = max(rect.bottom, hole.bottom), min(rect.top, hole.top)
    if lx >= rx or bz >= tz:
        return (rect,)
    return tuple(piece for piece in (
        Opening(rect.left, lx, rect.bottom, rect.top),
        Opening(rx, rect.right, rect.bottom, rect.top),
        Opening(lx, rx, rect.bottom, bz),
        Opening(lx, rx, tz, rect.top),
    ) if piece.right - piece.left > 1e-6 and piece.top - piece.bottom > 1e-6)


def coursed_wall(*, width: float, height: float, thickness: float,
                 openings: tuple[Opening, ...] = (), block_width: float = 1.02,
                 course_height: float = 0.49, joint: float = 0.012,
                 prefix: str = "masonry", material: str = "wall-stone") -> tuple[MeshInstance, ...]:
    """Staggered real masonry blocks, with actual voids for doors and windows.

    Openings can interrupt a course. Each resulting segment remains a distinct
    manifold stone, including partial stones around lintels and jambs.
    """
    width = _finite_number(width, "wall width", 0.2, 120)
    height = _finite_number(height, "wall height", 0.2, 80)
    thickness = _finite_number(thickness, "wall thickness", 0.05, 5)
    block_width = _finite_number(block_width, "block width", 0.2, 4)
    course_height = _finite_number(course_height, "course height", 0.15, 2)
    joint = _finite_number(joint, "mortar joint", 0, 0.05)
    if joint >= min(block_width, course_height) / 4:
        raise GeometryError("Mortar joint is too large for the coursing")
    if not isinstance(openings, tuple) or len(openings) > 64:
        raise GeometryError("Openings must be a tuple of at most 64 rectangles")
    for index, hole in enumerate(openings):
        if not isinstance(hole, Opening):
            raise GeometryError("Openings must use Opening rectangles")
        for value in (hole.left, hole.right, hole.bottom, hole.top):
            _finite_number(value, f"opening {index}", 0, max(width, height))
        if not (0 <= hole.left < hole.right <= width and 0 <= hole.bottom < hole.top <= height):
            raise GeometryError("Opening must be inside the wall and have positive area")
        if any(hole.left < other.right and hole.right > other.left
               and hole.bottom < other.top and hole.top > other.bottom for other in openings[:index]):
            raise GeometryError("Wall openings may not overlap")
    count = math.ceil(height / course_height)
    if count * math.ceil(width / block_width + 2) > 1800:
        raise GeometryError("Wall course count exceeds the 1800-block budget")
    result: list[MeshInstance] = []
    for course in range(count):
        bottom, top = course * course_height, min(height, (course + 1) * course_height)
        offset = block_width / 2 if course % 2 else 0
        col, start = 0, -offset
        while start < width:
            left, right = max(0.0, start), min(width, start + block_width)
            if right > left:
                sections = (Opening(left, right, bottom, top),)
                for hole in openings:
                    sections = tuple(piece for section in sections for piece in _subtract(section, hole))
                for segment, section in enumerate(sections):
                    sx = section.right - section.left - joint
                    sz = section.top - section.bottom - joint
                    if sx < 0.045 or sz < 0.045:
                        continue
                    result.append(place_mesh(box_mesh(sx, thickness, sz),
                        name=f"{prefix}-{course:02d}-{col:02d}-{segment:02d}",
                        origin=((section.left + section.right) / 2, 0,
                                (section.bottom + section.top) / 2), material=material))
            col += 1
            start += block_width
    return tuple(result)


def _part(name: str, dimensions: tuple[float, float, float], origin: tuple[float, float, float],
          material: str) -> MeshInstance:
    return place_mesh(box_mesh(*dimensions), name=name, origin=origin, material=material)


def window_assembly(*, prefix: str, width: float = 2.0, height: float = 2.5,
                    columns: int = 2, rows: int = 2,
                    frame: float = 0.12) -> tuple[MeshInstance, ...]:
    """A recessed glazing unit, thin separate panes, mullions and stone surround.

    Local origin is the center of the opening. The wall's front face is Y=-0.25.
    The dark recess, real glazing and trim lie at different physical depths.
    """
    width = _finite_number(width, "window width", 0.5, 9)
    height = _finite_number(height, "window height", 0.5, 9)
    frame = _finite_number(frame, "window frame", 0.025, 0.4)
    if type(columns) is not int or not 1 <= columns <= 6 or type(rows) is not int or not 1 <= rows <= 6:
        raise GeometryError("Window panes require 1–6 rows and 1–6 columns")
    if frame * 3 >= min(width / columns, height / rows):
        raise GeometryError("Window frame would obscure the glazing")
    w, h = width / 2, height / 2
    parts = [
        _part(f"{prefix}-recess", (width - frame, 0.035, height - frame),
              (0, 0.14, 0), "recess-dark"),
        _part(f"{prefix}-glass", (width - frame * 1.5, 0.012, height - frame * 1.5),
              (0, -0.13, 0), "window-glass"),
        _part(f"{prefix}-stile-left", (frame, 0.16, height), (-w + frame/2, -0.24, 0), "window-metal"),
        _part(f"{prefix}-stile-right", (frame, 0.16, height), (w - frame/2, -0.24, 0), "window-metal"),
        _part(f"{prefix}-rail-top", (width - 2*frame, 0.16, frame),
              (0, -0.24, h - frame/2), "window-metal"),
        _part(f"{prefix}-rail-bottom", (width - 2*frame, 0.16, frame),
              (0, -0.24, -h + frame/2), "window-metal"),
        _part(f"{prefix}-stone-sill", (width + 0.43, 0.44, 0.15),
              (0, -0.36, -h - 0.055), "trim-stone"),
        _part(f"{prefix}-stone-lintel", (width + 0.43, 0.40, 0.20),
              (0, -0.32, h + 0.10), "trim-stone"),
        _part(f"{prefix}-jamb-left", (0.17, 0.22, height + 0.22),
              (-w - 0.12, -0.31, 0), "trim-stone"),
        _part(f"{prefix}-jamb-right", (0.17, 0.22, height + 0.22),
              (w + 0.12, -0.31, 0), "trim-stone"),
    ]
    for col in range(1, columns):
        x = -w + col * width / columns
        parts.append(_part(f"{prefix}-mullion-{col:02d}", (frame*0.72, 0.13, height - 2*frame),
                           (x, -0.26, 0), "window-metal"))
    for row in range(1, rows):
        z = -h + row * height / rows
        parts.append(_part(f"{prefix}-transom-{row:02d}",
                           (width - 2*frame, 0.13, frame*0.72),
                           (0, -0.26, z), "window-metal"))
    return tuple(parts)


def door_assembly(*, prefix: str, width: float = 2.2, height: float = 3.65) -> tuple[MeshInstance, ...]:
    """Panelled timber entrance with stone jambs, raised moldings and hardware."""
    width = _finite_number(width, "door width", 0.8, 6)
    height = _finite_number(height, "door height", 1.6, 9)
    w, h = width / 2, height / 2
    parts = [
        _part(f"{prefix}-slab", (width - 0.13, 0.15, height - 0.09),
              (0, 0.035, 0), "door-timber"),
        _part(f"{prefix}-stone-left", (0.25, 0.53, height + 0.30),
              (-w - 0.12, -0.24, 0.06), "trim-stone"),
        _part(f"{prefix}-stone-right", (0.25, 0.53, height + 0.30),
              (w + 0.12, -0.24, 0.06), "trim-stone"),
        _part(f"{prefix}-stone-lintel", (width + 0.70, 0.57, 0.26),
              (0, -0.26, h + 0.19), "trim-stone"),
        _part(f"{prefix}-threshold", (width + 0.48, 0.68, 0.13),
              (0, -0.32, -h - 0.05), "foundation"),
        _part(f"{prefix}-center-rail", (width - 0.30, 0.055, 0.13),
              (0, -0.09, 0), "door-timber"),
        _part(f"{prefix}-escutcheon", (0.10, 0.035, 0.27),
              (w - 0.29, -0.115, 0.02), "bronze-metal"),
    ]
    for row in (-1, 1):
        for col in (-1, 1):
            x, z = col * width * 0.245, row * height * 0.245
            pw, ph = width * 0.38, height * 0.38
            parts.append(_part(f"{prefix}-panel-{row+1}-{col+1}", (pw, 0.045, ph),
                               (x, -0.095, z), "door-timber-light"))
            for side in (-1, 1):
                parts.append(_part(f"{prefix}-panel-stile-{row+1}-{col+1}-{side+1}",
                                   (0.045, 0.066, ph+0.045),
                                   (x + side * (pw/2+0.025), -0.12, z), "door-timber"))
                parts.append(_part(f"{prefix}-panel-rail-{row+1}-{col+1}-{side+1}",
                                   (pw+0.09, 0.066, 0.045),
                                   (x, -0.12, z + side * (ph/2+0.025)), "door-timber"))
    knob = revolve_profile(((0.045, 0), (0.052, 0.028), (0.062, 0.07), (0.042, 0.10)),
                           radial_segments=20)
    parts.append(place_mesh(knob, name=f"{prefix}-round-knob", origin=(w - 0.29, -0.14, 0.02),
                            axis_x=(1, 0, 0), axis_y=(0, 0, 1), axis_z=(0, -1, 0),
                            material="bronze-metal"))
    return tuple(parts)


def cornice_extrusion(length: float, projection: float = 0.42, height: float = 0.42) -> Mesh:
    """A non-flat molding profile, extruded along the building's +X axis.

    Local profile X is exterior Y, profile Y is world Z. Place with the
    canonical cornice axes (Y, Z, X) to align the part along a facade.
    """
    length = _finite_number(length, "cornice length", 0.2, 150)
    projection = _finite_number(projection, "cornice projection", 0.06, 3)
    height = _finite_number(height, "cornice height", 0.06, 3)
    p, h = projection, height
    return extrude_profile(((0, 0), (-p*0.72, 0), (-p*0.72, h*0.15),
                            (-p, h*0.26), (-p, h*0.47), (-p*0.74, h*0.58),
                            (-p*0.88, h*0.76), (-p*0.52, h), (0, h)), length)


def dentil_row(*, prefix: str, length: float, count: int, tooth_width: float = 0.23,
               tooth_depth: float = 0.27, tooth_height: float = 0.26) -> tuple[MeshInstance, ...]:
    """Repeated stone teeth for the underside of a classical cornice."""
    length = _finite_number(length, "dentil row length", 0.3, 150)
    if type(count) is not int or not 1 <= count <= 200:
        raise GeometryError("Dentil count must be an integer in [1, 200]")
    for label, value in (("tooth width", tooth_width), ("tooth depth", tooth_depth),
                         ("tooth height", tooth_height)):
        _finite_number(value, label, 0.01, 2)
    if count * tooth_width > length * 0.88:
        raise GeometryError("Dentils must have a visible gap between teeth")
    step = length / count
    tooth = box_mesh(tooth_width, tooth_depth, tooth_height)
    return tuple(place_mesh(tooth, name=f"{prefix}-{i:03d}",
                            origin=((i + 0.5) * step, 0, 0), material="trim-stone")
                 for i in range(count))


def rosette_relief(*, prefix: str, diameter: float = 0.65,
                   petals: int = 12, relief_depth: float = 0.075) -> tuple[MeshInstance, ...]:
    """A modular carved stone rosette with radial leaves and raised veins.

    The decoration lies in local XY and projects along +Z. Mount it on a
    vertical facade using basis axes (world X, world Z, world -Y). Each leaf
    is a real manifold relief mesh rather than a normal-map-only decal.
    """
    diameter = _finite_number(diameter, "rosette diameter", 0.15, 3)
    relief_depth = _finite_number(relief_depth, "rosette relief", 0.01, 0.5)
    if type(petals) is not int or not 6 <= petals <= 24:
        raise GeometryError("A rosette needs 6–24 carved petals")
    radius = diameter / 2
    backing = revolve_profile(((radius * 0.92, 0),
                               (radius, relief_depth * 0.28),
                               (radius * 0.96, relief_depth * 0.58)), radial_segments=48)
    boss = revolve_profile(((radius * 0.15, 0),
                            (radius * 0.23, relief_depth * 0.62),
                            (radius * 0.16, relief_depth * 1.4)), radial_segments=32)
    leaf = extrude_profile(((radius*.20, 0), (radius*.38, -radius*.14),
                            (radius*.59, -radius*.17), (radius*.86, 0),
                            (radius*.59, radius*.17), (radius*.38, radius*.14)),
                           relief_depth * 0.74)
    vein = extrude_profile(((radius*.32, -radius*.027),
                            (radius*.76, 0), (radius*.32, radius*.027)),
                           relief_depth * 0.16)
    result = [
        place_mesh(backing, name=f"{prefix}-backing", material="ornament-stone"),
        place_mesh(boss, name=f"{prefix}-boss", origin=(0, 0, relief_depth*0.5),
                   material="ornament-highlight"),
    ]
    for index in range(petals):
        angle = math.tau * index / petals
        axes = {"axis_x": (math.cos(angle), math.sin(angle), 0),
                "axis_y": (-math.sin(angle), math.cos(angle), 0),
                "axis_z": (0, 0, 1)}
        result.append(place_mesh(leaf, name=f"{prefix}-leaf-{index:02d}",
                                 origin=(0, 0, relief_depth * 0.37),
                                 material="ornament-stone", **axes))
        result.append(place_mesh(vein, name=f"{prefix}-vein-{index:02d}",
                                 origin=(0, 0, relief_depth * 1.11),
                                 material="ornament-highlight", **axes))
    return tuple(result)


@lru_cache(maxsize=128)
def fluted_column_shaft(*, height: float = 3.55, base_radius: float = 0.36,
                        top_radius: float = 0.31, flutes: int = 24,
                        flute_depth: float = 0.020, entasis: float = 0.012) -> Mesh:
    """A single watertight shaft with physically concave flutes and entasis.

    Flutes deform the actual cylindrical perimeter, never floating cylinder
    strips. Set flutes=0 for a smooth Tuscan-inspired column. The shaft is
    drawn from Z=0 to height with capped ends; column assembly adds the base
    and capital as separately editable components.
    """
    height = _finite_number(height, "shaft height", 0.3, 40)
    base_radius = _finite_number(base_radius, "shaft base radius", 0.08, 5)
    top_radius = _finite_number(top_radius, "shaft top radius", 0.08, 5)
    flute_depth = _finite_number(flute_depth, "flute depth", 0, 0.2)
    entasis = _finite_number(entasis, "entasis", 0, 0.2)
    if type(flutes) is not int or (flutes != 0 and not 8 <= flutes <= 32):
        raise GeometryError("Flutes must be 0 or an integer in [8, 32]")
    if flute_depth >= min(base_radius, top_radius) * 0.22:
        raise GeometryError("Flute depth would cut too deeply into the shaft")
    if entasis >= min(base_radius, top_radius) * 0.2:
        raise GeometryError("Entasis exceeds the classical shaft radius budget")
    sectors = max(48, flutes * 8)
    levels = (0.0, 0.035, 0.11, 0.26, 0.45, 0.63, 0.8, 0.94, 1.0)
    vertices: list[tuple[float, float, float]] = []
    for t in levels:
        outer = base_radius + (top_radius - base_radius) * t + entasis * math.sin(math.pi * t)
        for sector in range(sectors):
            angle = math.tau * sector / sectors
            groove = (flute_depth * (0.5 + 0.5 * math.cos(flutes * angle)) ** 1.65
                      if flutes else 0.0)
            radius = outer - groove
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), height * t))
    start_center, end_center = len(vertices), len(vertices) + 1
    vertices.extend(((0.0, 0.0, 0.0), (0.0, 0.0, height)))
    faces: list[tuple[int, ...]] = []
    for ring in range(len(levels) - 1):
        lower, upper = ring * sectors, (ring + 1) * sectors
        for index in range(sectors):
            following = (index + 1) % sectors
            faces.append((lower + index, lower + following, upper + following, upper + index))
    top = (len(levels) - 1) * sectors
    for index in range(sectors):
        following = (index + 1) % sectors
        faces.append((start_center, following, index))
        faces.append((end_center, top + index, top + following))
    return Mesh(tuple(vertices), tuple(faces))


@lru_cache(maxsize=128)
def volute_scroll(*, radius: float = 0.18, tube_radius: float = 0.027,
                  turns: float = 1.25, segments: int = 96) -> Mesh:
    """Closed spiral relief tube in the X/Z plane, suitable for Ionic capitals.

    The spiral decreases its radius to leave clear space between turns; a
    circular cross-section creates real highlight/shadow volume in closeups.
    """
    radius = _finite_number(radius, "volute radius", 0.04, 3.0)
    tube_radius = _finite_number(tube_radius, "volute tube radius", 0.004, 0.3)
    turns = _finite_number(turns, "volute turns", 0.65, 1.75)
    if type(segments) is not int or not 32 <= segments <= 192:
        raise GeometryError("Volute segments must be 32–192")
    if tube_radius * 2 >= radius * 0.74 / turns:
        raise GeometryError("Volute tube is too thick for its spiral turn spacing")
    around = 12
    rings: list[tuple[float, float, float, float]] = []
    for index in range(segments + 1):
        t = index / segments
        angle = math.tau * turns * t
        r = radius * (1 - 0.74 * t)
        dr = -radius * 0.74
        dx = dr * math.cos(angle) - r * math.sin(angle) * math.tau * turns
        dz = dr * math.sin(angle) + r * math.cos(angle) * math.tau * turns
        length = math.hypot(dx, dz)
        rings.append((r * math.cos(angle), r * math.sin(angle), dx / length, dz / length))
    vertices: list[tuple[float, float, float]] = []
    for x, z, tx, tz in rings:
        for index in range(around):
            angle = math.tau * index / around
            vertices.append((x - tz * tube_radius * math.cos(angle),
                             tube_radius * math.sin(angle),
                             z + tx * tube_radius * math.cos(angle)))
    start_cap, end_cap = len(vertices), len(vertices) + 1
    vertices.extend(((rings[0][0], 0, rings[0][1]),
                     (rings[-1][0], 0, rings[-1][1])))
    faces: list[tuple[int, ...]] = []
    for ring in range(segments):
        lower, upper = ring * around, (ring + 1) * around
        for index in range(around):
            following = (index + 1) % around
            faces.append((lower + index, upper + index, upper + following, lower + following))
    for index in range(around):
        following = (index + 1) % around
        faces.append((start_cap, index, following))
        faces.append((end_cap, segments * around + following, segments * around + index))
    return Mesh(tuple(vertices), tuple(faces))


@lru_cache(maxsize=128)
def _classical_lathed_moldings(height: float, radius: float) -> tuple[Mesh, Mesh, Mesh, Mesh]:
    """Share identical base/neck/echinus mesh data across repeated orders."""
    lower_base = revolve_profile(((radius*1.19, 0.16), (radius*1.29, 0.205),
                                  (radius*1.31, 0.265), (radius*1.17, 0.325)),
                                 radial_segments=64)
    shaft_end = height - 0.58
    neck = revolve_profile(((radius*0.87, shaft_end-0.01),
                            (radius*0.94, shaft_end+0.055),
                            (radius*0.94, shaft_end+0.095)), radial_segments=64)
    bell = revolve_profile(((radius*0.91, height-0.50),
                            (radius*0.98, height-0.40),
                            (radius*1.23, height-0.27),
                            (radius*1.33, height-0.20)), radial_segments=64)
    cushion = revolve_profile(((radius*0.55, 0),
                               (radius*0.62, radius*0.06),
                               (radius*0.62, radius*0.22),
                               (radius*0.56, radius*0.30)), radial_segments=64)
    return lower_base, neck, bell, cushion


def classical_column(*, prefix: str, height: float = 4.55,
                     base_radius: float = 0.36,
                     order: str = "ionic") -> tuple[MeshInstance, ...]:
    """Composable Tuscan, Doric or Ionic-inspired column details in meters.

    Uses actual fluted solids, subtle entasis, moulded bases/echinus/abacus,
    and front/back spiral volutes on Ionic capitals. The orders are reusable
    *studies*, not measured archaeological reconstructions.
    """
    height = _finite_number(height, "column height", 2.0, 24)
    radius = _finite_number(base_radius, "column radius", 0.16, 2.5)
    if order not in ("tuscan", "doric", "ionic"):
        raise GeometryError("Column order must be tuscan, doric or ionic")
    if height < radius * 7:
        raise GeometryError("Column height must be at least seven shaft radii")
    parts: list[MeshInstance] = []
    shaft_start = 0.34 if order != "doric" else 0.045
    shaft_end = height - 0.58
    lower_base, neck, bell, cushion = _classical_lathed_moldings(height, radius)
    if order != "doric":
        parts.append(_part(f"{prefix}-base-plinth", (radius*2.75, radius*2.75, 0.16),
                           (0, 0, 0.08), "limestone-light"))
        parts.append(place_mesh(lower_base, name=f"{prefix}-base-torus",
                                material="limestone"))
    flutes = 0 if order == "tuscan" else 20 if order == "doric" else 24
    shaft = fluted_column_shaft(height=shaft_end-shaft_start, base_radius=radius,
                                top_radius=radius*0.875, flutes=flutes,
                                flute_depth=radius*0.055 if flutes else 0.0,
                                entasis=radius*0.035)
    parts.append(place_mesh(shaft, name=f"{prefix}-shaft", origin=(0, 0, shaft_start),
                            material="limestone"))
    parts.append(place_mesh(neck, name=f"{prefix}-capital-necking",
                            material="limestone-light"))
    parts.append(place_mesh(bell, name=f"{prefix}-capital-echinus",
                            material="limestone"))
    if order == "ionic":
        band = box_mesh(radius*3.05, radius*1.67, 0.11)
        parts.append(place_mesh(band, name=f"{prefix}-capital-volute-band",
                                origin=(0, 0, height-0.30), material="limestone-light"))
        scroll = volute_scroll(radius=radius*0.53, tube_radius=radius*0.067)
        for side in ("front", "back"):
            front = side == "front"
            facing_axes = ({"axis_x": (1, 0, 0), "axis_y": (0, 0, 1),
                            "axis_z": (0, -1, 0)} if front else
                           {"axis_x": (1, 0, 0), "axis_y": (0, 0, -1),
                            "axis_z": (0, 1, 0)})
            for index, x in enumerate((-radius*1.12, radius*1.12)):
                # A curved stone cushion behind the spiral makes this read as
                # a carved Ionic capital, rather than free-floating metal curls.
                parts.append(place_mesh(cushion,
                                        name=f"{prefix}-volute-backing-{side}-{index}",
                                        origin=(x, (-1 if front else 1)*radius*0.89,
                                                height-0.35),
                                        material="limestone", **facing_axes))
                # Mirror the outer scroll on each side of the capital while
                # keeping a right-handed rigid frame. Otherwise both spirals
                # curl in the same direction and the silhouette is lopsided.
                axes = ({"axis_x": (-1, 0, 0), "axis_y": (0, -1, 0),
                         "axis_z": (0, 0, 1)} if index == 0 else {})
                parts.append(place_mesh(scroll, name=f"{prefix}-volute-{side}-{index}",
                                        origin=(x, (-1 if front else 1)*radius*1.25,
                                                height-0.35), material="limestone-light",
                                        **axes))
    abacus = box_mesh(radius*3.15, radius*2.82, 0.16)
    parts.append(place_mesh(abacus, name=f"{prefix}-capital-abacus",
                            origin=(0, 0, height-0.08), material="limestone-light"))
    return tuple(parts)


@lru_cache(maxsize=128)
def turned_stone_baluster(*, height: float = 0.72, radius: float = 0.12) -> Mesh:
    """Solid turned-stone spindle with necks, swelling belly and square-cut ends."""
    height = _finite_number(height, "baluster height", 0.25, 3.0)
    radius = _finite_number(radius, "baluster radius", 0.035, 0.6)
    if height < radius * 4.5:
        raise GeometryError("Baluster height must be at least 4.5 radii")
    shape = ((0.60, 0), (0.85, 0.035), (0.90, 0.095), (0.63, 0.14),
             (0.54, 0.23), (0.78, 0.35), (1.0, 0.47), (1.0, 0.54),
             (0.78, 0.67), (0.54, 0.79), (0.63, 0.87), (0.90, 0.93),
             (0.85, 0.975), (0.60, 1.0))
    return revolve_profile(tuple((radius * r, height * z) for r, z in shape),
                           radial_segments=48)


def balustrade_assembly(*, prefix: str, length: float = 3.30,
                        height: float = 1.10, baluster_count: int = 7,
                        depth: float = 0.30) -> tuple[MeshInstance, ...]:
    """Modular stone railing: shaped spindles, continuous rails and end piers.

    Centered on local X, with its footing at Z=0 and facade facing -Y.
    Each element is an individually selectable, watertight mesh. Structural
    clearances are checked so repeated balusters cannot intersect each other.
    """
    length = _finite_number(length, "balustrade length", 0.8, 50)
    height = _finite_number(height, "balustrade height", 0.55, 3.0)
    depth = _finite_number(depth, "balustrade depth", 0.16, 1.0)
    if type(baluster_count) is not int or not 2 <= baluster_count <= 100:
        raise GeometryError("baluster_count must be an integer from 2 to 100")
    post_width = min(depth, height * 0.26)
    radius = min(depth * 0.38, height * 0.11)
    inner_span = length - 2 * post_width
    spacing = inner_span / (baluster_count + 1)
    if spacing <= radius * 2.2:
        raise GeometryError("Balusters would overlap: increase length or reduce count")
    rail_bottom = height * 0.19
    rail_top = height * 0.82
    spindle_height = rail_top - rail_bottom + height * 0.018
    spindle = turned_stone_baluster(height=spindle_height, radius=radius)

    items: list[MeshInstance] = [
        place_mesh(box_mesh(length, depth, rail_bottom),
                   name=f"{prefix}-bottom-rail", origin=(0, 0, rail_bottom/2),
                   material="limestone"),
        place_mesh(box_mesh(length, depth*1.12, height-rail_top),
                   name=f"{prefix}-coping-rail", origin=(0, 0, (height+rail_top)/2),
                   material="limestone-light"),
    ]
    for index in range(baluster_count):
        x = -inner_span/2 + (index+1)*spacing
        items.append(place_mesh(spindle, name=f"{prefix}-baluster-{index:02d}",
                                origin=(x, 0, rail_bottom-height*0.009),
                                material="limestone-light"))
    for side, sign in (("left", -1), ("right", 1)):
        x = sign * (length - post_width) / 2
        items.append(place_mesh(box_mesh(post_width, depth*1.25, height),
                                name=f"{prefix}-{side}-post", origin=(x, 0, height/2),
                                material="limestone"))
        items.append(place_mesh(box_mesh(post_width*1.2, depth*1.34, height*0.10),
                                name=f"{prefix}-{side}-post-cap",
                                origin=(x, 0, height*0.98), material="limestone-light"))
    return tuple(items)


def gutter_channel(length: float, *, radius: float = 0.105,
                   wall: float = 0.018, segments: int = 14) -> Mesh:
    """Half-round metal rain gutter with a real open U-shaped channel.

    The cross-section is annular, not a dark stripe texture. Place its local
    axes (Y, Z, X) so the trough extends along a roof's horizontal eave.
    """
    length = _finite_number(length, "gutter length", 0.2, 150)
    radius = _finite_number(radius, "gutter radius", 0.025, 0.5)
    wall = _finite_number(wall, "gutter wall", 0.003, 0.08)
    if wall >= radius / 2:
        raise GeometryError("Gutter metal wall must be thinner than half its radius")
    if type(segments) is not int or not 8 <= segments <= 64:
        raise GeometryError("Gutter radial segments must be 8–64")
    angles = [math.pi + math.pi * i / segments for i in range(segments + 1)]
    outer = [(radius * math.cos(angle), radius * math.sin(angle)) for angle in angles]
    inner = [((radius-wall) * math.cos(angle), (radius-wall) * math.sin(angle))
             for angle in reversed(angles)]
    return extrude_profile(tuple(outer + inner), length)


def rainwater_system(*, width: float, depth: float, eave_z: float,
                     overhang: float = 0.55) -> tuple[MeshInstance, ...]:
    """Reusable half-round gutters, front downspouts and connection elbows."""
    width = _finite_number(width, "building width", 2, 120)
    depth = _finite_number(depth, "building depth", 2, 80)
    eave_z = _finite_number(eave_z, "eave height", 1, 120)
    overhang = _finite_number(overhang, "roof overhang", 0.1, 2)
    axes = {"axis_x": (0, 1, 0), "axis_y": (0, 0, 1), "axis_z": (1, 0, 0)}
    trough = gutter_channel(width + overhang*2)
    parts = [
        place_mesh(trough, name="gutter-front-channel",
                   origin=(-overhang, -depth/2-overhang-0.015, eave_z+0.01),
                   material="rain-metal", **axes),
        place_mesh(trough, name="gutter-back-channel",
                   origin=(-overhang, depth/2+overhang+0.015, eave_z+0.01),
                   material="rain-metal", **axes),
    ]
    downspout = revolve_profile(((0.062, 0), (0.062, eave_z - 0.17)), radial_segments=16)
    collar = revolve_profile(((0.080, 0), (0.080, 0.055)), radial_segments=16)
    y = -depth/2 - 0.32
    for i, x in enumerate((0.26, width-0.26)):
        parts.append(place_mesh(downspout, name=f"gutter-downspout-{i:02d}",
                                origin=(x, y, 0.06), material="rain-metal"))
        # Elbow bridges the overhanging gutter and the wall-mounted pipe.
        connection_y = (-depth/2 - overhang - 0.015 + y) / 2
        connection_span = abs(-depth/2 - overhang - 0.015 - y) + 0.1
        parts.append(_part(f"gutter-elbow-{i:02d}", (0.135, connection_span, 0.135),
                           (x, connection_y, eave_z-0.045), "rain-metal"))
        for j, z in enumerate((1.1, eave_z/2, eave_z-0.85)):
            parts.append(place_mesh(collar, name=f"gutter-collar-{i:02d}-{j:02d}",
                                    origin=(x, y, z), material="rain-metal"))
    return tuple(parts)


def gable_roof(*, width: float, depth: float, eave_z: float, rise: float,
               overhang: float = 0.5, panel_thickness: float = 0.12,
               tile_rows: int = 8, tile_columns: int = 24) -> tuple[MeshInstance, ...]:
    """Closed pitched slabs, triangular gables, individual overlapping tile rows.

    Ridge runs in +X. Tiles follow the actual roof slope using orthonormal
    bases, rather than being flattened onto a horizontal roof placeholder.
    """
    width = _finite_number(width, "roof width", 2, 120)
    depth = _finite_number(depth, "roof depth", 2, 80)
    eave_z = _finite_number(eave_z, "eave height", 1, 120)
    rise = _finite_number(rise, "roof rise", 0.3, 30)
    overhang = _finite_number(overhang, "roof overhang", 0.1, 2)
    panel_thickness = _finite_number(panel_thickness, "roof panel thickness", 0.03, 1)
    if (type(tile_rows) is not int or not 2 <= tile_rows <= 30
            or type(tile_columns) is not int or not 2 <= tile_columns <= 100
            or tile_rows * tile_columns * 2 > 2000):
        raise GeometryError("Tile grid must be 2–30 rows, 2–100 columns and <= 2000 tiles")
    half = depth / 2 + overhang
    ridge = eave_z + rise
    roof_span = width + 2 * overhang
    axes = {"axis_x": (0, 1, 0), "axis_y": (0, 0, 1), "axis_z": (1, 0, 0)}
    front = extrude_profile(((-half, eave_z), (0, ridge), (0, ridge-panel_thickness),
                             (-half, eave_z-panel_thickness)), roof_span)
    back = extrude_profile(((0, ridge), (half, eave_z),
                            (half, eave_z-panel_thickness), (0, ridge-panel_thickness)), roof_span)
    parts = [
        place_mesh(front, name="roof-front-slab", origin=(-overhang, 0, 0),
                   material="roof-slate", **axes),
        place_mesh(back, name="roof-back-slab", origin=(-overhang, 0, 0),
                   material="roof-slate", **axes),
    ]
    gable = extrude_profile(((-depth/2, eave_z-0.025),
                            (depth/2, eave_z-0.025), (0, eave_z + rise - 0.13)), 0.27)
    parts.extend((
        place_mesh(gable, name="roof-gable-left", origin=(0, 0, 0),
                   material="wall-stone", **axes),
        place_mesh(gable, name="roof-gable-right", origin=(width - 0.27, 0, 0),
                   material="wall-stone", **axes),
    ))
    slope_len = math.hypot(half, rise)
    sin_theta, cos_theta = rise / slope_len, half / slope_len
    step_x = roof_span / tile_columns
    step_s = slope_len / tile_rows
    for side in ("front", "back"):
        sy = -1 if side == "front" else 1
        slope_axis = (0, cos_theta, -sy * sin_theta)
        normal_axis = (0, sy * sin_theta, cos_theta)
        for row in range(tile_rows):
            t = (row + 0.5) / tile_rows
            y = (-half + t * half) if side == "front" else (t * half)
            z = (eave_z + rise * t) if side == "front" else (ridge - rise * t)
            for col in range(tile_columns):
                left = -overhang + col * step_x - (step_x * 0.5 if row % 2 else 0)
                right = left + step_x
                left, right = max(-overhang, left), min(width + overhang, right)
                if right - left < 0.035:
                    continue
                x = (left + right) / 2
                tile = box_mesh(right - left - min(0.012, (right-left)*0.15),
                                step_s * 0.97, 0.035)
                # Stable integer mixing avoids unnaturally regular diagonal
                # stripes caused by (row+col) % N material alternation.
                shade = ((col * 73856093) ^ (row * 19349663)
                         ^ (83492791 if side == "back" else 0)) & 0xFFFFFFFF
                shade ^= shade >> 13
                shade = (shade * 0x85EBCA6B) & 0xFFFFFFFF
                shade ^= shade >> 16
                parts.append(place_mesh(tile, name=f"roof-{side}-tile-{row:02d}-{col:02d}",
                    origin=(x, y + normal_axis[1]*0.025, z + normal_axis[2]*0.025),
                    axis_x=(1, 0, 0), axis_y=slope_axis, axis_z=normal_axis,
                    material="roof-slate-light" if shade % 13 == 0 else "roof-slate"))
    # Ridge cap hides the meeting seam between pitched panel edges.
    parts.append(_part("roof-ridge-cap", (roof_span, 0.20, 0.16),
                       (width/2, 0, ridge+0.035), "roof-slate-light"))
    return tuple(parts)
