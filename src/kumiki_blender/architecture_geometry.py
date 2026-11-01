"""Blender-independent, manifold architectural solids and lightweight instancing.

All geometry uses meters and a right-handed, Z-up coordinate system. Shapes
are closed volumes; profile caps are triangulated (including concave outlines)
instead of relying on a renderer to guess their topology.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
_EPS = 1e-10
_IDENTITY: tuple[Vec3, Vec3, Vec3] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


class GeometryError(ValueError):
    """An architectural profile or solid violates topology/size constraints."""


@dataclass(frozen=True)
class Mesh:
    vertices: tuple[Vec3, ...]
    faces: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MeshReport:
    vertices: int
    faces: int
    boundary_edges: int
    nonmanifold_edges: int
    volume_m3: float


@dataclass(frozen=True)
class MeshInstance:
    name: str
    mesh: Mesh
    origin: Vec3
    axis_x: Vec3
    axis_y: Vec3
    axis_z: Vec3
    material: str


def _finite_number(value: object, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise GeometryError(f"{label} must be finite")
    result = float(value)
    if not minimum <= result <= maximum:
        raise GeometryError(f"{label} must be within [{minimum}, {maximum}]")
    return result


def _vector(value: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise GeometryError(f"{label} must have {size} coordinates")
    return tuple(_finite_number(v, f"{label}[{i}]", -100_000, 100_000) for i, v in enumerate(value))


def _cross2(a: Vec2, b: Vec2, c: Vec2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _signed_area(points: Sequence[Vec2]) -> float:
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1]
                     for a, b in zip(points, (*points[1:], points[0])))


def _on_segment(a: Vec2, b: Vec2, p: Vec2) -> bool:
    return (abs(_cross2(a, b, p)) <= _EPS
            and min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS
            and min(a[1], b[1]) - _EPS <= p[1] <= max(a[1], b[1]) + _EPS)


def _intersects(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool:
    ab_c, ab_d, cd_a, cd_b = _cross2(a, b, c), _cross2(a, b, d), _cross2(c, d, a), _cross2(c, d, b)
    if ((ab_c > _EPS and ab_d < -_EPS or ab_c < -_EPS and ab_d > _EPS)
            and (cd_a > _EPS and cd_b < -_EPS or cd_a < -_EPS and cd_b > _EPS)):
        return True
    return any((_on_segment(a, b, c), _on_segment(a, b, d),
                _on_segment(c, d, a), _on_segment(c, d, b)))


def _profile(points: Sequence[Vec2]) -> tuple[Vec2, ...]:
    if not isinstance(points, (tuple, list)) or not 3 <= len(points) <= 512:
        raise GeometryError("A profile requires 3–512 vertices")
    vertices = tuple(_vector(p, 2, "profile point") for p in points)
    if len(set(vertices)) != len(vertices):
        raise GeometryError("Profile vertices must be unique")
    n = len(vertices)
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        if math.dist(a, b) < 1e-6:
            raise GeometryError("Profile edges must have nonzero length")
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _intersects(a, b, vertices[j], vertices[(j + 1) % n]):
                raise GeometryError("Profile self-intersects or touches itself")
    area = _signed_area(vertices)
    if abs(area) < _EPS:
        raise GeometryError("Profile has zero area")
    if any(abs(_cross2(vertices[i - 1], vertices[i], vertices[(i + 1) % n])) < _EPS for i in range(n)):
        raise GeometryError("Remove collinear profile corners")
    return vertices if area > 0 else tuple(reversed(vertices))


def _inside_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool:
    return (_cross2(a, b, p) >= -_EPS and _cross2(b, c, p) >= -_EPS
            and _cross2(c, a, p) >= -_EPS)


def _triangulate(points: tuple[Vec2, ...]) -> list[tuple[int, int, int]]:
    """Ear-clipping for simple concave, counter-clockwise polygons."""
    remaining = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        for index in range(len(remaining)):
            a, b, c = (remaining[(index - 1) % len(remaining)],
                       remaining[index], remaining[(index + 1) % len(remaining)])
            if _cross2(points[a], points[b], points[c]) <= _EPS:
                continue
            if any(_inside_triangle(points[other], points[a], points[b], points[c])
                   for other in remaining if other not in (a, b, c)):
                continue
            triangles.append((a, b, c))
            remaining.pop(index)
            break
        else:
            raise GeometryError("Could not triangulate profile without degeneracy")
    triangles.append(tuple(remaining))
    return triangles


def extrude_profile(points_xy: Sequence[Vec2], depth: float) -> Mesh:
    """Extrude any *simple* planar polygon from Z=0 to Z=depth."""
    depth = _finite_number(depth, "depth", 1e-5, 10_000)
    points = _profile(points_xy)
    triangles = _triangulate(points)
    count = len(points)
    vertices = tuple((x, y, z) for z in (0.0, depth) for x, y in points)
    faces: list[tuple[int, ...]] = []
    faces.extend((c, b, a) for a, b, c in triangles)  # Bottom points toward -Z.
    faces.extend((a + count, b + count, c + count) for a, b, c in triangles)
    faces.extend((i, (i + 1) % count, (i + 1) % count + count, i + count)
                 for i in range(count))
    return Mesh(vertices, tuple(faces))


def revolve_profile(points_rz: Sequence[Vec2], *, radial_segments: int = 32) -> Mesh:
    """Turn a positive-radius, upward R/Z silhouette into a capped solid.

    Use for columns, balusters, plinths, posts and decorative finials.
    """
    if not isinstance(points_rz, (tuple, list)) or not 2 <= len(points_rz) <= 512:
        raise GeometryError("A revolved profile requires 2–512 points")
    if type(radial_segments) is not int or not 8 <= radial_segments <= 256:
        raise GeometryError("radial_segments must be an integer from 8 to 256")
    profile = tuple(_vector(point, 2, "radius/height") for point in points_rz)
    for index, (radius, z) in enumerate(profile):
        if not 0.001 <= radius <= 10_000:
            raise GeometryError("Revolved radii must be positive")
        if index and z - profile[index - 1][1] < 1e-5:
            raise GeometryError("Revolved heights must be strictly increasing")
    vertices = tuple((radius * math.cos(math.tau * sector / radial_segments),
                      radius * math.sin(math.tau * sector / radial_segments), z)
                     for radius, z in profile for sector in range(radial_segments))
    faces: list[tuple[int, ...]] = [tuple(reversed(range(radial_segments)))]
    for ring in range(len(profile) - 1):
        lower, upper = ring * radial_segments, (ring + 1) * radial_segments
        for sector in range(radial_segments):
            nxt = (sector + 1) % radial_segments
            faces.append((lower + sector, lower + nxt, upper + nxt, upper + sector))
    last = (len(profile) - 1) * radial_segments
    faces.append(tuple(last + i for i in range(radial_segments)))
    return Mesh(vertices, tuple(faces))


def loft_elliptical_sections(sections: Sequence[tuple[float, float, float, float, float]], *,
                             axis: str = "z", radial_segments: int = 48,
                             subdivisions: int = 1,
                             cross_section_exponent: float = 2.0) -> Mesh:
    """Skin capped elliptical rings along X, Y or Z into a watertight solid.

    Each section is ``(axis_position, center_a, center_b, radius_a, radius_b)``.
    Cross-plane coordinates are (Y,Z) for X, (Z,X) for Y, and (X,Y) for Z.
    Radii can change independently to sculpt a taper, shoulders, anatomy,
    rounded masonry, or a carved stone silhouette without Blender booleans.
    """
    if axis not in ("x", "y", "z"):
        raise GeometryError("Loft axis must be x, y or z")
    if not isinstance(sections, (tuple, list)) or not 2 <= len(sections) <= 128:
        raise GeometryError("Loft requires 2–128 elliptical sections")
    if type(radial_segments) is not int or not 8 <= radial_segments <= 128:
        raise GeometryError("Loft radial_segments must be an integer from 8 to 128")
    if type(subdivisions) is not int or not 1 <= subdivisions <= 8:
        raise GeometryError("Loft subdivisions must be an integer from 1 to 8")
    exponent = _finite_number(cross_section_exponent, "cross_section_exponent", 1.2, 6.0)
    rings = tuple(_vector(section, 5, "elliptical section") for section in sections)
    for index, section in enumerate(rings):
        if min(section[3:]) < 0.001:
            raise GeometryError("Elliptical radii must be at least 0.001 m")
        if index and section[0] - rings[index - 1][0] < 1e-5:
            raise GeometryError("Loft sections must strictly advance along their axis")

    if subdivisions > 1:
        # Nonuniform cubic Hermite interpolation for organic sculpture;
        # axial positions remain linear/strictly ordered. Tangents use
        # neighboring section locations, so rounded anatomy retains its
        # silhouette without Blender-specific subdivision modifiers.
        dense = []
        for i in range(len(rings)-1):
            low, high = rings[i], rings[i+1]
            delta = high[0]-low[0]
            for step in range(subdivisions):
                u = step/subdivisions
                h00 = 2*u**3 - 3*u*u + 1
                h10 = u**3 - 2*u*u + u
                h01 = -2*u**3 + 3*u*u
                h11 = u**3 - u*u
                values = [low[0] + delta*u]
                for field in range(1, 5):
                    before = rings[max(0, i-1)]
                    after = rings[min(len(rings)-1, i+2)]
                    d0 = (high[field]-before[field])/(high[0]-before[0])*delta
                    d1 = (after[field]-low[field])/(after[0]-low[0])*delta
                    value = (h00*low[field] + h10*d0 +
                             h01*high[field] + h11*d1)
                    values.append(max(0.001, value) if field >= 3 else value)
                dense.append(tuple(values))
        dense.append(rings[-1])
        rings = tuple(dense)

    vertices: list[Vec3] = []
    for p, a, b, ra, rb in rings:
        for sector in range(radial_segments):
            angle = math.tau * sector / radial_segments
            # Exponent 2 is an ellipse. Higher exponents build flatter sides
            # and squarer shoulders for carved monuments and bedrock masses,
            # without introducing discontinuous edges or empty open shells.
            c, s = math.cos(angle), math.sin(angle)
            u = ra * math.copysign(abs(c)**(2/exponent), c)
            v = rb * math.copysign(abs(s)**(2/exponent), s)
            vertices.append(((p, a+u, b+v) if axis == "x" else
                             (b+v, p, a+u) if axis == "y" else
                             (a+u, b+v, p)))
    faces: list[tuple[int, ...]] = [tuple(reversed(range(radial_segments)))]
    for ring in range(len(rings)-1):
        low, high = ring*radial_segments, (ring+1)*radial_segments
        for sector in range(radial_segments):
            nxt = (sector+1) % radial_segments
            faces.append((low+sector, low+nxt, high+nxt, high+sector))
    last = (len(rings)-1)*radial_segments
    faces.append(tuple(last+i for i in range(radial_segments)))
    return Mesh(tuple(vertices), tuple(faces))


def semicircular_arch(inner_radius: float, outer_radius: float, depth: float, *, segments: int = 32) -> Mesh:
    """A closed semicircular archivolt, with an actual open central void.

    This uses a concave planar U-profile, not a boolean or a floating shell.
    In local coordinates the arch rises along +Y and thickness spans +Z.
    """
    inner_radius = _finite_number(inner_radius, "inner_radius", 0.01, 10000)
    outer_radius = _finite_number(outer_radius, "outer_radius", 0.01, 10000)
    if outer_radius - inner_radius < 1e-4:
        raise GeometryError("outer_radius must exceed inner_radius by at least 0.0001 m")
    if type(segments) is not int or not 8 <= segments <= 128:
        raise GeometryError("segments must be an integer from 8 to 128")
    outer = [(outer_radius * math.cos(math.pi * i / segments),
              outer_radius * math.sin(math.pi * i / segments)) for i in range(segments + 1)]
    inner = [(inner_radius * math.cos(math.pi * i / segments),
              inner_radius * math.sin(math.pi * i / segments)) for i in range(segments, -1, -1)]
    return extrude_profile(tuple(outer + inner), depth)


def arch_voussoirs(inner_radius: float, outer_radius: float, depth: float, *, stone_count: int = 11,
                   joint_gap_m: float = 0.006, segments_per_stone: int = 4) -> tuple[Mesh, ...]:
    """Build independent wedge-shaped masonry blocks with visible radial joints.

    The gap is specified as an arc distance at the mean radius. Each block is
    a real manifold solid; no surface-only seam or shader trick is involved.
    """
    inner = _finite_number(inner_radius, "inner_radius", 0.01, 10000)
    outer = _finite_number(outer_radius, "outer_radius", 0.01, 10000)
    depth = _finite_number(depth, "depth", 1e-5, 10000)
    if outer - inner < 1e-4:
        raise GeometryError("outer_radius must exceed inner_radius")
    if type(stone_count) is not int or not 3 <= stone_count <= 64:
        raise GeometryError("stone_count must be an integer from 3 to 64")
    if type(segments_per_stone) is not int or not 2 <= segments_per_stone <= 16:
        raise GeometryError("segments_per_stone must be in [2, 16]")
    gap = _finite_number(joint_gap_m, "joint_gap_m", 0, 1000) / ((inner + outer) / 2)
    sweep = math.pi / stone_count
    if gap > sweep * 0.2:
        raise GeometryError("Joint gap is too wide for arch stones")
    pieces = []
    for stone in range(stone_count):
        start = sweep * stone + gap / 2
        end = sweep * (stone + 1) - gap / 2
        angles = [start + (end - start) * i / segments_per_stone for i in range(segments_per_stone + 1)]
        outer_arc = [(outer * math.cos(theta), outer * math.sin(theta)) for theta in angles]
        inner_arc = [(inner * math.cos(theta), inner * math.sin(theta)) for theta in reversed(angles)]
        pieces.append(extrude_profile(tuple(outer_arc + inner_arc), depth))
    return tuple(pieces)


def pointed_arch_voussoirs(half_span: float, rise: float, band_width: float, depth: float, *,
                           stone_count: int = 11, joint_gap_m: float = 0.006,
                           segments_per_stone: int = 6) -> tuple[Mesh, ...]:
    """Independent, mortar-gapped Gothic arch stones with two true circular arcs.

    The intrados rises from (+/-half_span, 0) to (0, rise). Each side is a
    different circular arc, meeting at an actual pointed apex; the extrados
    uses concentric, wider arcs. Coordinates are local XY with depth along Z.
    No booleans, nonphysical planar decals, or overlapping arch-sized slabs.
    """
    a = _finite_number(half_span, "half_span", 0.1, 1000)
    h = _finite_number(rise, "rise", 0.1, 1000)
    t = _finite_number(band_width, "band_width", 0.01, 100)
    depth = _finite_number(depth, "depth", 1e-5, 10000)
    if h <= a * 1.01:
        raise GeometryError("Pointed arch rise must exceed the half-span by at least 1%")
    if h > a * 5:
        raise GeometryError("Pointed arch rise cannot exceed five times the half-span")
    if t >= a * 0.65:
        raise GeometryError("Pointed arch band is too wide for its span")
    if type(stone_count) is not int or not 3 <= stone_count <= 64:
        raise GeometryError("stone_count must be an integer from 3 to 64")
    if type(segments_per_stone) is not int or not 2 <= segments_per_stone <= 16:
        raise GeometryError("segments_per_stone must be in [2, 16]")

    center_offset = (h*h - a*a) / (2*a)
    inner_radius = a + center_offset
    outer_radius = inner_radius + t
    # Parametrize each side by its own arc angle, reserving u=0.5 for
    # the apex. Stone boundaries occur at matching fractions of arc length.
    inner_half_sweep = math.pi - math.acos(-center_offset / inner_radius)
    mean_half_length = (inner_radius + outer_radius) / 2 * inner_half_sweep
    joint = _finite_number(joint_gap_m, "joint_gap_m", 0, 1000) / (4 * mean_half_length)
    if joint > 0.10 / stone_count:
        raise GeometryError("Joint gap is too wide for pointed arch stones")

    def curve(u: float, radius: float) -> Vec2:
        if u <= 0.5:
            apex_angle = math.acos(-center_offset / radius)
            angle = math.pi + (apex_angle - math.pi) * u * 2
            return (center_offset + radius * math.cos(angle), radius * math.sin(angle))
        apex_angle = math.acos(center_offset / radius)
        angle = apex_angle * (2 - 2*u)
        return (-center_offset + radius * math.cos(angle), radius * math.sin(angle))

    stones: list[Mesh] = []
    for index in range(stone_count):
        start, end = index/stone_count + joint, (index+1)/stone_count - joint
        steps = [start + (end-start)*i/segments_per_stone for i in range(segments_per_stone+1)]
        if start < 0.5 < end and not any(abs(u-0.5) < 1e-10 for u in steps):
            steps.append(0.5)
            steps.sort()
        outer_arc = [curve(u, outer_radius) for u in steps]
        inner_arc = [curve(u, inner_radius) for u in reversed(steps)]
        stones.append(extrude_profile(tuple(outer_arc + inner_arc), depth))
    return tuple(stones)


def audit_mesh(mesh: Mesh) -> MeshReport:
    """Report topology errors and signed volume; malformed indexing raises."""
    if not isinstance(mesh, Mesh) or not mesh.vertices or not mesh.faces:
        raise GeometryError("Mesh has no geometry")
    for v in mesh.vertices:
        _vector(v, 3, "mesh vertex")
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    volume = 0.0
    for face in mesh.faces:
        if len(face) < 3 or len(set(face)) != len(face) or any(type(i) is not int or not 0 <= i < len(mesh.vertices) for i in face):
            raise GeometryError("Face has repeated or out-of-range vertex indices")
        normal_sum = [0.0, 0.0, 0.0]
        for i, j in zip(face, (*face[1:], face[0])):
            a, b = mesh.vertices[i], mesh.vertices[j]
            normal_sum[0] += (a[1] - b[1]) * (a[2] + b[2])
            normal_sum[1] += (a[2] - b[2]) * (a[0] + b[0])
            normal_sum[2] += (a[0] - b[0]) * (a[1] + b[1])
            edges.setdefault((min(i, j), max(i, j)), []).append((i, j))
        if math.sqrt(sum(n * n for n in normal_sum)) < _EPS:
            raise GeometryError("Mesh contains a zero-area face")
        a = mesh.vertices[face[0]]
        for index in range(1, len(face) - 1):
            b, c = mesh.vertices[face[index]], mesh.vertices[face[index + 1]]
            volume += (a[0] * (b[1] * c[2] - b[2] * c[1])
                       + a[1] * (b[2] * c[0] - b[0] * c[2])
                       + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6
    boundary = sum(len(pair) == 1 for pair in edges.values())
    nonmanifold = sum(len(pair) > 2 or (len(pair) == 2 and pair[0] == pair[1]) for pair in edges.values())
    return MeshReport(len(mesh.vertices), len(mesh.faces), boundary, nonmanifold, volume)


def place_mesh(mesh: Mesh, *, name: str, origin: Vec3 = (0, 0, 0),
               axis_x: Vec3 = _IDENTITY[0], axis_y: Vec3 = _IDENTITY[1],
               axis_z: Vec3 = _IDENTITY[2], material: str = "painted-steel") -> MeshInstance:
    """Record a right-handed rigid transform without duplicating mesh data."""
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", name) or len(name) > 100:
        raise GeometryError("Instance name must be a compact identifier")
    if not isinstance(material, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", material):
        raise GeometryError("Material name must be a catalog identifier")
    origin = _vector(origin, 3, "origin")
    x, y, z = (_vector(axis, 3, "basis axis") for axis in (axis_x, axis_y, axis_z))
    if any(abs(sum(c * c for c in axis) - 1) > 1e-6 for axis in (x, y, z)):
        raise GeometryError("Basis axes must have unit length")
    if any(abs(sum(a * b for a, b in zip(aa, bb))) > 1e-6 for aa, bb in ((x, y), (y, z), (z, x))):
        raise GeometryError("Basis axes must be perpendicular")
    cross = (x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0])
    if math.dist(cross, z) > 1e-6:
        raise GeometryError("Basis axes must be right-handed")
    return MeshInstance(name, mesh, origin, x, y, z, material)


def repeat_linear(mesh: Mesh, *, prefix: str, count: int, step: Vec3,
                  origin: Vec3 = (0, 0, 0), material: str = "painted-steel") -> tuple[MeshInstance, ...]:
    """Create named, reusable instances at fixed meter spacing."""
    if type(count) is not int or not 1 <= count <= 1000:
        raise GeometryError("count must be in [1, 1000]")
    step, origin = _vector(step, 3, "step"), _vector(origin, 3, "origin")
    if count > 1 and sum(value * value for value in step) < 1e-12:
        raise GeometryError("Repeated instances need a nonzero step")
    return tuple(place_mesh(mesh, name=f"{prefix}-{i:03d}",
                            origin=tuple(origin[j] + i * step[j] for j in range(3)), material=material)
                 for i in range(count))


def world_vertices(instance: MeshInstance) -> tuple[Vec3, ...]:
    """Expand an instance into world coordinates for bounds and tests only."""
    return tuple(tuple(instance.origin[k] + p[0] * instance.axis_x[k]
                       + p[1] * instance.axis_y[k] + p[2] * instance.axis_z[k]
                       for k in range(3)) for p in instance.mesh.vertices)
