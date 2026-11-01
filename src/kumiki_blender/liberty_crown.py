"""Perforated classical Liberty diadem and seven faceted copper rays.

Unlike the legacy dark vertical rods, this annular shell has *real* through
openings.  It is a sculptural approximation at the base figure's scale, not a
measured restoration of the monument's copper skin or internal structure.
"""

from __future__ import annotations

import math

from .architecture_geometry import Mesh


WINDOW_COUNT = 25
RAY_COUNT = 7
LOWER_RAIL = 28.62
UPPER_RAIL = 29.28


def crown_band() -> Mesh:
    """Closed elliptical annulus with 25 true rectangular open window cells."""
    # Each angular interval is a narrow mullion followed by a wider aperture.
    # The top and bottom rails continue over both interval types. All exterior,
    # interior and opening wall faces share vertices along their common edges.
    angles = tuple(math.tau*k/WINDOW_COUNT + offset
                   for k in range(WINDOW_COUNT) for offset in (0., .072))
    levels = (LOWER_RAIL, 28.76, 29.11, UPPER_RAIL)
    column_count = len(angles)
    verts: list[tuple[float, float, float]] = []
    indices: dict[tuple[int, int, int], int] = {}

    def at(column: int, level: int, shell: int) -> int:
        column %= column_count
        key = (column, level, shell)
        if key not in indices:
            theta = angles[column]
            rx, ry = ((1.50, .985) if shell == 0 else (1.37, .855))
            indices[key] = len(verts)
            verts.append((rx*math.cos(theta), .22+ry*math.sin(theta), levels[level]))
        return indices[key]

    def occupied(col: int, row: int) -> bool:
        return 0 <= row <= 2 and (row != 1 or col % 2 == 0)

    faces: list[tuple[int, ...]] = []
    for column in range(column_count):
        following = (column+1) % column_count
        for row in range(3):
            if not occupied(column, row):
                continue
            a, b = at(column, row, 0), at(following, row, 0)
            c, d = at(following, row+1, 0), at(column, row+1, 0)
            ai, bi = at(column, row, 1), at(following, row, 1)
            ci, di = at(following, row+1, 1), at(column, row+1, 1)
            faces.extend(((a, b, c, d), (di, ci, bi, ai)))
            # Oriented exterior perimeter edges become the thickness walls of
            # the 25 apertures and both circular rims.
            edges = (
                ((a, b), (ai, bi), not occupied(column, row-1)),
                ((b, c), (bi, ci), not occupied(following, row)),
                ((c, d), (ci, di), not occupied(column, row+1)),
                ((d, a), (di, ai), not occupied(column-1, row)),
            )
            for (p, q), (pi, qi), boundary in edges:
                if boundary:
                    faces.append((p, pi, qi, q))
    return Mesh(tuple(verts), tuple(faces))


def crown_rim(*, upper: bool) -> Mesh:
    """A continuous beaded edge (elliptical torus), with no seam-end caps."""
    around, across = 128, 12
    z = UPPER_RAIL if upper else LOWER_RAIL
    width = .084 if upper else .076
    vertices = []
    for i in range(around):
        theta = math.tau*i/around
        for j in range(across):
            phi = math.tau*j/across
            vertices.append(((1.465+width*math.cos(phi))*math.cos(theta),
                             .22+(.925+width*math.cos(phi))*math.sin(theta),
                             z+width*math.sin(phi)))
    faces = []
    for i in range(around):
        for j in range(across):
            faces.append((i*across+j,
                          ((i+1)%around)*across+j,
                          ((i+1)%around)*across+(j+1)%across,
                          i*across+(j+1)%across))
    return Mesh(tuple(vertices), tuple(faces))


def crown_ray(index: int) -> Mesh:
    """A broad, thin tapering copper blade instead of a rounded spike."""
    if not 0 <= index < RAY_COUNT:
        raise ValueError("Crown ray index must be between zero and six")
    degrees = (9, 36, 63, 90, 117, 144, 171)[index]
    theta = math.radians(degrees)
    start_x = 1.25*math.cos(theta)
    start_z = 29.20+.18*math.sin(theta)
    end_x = 3.48*math.cos(theta)
    end_z = 29.08+3.55*math.sin(theta)
    dx, dz = end_x-start_x, end_z-start_z
    length = math.hypot(dx, dz)
    side = (-dz/length, dx/length)
    rings = []
    for t, width, thickness in (
        (0., .46, .150), (.10, .57, .140), (.25, .42, .120),
        (.53, .29, .090), (.78, .15, .060), (1., .014, .026),
    ):
        x, z = start_x+dx*t, start_z+dz*t
        # A raised center fold and shallow back give the copper ray six
        # distinct surfaces, rather than one featureless paper-flat front.
        for lateral, depth in ((-1, -.14), (0, -.79), (1, -.14),
                               (1, .48), (0, .34), (-1, .48)):
            rings.append((x+side[0]*width*lateral/2,
                          .16+thickness*depth,
                          z+side[1]*width*lateral/2))
    ring_size = 6
    faces = [tuple(range(ring_size))]
    for segment in range(5):
        for k in range(ring_size):
            faces.append((segment*ring_size+k, (segment+1)*ring_size+k,
                          (segment+1)*ring_size+(k+1)%ring_size,
                          segment*ring_size+(k+1)%ring_size))
    faces.append(tuple(reversed(range(5*ring_size, 6*ring_size))))
    return Mesh(tuple(rings), tuple(faces))


def crown_detail_parts() -> tuple[tuple[str, Mesh, str], ...]:
    parts: list[tuple[str, Mesh, str]] = [
        ("crown-diadem-perforated", crown_band(), "patina-shadow"),
        ("crown-rim-lower", crown_rim(upper=False), "patina"),
        ("crown-rim-upper", crown_rim(upper=True), "patina-light"),
    ]
    parts.extend((f"crown-ray-{i}", crown_ray(i), "patina-light")
                 for i in range(RAY_COUNT))
    return tuple(parts)
