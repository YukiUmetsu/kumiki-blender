"""Reference-led sandal, broken shackle and chain detail for Liberty's base.

The NPS identifies a raised right heel, a broken shackle beside that foot,
and a chain reappearing at the left foot with a fractured end link. HAER 255
and 256 show the actual installed, riveted flattened copper straps and toes.

These are original **illustrative** modeled forms, not a measured surface or
the National Park Service's laser scan. The statue faces -Y and her right is
world -X. All surfaces are closed, editable mesh solids in meters.
"""

from __future__ import annotations

import math

from .architecture_geometry import Mesh


def _rounded_section(angle: float, exponent: float = .63) -> tuple[float, float]:
    """A softened rectangular profile, distinct from a cylindrical chain."""
    c, s = math.cos(angle), math.sin(angle)
    return (math.copysign(abs(c)**exponent, c),
            math.copysign(abs(s)**exponent, s))


def _band_sweep(path: tuple[tuple[float, float, float], ...], *,
                width: float, height: float, closed: bool = False,
                sectors: int = 12,
                taper: tuple[float, ...] | None = None) -> Mesh:
    """Broad copper strap along an XY path, with bevelled square-ish edges."""
    if len(path) < 3 or min(width, height) <= 0:
        raise ValueError("strap requires at least three points and positive radii")
    count = len(path)
    if taper is not None and (len(taper) != count or min(taper) <= 0):
        raise ValueError("taper must contain one positive radius factor per path point")
    vertices: list[tuple[float, float, float]] = []
    for i, (x, y, z) in enumerate(path):
        a = path[(i-1) % count] if closed or i else path[0]
        b = path[(i+1) % count] if closed or i < count-1 else path[-1]
        dx, dy = b[0]-a[0], b[1]-a[1]
        mag = math.hypot(dx, dy)
        if mag < 1e-9:
            raise ValueError("strap path has a degenerate tangent")
        nx, ny = dy/mag, -dx/mag
        radius = taper[i] if taper is not None else 1.
        for j in range(sectors):
            radial, vertical = _rounded_section(math.tau*j/sectors)
            vertices.append((x+nx*width*radius*radial,
                             y+ny*width*radius*radial,
                             z+height*radius*vertical))
    faces: list[tuple[int, ...]] = []
    for i in range(count if closed else count-1):
        nxt = (i+1) % count
        for j in range(sectors):
            k = (j+1) % sectors
            faces.append((i*sectors+j, nxt*sectors+j,
                          nxt*sectors+k, i*sectors+k))
    if not closed:
        faces.insert(0, tuple(range(sectors)))
        last = (count-1)*sectors
        faces.append(tuple(last+j for j in reversed(range(sectors))))
    return Mesh(tuple(vertices), tuple(faces))


def _horizontal_link(cx: float, cy: float, cz: float, *,
                     long_radius: float = .48, short_radius: float = .27,
                     angle: float = 0., broken: bool = False,
                     tilt: float = 0.) -> Mesh:
    """An oval, broad-section link, optionally split with visibly open ends."""
    n = 72
    start, sweep = ((.23, math.tau-.48) if broken else (0., math.tau))
    path = []
    for j in range(n):
        t = start+sweep*(j/(n-1) if broken else j/n)
        u, v = long_radius*math.cos(t), short_radius*math.sin(t)
        x = cx+math.cos(angle)*u-math.sin(angle)*v
        y = cy+math.sin(angle)*u+math.cos(angle)*v
        z = cz+tilt*math.sin(t)
        path.append((x, y, z))
    # The installed copper appears as a flattened rectangular band, not the
    # thick circular iron torus commonly used for generic 3D chains.
    return _band_sweep(tuple(path), width=.125, height=.060, closed=not broken)


def _sandal_foot(cx: float, *, raised: bool) -> Mesh:
    """Continuous asymmetric forefoot/instep volume, behind separate toes."""
    # Cross-sections run from exposed toes toward the hem; right heel lifts.
    lift = .14 if raised else .025
    sections = (
        (-2.72, 9.79, .13, .048),
        (-2.57, 9.81, .48, .092),
        (-2.30, 9.86, .61, .145),
        (-2.01, 9.94, .56, .160),
        (-1.73, 10.04+lift*.3, .46, .162),
        (-1.47, 10.10+lift, .33, .140),
        (-1.29, 10.14+lift, .18, .100),
    )
    sectors = 32
    vertices = []
    for y, zc, half_width, height in sections:
        for j in range(sectors):
            t = math.tau*j/sectors
            # The upper instep is fuller than the under-sandal sole.
            s = math.sin(t)
            vertices.append((cx+half_width*math.cos(t), y,
                             zc+height*(s if s >= 0 else .62*s)))
    faces = [tuple(range(sectors))]
    for i in range(len(sections)-1):
        for j in range(sectors):
            k = (j+1) % sectors
            faces.append((i*sectors+j, (i+1)*sectors+j,
                          (i+1)*sectors+k, i*sectors+k))
    last = (len(sections)-1)*sectors
    faces.append(tuple(last+j for j in reversed(range(sectors))))
    return Mesh(tuple(vertices), tuple(faces))


def _toe(cx: float, index: int, *, right: bool) -> Mesh:
    """Individual knuckle-to-tip tapered anatomical lobe, not a painted mark."""
    medial = 1 if right else -1
    # Five distinct toes: thumb/big toe sits toward the body's centreline.
    offsets = (-.39, -.22, -.035, .16, .375)
    radii = (.105, .121, .139, .163, .194)
    x = cx+medial*offsets[index]
    y_tip = -2.53-.038*index
    radius = radii[index]
    path = (
        (x, y_tip+.36, 9.89),
        (x+medial*.008, y_tip+.235, 9.90),
        (x+medial*.014, y_tip+.108, 9.90),
        (x+medial*.011, y_tip+.026, 9.88),
        (x, y_tip, 9.87),
    )
    # Square-ish smooth bands approximate sculpted, blunt toe volumes.
    return _band_sweep(path, width=radius, height=radius*.83, sectors=20,
                       taper=(.78, 1.08, 1.0, .74, .14))


def _sandal_sole(cx: float, *, raised: bool) -> Mesh:
    """A double-curved leather-style sandal sole beneath the exposed toes."""
    z = 9.745+(.060 if raised else 0.)
    n = 64
    path = tuple((cx+.58*math.cos(math.tau*j/n),
                  -2.18+.70*math.sin(math.tau*j/n), z)
                 for j in range(n))
    return _band_sweep(path, width=.100, height=.044, closed=True, sectors=12)


def foot_detail_parts() -> tuple[tuple[str, Mesh, str], ...]:
    """Independent optional components, located on the statue's copper base."""
    result: list[tuple[str, Mesh, str]] = []
    for side, cx, raised in (("right", -1.30, True), ("left", 1.30, False)):
        result.append((f"feet-{side}-foot", _sandal_foot(cx, raised=raised), "patina"))
        result.append((f"feet-{side}-sole-rim", _sandal_sole(cx, raised=raised), "patina-shadow"))
        for i in range(5):
            result.append((f"feet-{side}-toe-{i}", _toe(cx, i, right=raised), "patina"))

    # At the right foot the open iron-size copper shackle is a thick curved
    # strap. Its end points remain separated: it is not a closed torus.
    path = []
    for i in range(60):
        t = math.radians(24)+math.radians(311)*i/59
        path.append((-.43+.65*math.cos(t), -2.19+.40*math.sin(t),
                     9.86+.035*math.sin(t*1.6)))
    result.append(("feet-right-broken-shackle", _band_sweep(
        tuple(path), width=.094, height=.260, sectors=16), "patina-shadow"))

    # The two visible chains are purposefully discontinuous where the NPS
    # describes them vanishing beneath the drapery.
    right_links = ((-2.41, -2.43, .09), (-1.53, -2.42, -.07))
    for i, (x, y, tilt) in enumerate(right_links):
        result.append((f"feet-right-chain-{i:02d}", _horizontal_link(
            x, y, 9.75, tilt=tilt), "patina-shadow"))
    left_links = ((1.40, -2.42, -.09), (2.37, -2.39, .06))
    for i, (x, y, tilt) in enumerate(left_links):
        result.append((f"feet-left-chain-{i:02d}" + ("-broken" if i == 1 else ""),
                       _horizontal_link(x, y, 9.77, tilt=tilt, broken=i == 1),
                       "patina-shadow"))
    return tuple(result)
