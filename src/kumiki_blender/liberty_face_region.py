"""Pure geometric boundaries for the anatomical Liberty face study.

Coordinates are in the centered, front=-Y scan frame. A gradual outer band
registers the sculpted facial anatomy against captured forehead/temple skin.
"""

from __future__ import annotations

import math


def face_radius(x: float, z: float) -> float:
    return math.hypot((x + .30) / .54, (z - 8.56) / .64)


def face_weight(radius: float) -> float:
    t = min(1.0, max(0.0, (1.0 - radius) / .30))
    return t * t * (3.0 - 2.0 * t)


def face_jawline_z(x: float) -> float:
    """Parabolic cut below the chin, excluding the donor's long neck slab."""
    return 7.92 + .55 * ((x + .30) / .54) ** 2


def keep_face_vertex(x: float, y: float, z: float) -> bool:
    return face_radius(x, z) < 1.0 and y < -.14 and z >= face_jawline_z(x)


def conform_depth(donor_y: float, captured_y: float | None, radius: float) -> float:
    if captured_y is None:
        return donor_y
    weight = face_weight(radius)
    # The front is negative Y; a small clearance prevents Z-fighting where
    # the separate source and donor meshes meet at the transition rim.
    return donor_y * weight + (captured_y - .004) * (1.0 - weight)


def sculpt_classical_face(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Calm the oversized CC0 eye aperture and form a restrained brow shelf.

    Coordinate-space adjustments are mirrored about the monument face center.
    They intentionally preserve the donor's real eyelid topology, and do not
    introduce separate rings or other fake anatomy.
    """
    if abs(x + .30) > .62 or not 8.50 < z < 9.04:
        return x, y, z
    from_eye = min(abs(x + .55), abs(x + .05))
    horizontal = math.exp(-((from_eye / .21) ** 4))
    aperture = horizontal * math.exp(-(((z - 8.76) / .16) ** 4))
    brow = horizontal * math.exp(-(((z - 8.92) / .07) ** 2))
    return x, y - .044 * brow, 8.76 + (z - 8.76) * (1.0 - .34 * aperture)


def wrap_limits(x: float) -> tuple[float, float]:
    """Trim the donor at the chin/upper neck rather than its flared shoulders."""
    lateral = min(1., abs(x + .30) / .80)
    return 8.13 + .17*lateral*lateral, 9.24 - .10*lateral*lateral


def keep_wrap_vertex(x: float, y: float, z: float) -> bool:
    """Keep cheeks and temples; discard lower neck, rear skull and top slab."""
    bottom, top = wrap_limits(x)
    return abs(x + .30) < .80 and y < .18 and bottom < z < top


def wrap_donor_weight(x: float, z: float) -> float:
    """Smoothly move the expanded donor border toward the scanned skin.

    Unlike the original front-only ellipse, the extended side surface carries
    the anatomic cheek through to the hair. Near the lower neck, scalp and
    extreme temples, captured geometry takes over gradually.
    """
    bottom, top = wrap_limits(x)

    def ease(value: float) -> float:
        t = min(1., max(0., value))
        return t*t*(3.-2.*t)

    lateral = ease((.80-abs(x+.30))/.27)
    lower = ease((z-bottom)/.33)
    upper = ease((top-z)/.24)
    return lateral*lower*upper
