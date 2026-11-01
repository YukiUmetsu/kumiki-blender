"""Reference-led facial relief for the licensed Liberty bronze miniature scan.

This is intentionally a sculpt of the *existing* captured copper surface, not
an assembly of floating eyes or facial cylinders. The source does not contain
the monument's true facial microgeometry; these fields give an art-directable
first anatomical pass, still subject to matched-camera artist review.

Coordinates are local to the fixed, attribution-locked jerryfisher scan:
base at Z=0, torch at Z=12.44, eyes near Z=8.75, negative Y toward camera.
All dimensions below are in those source units, **not historical metres**.
"""

from __future__ import annotations

import math


def _gaussian(value: float, center: float, radius: float) -> float:
    return math.exp(-((value - center) / radius) ** 2)


def _fade(edge_a: float, edge_b: float, value: float) -> float:
    """Smooth masking without sharp polygonal transitions."""
    t = max(0.0, min(1.0, (value - edge_a) / (edge_b - edge_a)))
    return t * t * (3.0 - 2.0 * t)


def face_relief_offset(x: float, y: float, z: float) -> float:
    """Carve eyelid edges, sockets, nostrils, lips and forehead into mesh Y.

    Positive values recess the front-facing copper surface; negative values
    lift it toward the camera. Soft transitions preserve the captured skull.
    The periocular areas use curved almond seams with distinct upper/lower
    ledges rather than flat Gaussian dots. The raised brow shields the eyes.
    """
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("Facial sculpt coordinates must be finite")
    if not (-.94 < x < .25 and -.95 < y < -.14 and 7.91 < z < 9.02):
        return 0.0

    # Suppress contributions on rear hair, ears and glancing-side polygons.
    region = (_fade(-.94, -.78, x) * (1 - _fade(.10, .25, x))
              * _fade(7.91, 8.04, z) * (1 - _fade(8.95, 9.02, z))
              * _fade(.19, .42, -y))
    if region < 1e-6:
        return 0.0

    d = 0.0
    for eye_x in (-.53, -.19):
        u = (x - eye_x) / .162
        v = max(-1.0, min(1.0, u))
        span = 1 - _fade(.79, 1.04, abs(u))
        if span <= 0:
            continue

        # The lids are asymmetrical: high medial upper arc, low outer corner.
        upper = 8.755 + .033 * (1 - v*v) - .010 * v
        lower = 8.725 - .030 * (1 - v*v) - .007 * v
        middle = (upper + lower) * .5
        # Thick eyelid shelves physically project over the eyeball.
        d += span * (
            +.017 * _gaussian(z, upper, .010)   # upper lash-line incision
            -.041 * _gaussian(z, upper + .023, .020)  # upper tarsal shelf
            +.013 * _gaussian(z, lower, .010)   # lower lid incision
            -.028 * _gaussian(z, lower - .018, .017)  # lower lid ledge
            -.041 * _gaussian(z, middle, .045)  # restrained bronze eye surface
            -.036 * _gaussian(z, 8.861, .044)  # heavy brow bone
            +.018 * _gaussian(z, 8.827, .016)  # brow fold / sulcus
            -.011 * _gaussian(z, 8.659, .037)  # upper cheek shelf
        )
        # A very shallow, real iris ring is appropriate for unpainted bronze.
        iris_r = math.hypot((x-eye_x)/.047, (z-8.741)/.029)
        d += span * .010 * _gaussian(iris_r, 1.0, .12)
        # Tear-duct hollow at the nasal end; avoid four circular eye blobs.
        nasal_direction = 1.0 if eye_x < -.35 else -1.0
        d += .025 * _gaussian(u, nasal_direction*.88, .13) * _gaussian(z, 8.735, .021)

    # Nasal root/glabella, tapered ridge, fleshy tip, alae and carved nares.
    bridge_mask = _fade(8.36, 8.52, z) * (1 - _fade(8.74, 8.81, z))
    d -= .041 * _gaussian(x, -.353, .051) * bridge_mask
    d -= .047 * _gaussian(x, -.353, .082) * _gaussian(z, 8.405, .060)
    for side in (-1, 1):
        ala_x = -.353 + side * .095
        d -= .029 * _gaussian(x, ala_x, .049) * _gaussian(z, 8.378, .039)
        d += .029 * _gaussian(x, ala_x + side*.014, .023) * _gaussian(z, 8.353, .020)
    d += .016 * _gaussian(x, -.353, .021) * _gaussian(z, 8.334, .039)

    # Double philtrum ridge and its central channel above Cupid's bow.
    for side in (-1, 1):
        d -= .011 * _gaussian(x, -.353 + side*.039, .017) * _gaussian(z, 8.317, .044)
    d += .010 * _gaussian(x, -.353, .020) * _gaussian(z, 8.318, .052)

    mouth_u = (x + .353) / .217
    mouth_span = 1 - _fade(.83, 1.06, abs(mouth_u))
    if mouth_span > 0:
        # Upper lip's Cupid bow and the nearly level central mouth fissure.
        cupid = .009 * (1 - min(1.0, abs(mouth_u) / .34))
        seam = 8.244 + cupid + .007 * mouth_u * mouth_u
        d += mouth_span * (
            +.034 * _gaussian(z, seam, .011)  # gently recessed lip split
            -.034 * _gaussian(z, seam + .030, .020)  # sculpted upper lip
            -.033 * _gaussian(z, seam - .041, .026)  # fuller lower lip
            +.012 * _gaussian(z, seam - .082, .019)  # mentolabial crease
        )
        d += .027 * _gaussian(abs(mouth_u), .94, .17) * _gaussian(z, seam, .024)
    d -= .025 * _gaussian(x, -.353, .19) * _gaussian(z, 8.097, .069)

    # A restrained malar plane gives the cheek a carved, non-balloon contour.
    for cheek_x in (-.60, -.12):
        d -= .016 * _gaussian(x, cheek_x, .103) * _gaussian(z, 8.52, .080)
    # The source already contains broad facial masses. Full displacement would
    # carve unrealistic trenches; model a controlled *incremental* sculpt.
    return region * d * .15


def hair_relief_offset(x: float, y: float, z: float) -> float:
    """Scored comb-direction striae across the captured forward forelock.

    These are anisotropic raised/lowered creases, *not* loose parallel tubes.
    Their varying phase follows the broad sweep around the central part.
    """
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("Hair sculpt coordinates must be finite")
    if not (-.85 < x < .24 and -.85 < y < -.18 and 8.97 < z < 9.39):
        return 0.0
    mask = (_fade(-.85, -.69, x) * (1-_fade(.10, .24, x))
            * _fade(8.97, 9.055, z) * (1-_fade(9.30, 9.39, z))
            * _fade(.22, .39, -y))
    if mask <= 0:
        return 0.0
    # Swept combing: closely spaced ridges change direction toward temples.
    sweep = x + .058 * math.sin((z-9.10)*12.0) + .018*math.sin((z-9.10)*25)
    strand = math.sin((2*math.pi/.032)*sweep)
    fine = math.sin((2*math.pi/.012)*sweep + z*3.0)
    # Restrained chased detail; keep the broad swept hair captured by the scan.
    return mask * (-.0024 * strand - .0006 * fine)


def sculpt_relief_offset(x: float, y: float, z: float) -> float:
    """Total scan-surface offset for the anatomical and combed sculpt pass."""
    return face_relief_offset(x, y, z) + hair_relief_offset(x, y, z)
