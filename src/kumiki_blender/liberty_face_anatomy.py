"""Continuous anatomical relief for the high-resolution procedural Liberty head.

All features deform one closed cranial surface rather than adding detached
eyelid cords, lip rods, nose lozenges or fake pupil tubes. Positions are in
the existing illustrative statue's *local* metre coordinates, not a measured
photogrammetry reconstruction. Realistic likeness still needs artist review.
"""

from __future__ import annotations

import math

from .architecture_geometry import Mesh


def _bell(value: float, center: float, width: float) -> float:
    return math.exp(-((value-center)/width)**2)


def _ease(value: float) -> float:
    t = min(1., max(0., value))
    return t*t*(3.-2.*t)


def face_projection(x: float, z: float) -> float:
    """Signed amount the copper skin projects toward the front (-Y).

    Negative relief carves a shallow cavity. Landmark fields overlap with
    smooth compact falloff to avoid open facial seams and visible pasted-on
    primitives. The lower eyelids carry a softer roll than the upper lids.
    """
    if not math.isfinite(x) or not math.isfinite(z):
        raise ValueError("Facial landmark coordinates must be finite")
    if abs(x) > 1.18 or not 25.02 < z < 28.24:
        return 0.
    d = 0.

    # The cheeks and mandibular shelf are planes, not spherical cheek beads.
    for sign in (-1, 1):
        d += .115*_bell(x, sign*.66, .38)*_bell(z, 26.54, .42)
        d += .065*_bell(x, sign*.78, .34)*_bell(z, 25.93, .33)

    for sign in (-1, 1):
        cx = sign*.46
        u = (x-cx)/.315
        span = _ease((1.05-abs(u))/.29)
        if span <= 0:
            continue
        curve = math.sqrt(max(0., 1.-min(1., u*u)))
        upper = 27.266+.068*curve-.020*u
        lower = 27.190-.055*curve-.012*u
        middle = .5*(upper+lower)
        # Carve a broad orbit, then build eyelids into its rim; the visible
        # eye surface stays shallow like a repoussé copper sculpture.
        d -= .152*_bell(u, 0., .82)*_bell(z, middle, .094)
        d += span*.065*_bell(z, upper, .023)
        d += span*.041*_bell(z, lower, .025)
        d += span*.016*_bell(z, upper+.075, .062)
        d += .073*_bell(x, cx, .30)*_bell(z, 27.55, .120)
        # A small recessed iris is readable without a freestanding eyeball.
        d -= .039*_bell(x, cx, .088)*_bell(z, middle, .051)
        # Small tear duct and subtle lower socket-to-cheek transition.
        d -= .025*_bell(x, cx-sign*.25, .054)*_bell(z, 27.18, .047)
        d += .038*_bell(x, cx, .25)*_bell(z, 27.04, .087)

    # A single bridge becomes the tip, asymmetric nostril wings and alar
    # grooves without a separate floating bulb or cylinder mesh.
    bridge = _ease((z-26.48)/.20)*(1.-_ease((z-27.40)/.19))
    d += .224*_bell(x, 0., .164)*bridge
    d += .332*_bell(x, 0., .155)*_bell(z, 26.53, .164)
    for sign in (-1, 1):
        d += .078*_bell(x, sign*.191, .092)*_bell(z, 26.47, .105)
        d -= .105*_bell(x, sign*.174, .067)*_bell(z, 26.398, .046)
        d += .025*_bell(x, sign*.058, .030)*_bell(z, 26.27, .105)
    d -= .021*_bell(x, 0., .038)*_bell(z, 26.28, .10)

    # Bow-shaped fissure and lip rolls are all displacements of the skin.
    if abs(x) < .51:
        u = x/.43
        lip = _ease((1.13-abs(u))/.28)
        cupid = .019*_bell(abs(u), .24, .17)
        fissure = 25.918+cupid+.034*u*u
        roll = math.sqrt(max(0., 1.-min(1., u*u)))
        upper = fissure+.056*roll
        lower = fissure-.052*roll
        d += lip*(.053*_bell(z, upper, .049)
                  +.047*_bell(z, lower, .054)
                  -.069*_bell(z, fissure, .018))
        d -= .029*_bell(abs(u), .92, .14)*_bell(z, fissure, .050)
        d -= .035*_bell(x, 0., .33)*_bell(z, fissure-.14, .045)

    # Chin and lower mandible: sustained vertical mass, not an attached ball.
    d += .132*_bell(x, 0., .47)*_bell(z, 25.43, .20)
    d -= .018*_bell(x, 0., .10)*_bell(z, 25.34, .065)
    d += .048*_bell(x, 0., .63)*_bell(z, 27.92, .33)

    # Fade into untouched scalp and neck surfaces with no cut edge.
    return d*_ease((z-25.02)/.28)*(1.-_ease((z-28.00)/.24))


def sculpted_ear(sign: int) -> Mesh:
    """Closed curved pinna with a real concave concha and raised helix.

    A side-facing Y/Z bowl integrates into the broad skull instead of an
    oval wedge plus freestanding ear-outline rod. These are artist-sculpted
    illustrative proportions, not licensed photogrammetry measurements.
    """
    if sign not in (-1, 1):
        raise ValueError("Ear side must be -1 or 1")
    around, rings = 64, 9
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    def point(radius: float, angle: float, *, front: bool) -> tuple[float, float, float]:
        shaped_radius = radius*(1.+.065*math.sin(angle+.45)
                                +.045*math.cos(2*angle-.2))
        y = .11+.258*shaped_radius*math.cos(angle)+.018*radius*math.sin(2*angle)
        z = 26.85+.51*shaped_radius*math.sin(angle)+.040*radius*math.cos(angle)
        # C-shaped helix, lower lobe, and an offset antihelix encircle the
        # concha. The anterior/lower interruption is deliberate: a uniform
        # concentric tube reads as a decorative ring rather than a pinna.
        notch = _bell(math.atan2(math.sin(angle-3.98),
                                 math.cos(angle-3.98)), 0., .40)
        helix = .175*_bell(radius, .76, .125)*(1.-.77*notch)
        antihelix = .067*_bell(radius, .46, .13)*(
            1.-.43*_bell(math.atan2(math.sin(angle-1.3),
                                    math.cos(angle-1.3)), 0., .56))
        tragus = .061*_bell(radius, .58, .17)*_bell(
            math.atan2(math.sin(angle-math.pi),
                       math.cos(angle-math.pi)), 0., .33)
        x = (1.06 + helix + antihelix + tragus
             - .048*_bell(radius, .23, .17) + .063*radius) if front else .970
        return (sign*x, y, z)

    vertices.append(point(0., 0., front=True))
    for ring in range(1, rings+1):
        for j in range(around):
            vertices.append(point(ring/rings, math.tau*j/around, front=True))
    back_center = len(vertices)
    vertices.append(point(0., 0., front=False))
    for ring in range(1, rings+1):
        for j in range(around):
            vertices.append(point(ring/rings, math.tau*j/around, front=False))

    def build_surface(center: int, first_ring: int, *, outward: bool):
        for j in range(around):
            nxt = (j+1)%around
            face = (center, first_ring+j, first_ring+nxt)
            faces.append(face if outward else tuple(reversed(face)))
        for ring in range(rings-1):
            lower = first_ring+ring*around
            upper = lower+around
            for j in range(around):
                nxt = (j+1)%around
                face = (lower+j, upper+j, upper+nxt, lower+nxt)
                faces.append(face if outward else tuple(reversed(face)))

    build_surface(0, 1, outward=True)
    build_surface(back_center, back_center+1, outward=False)
    front_outer = 1+(rings-1)*around
    back_outer = back_center+1+(rings-1)*around
    for j in range(around):
        nxt = (j+1)%around
        faces.append((front_outer+j, back_outer+j,
                      back_outer+nxt, front_outer+nxt))
    if sign < 0:
        faces = [tuple(reversed(face)) for face in faces]
    return Mesh(tuple(vertices), tuple(faces))
