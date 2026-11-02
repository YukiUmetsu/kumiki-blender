"""Keystone-shaped copper tablet and a continuous, curled supporting left hand.

All coordinates are in the unscaled procedural statue's frame (front is -Y).
This is a photo-guided sculptural study, not survey geometry.  The approximate
tablet proportions follow the National Park Service's 7.19 x 4.14 x 0.61 m
measurements, reduced into the existing illustration's smaller head scale.
"""

from __future__ import annotations

from .architecture_geometry import Mesh


TABLET_BOTTOM = 20.24
TABLET_TOP = 24.38
TABLET_CENTER_Z = 22.31
TABLET_BOTTOM_WIDTH = 1.84
TABLET_TOP_WIDTH = 2.36


def tablet_center_x(z: float) -> float:
    return 3.27 + .095*(z-TABLET_CENTER_Z)


def tablet_width(z: float) -> float:
    t = (z-TABLET_BOTTOM)/(TABLET_TOP-TABLET_BOTTOM)
    return TABLET_BOTTOM_WIDTH + t*(TABLET_TOP_WIDTH-TABLET_BOTTOM_WIDTH)


def tablet_right_edge(z: float) -> float:
    return tablet_center_x(z)+.5*tablet_width(z)


def tablet_front_y(z: float) -> float:
    return -1.13 + .035*(z-TABLET_CENTER_Z)


def tablet_keystone_mesh() -> Mesh:
    """A closed, chamfered, shallow wedge rather than a plain rectangular box."""
    rings = []
    for y_offset, inset in ((0., .045), (.055, 0.), (.325, 0.), (.38, .045)):
        contour = []
        for z, sign in ((TABLET_BOTTOM+inset, -1),
                        (TABLET_BOTTOM+inset, 1),
                        (TABLET_TOP-inset, 1),
                        (TABLET_TOP-inset, -1)):
            contour.append((tablet_center_x(z) + sign*(tablet_width(z)/2-inset),
                            tablet_front_y(z) + y_offset, z))
        rings.extend(contour)
    faces: list[tuple[int, ...]] = [(0, 1, 2, 3)]
    for ring in range(3):
        for side in range(4):
            nxt = (side+1) % 4
            faces.append((ring*4+side, (ring+1)*4+side,
                          (ring+1)*4+nxt, ring*4+nxt))
    faces.append((15, 14, 13, 12))
    return Mesh(tuple(rings), tuple(faces))


def tablet_detail_parts() -> tuple[tuple[str, Mesh, str], ...]:
    """Return independent, editable slab/hand meshes for the optional revision."""
    # Import only after liberty's sweep helpers have been defined; liberty
    # intentionally pulls this optional component in at assembly time.
    from .liberty import _loft, _tube

    parts: list[tuple[str, Mesh, str]] = [("tablet-slab", tablet_keystone_mesh(), "patina-light")]
    parts.append(("tablet-hand-wrist", _tube([
        (3.30, -.51, 22.26), (3.58, -.61, 22.27),
        (3.88, -.70, 22.30), (4.21, -.78, 22.34),
    ], [.285, .27, .25, .275], 24), "patina"))
    parts.append(("tablet-hand-palm", _loft((
        (21.79, 4.42, -.78, .11, .14),
        (21.97, 4.42, -.79, .25, .21),
        (22.28, 4.43, -.80, .34, .26),
        (22.59, 4.41, -.79, .33, .26),
        (22.83, 4.34, -.78, .20, .20),
    ), 48), "patina"))

    # Unequal fingers emerge from the knuckle pad, curl around the narrow edge
    # in plan view, and return onto the visible face.  The fingertip surface
    # slightly overlaps the tablet, avoiding four free-floating pegs.
    lengths = (.13, .03, -.04, .04)  # pinky, ring, middle, index
    girths = (.83, .93, 1., .94)
    for index, z in enumerate((21.96, 22.16, 22.36, 22.56)):
        spread = .017*(index-1.5)
        length = lengths[index]
        parts.append((f"tablet-grip-finger-{index}", _tube([
            (4.43+spread, -.79, z),
            (4.54+spread, -.83, z+.025),
            (4.57+spread, -.98, z+.065),
            (4.48+spread, -1.12, z+.095),
            (4.32+spread+length*.3, -1.23, z+.10),
            (4.16+spread+length*.7, -1.22, z+.074),
            (4.02+spread+length, -1.145, z+.025),
        ], [r*girths[index] for r in (.127, .135, .126, .106, .092, .071, .041)],
        20), "patina"))
    parts.append(("tablet-grip-thumb", _tube([
        (4.20, -.69, 22.79), (4.30, -.81, 22.88),
        (4.36, -1.02, 22.99), (4.30, -1.17, 23.10),
        (4.16, -1.22, 23.17), (4.04, -1.145, 23.22),
    ], [.155, .148, .133, .106, .075, .044], 20), "patina"))
    return tuple(parts)
