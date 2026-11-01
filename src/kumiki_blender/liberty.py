"""Procedural multi-angle study of Bartholdi's Liberty Enlightening the World.

All forms are closed, named meshes in meters. This is an original sculptural
interpretation of the supplied photographic references, not a surveyed scan.
The front of the statue is -Y; the raised torch is on the viewer's left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .architecture_details import box_mesh
from .architecture_geometry import Mesh, MeshInstance, place_mesh, world_vertices


class LibertyError(ValueError):
    """Invalid sculptural assembly parameters."""


@dataclass(frozen=True)
class LibertyStudy:
    asset_id: str
    instances: tuple[MeshInstance, ...]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    part_counts: tuple[tuple[str, int], ...]
    parameters: tuple[tuple[str, float | bool], ...]


def _gauss(value: float, position: float, width: float) -> float:
    return math.exp(-((value-position)/width)**2)


def _loft(sections: tuple[tuple[float, float, float, float, float], ...],
          sectors: int = 32) -> Mesh:
    """Z-axis lathe with a sculptable offset elliptical outline."""
    from .architecture_geometry import loft_elliptical_sections
    return loft_elliptical_sections(sections, radial_segments=sectors, subdivisions=3)


def _tube(points: list[tuple[float, float, float]], radii: float | list[float],
          segments: int = 10) -> Mesh:
    """Capped organic tube following a three-dimensional path, not a decal."""
    if len(points) < 2:
        raise LibertyError("A swept sculptural part requires at least two points")
    rr = [radii] * len(points) if isinstance(radii, (int, float)) else radii
    if len(rr) != len(points) or min(rr) <= 0:
        raise LibertyError("Sweep radii must be positive and match centerline")
    vertices = []
    for i, p in enumerate(points):
        left, right = points[max(0, i-1)], points[min(len(points)-1, i+1)]
        d = [b-a for a, b in zip(left, right)]
        norm = math.sqrt(sum(c*c for c in d))
        if norm < 1e-8:
            raise LibertyError("Sweep centerline has coincident endpoints")
        t = tuple(c/norm for c in d)
        ref = (0, 1, 0) if abs(t[1]) < .85 else (0, 0, 1)
        a = (t[1]*ref[2]-t[2]*ref[1], t[2]*ref[0]-t[0]*ref[2], t[0]*ref[1]-t[1]*ref[0])
        length = math.sqrt(sum(c*c for c in a))
        a = tuple(c/length for c in a)
        b = (t[1]*a[2]-t[2]*a[1], t[2]*a[0]-t[0]*a[2], t[0]*a[1]-t[1]*a[0])
        for sector in range(segments):
            theta = math.tau*sector/segments
            vertices.append(tuple(p[k]+rr[i]*(math.cos(theta)*a[k]+math.sin(theta)*b[k])
                                  for k in range(3)))
    faces = [tuple(reversed(range(segments)))]
    for ring in range(len(points)-1):
        for sector in range(segments):
            nxt = (sector+1) % segments
            faces.append((ring*segments+sector, ring*segments+nxt,
                          (ring+1)*segments+nxt, (ring+1)*segments+sector))
    last = (len(points)-1)*segments
    faces.append(tuple(range(last, last+segments)))
    return Mesh(tuple(vertices), tuple(faces))


def _cloth_panel(rows: list[tuple[float, float, float, float, float]],
                 *, folds: int = 7, thickness: float = .07) -> Mesh:
    """Closed variable-width draped sheet; row = (z, left x, right x, front y, hang)."""
    n = 33
    front = []
    for z, left, right, y, hang in rows:
        for index in range(n):
            t = index/(n-1)
            x = left+(right-left)*t
            # Deep, uneven tension folds and scalloped hem in real geometry.
            wave = (math.sin(t*math.pi*folds + z*.35) +
                    .31*math.sin(t*math.pi*(folds+3)-z*.19))
            front.append((x, y-.22*wave - hang*math.sin(t*math.pi),
                          z + .09*math.sin(t*math.pi*2 + z*.18)))
    total = len(front)
    vertices = front+[(x, y+thickness, z) for x, y, z in front]
    faces = []
    for row in range(len(rows)-1):
        for col in range(n-1):
            a = row*n+col
            b = a+n
            faces.append((a, b, b+1, a+1))
            faces.append((total+a+1, total+b+1, total+b, total+a))
    for col in range(n-1):
        faces.append((total+col+1, total+col, col, col+1))
        a = (len(rows)-1)*n+col
        faces.append((total+a, total+a+1, a+1, a))
    for row in range(len(rows)-1):
        a, b = row*n, (row+1)*n
        faces.append((a, total+a, total+b, b))
        a, b = row*n+n-1, (row+1)*n+n-1
        faces.append((b, total+b, total+a, a))
    return Mesh(tuple(vertices), tuple(tuple(reversed(face)) for face in faces))


def _face() -> Mesh:
    """Anatomical cranial mesh: oval jaw, projecting brows, cheeks, lips, chin."""
    sections = []
    for i in range(61):
        z = 24.93+i*(3.52/60)
        t = (z-24.93)/3.52
        radius = (.70 + .52*math.sin(math.pi*t)**.6) * (1-.13*t)
        depth = .78+.18*math.sin(math.pi*t)
        sections.append((z, 0, .25, max(.06, radius), max(.06, depth)))
    mesh = _loft(tuple(sections), sectors=96)
    sculpted = []
    for x, y, z in mesh.vertices:
        front = max(0., -y+.08)
        mask = min(1., front/.7)**2
        cheek = sum(.13*_gauss(x, p, .43)*_gauss(z, 26.57, .40) for p in (-.66, .66))
        brows = sum(.18*_gauss(x, p, .43)*_gauss(z, 27.55, .16) for p in (-.45, .45))
        sockets = sum(.18*_gauss(x, p, .29)*_gauss(z, 27.22, .20) for p in (-.45, .45))
        forehead = .11*_gauss(x, 0, .77)*_gauss(z, 27.95, .48)
        chin = .14*_gauss(x, 0, .49)*_gauss(z, 25.39, .27)
        mouth = .08*_gauss(x, 0, .45)*_gauss(z, 25.92, .17)
        y -= mask*(cheek+brows+forehead+chin+mouth-sockets)
        # Lower face tapers to a distinctly long and squared chin.
        x *= 1-.11*_gauss(z, 25.05, .38)
        sculpted.append((x, y, z))
    return Mesh(tuple(sculpted), mesh.faces)


def _gown() -> Mesh:
    """Continuous, asymmetric sculptural robe with baked-in deep folds."""
    sections = []
    for row in range(97):
        z = 9.83+row*(14.75/96)
        t = (z-9.83)/14.75
        radius_x = 2.32 + .56*math.sin(t*math.pi) - .44*t
        radius_y = 1.66 + .32*math.sin(t*math.pi)-.17*t
        sections.append((z, radius_x, radius_y, t))
    n = 128
    vertices = []
    for z, rx, ry, t in sections:
        for j in range(n):
            a = math.tau*j/n
            fold = (.18*math.sin(23*a + z*.15) +
                    .085*math.sin(37*a-z*.31)+.028*math.sin(61*a+z*.51))
            amplitude = .47 + .53*(1-t)
            x = (rx+fold*amplitude)*math.cos(a) + .12*math.sin(t*math.pi)
            y = (ry+fold*amplitude)*math.sin(a)
            vertices.append((x, y, z + .035*math.sin(a*13+z*.20)))
    faces = [tuple(reversed(range(n)))]
    for i in range(len(sections)-1):
        for j in range(n):
            faces.append((i*n+j, i*n+(j+1)%n,
                          (i+1)*n+(j+1)%n, (i+1)*n+j))
    faces.append(tuple(range((len(sections)-1)*n, len(sections)*n)))
    return Mesh(tuple(vertices), tuple(faces))


def make_liberty_study(*, scale: float = 1., include_pedestal: bool = True,
                       detailed_tablet: bool = False,
                       detailed_crown: bool = False) -> LibertyStudy:
    """Make the original, reference-guided statue and optional 8.9m pedestal."""
    if (isinstance(scale, bool) or not isinstance(scale, (int, float)) or
            not math.isfinite(scale) or not .05 <= scale <= 20):
        raise LibertyError("scale must be a finite value between 0.05 and 20")
    if type(include_pedestal) is not bool:
        raise LibertyError("include_pedestal must be a boolean")
    if type(detailed_tablet) is not bool:
        raise LibertyError("detailed_tablet must be a boolean")
    if type(detailed_crown) is not bool:
        raise LibertyError("detailed_crown must be a boolean")
    result: list[MeshInstance] = []

    def put(name: str, mesh: Mesh, material: str = "patina") -> None:
        result.append(place_mesh(mesh, name=f"liberty-{name}", material=material))

    def box(name: str, dims: tuple[float, float, float], center: tuple[float, float, float],
            material: str = "patina") -> None:
        from .architecture_geometry import Mesh
        x, y, z = center
        mesh = box_mesh(*dims)
        put(name, Mesh(tuple((a+x, b+y, c+z) for a, b, c in mesh.vertices), mesh.faces), material)

    if include_pedestal:
        box("pedestal-footing", (9.65, 9.65, .56), (0, 0, .28), "granite")
        box("pedestal-plinth", (9.15, 9.15, .44), (0, 0, .77), "limestone")
        box("pedestal-subbase", (8.28, 8.28, .36), (0, 0, 1.20), "granite")
        box("pedestal-core", (7.62, 7.62, 6.36), (0, 0, 4.55), "masonry")
        for name, width, z, h in (
            ("pedestal-lower-cornice", 8.5, 1.50, .24),
            ("pedestal-belt-course", 8.11, 3.52, .18),
            ("pedestal-architrave", 8.65, 7.87, .28),
            ("pedestal-crown-cornice", 9.33, 8.18, .39),
            ("pedestal-attic", 8.55, 8.50, .30),
            ("pedestal-statue-plinth", 6.63, 8.88, .48),
        ):
            box(name, (width, width, h), (0, 0, z), "limestone")
        for face in range(4):
            # Four identical developed facades, with masonry pilasters and
            # actual depressed dark window recesses, not printed rectangles.
            for idx, along in enumerate((-2.92, -1.75, -.58, .58, 1.75, 2.92)):
                x, y = ((along, -3.89) if face == 0 else
                        (3.89, along) if face == 1 else
                        (along, 3.89) if face == 2 else (-3.89, along))
                dims = (.14, .32, 2.40) if face % 2 == 0 else (.32, .14, 2.40)
                box(f"pedestal-pilaster-{face:01d}-{idx:02d}", dims,
                    (x, y, 5.95), "limestone")
            for idx, along in enumerate((-2.25, 0., 2.25)):
                x, y = ((along, -3.823) if face == 0 else
                        (3.823, along) if face == 1 else
                        (along, 3.823) if face == 2 else (-3.823, along))
                dims = (.78, .04, 1.83) if face % 2 == 0 else (.04, .78, 1.83)
                box(f"pedestal-dark-recess-{face}-{idx}", dims, (x, y, 5.95), "recess")
                cap = (.95, .26, .14) if face % 2 == 0 else (.26, .95, .14)
                box(f"pedestal-window-lintel-{face}-{idx}", cap,
                    (x, y, 6.95), "limestone")
            for index in range(20):
                along = -3.5 + 7*index/19
                x, y = ((along, -4.24) if face == 0 else (4.24, along) if face == 1
                        else (along, 4.24) if face == 2 else (-4.24, along))
                box(f"pedestal-dentil-{face}-{index:02d}", (.24, .24, .21),
                    (x, y, 7.79), "limestone")
    else:
        # Exact statue parts are translated to ground only at the end.
        pass

    box("statue-footplate", (6.36, 5.78, .28), (0, .02, 9.44), "patina")
    put("gown-underrobe", _gown())

    # The overfold is a descending diagonal sash, with a detached rolled hem.
    put("gown-front-overfold", _cloth_panel([
        (13.18, -.95, 2.11, -2.02, .32),
        (14.05, -1.35, 2.14, -2.12, .24),
        (15.10, -1.64, 2.05, -2.15, .22),
        (16.65, -1.93, 1.81, -2.20, .34),
        (18.15, -1.91, 1.38, -2.19, .27),
        (19.80, -1.66, .89, -2.14, .25),
        (21.50, -1.35, .43, -1.97, .16),
        (23.00, -1.05, -.15, -1.71, .10),
        (24.18, -.98, -.61, -1.24, .05),
    ], folds=8), "patina-light")
    # Long vertical tunic fluting and secondary hanging textile edges.
    for i in range(26):
        x = -2.10 + i*.163
        points = []
        for j in range(15):
            z = 9.98 + j*.83
            rx = 2.34 + .54*math.sin((z-9.83)/14.75*math.pi)
            t = min(.94, x/max(rx, 1))
            y = -1.82*math.sqrt(max(.05, 1-t*t))-.13
            points.append((x+.12*math.sin(z*.49+i*.65), y-.06*math.sin(i*.9+z*.31), z))
        put(f"gown-pleat-{i:02d}", _tube(points, [.024+.012*(1-j/14) for j in range(15)], 8),
            "patina-shadow" if i%3 == 0 else "patina-light")
    # Rolled hem seams tracking the diagonal drapery and a trailing back cape.
    put("gown-overfold-hem", _tube([
        (-1.1, -2.04, 13.15), (-.28, -2.25, 12.95), (.53, -2.36, 13.38),
        (1.31, -2.26, 14.28), (2.07, -2.13, 15.10),
    ], [.095, .115, .108, .085, .045]), "patina-shadow")
    put("gown-back-cape", _cloth_panel([
        (12.7, -.7, 1.85, 2.04, .30), (15.4, -1.8, 2.0, 2.12, .27),
        (18.6, -2.1, 1.7, 2.22, .36), (21.3, -1.45, 1.8, 2.00, .28),
        (24.1, -.9, 1.5, 1.50, .14),
    ], folds=11), "patina-shadow")
    # Shoulders and the asymmetric puffed torch sleeve.
    put("shoulder-torso", _loft(((22.75, 0, 0, 1.8, 1.0),
                                 (23.60, 0, 0, 2.43, 1.10),
                                 (24.58, -.05, .0, 1.95, .97)), 48))
    put("sleeve-torch", _loft(((22.45, -2.13, .06, .66, .72),
                              (23.42, -2.31, .05, .99, .85),
                              (24.60, -2.40, .06, .73, .73),
                              (25.12, -2.52, .03, .49, .50)), 48), "patina-light")
    for i in range(9):
        y = -.63+i*.155
        put(f"sleeve-torch-wrinkle-{i:02d}", _tube([
            (-3.11, y, 23.7+i*.06), (-2.61, y-.10, 23.0+i*.045),
            (-2.0, y+.06, 22.45+i*.035),
        ], [.045, .083, .024]), "patina-shadow")

    # Arm, wrist and knuckles trace an anatomically believable upward line.
    put("arm-raised", _tube([
        (-2.60, -.08, 24.58), (-2.91, -.06, 26.2), (-3.16, -.05, 28.45),
        (-3.32, -.02, 30.55), (-3.46, -.03, 32.47),
    ], [.60, .52, .42, .31, .25], 24))
    put("arm-raised-palm", _loft(((32.20, -3.46, -.03, .30, .26),
                                  (32.60, -3.46, -.02, .32, .27),
                                  (33.14, -3.46, .0, .27, .25)), 24))
    # Four articulated fingers clasp the *circumference* rather than forming
    # four vertical rods painted onto the shaft. Uneven knuckle swells and
    # tapered distal phalanges make the silhouette read like a copper hand.
    for i in range(4):
        x = -3.77+i*.17
        fingertip_z = 33.50 - .025*abs(i-1.5)
        put(f"torch-grip-finger-{i}", _tube([
            (x, -.19, 32.76), (x-.025, -.29, 32.85),
            (x-.063, -.355, 33.00), (x-.094, -.36, 33.13),
            (x-.087, -.305, 33.28), (x-.055, -.205, 33.40),
            (x+.011, -.075, fingertip_z-.025),
            (x+.075, .07, fingertip_z),
        ], [.112, .121, .129, .112, .098, .079, .059, .038], 20))
    put("torch-grip-thumb", _tube([
        (-3.12, -.005, 32.64), (-3.18, -.13, 32.71),
        (-3.29, -.27, 32.82), (-3.41, -.35, 32.94),
        (-3.51, -.38, 33.05), (-3.60, -.32, 33.13),
        (-3.64, -.23, 33.17),
    ], [.17, .17, .155, .139, .117, .090, .059], 20))

    # Other arm bends at elbow and supports the inscription tablet.
    put("arm-tablet", _tube([
        (2.0, -.02, 23.53), (2.55, -.07, 22.08),
        (2.93, -.31, 21.76), (3.38, -.52, 22.26),
    ], [.55, .49, .40, .29], 20))
    put("arm-tablet-sleeve", _cloth_panel([
        (19.9, 2.02, 3.24, -.46, .18), (21.1, 1.78, 3.38, -.53, .24),
        (22.4, 1.62, 2.99, -.46, .31), (23.7, 1.50, 2.58, -.24, .10),
    ], folds=5), "patina-shadow")
    if detailed_tablet:
        from .liberty_tablet import tablet_detail_parts
        for name, mesh, material in tablet_detail_parts():
            put(name, mesh, material)
    else:
        # Legacy tablet is retained for backwards-compatible baseline studies.
        source = box_mesh(1.64, .30, 3.13)
        put("tablet-slab", Mesh(tuple((x+3.15+.15*z, y-.81, z+22.37+.10*x)
                                      for x, y, z in source.vertices), source.faces), "patina-light")
        box("tablet-top-bead", (1.72, .37, .115), (3.37, -.83, 23.96), "patina-shadow")
        for i in range(4):
            put(f"tablet-grip-finger-{i}", _tube([
                (3.87, -1.10, 21.60+i*.29), (4.00, -1.19, 21.78+i*.29),
                (3.95, -.97, 21.97+i*.29),
            ], [.115, .105, .082]), "patina")

    # Neck, long classical face and rounded crown of hair.
    put("neck", _loft(((23.83, 0, .15, .67, .57), (24.66, 0, .20, .61, .52),
                       (25.44, 0, .22, .78, .62)), 48))
    put("head-face", _face())
    put("head-chin", _loft(((25.02, 0, -.33, .31, .22),
                            (25.23, 0, -.42, .48, .29),
                            (25.45, 0, -.35, .56, .29)), 36))
    put("head-nose", _loft(((26.35, 0, -.99, .14, .22),
                            (26.52, 0, -1.06, .24, .28),
                            (26.70, 0, -1.03, .19, .23),
                            (27.45, 0, -.80, .12, .10)), 48))
    # Eyes: actual inset almond solids, upper/lower lid cords, brows and inner tear ducts.
    for label, sign in (("left", -1), ("right", 1)):
        x = sign*.47
        put(f"head-{label}-eye", _loft(((27.16, x, -.72, .24, .075),
                                       (27.27, x, -.73, .27, .09),
                                       (27.35, x, -.73, .22, .07)), 40))
        for lid, z, height in (("upper", 27.36, .07), ("lower", 27.16, -.039)):
            points = [(x-.32+j*.107, -.85-.035*math.sin(j*math.pi/6),
                       z+height*math.sin(j*math.pi/6)) for j in range(7)]
            put(f"head-{label}-{lid}-lid", _tube(points, [.034]*7, 10), "patina-shadow")
        put(f"head-{label}-brow", _tube([
            (x-.35, -.82, 27.55), (x-.19, -.89, 27.65),
            (x+.10, -.89, 27.66), (x+.35, -.80, 27.56)
        ], [.035, .062, .058, .026]), "patina-shadow")
        put(f"head-{label}-ear", _loft(((26.34, sign*1.10, .10, .20, .18),
                                       (26.88, sign*1.12, .11, .30, .27),
                                       (27.33, sign*1.08, .12, .22, .17)), 32))
        put(f"head-{label}-ear-helix", _tube([
            (sign*1.20, -.11, 26.45), (sign*1.34, -.04, 26.72),
            (sign*1.39, .01, 27.07), (sign*1.24, -.01, 27.31),
        ], [.051, .077, .065, .035]), "patina-shadow")
        put(f"head-{label}-nostril", _loft(((26.36, sign*.17, -1.22, .052, .05),
                                           (26.44, sign*.17, -1.20, .064, .07)), 24), "patina-shadow")
    # Lips are three distinct relief bands, not an expression texture.
    put("head-upper-lip", _tube([
        (-.43, -.79, 25.95), (-.22, -.89, 26.01), (0, -.92, 25.97),
        (.20, -.89, 26.01), (.43, -.79, 25.95),
    ], [.035, .053, .060, .053, .035], 12), "patina-light")
    put("head-lower-lip", _tube([
        (-.40, -.79, 25.85), (-.21, -.86, 25.81),
        (0, -.91, 25.80), (.22, -.86, 25.81), (.40, -.79, 25.85),
    ], [.029, .051, .063, .051, .029], 12), "patina-light")
    put("head-mouth-crease", _tube([
        (-.38, -.87, 25.91), (0, -.94, 25.89), (.38, -.87, 25.91),
    ], [.012, .014, .012], 10), "patina-shadow")
    put("hair-scalp", _loft(((27.54, 0, .31, 1.00, .78),
                             (28.20, 0, .28, 1.23, .90),
                             (28.78, 0, .28, 1.18, .93),
                             (29.06, 0, .28, .63, .53),
                             (29.15, 0, .28, .18, .18)), 60), "patina-shadow")
    if detailed_crown:
        from .liberty_hair import sculpted_forelock
        put("hair-forelock-sculpt", sculpted_forelock(), "patina-shadow")
    else:
        # The legacy forehead mass and separately swept hair wires remain
        # reproducible for comparison with the newer integrated relief.
        put("hair-swept-forelock", _tube([
            (-1.19, -.37, 28.30), (-.94, -.76, 28.25),
            (-.49, -.91, 28.33), (.02, -.94, 28.44),
            (.49, -.87, 28.49), (.94, -.61, 28.41), (1.18, -.27, 28.28),
        ], [.20, .29, .32, .34, .31, .28, .18], 32))
        for index in range(26):
            t = (index+0.5)/26
            x = -1.03+2.06*t
            points = [
                (x-.13, -1.017, 28.09+.10*math.cos(t*math.pi)),
                (x-.19, -1.14, 28.23+.12*math.sin(t*math.pi)),
                (x-.13, -1.16, 28.38+.12*math.cos(t*math.pi)),
                (x+.05, -1.03, 28.60+.05*math.sin(t*math.pi)),
                (x+.20, -.77, 28.78),
            ]
            put(f"hair-strand-{index:02d}", _tube(points, [.014, .023, .026, .022, .012], 10),
                "patina-light" if index%3 else "patina-shadow")
    for sign, label in ((-1, "left"), (1, "right")):
        put(f"hair-{label}-temple-volume", _tube([
            (sign*1.0, -.55, 28.10), (sign*1.25, -.34, 27.97),
            (sign*1.37, -.10, 27.68), (sign*1.28, .09, 27.34),
        ], [.22, .28, .27, .13], 24), "patina-shadow")
        if not detailed_crown:
            for i in range(6):
                z = 27.92-i*.16
                put(f"hair-{label}-temple-wave-{i:02d}", _tube([
                    (sign*1.03, -.34, z), (sign*1.32, -.24, z-.08),
                    (sign*1.37, -.08, z-.21), (sign*1.21, .09, z-.28),
                ], [.022, .032, .029, .011], 10), "patina-light")
        put(f"hair-{label}-shoulder-tress", _tube([
            (sign*1.24, .34, 27.78), (sign*1.38, .55, 26.46),
            (sign*1.64, .50, 25.2), (sign*1.82, .48, 24.22),
        ], [.22, .24, .22, .075], 20))
    put("hair-bun", _loft(((26.64, 0, 1.00, .48, .43),
                            (27.05, 0, 1.05, .77, .50),
                            (27.58, 0, 1.02, .75, .48),
                            (27.93, 0, .92, .32, .23)), 48), "patina-shadow")

    if detailed_crown:
        from .liberty_crown import crown_detail_parts
        for name, mesh, material in crown_detail_parts():
            put(name, mesh, material)
    else:
        # Legacy ray rods and dark window inserts remain the default geometry.
        for index, degrees in enumerate((9, 36, 63, 90, 117, 144, 171)):
            theta = math.radians(degrees)
            start = (.94*math.cos(theta), .18, 28.83+.29*math.sin(theta))
            end = (3.47*math.cos(theta), .20, 29.05+3.55*math.sin(theta))
            middle = tuple(start[k]*.43+end[k]*.57 for k in range(3))
            put(f"crown-ray-{index}", _tube([start, middle, end], [.25, .155, .028], 6),
                "patina-light")
        for z, radius, width in ((28.68, 1.45, .095), (29.23, 1.46, .105)):
            ring = [(radius*math.cos(math.tau*i/72), .22+radius*.64*math.sin(math.tau*i/72), z)
                    for i in range(73)]
            put(f"crown-rim-{str(z).replace('.', '-')}", _tube(ring, width, 8))
        for index in range(27):
            theta = math.tau*index/27
            x = 1.46*math.cos(theta)
            y = .22+.935*math.sin(theta)
            put(f"crown-window-mullion-{index:02d}", _tube([
                (x, y, 28.72), (x*1.012, y*1.012, 29.20),
            ], [.043, .047], 8), "patina-shadow")
            put(f"crown-window-dark-{index:02d}", _tube([
                (x*.965, y-.027, 28.77), (x*.965, y-.027, 29.14),
            ], [.055, .060], 8), "recess")

    # Cast torch shaft, its concentric support dish and pierced brass gallery.
    put("torch-handle", _loft(((32.70, -3.45, 0, .15, .15),
                               (33.06, -3.45, 0, .20, .20),
                               (33.20, -3.45, 0, .28, .28),
                               (33.43, -3.45, 0, .25, .25),
                               (33.87, -3.45, 0, .20, .20),
                               (34.05, -3.45, 0, .31, .31)), 48), "patina-shadow")
    for i in range(12):
        theta = math.tau*i/12
        x, y = -3.45+.232*math.cos(theta), .232*math.sin(theta)
        put(f"torch-flute-{i:02d}", _tube([(x, y, 33.25), (x, y, 33.81)], [.022, .022], 6),
            "patina-light")
    for z, radius, thickness in ((33.89, .28, .073), (34.15, .75, .10),
                                  (34.28, .82, .13), (34.53, .77, .09)):
        put(f"torch-gallery-ring-{str(z).replace('.', '-')}", _loft((
            (z-.04, -3.45, 0, radius-thickness/3, radius-thickness/3),
            (z+.04, -3.45, 0, radius, radius)), 48), "patina")
    for i in range(20):
        theta = math.tau*i/20
        x, y = -3.45+.74*math.cos(theta), .74*math.sin(theta)
        put(f"torch-gallery-baluster-{i:02d}", _tube([
            (x, y, 34.30), (x, y, 34.50),
        ], [.041, .035], 8), "patina-shadow")
    # The flame is layered sculptural fire with irregular whipping tongues.
    put("torch-flame-core", _loft(((34.56, -3.45, 0, .43, .40),
                                   (34.85, -3.42, -.02, .49, .43),
                                   (35.20, -3.37, .0, .39, .32),
                                   (35.57, -3.33, .08, .22, .21),
                                   (35.87, -3.24, .11, .06, .06)), 48), "flame-gold")
    for i in range(9):
        theta = math.tau*i/9
        x = -3.45+.33*math.cos(theta)
        y = .28*math.sin(theta)
        put(f"torch-flame-tongue-{i:02d}", _tube([
            (x, y, 34.74), (x+.13*math.cos(theta), y+.1*math.sin(theta), 35.10),
            (x+.20*math.cos(theta), y+.17*math.sin(theta), 35.34+.17*(i%3)),
            (x+.26*math.cos(theta), y+.21*math.sin(theta), 35.56+.14*(i%4)),
        ], [.18, .15, .09, .012], 10), "flame-gold")

    # Keep every part in world-space and rescale about its ground datum. The
    # pedestal-off version drops the statue as one assembly to Z=0.
    shift = 0. if include_pedestal else -9.30
    scaled = []
    for part in result:
        mesh = Mesh(tuple((x*scale, y*scale, (z+shift)*scale)
                          for x, y, z in part.mesh.vertices), part.mesh.faces)
        scaled.append(place_mesh(mesh, name=part.name, material=part.material))
    verts = [v for part in scaled for v in world_vertices(part)]
    bounds_min = tuple(min(p[k] for p in verts) for k in range(3))
    bounds_max = tuple(max(p[k] for p in verts) for k in range(3))
    families = ("pedestal", "gown", "head", "hair", "crown", "torch", "tablet", "arm", "sleeve")
    counts = tuple((family, sum(part.name.startswith("liberty-"+family) for part in scaled))
                   for family in families)
    return LibertyStudy("statue-of-liberty-procedural", tuple(scaled), bounds_min,
                        bounds_max, counts, (("scale", float(scale)),
                                             ("include_pedestal", include_pedestal),
                                             ("detailed_tablet", detailed_tablet),
                                             ("detailed_crown", detailed_crown)))
