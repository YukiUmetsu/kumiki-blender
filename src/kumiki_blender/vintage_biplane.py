"""Original, reference-guided 1920s-style red biplane, without Blender imports.

Blender coordinates: X spans the wings, the nose points toward -Y, Z is up.
This is a dimensional interpretation of the supplied orthographic illustration,
not a measured reproduction or a claim about any particular historical aircraft.
Every part is a real closed volume, named for editable component/rig workflows.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .architecture_geometry import Mesh, loft_elliptical_sections

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class AirplanePart:
    name: str
    mesh: Mesh
    material: str
    group: str
    rig: str | None = None


@dataclass(frozen=True)
class BiplaneModel:
    asset_id: str
    parts: tuple[AirplanePart, ...]
    pivots: dict[str, Vec3]
    bounds_min: Vec3
    bounds_max: Vec3
    part_counts: dict[str, int]


def _oriented(vertices: list[Vec3], faces: list[tuple[int, ...]]) -> Mesh:
    """Close a constructed part and normalize outward orientation by volume."""
    volume = 0.0
    for face in faces:
        a = vertices[face[0]]
        for i in range(1, len(face) - 1):
            b, c = vertices[face[i]], vertices[face[i + 1]]
            volume += (a[0]*(b[1]*c[2]-b[2]*c[1])
                       + a[1]*(b[2]*c[0]-b[0]*c[2])
                       + a[2]*(b[0]*c[1]-b[1]*c[0])) / 6.0
    if volume < 0:
        faces = [tuple(reversed(face)) for face in faces]
    return Mesh(tuple(vertices), tuple(faces))


def _tube(a: Vec3, b: Vec3, radius: float, sides: int = 10) -> Mesh:
    direction = tuple(b[i] - a[i] for i in range(3))
    length = math.sqrt(sum(v*v for v in direction))
    if length < 1e-6:
        raise ValueError("A structural tube must have non-zero length")
    forward = tuple(v/length for v in direction)
    ref = (0, 0, 1) if abs(forward[2]) < .91 else (0, 1, 0)
    cross = lambda u, v: (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    across = cross(forward, ref)
    norm = math.sqrt(sum(v*v for v in across))
    u = tuple(v/norm for v in across)
    v = cross(forward, u)
    vertices = [tuple(origin[k] + radius*(math.cos(math.tau*j/sides)*u[k]
                          + math.sin(math.tau*j/sides)*v[k]) for k in range(3))
                for origin in (a, b) for j in range(sides)]
    faces = [(j, (j+1)%sides, sides+(j+1)%sides, sides+j) for j in range(sides)]
    faces += [tuple(reversed(range(sides))), tuple(range(sides, 2*sides))]
    return _oriented(vertices, faces)


def _curved_rib(points: tuple[Vec3, ...], radius: float = .003) -> Mesh:
    """Single continuous rib tape following camber instead of 7 disconnected rods."""
    sides = 8
    vertices: list[Vec3] = []
    for p in points:
        for j in range(sides):
            angle = math.tau*j/sides
            # Rib runs in Y; X and Z are the stable normal plane.
            vertices.append((p[0]+radius*math.cos(angle),
                             p[1], p[2]+radius*math.sin(angle)))
    faces = [(i*sides+j, (i+1)*sides+j, (i+1)*sides+(j+1)%sides,
              i*sides+(j+1)%sides)
             for i in range(len(points)-1) for j in range(sides)]
    faces += [tuple(range(sides)),
              tuple(reversed(range((len(points)-1)*sides, len(points)*sides)))]
    return _oriented(vertices, faces)


def _loft_y(sections: tuple[tuple[float, float, float, float, float], ...],
            sides: int = 32) -> Mesh:
    """Fuselage panels: (longitudinal Y, X center, Z center, X radius, Z radius)."""
    # The reusable loft kernel's Y cross-plane order is (Z, X), not (X, Z).
    return loft_elliptical_sections(tuple((y, cz, cx, rz, rx)
                                          for y, cx, cz, rx, rz in sections),
                                    axis="y", radial_segments=sides,
                                    subdivisions=3, cross_section_exponent=2.45)


def _foil(u: float) -> float:
    """Rounded vintage fabric airfoil; local chord position 0=leading, 1=trailing."""
    return .2969*math.sqrt(max(0, u)) - .126*u - .3516*u*u + .2843*u**3 - .1015*u**4


def _fixed_trailing_fraction(x: float, span: float, cut_at: float) -> float:
    """End an aileron before the last rounded wingtip without an open notch."""
    if x <= cut_at:
        return 1.
    if x < cut_at+.03:
        return 1. - .265*(x-cut_at)/.03
    if x <= span-.63:
        return .735
    q = min(1., (x-(span-.63))/.20)
    return .735+.265*q*q*(3-2*q)


def _wing_half(side: int, *, span: float, chord: float, center_y: float,
               center_z: float, x_start: float = .10, x_end: float | None = None,
               leading: float = 0., trailing: float = 1., rib_pitch: float = .255,
               cut_at: float | None = None, sweep_m: float = .025) -> Mesh:
    """A solid, cambered and rib-tensioned fabric half-wing, including capped tips.

    The outboard main portion stops at the hinge line to make a genuine,
    detached aileron; moving surfaces do not overlap buried full-chord wings.
    """
    if x_end is None:
        x_end = span
    count = max(3, math.ceil((x_end - x_start) / (rib_pitch/2)))
    span_points = [x_start + (x_end-x_start)*i/count for i in range(count+1)]
    if cut_at is not None:
        span_points = sorted(set((*span_points, cut_at, cut_at+.03,
                                  span-.63, span-.43,
                                  *(span-.43+.043*j for j in range(11)))))
    # Added stations are real surface sections, not spare vertices: stitch
    # every station through the tip, including the aileron transition.
    count = len(span_points)-1
    n = 28
    vertices: list[Vec3] = []
    for i, x in enumerate(span_points):
        # Rounded planform: the last 43 cm follow an elliptical tip curve.
        tip = min(1., max(0., (x-(span-.43))/.43))
        tip_scale = max(.025, math.sqrt(max(0., 1-tip*tip)))
        # Cloth is taut at each wooden rib and slightly sunken between ribs.
        tension = 1.0 + (.018 if i % 2 == 0 else -.012)
        actual_trailing = (_fixed_trailing_fraction(x, span, cut_at)
                           if cut_at is not None else trailing)
        u_top = [leading+(actual_trailing-leading)*j/14 for j in range(15)]
        contour = u_top + list(reversed(u_top[1:-1]))
        for j, u in enumerate(contour):
            top = j < len(u_top)
            thickness = .48*chord*_foil(u)*tension*tip_scale
            thickness += .0024  # finite trailing edge and full manifold caps
            camber = .018*chord*math.sin(math.pi*u)
            # Wing leading edges may sweep, while the elevator hinge at 70%
            # chord stays straight along X for a mechanically honest rig.
            sweep = sweep_m*(x/span)**2*max(0., (.70-u)/.70)
            y = center_y + (u-.5)*chord*tip_scale + sweep
            z = center_z + camber + (thickness if top else -thickness)
            vertices.append((side*x, y, z))
    faces = []
    for i in range(count):
        for j in range(n):
            k = (j+1)%n
            faces.append((i*n+j, (i+1)*n+j, (i+1)*n+k, i*n+k))
    faces += [tuple(range(n)), tuple(reversed(range(count*n, (count+1)*n)))]
    return _oriented(vertices, faces)


def _plate_yz(profile: tuple[tuple[float, float], ...], width: float) -> Mesh:
    """A thick, organically profiled fin with full front and rear cap faces."""
    n = len(profile)
    vertices = [(x, y, z) for x in (-width/2, width/2) for y, z in profile]
    faces = [(j, (j+1)%n, n+(j+1)%n, n+j) for j in range(n)]
    faces += [tuple(reversed(range(n))), tuple(range(n, 2*n))]
    return _oriented(vertices, faces)


def _windscreen() -> Mesh:
    """Transparent slanted glass sheet with a physical 8 mm thick edge."""
    corners = ((-.32, -.92, 1.81), (.32, -.92, 1.81),
               (.25, -1.03, 2.035), (-.25, -1.03, 2.035))
    vertices = [(*p[:1], p[1]+offset, p[2]) for offset in (-.004, .004)
                for p in corners]
    faces = [(3, 2, 1, 0), (4, 5, 6, 7)]
    faces += [(i, (i+1)%4, 4+(i+1)%4, 4+i) for i in range(4)]
    return _oriented(vertices, faces)


def _cowling_access_panel(side: int) -> Mesh:
    """Curved riveted removable service cover that actually hugs the cowl."""
    # Two spanwise strips along the longitudinal body, curved around the
    # ellipse rather than clipping a flat rectangular plate into the skin.
    vertices: list[Vec3] = []
    for radial_offset in (.003, .012):
        for y in (-2.87, -2.18):
            for z in (1.25, 1.33, 1.43, 1.53, 1.61):
                hull_x = .48*math.sqrt(1-((z-1.43)/.47)**2)
                vertices.append((side*(hull_x+radial_offset), y, z))
    faces: list[tuple[int, ...]] = []
    # Thin cap/sheet with 4x1 pieces on each face and closed edges.
    for j in range(4):
        faces.append((j, 5+j, 6+j, 1+j))
        faces.append((10+j, 11+j, 16+j, 15+j))
        faces.append((j, j+1, 11+j, 10+j))
        faces.append((5+j, 15+j, 16+j, 6+j))
    faces.extend([(0, 10, 15, 5), (4, 9, 19, 14)])
    return _oriented(vertices, faces)


def _torus(center: Vec3, major: float, tube: float) -> Mesh:
    """Real toroidal rubber tire, axle parallel to X; no hollow-disc shortcut."""
    v: list[Vec3] = []
    rings, sides = 36, 12
    for a in range(rings):
        phi = math.tau*a/rings
        for b in range(sides):
            theta = math.tau*b/sides
            rr = major+tube*math.sin(theta)
            v.append((center[0]+tube*math.cos(theta),
                      center[1]+rr*math.cos(phi), center[2]+rr*math.sin(phi)))
    f = [(a*sides+b, ((a+1)%rings)*sides+b,
          ((a+1)%rings)*sides+(b+1)%sides, a*sides+(b+1)%sides)
         for a in range(rings) for b in range(sides)]
    return _oriented(v, f)


def _prop_blade(direction: int) -> Mesh:
    sections = []
    for i in range(11):
        radius = .12 + 1.14*i/10
        bulb = math.sin(math.pi*(.10+.87*i/10))**.65
        width = .015+.125*bulb
        sections.append((1.48+direction*radius, .045*math.sin(i/10*math.pi),
                         -3.52+.018*i/10, width, .032-.01*i/10))
    # Loft around Z; section fields are (Z, X center, Y center, X radius, Y radius).
    return loft_elliptical_sections(tuple(sorted(sections)), axis="z", radial_segments=16)


def animation_keyframes() -> dict[str, tuple[tuple[int, float], ...]]:
    """A 96-frame demonstration loop, in radians, with separate mechanical axes."""
    degrees = math.radians
    aileron_left = ((1, 0.), (25, degrees(13)), (49, 0.),
                    (73, degrees(-13)), (97, 0.))
    aileron_right = tuple((frame, -angle) for frame, angle in aileron_left)
    return {
        "propeller": ((1, 0.), (97, math.tau*8)),
        "upper-left-aileron": aileron_left,
        "upper-right-aileron": aileron_right,
        "lower-left-aileron": aileron_left,
        "lower-right-aileron": aileron_right,
        "elevator": ((1, 0.), (25, degrees(7)), (49, 0.), (73, degrees(-7)), (97, 0.)),
        "rudder": ((1, 0.), (25, degrees(9)), (49, 0.), (73, degrees(-9)), (97, 0.)),
        "left-wheel": ((1, 0.), (97, math.tau*3)),
        "right-wheel": ((1, 0.), (97, math.tau*3)),
    }


def make_biplane() -> BiplaneModel:
    """Create a detailed, deterministic multiview aircraft component assembly."""
    parts: list[AirplanePart] = []

    def add(name: str, mesh: Mesh, material: str, group: str, rig: str | None = None) -> None:
        parts.append(AirplanePart(name, mesh, material, group, rig))

    def rod(name: str, a: Vec3, b: Vec3, radius: float,
            material: str, group: str, rig: str | None = None) -> None:
        add(name, _tube(a, b, radius), material, group, rig)

    pivots: dict[str, Vec3] = {
        "propeller": (0, -3.52, 1.48),
        "upper-left-aileron": (-3.57, -1.216, 2.65),
        "upper-right-aileron": (3.57, -1.216, 2.65),
        "lower-left-aileron": (-3.57, -1.067, 1.55),
        "lower-right-aileron": (3.57, -1.067, 1.55),
        "elevator": (0, 3.265, 1.17),
        "rudder": (0, 3.49, 1.70),
        "left-wheel": (-1.09, -2.15, .43),
        "right-wheel": (1.09, -2.15, .43),
    }

    # Front cowling, separate cockpit opening, narrowing turtledeck and tail.
    add("fuselage-forward", _loft_y((
        (-3.19, 0, 1.46, .39, .42), (-3.00, 0, 1.44, .46, .45),
        (-2.40, 0, 1.43, .49, .47), (-1.65, 0, 1.42, .47, .46),
        (-.95, 0, 1.40, .43, .43), (-.82, 0, 1.40, .43, .43))), "red-lacquer", "fuselage")
    add("fuselage-tail-taper", _loft_y((
        (-.22, 0, 1.40, .43, .42), (.15, 0, 1.38, .42, .42),
        (.75, 0, 1.31, .36, .34), (1.55, 0, 1.24, .28, .26),
        (2.50, 0, 1.19, .18, .16), (3.38, 0, 1.15, .105, .13))), "red-lacquer", "fuselage")
    # Dark, visibly recessed cockpit pit and the leather padded oval coaming.
    add("cockpit-well", _loft_y(((-.81, 0, 1.27, .33, .28),
                                (-.25, 0, 1.27, .33, .28))), "cockpit-dark", "cockpit")
    for side in (-1, 1):
        rod(f"cockpit-side-coaming-{side}", (side*.39, -.88, 1.81),
            (side*.39, -.20, 1.81), .045, "saddle-leather", "cockpit")
    rod("cockpit-front-coaming", (-.39, -.88, 1.81), (.39, -.88, 1.81), .045, "saddle-leather", "cockpit")
    rod("cockpit-rear-coaming", (-.39, -.20, 1.81), (.39, -.20, 1.81), .045, "saddle-leather", "cockpit")
    for side, label in ((-1, "left"), (1, "right")):
        add(f"cockpit-{label}-side-fairing", _loft_y((
            (-.83, side*.35, 1.42, .105, .385),
            (-.54, side*.35, 1.42, .11, .39),
            (-.22, side*.34, 1.42, .10, .375))),
            "red-lacquer", "cockpit")
    add("cockpit-seat", _loft_y(((-.40, 0, 1.23, .24, .16),
                                (-.27, 0, 1.23, .24, .16))), "saddle-leather", "cockpit")
    rod("cockpit-stick", (0, -.62, 1.12), (0, -.62, 1.70), .018, "brass", "cockpit")
    rod("cockpit-instrument-crossbar", (-.25, -.82, 1.61),
        (.25, -.82, 1.61), .052, "cockpit-dark", "cockpit")
    for i, x in enumerate((-.15, 0, .15)):
        rod(f"cockpit-instrument-dial-{i}", (x, -.79, 1.63),
            (x, -.76, 1.63), .041, "brass", "cockpit")
        rod(f"cockpit-instrument-face-{i}", (x, -.751, 1.63),
            (x, -.745, 1.63), .026, "engine-dark", "cockpit")
    rod("windscreen-left-post", (-.30, -.94, 1.80), (-.25, -1.03, 2.04), .015, "brass", "cockpit")
    rod("windscreen-right-post", (.30, -.94, 1.80), (.25, -1.03, 2.04), .015, "brass", "cockpit")
    rod("windscreen-top", (-.25, -1.03, 2.04), (.25, -1.03, 2.04), .016, "brass", "cockpit")
    add("windscreen-glass", _windscreen(), "windscreen-glass", "cockpit")

    # Five-sided front engine intake, rim stack and two cylinders emerging from cowl.
    add("engine-cowl-front-rim", _loft_y(((-3.255, 0, 1.48, .40, .42),
                                         (-3.18, 0, 1.48, .43, .44))), "deep-red", "engine")
    add("engine-black-radiator", _loft_y(((-3.288, 0, 1.48, .30, .31),
                                          (-3.249, 0, 1.48, .30, .31))), "engine-dark", "engine")
    for j in range(8):
        x = -.24+j*.068
        rod(f"engine-grille-bar-{j:02d}", (x, -3.299, 1.23),
            (x, -3.299, 1.72), .008, "aged-steel", "engine")
    for side in (-1, 1):
        add(f"exposed-exhaust-cylinder-{side}", _loft_y((
            (-2.76, side*.24, 1.82, .065, .18),
            (-2.53, side*.24, 1.82, .065, .18))), "engine-dark", "engine")
        rod(f"exhaust-pipe-{side}", (side*.37, -2.63, 1.56),
            (side*.51, -1.82, 1.55), .033, "aged-steel", "engine")
        label = "left" if side < 0 else "right"
        add(f"engine-{label}-access-panel", _cowling_access_panel(side),
            "deep-red", "engine")
        for j in range(6):
            y = -2.77+j*.105
            rod(f"engine-{label}-louver-{j:02d}",
                (side*.481, y, 1.34), (side*.480, y+.018, 1.55),
                .014, "aged-steel", "engine")
    for j in range(22):
        ang = math.tau*j/22
        x, z = .42*math.cos(ang), 1.48+.43*math.sin(ang)
        rod(f"cowl-rivet-{j:02d}", (x, -3.23, z), (x, -3.20, z),
            .012, "brass", "fastener")

    for level, span, chord, cy, cz in (("upper", 4.83, 1.77, -1.65, 2.65),
                                       ("lower", 4.66, 1.48, -1.43, 1.55)):
        add(f"{level}-center-wing", _wing_half(1, span=span,
            chord=chord, center_y=cy, center_z=cz, x_start=-.14,
            x_end=.14, rib_pitch=.255), "fabric-red", "wing")
        for side, name in ((-1, "left"), (1, "right")):
            add(f"{level}-{name}-wing", _wing_half(side, span=span, chord=chord,
                  center_y=cy, center_z=cz, rib_pitch=.255, cut_at=3.58),
                "fabric-red", "wing")
            # Hinge begins at 73.5% chord, a real movable fabric-covered flap.
            add(f"{level}-{name}-aileron", _wing_half(side, span=span,
                  chord=chord, center_y=cy, center_z=cz, x_start=3.60, x_end=span-.63,
                  leading=.745, trailing=1., rib_pitch=.255),
                "fabric-red", "aileron", f"{level}-{name}-aileron")
            for i in range(1, 18):
                x = i*.255
                if x >= span-.28:
                    continue
                # Tape follows the actual top-surface camber, not a floating
                # straight beam; subtle ribs visible under studio highlights.
                main_trailing = _fixed_trailing_fraction(x, span, 3.58)
                def point(u: float) -> Vec3:
                    return (side*x, cy+(u-.5)*chord,
                            cz + .018*chord*math.sin(math.pi*u)
                            + .48*chord*_foil(u)*1.028 + .0025)
                add(f"{level}-{name}-rib-{i:02d}",
                    _curved_rib(tuple(point(main_trailing*j/16) for j in range(17))),
                    "rib-red", "wing-rib")
                if 3.60 <= x <= span-.63:
                    add(f"{level}-{name}-aileron-rib-{i:02d}",
                        _curved_rib(tuple(point(.745 + (1-.745)*j/8) for j in range(9))),
                        "rib-red", "wing-rib", f"{level}-{name}-aileron")
            # Black trailing-edge hinge bead emphasizes control-surface separation.
            rod(f"{level}-{name}-aileron-hinge", (side*3.61,
                cy+(.74-.5)*chord, cz), (side*(span-.64),
                cy+(.74-.5)*chord, cz), .010, "deep-red", "hinge")

    # Vertical cabane and outer interplane struts: weathered varnished timber.
    for side, name in ((-1, "left"), (1, "right")):
        for station in (.63, 2.15, 3.92):
            for edge, yoffset in (("front", -.43), ("rear", .44)):
                lower_x = (.39 if station == .63 else station)*side
                upper_x = station*side
                rod(f"{name}-interplane-{station:.2f}-{edge}",
                    (lower_x, -1.43+yoffset*.86, 1.60),
                    (upper_x, -1.65+yoffset, 2.61), .045,
                    "varnished-wood", "strut")
        for bay, (a, b) in enumerate(((.68, 2.15), (2.15, 3.92))):
            for diagonal, (x0, x1) in enumerate(((a, b), (b, a))):
                for edge, y0, y1 in (("leading", -1.83, -2.07),
                                      ("trailing", -1.02, -.91)):
                    rod(f"{name}-bay-{bay}-{diagonal}-{edge}-wire",
                        (side*x0, y0, 1.60), (side*x1, y1, 2.61),
                        .009, "cable-steel", "flying-wire")
        # Internal cabane X-bracing between fuselage and upper center panel.
        for k in (0, 1):
            rod(f"{name}-cabane-wire-{k}",
                (side*.43, -1.13 if k else -2.05, 1.85),
                (side*.63, -2.09 if k else -1.13, 2.62),
                .010, "cable-steel", "flying-wire")

    # Rounded horizontal stabilizer and trailing, hinged elevators.
    for side, name in ((-1, "left"), (1, "right")):
        add(f"tail-{name}-stabilizer", _wing_half(side, span=1.88,
            chord=.94, center_y=3.07, center_z=1.17, x_start=.095,
            leading=0., trailing=.68, rib_pitch=.20, sweep_m=.35), "fabric-red", "tail")
        add(f"tail-{name}-elevator", _wing_half(side, span=1.88,
            chord=.94, center_y=3.07, center_z=1.17, x_start=.095,
            leading=.70, trailing=1., rib_pitch=.20, sweep_m=.35),
            "fabric-red", "elevator", "elevator")
        rod(f"tail-{name}-brace", (side*1.1, 3.03, 1.13),
            (side*.20, 2.40, 1.06), .019, "aged-steel", "strut")
    add("vertical-tail-fin", _plate_yz(((2.01, 1.26), (2.18, 1.36),
        (2.44, 1.55), (2.71, 1.76), (2.94, 1.95), (3.15, 2.10),
        (3.32, 2.17), (3.43, 2.17), (3.48, 2.11), (3.48, 1.45),
        (3.29, 1.22)), .13), "fabric-red", "tail")
    add("rudder", _plate_yz(((3.49, 2.19), (3.63, 2.11), (3.68, 1.80),
                              (3.69, 1.30), (3.50, 1.16), (3.49, 1.42)), .12),
        "fabric-red", "rudder", "rudder")
    rod("rudder-hinge-pin", (0, 3.485, 1.36), (0, 3.485, 2.16),
        .013, "brass", "hinge")

    # Sprung fixed undercarriage: angled timber legs, axle, wire bracing.
    for side, name in ((-1, "left"), (1, "right")):
        wheel = (side*1.09, -2.15, .43)
        rod(f"{name}-landing-leg", (side*.33, -1.98, 1.42), wheel,
            .067, "varnished-wood", "landing-gear")
        rod(f"{name}-gear-rear-brace", (side*.40, -1.28, 1.35), wheel,
            .036, "varnished-wood", "landing-gear")
        rod(f"{name}-wheel-axle", (side*.92, -2.15, .43),
            (side*1.10, -2.15, .43), .052, "aged-steel", "landing-gear")
        add(f"{name}-rubber-tire", _torus(wheel, .31, .095),
            "aged-rubber", "wheel", f"{name}-wheel")
        # Lacquered hub is thicker than the wheel flange and can turn with it.
        add(f"{name}-red-wheel-hub", _tube((side*1.155, -2.15, .43),
            (side*1.035, -2.15, .43), .27, sides=32),
            "deep-red", "wheel", f"{name}-wheel")
        rod(f"{name}-hub-fastener", (side*1.17, -2.15, .43),
            (side*1.19, -2.15, .43), .038, "brass", "fastener", f"{name}-wheel")
        # Pressed metal hub detailing is parented to the wheel pivot, so
        # rotation is perceptible even though the period-correct tire is smooth.
        add(f"{name}-hub-trim-ring", _torus((side*1.17, -2.15, .43),
            .238, .010), "brass", "wheel-trim", f"{name}-wheel")
        for rivet in range(8):
            theta = math.tau*rivet/8
            y = -2.15+.18*math.cos(theta)
            z = .43+.18*math.sin(theta)
            rod(f"{name}-hub-rivet-{rivet:02d}",
                (side*1.16, y, z), (side*1.192, y, z),
                .012, "brass", "fastener", f"{name}-wheel")
        # One asymmetric brass balance clip gives the rolling tire a readable
        # reference point; unlike a fake rotating texture, it is physical mesh.
        theta = math.pi/5
        rod(f"{name}-wheel-balance-weight",
            (side*1.184, -2.15+.216*math.cos(theta), .43+.216*math.sin(theta)),
            (side*1.184, -2.15+.267*math.cos(theta), .43+.267*math.sin(theta)),
            .018, "brass", "wheel-trim", f"{name}-wheel")
    rod("landing-axle-crossbar", (-1.09, -2.15, .43), (1.09, -2.15, .43),
        .047, "varnished-wood", "landing-gear")
    for side in (-1, 1):
        rod(f"landing-crosswire-{side}", (side*.36, -1.65, 1.36),
            (-side*1.09, -2.15, .43), .009, "cable-steel", "flying-wire")
    rod("tail-skid", (0, 3.20, 1.08), (0, 3.40, .87),
        .040, "varnished-wood", "landing-gear")

    # Two separately selectable propeller blades, independent bronze hub.
    for sign, name in ((1, "upper"), (-1, "lower")):
        add(f"propeller-{name}-blade", _prop_blade(sign),
            "prop-wood", "propeller-blade", "propeller")
    rod("propeller-brass-hub", (0, -3.59, 1.48), (0, -3.43, 1.48),
        .125, "brass", "propeller-hub", "propeller")
    rod("propeller-axle", (0, -3.44, 1.48), (0, -3.22, 1.48),
        .067, "engine-dark", "engine")

    vertices = [v for part in parts for v in part.mesh.vertices]
    counts = dict(sorted(Counter(part.group for part in parts).items()))
    return BiplaneModel("red-vintage-biplane", tuple(parts), pivots,
                        tuple(min(v[a] for v in vertices) for a in range(3)),
                        tuple(max(v[a] for v in vertices) for a in range(3)), counts)
