"""Original, parametric pyramid and lion-bodied royal monument studies.

These are Blender-independent geometric building blocks, NOT survey-accurate
copies of the monuments at Giza. Dimensions are meters; all meshes are closed.
Pyramid casing stones are separate solids, while the lion body, head, haunches
and paws use sculpted elliptical lofts rather than rectangular placeholders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .architecture_geometry import (
    GeometryError, Mesh, MeshInstance, Vec3, _finite_number,
    loft_elliptical_sections, place_mesh, world_vertices,
)
from .rock_weathering import erode_loft, rock_height_field, rough_bedrock_slab


class MonumentError(ValueError):
    """Invalid monument composition or dimensions."""


@dataclass(frozen=True)
class MonumentStudy:
    asset_id: str
    kind: str
    instances: tuple[MeshInstance, ...]
    part_counts: tuple[tuple[str, int], ...]
    bounds_min: Vec3
    bounds_max: Vec3
    parameters: tuple[tuple[str, float | int | bool], ...]


def pyramid_frustum(bottom_width: float, top_width: float, height: float) -> Mesh:
    """Closed, square-sided tapered stone mass, from Z=0 to Z=height."""
    b = _finite_number(bottom_width, "bottom_width", 0.05, 10000) / 2
    t = _finite_number(top_width, "top_width", 0.01, 10000) / 2
    h = _finite_number(height, "height", 0.01, 10000)
    if t > b:
        raise GeometryError("Pyramid frustum must not widen toward its top")
    return Mesh(((-b, -b, 0), (b, -b, 0), (b, b, 0), (-b, b, 0),
                 (-t, -t, h), (t, -t, h), (t, t, h), (-t, t, h)),
                ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
                 (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)))


def pyramid_casing_stone(bottom_width: float, top_width: float,
                         course_height: float, start: float, end: float, *,
                         inset: float = 0.2, joint: float = 0.015,
                         face_grid: int = 1, erosion_strength: float = 0.0,
                         seed: int = 0, height_offset: float = 0.0) -> Mesh:
    """One tapered, physically thick casing block for the pyramid's +Y face.

    ``start`` and ``end`` are fractions across a course's horizontal span.
    Rotate this right-handed solid around Z for the other three faces.
    Adjacent blocks have actual air joints, not drawn grid lines.
    """
    b = _finite_number(bottom_width, "bottom_width", 0.01, 10000)
    t = _finite_number(top_width, "top_width", 0.01, 10000)
    h = _finite_number(course_height, "course_height", 0.01, 100)
    inset = _finite_number(inset, "inset", 0.01, 10)
    joint = _finite_number(joint, "joint", 0, 1)
    u0 = _finite_number(start, "start", 0, 1)
    u1 = _finite_number(end, "end", 0, 1)
    if t > b or u1 - u0 <= 0.001:
        raise GeometryError("Casing stone needs a narrowing course and positive interval")
    if min(b, t) * (u1-u0) <= joint + 0.005:
        raise GeometryError("Casing block would collapse after its mortar joint")
    if inset >= t/2:
        raise GeometryError("Casing stone inset cannot reach the pyramid center")
    if type(face_grid) is not int or not 1 <= face_grid <= 12:
        raise GeometryError("Casing face_grid must be an integer from 1 to 12")
    erosion_strength = _finite_number(erosion_strength, "erosion_strength", 0, 0.5)
    height_offset = _finite_number(height_offset, "height_offset", -10000, 10000)
    if erosion_strength > min(inset*0.95, h*0.3):
        raise GeometryError("Casing relief must fit behind its stone edges")
    if type(seed) is not int or not 0 <= seed <= 100000:
        raise GeometryError("Casing seed must be an integer from 0 to 100000")
    x0, x1 = -b/2 + b*u0 + joint/2, -b/2 + b*u1 - joint/2
    x2, x3 = -t/2 + t*u0 + joint/2, -t/2 + t*u1 - joint/2
    y0, y1 = b/2, t/2
    # Four exterior corners followed by their recessed (interior) copies.
    verts = ((x0, y0, 0), (x1, y0, 0), (x3, y1, h), (x2, y1, h),
             (x0, y0-inset, 0), (x1, y0-inset, 0),
             (x3, y1-inset, h), (x2, y1-inset, h))
    faces = ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    if face_grid == 1:
        return Mesh(verts, faces)

    # A tessellated *physical* outside face provides silhouette-level chips
    # and light-catching relief rather than shader-only normal variation.
    # The inner surface stays planar against the solid tapered core.
    grid_w, grid_h = face_grid*2, face_grid
    row_width = grid_w+1
    layer_count = (grid_w+1)*(grid_h+1)
    points = []
    for outside in (True, False):
        for row in range(grid_h+1):
            v = row/grid_h
            left_x = x0*(1-v) + x2*v
            right_x = x1*(1-v) + x3*v
            yy = y0*(1-v) + y1*v
            zz = h*v
            for col in range(grid_w+1):
                u = col/grid_w
                xx = left_x*(1-u) + right_x*u
                dent = 0.0
                if outside:
                    edge = .18 + .82*math.sin(math.pi*u)*math.sin(math.pi*v)
                    dent = erosion_strength*edge*rock_height_field(
                        xx, yy, zz+height_offset, seed=seed,
                        strata_spacing=.31)
                points.append((xx, yy+(dent if outside else -inset), zz))
    all_faces: list[tuple[int, ...]] = []
    for row in range(grid_h):
        for col in range(grid_w):
            a = row*row_width+col
            b, c, d = a+1, a+row_width+1, a+row_width
            all_faces.append((d, c, b, a))
            all_faces.append((a+layer_count, b+layer_count,
                              c+layer_count, d+layer_count))
    perimeter = ([col for col in range(grid_w+1)] +
                 [row*row_width+grid_w for row in range(1, grid_h+1)] +
                 [grid_h*row_width+col for col in range(grid_w-1, -1, -1)] +
                 [row*row_width for row in range(grid_h-1, 0, -1)])
    for a, b in zip(perimeter, (*perimeter[1:], perimeter[0])):
        all_faces.append((a, b, b+layer_count, a+layer_count))
    return Mesh(tuple(points), tuple(all_faces))


def pyramid_assembly(*, prefix: str, base_width: float = 36.0,
                     height: float = 23.0, courses: int = 18,
                     block_width: float = 4.5, joint: float = 0.025,
                     casing_fraction: float = 0.18) -> tuple[MeshInstance, ...]:
    """A hollow-free pyramid core with individually editable four-face courses.

    Lower, light-colored Tura-inspired casing and upper weathered exterior
    can be controlled independently. Alternating courses offset vertical
    joints, and a few deterministic worn gaps expose the recessed core.
    """
    base = _finite_number(base_width, "base_width", 8, 1000)
    height = _finite_number(height, "height", 5, 700)
    target = _finite_number(block_width, "block_width", 0.4, 30)
    joint = _finite_number(joint, "joint", 0, 0.1)
    casing_fraction = _finite_number(casing_fraction, "casing_fraction", 0, 1)
    if type(courses) is not int or not 6 <= courses <= 128:
        raise GeometryError("courses must be an integer from 6 to 128")
    if height/base < 0.25 or height/base > 1.1:
        raise GeometryError("Pyramid height/base ratio must be in [0.25, 1.1]")
    cap_h = height / courses * 0.70
    work_h = height - cap_h
    cap_base = base * cap_h / height
    # A consistent tapered inner mass supports every exposed/weathered stone.
    inset = min(base*0.004, cap_base*0.25, 0.35)
    if joint > min(target/10, cap_base/12):
        raise GeometryError("Mortar joint is too wide for the upper courses")
    core = pyramid_frustum(base-2*inset, cap_base-2*inset, work_h)
    items = [place_mesh(core, name=f"{prefix}-core", material="core-limestone")]
    for course in range(courses):
        # Inset the tops/bottoms of consecutive courses to form a genuine
        # horizontal mortar seam above the recessed core, not a painted line.
        z0 = work_h * course / courses + (joint/2 if course else 0)
        z1 = work_h * (course+1) / courses - (joint/2 if course+1 < courses else 0)
        w0 = base*(1-z0/height)
        w1 = base*(1-z1/height)
        blocks = max(1, min(16, math.ceil(w1/target)))
        boundaries = ([0] + [(n+0.5)/blocks for n in range(blocks)] + [1]
                      if course % 2 else [n/blocks for n in range(blocks+1)])
        for face in range(4):
            angle = face*math.pi/2
            axes = {"axis_x": (round(math.cos(angle)), round(math.sin(angle)), 0),
                    "axis_y": (-round(math.sin(angle)), round(math.cos(angle)), 0),
                    "axis_z": (0, 0, 1)}
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                # Missing weathered blocks are an intentional, predictable
                # outer-casing loss. Solid core remains below the vacancy.
                covered = (course+0.5)/courses <= casing_fraction
                if not covered and (course*17+face*13+index*7) % 47 == 0:
                    continue
                stone = pyramid_casing_stone(
                    w0, w1, z1-z0, start, end, inset=inset, joint=joint,
                    face_grid=1 if covered else 5,
                    erosion_strength=0 if covered else min(.12, inset*.72),
                    height_offset=z0, seed=(course*71+face*13+index*3) % 100000)
                material = ("casing-limestone" if covered else
                            "weathered-limestone" if (course+face+index) % 3 else
                            "core-limestone")
                items.append(place_mesh(stone, name=f"{prefix}-course-{course:03d}-"
                                             f"face-{face}-stone-{index:02d}",
                                        origin=(0, 0, z0), material=material, **axes))
    cap = pyramid_frustum(cap_base, 0.045, cap_h)
    items.append(place_mesh(cap, name=f"{prefix}-capstone", origin=(0, 0, work_h),
                            material="casing-limestone"))
    return tuple(items)


def _sculpt_face_relief(mesh: Mesh, *, scale_x: float, scale_y: float,
                       scale_z: float) -> Mesh:
    """Carve shallow eye sockets, cheekbones and a worn, nose-less face plane.

    The original loft keeps its manifold faces. Local facial relief moves only
    the front-facing X surface, so the head remains a continuous closed solid.
    """
    vertices = []
    for px, py, pz in mesh.vertices:
        xx, yy, zz = px/scale_x, py/scale_y, pz/scale_z
        front = max(0.0, min(1.0, (-xx-4.00)/0.33))
        eye = math.exp(-((abs(yy)-.29)/.17)**2 - ((zz-4.035)/.115)**2)
        cheek = math.exp(-((abs(yy)-.35)/.19)**2 - ((zz-3.61)/.24)**2)
        broken_nose = math.exp(-(yy/.16)**2 - ((zz-3.72)/.16)**2)
        brow = math.exp(-((abs(yy)-.27)/.22)**2 - ((zz-4.20)/.10)**2)
        relief = front*(.085*eye + .028*broken_nose - .045*cheek - .023*brow)
        vertices.append((px+relief*scale_x, py, pz))
    return Mesh(tuple(vertices), mesh.faces)


def sphinx_assembly(*, prefix: str, length: float = 10.5,
                    width: float = 3.4, height: float = 4.6,
                    restored_nose: bool = False) -> tuple[MeshInstance, ...]:
    """Lion-bodied royal figure, assembled from shaped closed stone solids.

    -X is the front of its face and forepaws. This is an interpretive
    silhouette study, not a scan or archaeological restoration. A nose can
    be toggled explicitly; it is absent on the default present-day study.
    """
    length = _finite_number(length, "sphinx length", 3, 150)
    width = _finite_number(width, "sphinx width", 1.0, 50)
    height = _finite_number(height, "sphinx height", 1.5, 50)
    if type(restored_nose) is not bool:
        raise GeometryError("restored_nose must be boolean")
    x, y, z = length/10.5, width/3.4, height/4.6

    def along_x(points: tuple[tuple[float, float, float, float, float], ...], *,
                roughness: float = .045, seed: int = 17,
                exponent: float = 3.0) -> Mesh:
        mesh = loft_elliptical_sections(tuple((p*x, a*y, b*z, ra*y, rb*z)
                                              for p, a, b, ra, rb in points), axis="x",
                                        radial_segments=96, subdivisions=5,
                                        cross_section_exponent=exponent)
        return erode_loft(mesh, axis="x", radial_segments=96,
                          strength=roughness*min(y, z), seed=seed,
                          strata_spacing=.31*z)

    def along_z(points: tuple[tuple[float, float, float, float, float], ...], *,
                roughness: float = .025, seed: int = 19,
                exponent: float = 2.6) -> Mesh:
        mesh = loft_elliptical_sections(tuple((p*z, a*x, b*y, ra*x, rb*y)
                                              for p, a, b, ra, rb in points),
                                        radial_segments=96, subdivisions=5,
                                        cross_section_exponent=exponent)
        return erode_loft(mesh, axis="z", radial_segments=96,
                          strength=roughness*min(x, y), seed=seed,
                          strata_spacing=.29*z)

    def along_y(points: tuple[tuple[float, float, float, float, float], ...]) -> Mesh:
        return loft_elliptical_sections(tuple((p*y, a*z, b*x, ra*z, rb*x)
                                              for p, a, b, ra, rb in points), axis="y",
                                        radial_segments=32, subdivisions=3)

    items: list[MeshInstance] = []

    def part(name: str, mesh: Mesh, material: str = "sphinx-limestone") -> None:
        items.append(place_mesh(mesh, name=f"{prefix}-{name}", material=material))

    # The statue is hewn from a bedrock ridge, not perched on a tall plinth.
    # A low irregular terrain seam keeps the paws grounded in site material.
    part("bedrock", rough_bedrock_slab(length*1.065, width*1.37, height*.058,
                                       roughness=.10*min(x, y, z), seed=271),
         "bedrock-limestone")
    # Long recumbent lion back, tapering rump and forward shoulder masses.
    part("lion-torso", along_x(((-3.0, 0, 1.43, 0.73, 0.72),
                                (-2.5, 0, 1.45, 1.02, 0.85),
                                (-1.65, 0, 1.30, 1.19, 0.95),
                                (0.0, 0, 1.22, 1.30, 1.01),
                                (1.75, 0, 1.22, 1.30, 0.95),
                                (2.95, 0, 1.21, 1.14, 0.82),
                                (3.75, 0, 1.10, 0.76, 0.60),
                                (4.15, 0, 1.04, 0.20, 0.28)),
                               roughness=.16, seed=211, exponent=3.9),
         "sphinx-limestone")
    # A forward shoulder mass merges with the lion torso; the narrower neck
    # connects it to the headdress. The old single upstanding spheroid made the
    # figure read like a bust standing on a pedestal.
    part("chest", along_x(((-3.85, 0, 1.43, .48, .56),
                            (-3.54, 0, 1.43, .67, .72),
                            (-3.22, 0, 1.40, .83, .81),
                            (-2.76, 0, 1.32, 1.02, .94),
                            (-2.16, 0, 1.25, 1.17, .97)),
                          roughness=.13, seed=43, exponent=3.9))
    part("carved-neck", along_z(((1.77, -3.23, 0, .49, .70),
                                  (2.09, -3.28, 0, .48, .68),
                                  (2.44, -3.34, 0, .45, .60),
                                  (2.79, -3.38, 0, .43, .58),
                                  (3.07, -3.44, 0, .41, .60)),
                                 roughness=.073, seed=49, exponent=3.4))
    for side, sign in (("left", -1), ("right", 1)):
        side_y = sign*1.07
        part(f"{side}-forepaw", along_x(((-5.43, side_y, .32, .22, .19),
                                         (-5.12, side_y, .33, .33, .26),
                                         (-4.70, side_y, .35, .39, .30),
                                         (-4.03, side_y, .38, .43, .34),
                                         (-3.34, side_y, .43, .45, .37),
                                         (-2.75, side_y, .53, .45, .46),
                                         (-2.20, side_y, .74, .42, .61),
                                         (-1.90, side_y, .87, .37, .63)),
                                        roughness=.13, seed=61, exponent=4.1))
        for toe in range(3):
            toe_y = side_y + (toe-1)*.22
            part(f"{side}-toe-{toe}", along_x(((-5.49, toe_y, .24, .065, .095),
                                                (-5.32, toe_y, .24, .12, .14),
                                                (-4.94, toe_y, .25, .14, .15)),
                                               roughness=.017, seed=73),
                 "weathered-limestone")
        part(f"{side}-haunch", along_x(((1.90, sign*1.03, 1.03, .37, .56),
                                       (2.40, sign*1.04, 1.07, .62, .81),
                                       (3.14, sign*1.02, 1.03, .67, .84),
                                       (3.75, sign*.98, .84, .50, .65),
                                       (4.08, sign*.94, .60, .18, .31)),
                                      roughness=.135, seed=151, exponent=3.6))
    part("curved-tail", along_x(((2.1, -1.04, .61, .14, .15),
                                (2.7, -1.26, .58, .16, .17),
                                (3.4, -1.38, .65, .15, .15),
                                (3.86, -1.27, .74, .13, .12),
                                (4.13, -1.12, .82, .09, .09)),
                               roughness=.025, seed=156))
    # Broad rounded nemes headdress sits behind a separate sculpted face.
    part("nemes-hood", along_z(((2.70, -3.38, 0, .65, .80),
                                (3.22, -3.48, 0, .74, 1.00),
                                (3.76, -3.51, 0, .77, 1.08),
                                (4.16, -3.46, 0, .74, 1.03),
                                (4.47, -3.46, 0, .60, .84),
                                (4.60, -3.50, 0, .40, .66)),
                               roughness=.055, seed=89, exponent=3.8), "casing-limestone")
    face = along_z(((2.92, -3.85, 0, .32, .28),
                          (3.17, -3.90, 0, .47, .44),
                          (3.58, -3.90, 0, .54, .52),
                          (3.95, -3.86, 0, .54, .54),
                          (4.31, -3.84, 0, .48, .52),
                          (4.48, -3.85, 0, .35, .45)),
                         roughness=.019, seed=97, exponent=2.7)
    part("face", _sculpt_face_relief(face, scale_x=x, scale_y=y, scale_z=z))
    for side, sign in (("left", -1), ("right", 1)):
        part(f"nemes-{side}-lappet", along_z(((2.18, -3.42, sign*.83, .20, .24),
                                             (2.60, -3.46, sign*.85, .25, .29),
                                             (3.25, -3.51, sign*.89, .28, .29),
                                             (3.88, -3.49, sign*.91, .30, .25)),
                                            roughness=.040, seed=105, exponent=3.5),
             "casing-limestone")
        part(f"{side}-ear", along_z(((3.81, -3.83, sign*.58, .16, .10),
                                    (4.13, -3.80, sign*.62, .20, .18),
                                    (4.42, -3.79, sign*.61, .12, .10)),
                                   roughness=.023, seed=118))
        eye_y = sign*.28
        part(f"{side}-eye", along_y(((eye_y-.16, 4.015, -4.365, .016, .02),
                                    (eye_y, 4.02, -4.39, .075, .058),
                                    (eye_y+.16, 4.015, -4.365, .016, .02))),
             "weathered-limestone")
        part(f"{side}-brow", along_y(((eye_y-.20, 4.18, -4.35, .035, .025),
                                     (eye_y, 4.20, -4.40, .07, .065),
                                     (eye_y+.20, 4.18, -4.35, .035, .025))))
        for stripe in range(5):
            # Narrow relief collars follow the lappet's curved cross-section.
            # Flat overlay boxes previously gave the headdress a toy look.
            bottom = (2.43+stripe*.29)*z
            collar = loft_elliptical_sections(
                ((bottom, -3.48*x, sign*.87*y, .295*x, .302*y),
                 (bottom+.042*z, -3.48*x, sign*.87*y, .295*x, .302*y)),
                radial_segments=64, cross_section_exponent=3.5)
            part(f"{side}-hood-stripe-{stripe}", collar, "weathered-limestone")
    part("mouth-relief", along_y(((-.27, 3.40, -4.390, .015, .014),
                                  (0, 3.37, -4.410, .025, .028),
                                  (.27, 3.40, -4.390, .015, .014))),
         "sphinx-shadow")
    if restored_nose:
        part("optional-restored-nose", along_z(((3.52, -4.43, 0, .13, .17),
                                                (3.82, -4.52, 0, .19, .13),
                                                (4.18, -4.45, 0, .09, .12))))
    return tuple(items)


def make_monument_study(*, kind: str = "giza-study",
                        pyramid_base: float = 36.0, pyramid_height: float = 23.0,
                        pyramid_courses: int = 18, casing_fraction: float = 0.18,
                        sphinx_length: float = 10.5, restored_nose: bool = False) -> MonumentStudy:
    """Compose a pyramid, a lion-bodied royal figure or a staged pair."""
    if kind not in ("pyramid", "sphinx", "giza-study"):
        raise MonumentError("kind must be pyramid, sphinx or giza-study")
    # Validate the whole exposed recipe even if an unused optional model is
    # disabled. Otherwise a sphinx-only plan could emit non-finite JSON in
    # its manifest, or accept an unsafe setting that changes behavior later.
    pyramid_base = _finite_number(pyramid_base, "pyramid_base", 8, 1000)
    pyramid_height = _finite_number(pyramid_height, "pyramid_height", 5, 700)
    casing_fraction = _finite_number(casing_fraction, "casing_fraction", 0, 1)
    sphinx_length = _finite_number(sphinx_length, "sphinx_length", 3, 150)
    if type(pyramid_courses) is not int or not 6 <= pyramid_courses <= 128:
        raise MonumentError("pyramid_courses must be an integer from 6 to 128")
    if type(restored_nose) is not bool:
        raise MonumentError("restored_nose must be boolean")
    parts: list[MeshInstance] = []
    if kind in ("pyramid", "giza-study"):
        shift_y = 8.0 if kind == "giza-study" else 0.0
        for part in pyramid_assembly(prefix="pyramid", base_width=pyramid_base,
                                     height=pyramid_height, courses=pyramid_courses,
                                     casing_fraction=casing_fraction):
            parts.append(replace(part, origin=(part.origin[0], part.origin[1]+shift_y,
                                               part.origin[2])))
    if kind in ("sphinx", "giza-study"):
        shift_y = -(pyramid_base/2+8.0) if kind == "giza-study" else 0.0
        for part in sphinx_assembly(prefix="sphinx", length=sphinx_length,
                                    restored_nose=restored_nose):
            parts.append(replace(part, origin=(part.origin[0], part.origin[1]+shift_y,
                                               part.origin[2]+0.02)))
    mins = [float("inf")]*3
    maxs = [float("-inf")]*3
    for part in parts:
        for vertex in world_vertices(part):
            for axis in range(3):
                mins[axis] = min(mins[axis], vertex[axis])
                maxs[axis] = max(maxs[axis], vertex[axis])
    return MonumentStudy(
        asset_id=f"egyptian-{kind}-detail-study", kind=kind,
        instances=tuple(parts),
        part_counts=(("pyramid_parts", sum(p.name.startswith("pyramid-") for p in parts)),
                     ("sphinx_parts", sum(p.name.startswith("sphinx-") for p in parts))),
        bounds_min=tuple(mins), bounds_max=tuple(maxs),
        parameters=(("pyramid_base_m", pyramid_base), ("pyramid_height_m", pyramid_height),
                    ("pyramid_courses", pyramid_courses), ("casing_fraction", casing_fraction),
                    ("sphinx_length_m", sphinx_length), ("restored_nose", restored_nose)),
    )
