"""Original, parametric pyramid and lion-bodied royal monument studies.

These are Blender-independent geometric building blocks, NOT survey-accurate
copies of the monuments at Giza. Dimensions are meters; all meshes are closed.
Pyramid casing stones are separate solids, while the lion body, head, haunches
and paws use sculpted elliptical lofts rather than rectangular placeholders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .architecture_details import box_mesh
from .architecture_geometry import (
    GeometryError, Mesh, MeshInstance, Vec3, _finite_number,
    loft_elliptical_sections, place_mesh, world_vertices,
)


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
                         inset: float = 0.2, joint: float = 0.015) -> Mesh:
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
    x0, x1 = -b/2 + b*u0 + joint/2, -b/2 + b*u1 - joint/2
    x2, x3 = -t/2 + t*u0 + joint/2, -t/2 + t*u1 - joint/2
    y0, y1 = b/2, t/2
    # Four exterior corners followed by their recessed (interior) copies.
    verts = ((x0, y0, 0), (x1, y0, 0), (x3, y1, h), (x2, y1, h),
             (x0, y0-inset, 0), (x1, y0-inset, 0),
             (x3, y1-inset, h), (x2, y1-inset, h))
    faces = ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    return Mesh(verts, faces)


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
                stone = pyramid_casing_stone(w0, w1, z1-z0, start, end,
                                             inset=inset, joint=joint)
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

    def along_x(points: tuple[tuple[float, float, float, float, float], ...]) -> Mesh:
        return loft_elliptical_sections(tuple((p*x, a*y, b*z, ra*y, rb*z)
                                              for p, a, b, ra, rb in points), axis="x",
                                        radial_segments=64, subdivisions=3)

    def along_z(points: tuple[tuple[float, float, float, float, float], ...]) -> Mesh:
        return loft_elliptical_sections(tuple((p*z, a*x, b*y, ra*x, rb*y)
                                              for p, a, b, ra, rb in points),
                                        radial_segments=64, subdivisions=3)

    def along_y(points: tuple[tuple[float, float, float, float, float], ...]) -> Mesh:
        return loft_elliptical_sections(tuple((p*y, a*z, b*x, ra*z, rb*x)
                                              for p, a, b, ra, rb in points), axis="y",
                                        radial_segments=32, subdivisions=3)

    items: list[MeshInstance] = []

    def part(name: str, mesh: Mesh, material: str = "sphinx-limestone") -> None:
        items.append(place_mesh(mesh, name=f"{prefix}-{name}", material=material))

    part("bedrock", box_mesh(length*1.06, width*1.35, height*0.13), "bedrock-limestone")
    # Long recumbent lion back, tapering rump and forward shoulder masses.
    part("lion-torso", along_x(((-3.0, 0, 1.43, 0.73, 0.72),
                                (-2.5, 0, 1.45, 1.02, 0.85),
                                (-1.65, 0, 1.30, 1.19, 0.95),
                                (0.0, 0, 1.22, 1.30, 1.01),
                                (1.75, 0, 1.22, 1.30, 0.95),
                                (2.95, 0, 1.21, 1.14, 0.82),
                                (3.75, 0, 1.10, 0.76, 0.60),
                                (4.15, 0, 1.04, 0.20, 0.28))))
    part("chest", along_z(((0.45, -2.95, 0, 0.68, 0.85),
                           (0.85, -2.96, 0, 0.87, 1.03),
                           (1.50, -2.95, 0, 0.85, 0.99),
                           (2.20, -2.85, 0, 0.72, 0.85),
                           (2.72, -2.86, 0, 0.45, 0.61))))
    for side, sign in (("left", -1), ("right", 1)):
        side_y = sign*1.07
        part(f"{side}-forepaw", along_x(((-5.35, side_y, .35, .22, .21),
                                         (-5.10, side_y, .34, .35, .26),
                                         (-4.70, side_y, .37, .41, .30),
                                         (-3.8, side_y, .42, .42, .34),
                                         (-2.6, side_y, .61, .46, .50),
                                         (-1.85, side_y, 1.00, .38, .80))))
        for toe in range(3):
            toe_y = side_y + (toe-1)*.22
            part(f"{side}-toe-{toe}", along_x(((-5.49, toe_y, .24, .065, .095),
                                                (-5.32, toe_y, .24, .12, .14),
                                                (-4.94, toe_y, .25, .14, .15))),
                 "weathered-limestone")
        part(f"{side}-haunch", along_x(((1.90, sign*1.03, 1.03, .37, .56),
                                       (2.40, sign*1.04, 1.07, .62, .81),
                                       (3.14, sign*1.02, 1.03, .67, .84),
                                       (3.75, sign*.98, .84, .50, .65),
                                       (4.08, sign*.94, .60, .18, .31))))
    part("curved-tail", along_x(((2.1, -1.06, .65, .16, .17),
                                (2.8, -1.42, .60, .18, .18),
                                (3.7, -1.50, .67, .17, .16),
                                (4.4, -1.35, .82, .14, .12),
                                (4.75, -1.12, .97, .12, .12))))
    # Broad rounded nemes headdress sits behind a separate sculpted face.
    part("nemes-hood", along_z(((2.70, -3.38, 0, .65, .80),
                                (3.22, -3.48, 0, .74, 1.00),
                                (3.76, -3.51, 0, .77, 1.08),
                                (4.16, -3.46, 0, .74, 1.03),
                                (4.47, -3.46, 0, .60, .84),
                                (4.60, -3.50, 0, .40, .66))), "casing-limestone")
    part("face", along_z(((2.92, -3.85, 0, .32, .28),
                          (3.17, -3.90, 0, .47, .44),
                          (3.58, -3.90, 0, .54, .52),
                          (3.95, -3.86, 0, .54, .54),
                          (4.31, -3.84, 0, .48, .52),
                          (4.48, -3.85, 0, .35, .45))))
    for side, sign in (("left", -1), ("right", 1)):
        part(f"nemes-{side}-lappet", along_z(((2.18, -3.42, sign*.83, .20, .24),
                                             (2.60, -3.46, sign*.85, .25, .29),
                                             (3.25, -3.51, sign*.89, .28, .29),
                                             (3.88, -3.49, sign*.91, .30, .25))),
             "casing-limestone")
        part(f"{side}-ear", along_z(((3.81, -3.83, sign*.58, .16, .10),
                                    (4.13, -3.80, sign*.62, .20, .18),
                                    (4.42, -3.79, sign*.61, .12, .10))))
        eye_y = sign*.28
        part(f"{side}-eye", along_y(((eye_y-.16, 4.015, -4.365, .016, .02),
                                    (eye_y, 4.02, -4.39, .075, .058),
                                    (eye_y+.16, 4.015, -4.365, .016, .02))),
             "sphinx-shadow")
        part(f"{side}-brow", along_y(((eye_y-.20, 4.18, -4.35, .035, .025),
                                     (eye_y, 4.20, -4.40, .07, .065),
                                     (eye_y+.20, 4.18, -4.35, .035, .025))))
        for stripe in range(5):
            items.append(place_mesh(box_mesh(.40*x, .41*y, .045*z),
                                    name=f"{prefix}-{side}-hood-stripe-{stripe}",
                                    origin=(-3.71*x, sign*.90*y, (2.42+stripe*.29)*z),
                                    material="weathered-limestone"))
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
                                               part.origin[2]+0.30)))
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
