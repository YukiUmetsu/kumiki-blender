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
from .pyramid_survey import (
    KHUFU_ORIGINAL_BASE_M, KHUFU_ORIGINAL_HEIGHT_M,
    PETRIE_LAST_PRESERVED_COURSE, petrie_course_tops_m,
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
    parameters: tuple[tuple[str, float | int | bool | str], ...]


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
                    mineral = rock_height_field(xx, yy, zz+height_offset,
                                                seed=seed, strata_spacing=.31)
                    # High-frequency, asymmetrically broken arrises survive
                    # distant views better than a smooth sine-wave height map.
                    chips = math.sin(xx*13.7 + seed*.31)*math.cos(
                        (zz+height_offset)*17.9 - seed*.17)
                    pit = max(0.0, mineral*.72 + chips*.48)**2
                    interior = math.sin(math.pi*u)*math.sin(math.pi*v)
                    dent = erosion_strength*((.40+.60*interior)*(
                        mineral*.75 + chips*.34) - .66*pit)
                    # Damage boundary vertices *inward* from each mortar
                    # edge. The back face remains a solid, tapered wedge;
                    # the outside face has genuinely irregular chip outlines.
                    chip_depth = erosion_strength*.48*max(
                        0.0, abs(mineral*.65 + chips*.65)-.12)
                    if col == 0:
                        xx += chip_depth
                    elif col == grid_w:
                        xx -= chip_depth
                    if row == 0:
                        zz += chip_depth*.48
                    elif row == grid_h:
                        zz -= chip_depth*.48
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
                     casing_fraction: float = 0.04) -> tuple[MeshInstance, ...]:
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
    inset = min(base*0.008, cap_base*0.30, 0.45)
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
                    face_grid=1 if covered else 8,
                    erosion_strength=0 if covered else min(.215, inset*.78, (z1-z0)*.22),
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


def khufu_petrie_assembly(*, prefix: str,
                          last_course: int = PETRIE_LAST_PRESERVED_COURSE,
                          target_block_width: float = 4.2) -> tuple[MeshInstance, ...]:
    """Measured course-height study, with scalable batched exterior masonry.

    Course elevations come from Petrie's 1883 NE measurements. Course widths
    follow a *nominal* 230.4 m base / 146.5 m original apex envelope, so block
    positions, counts, erosion and individual cuts are illustrative, not a
    measured archaeological plan. The model ends in a flat damaged top,
    without the invented pristine capstone of the miniature preset.

    Four batches per course (one per cardinal face) keep the Blender scene
    below ~810 objects instead of tens of thousands of loose stone objects.
    Each batch is an ordinary mesh holding separate watertight stone wedges,
    with open mortar joints and physically chipped close-range stones.
    """
    tops = petrie_course_tops_m(last_course=last_course)
    target = _finite_number(target_block_width, "target_block_width", 1.5, 12)
    base = KHUFU_ORIGINAL_BASE_M
    apex = KHUFU_ORIGINAL_HEIGHT_M
    top_height = tops[-1]
    inset, joint = 0.42, 0.028
    top_width = base*(1-top_height/apex)
    core = pyramid_frustum(base-2*inset, top_width-2*inset, top_height)
    items: list[MeshInstance] = [
        place_mesh(core, name=f"{prefix}-core", material="core-limestone")
    ]

    previous_top = 0.0
    for course, course_top in enumerate(tops):
        # A level represents the *top* of one measured course. We retain a
        # narrow physical gap at both horizontal stone-bed joints.
        lower = previous_top + (joint/2 if course else 0.0)
        upper = course_top - (joint/2 if course < len(tops)-1 else 0.0)
        previous_top = course_top
        bottom_w = base*(1-lower/apex)
        upper_w = base*(1-upper/apex)
        count = max(2, math.ceil(upper_w/target))
        if course % 2:
            intervals = [0.0] + [(i+0.5)/count for i in range(count)] + [1.0]
        else:
            intervals = [i/count for i in range(count+1)]
        # Close masonry is physically roughened, but distant upper courses
        # retain real mortar joints with cheaper watertight wedge geometry.
        grid = 4 if course < 12 else (2 if course < 42 or course % 11 == 0 else 1)
        erosion = min(0.19, inset*.72, (upper-lower)*.23) if grid > 1 else 0.0
        for face in range(4):
            offset = face*math.pi/2
            axes = {
                "axis_x": (round(math.cos(offset)), round(math.sin(offset)), 0),
                "axis_y": (-round(math.sin(offset)), round(math.cos(offset)), 0),
                "axis_z": (0, 0, 1),
            }
            vertices: list[Vec3] = []
            faces: list[tuple[int, ...]] = []
            for stone_no, (start, end) in enumerate(zip(intervals, intervals[1:])):
                block = pyramid_casing_stone(
                    bottom_w, upper_w, upper-lower, start, end,
                    inset=inset, joint=joint, face_grid=grid,
                    erosion_strength=erosion, height_offset=lower,
                    seed=(course*71+face*19+stone_no*7) % 100000)
                index_offset = len(vertices)
                vertices.extend(block.vertices)
                faces.extend(tuple(i+index_offset for i in polygon) for polygon in block.faces)
            material = ("weathered-limestone" if course % 5 else "core-limestone")
            items.append(place_mesh(
                Mesh(tuple(vertices), tuple(faces)),
                name=f"{prefix}-course-{course+1:03d}-face-{face}-batch",
                origin=(0, 0, lower), material=material, **axes))
    return tuple(items)


def _sculpt_face_relief(mesh: Mesh, *, scale_x: float, scale_y: float,
                       scale_z: float) -> Mesh:
    """Carve anatomical planes and surviving facial features into the head.

    Relief is applied to the existing manifold face shell rather than layering
    protruding eye/brow blobs onto it. The present-day nose has a preserved
    bridge but a chipped, recessed lower tip; lips and chin project subtly
    from the jaw. All displacements are in *normalized* statue coordinates.
    """
    vertices = []
    for px, py, pz in mesh.vertices:
        xx, yy, zz = px/scale_x, py/scale_y, pz/scale_z
        front = max(0.0, min(1.0, (-xx-3.96)/.31))
        front = front*front*(3.0-2.0*front)
        lateral = abs(yy)
        # Real surviving lids are shallow worn creases, not two deep black
        # sockets. Cycles frontal QA also exposed the triangular dark hole
        # made by the previous "broken tip" displacement: the broken nose
        # should end in a scarred *stub*, not look punched through.
        # Landmark spacing follows the reference's long, almost trapezoidal
        # face: convex brows, individually bounded almond eyes, surviving
        # nasal bridge, small lips and a distinct projecting chin. An older
        # super-Gaussian eye mask had a wide flat center, hence the two
        # featureless dark rectangular slots seen in the first Cycles QA.
        eye = math.exp(-((lateral-.302)/.151)**2*1.6
                       - ((zz-4.073)/.058)**2*1.6)
        upper_lid = math.exp(-((lateral-.305)/.154)**2*1.6
                             - ((zz-4.132)/.031)**2)
        lower_lid = math.exp(-((lateral-.305)/.153)**2*1.5
                             - ((zz-4.015)/.030)**2)
        brow = math.exp(-((lateral-.320)/.180)**2*1.4
                        - ((zz-4.209)/.057)**2)
        cheekbone = math.exp(-((lateral-.385)/.170)**2 - ((zz-3.800)/.170)**2)
        hollow = math.exp(-((lateral-.505)/.118)**2 - ((zz-3.963)/.174)**2)
        nose_root = math.exp(-(yy/.120)**2 - ((zz-4.056)/.156)**2)
        bridge = math.exp(-(yy/.112)**2 - ((zz-3.923)/.202)**2)
        nose_flanks = math.exp(-((lateral-.128)/.067)**2
                               - ((zz-3.816)/.130)**2)
        broken_tip = math.exp(-(yy/.123)**4 - ((zz-3.727)/.082)**2)
        upper_lip = math.exp(-(yy/.204)**4 - ((zz-3.435)/.047)**2)
        # The slightly bowed, asymmetrically chipped mouth line is narrower
        # than the broad mineral bedding cuts. A .010 m cut disappeared in
        # Blender's .019 m head-voxel union; this groove is true geometry.
        mouth_height = 3.375+.019*(yy/.22)**2+.008*yy
        mouth_cut = math.exp(-(yy/.226)**8
                             - ((zz-mouth_height)/.022)**2)
        lower_lip = math.exp(-(yy/.202)**4 - ((zz-3.325)/.044)**2)
        philtrum = math.exp(-((lateral-.067)/.037)**2
                             - ((zz-3.532)/.075)**2)
        chin = math.exp(-(yy/.262)**4 - ((zz-3.115)/.110)**2)
        jaw = math.exp(-((lateral-.347)/.158)**2 - ((zz-3.255)/.206)**2)
        # Subtle surviving vertical folds cross the exposed upper forehead.
        # Crown-only incisions were mostly buried behind the lofted face in
        # the voxel union, leaving a visually blank helmet-like brow.
        forehead = max(0., min(1., (zz-4.23)/.10, (4.43-zz)/.055))
        forehead *= max(0., min(1., (.58-lateral)/.075))
        fold = max(0., math.cos(yy*26.5+.14*math.sin(zz*8)))**10
        wear = rock_height_field(px*4.7, py*4.7, pz*4.7,
                                 seed=301, strata_spacing=.42)
        eye_chip = math.exp(-((yy+.315)/.116)**2
                            - ((zz-4.109)/.085)**2)
        broken_cheek = math.exp(-((yy-.418)/.132)**2
                                - ((zz-3.801)/.118)**2)
        # X points toward the rear: negative means a projected cheek/nasal
        # bridge, positive means recesses. The bridge ends in an uneven,
        # visibly broken nasal scar rather than an intact spheroidal nose.
        relief = front*(.013*eye - .039*upper_lid - .023*lower_lid
                        - .047*brow - .035*cheekbone + .015*hollow
                        - .021*nose_root - .112*bridge + .010*nose_flanks
                        + (.007 + .007*wear)*broken_tip
                        - .024*upper_lip + .018*mouth_cut
                        - .020*lower_lip - .019*philtrum
                        - .058*chin - .035*jaw
                        + .041*forehead*fold + .013*wear*(1-.45*eye)
                        + (.035+.009*wear)*eye_chip
                        + (.030+.011*wear)*broken_cheek)
        # Angular jaw and temporal narrowing break the toy-like oval facial
        # silhouette, while the back half of the head stays untouched.
        jaw_width = .022*jaw - .015*hollow
        vertices.append((px+relief*scale_x,
                         py + math.copysign(jaw_width*scale_y*front, yy), pz))
    return Mesh(tuple(vertices), mesh.faces)


def _elongate_anatomical_face(mesh: Mesh, *, scale_y: float,
                             scale_z: float) -> Mesh:
    """Stretch *finished-world* facial proportions without stretching the lion.

    The reference face was evaluated in source coordinates while the statue's
    final Z scaling was only 3.6/4.6. A 1.26 source height/width ratio thus
    rendered at barely 1.0 in actual meters: a squat human face. Stretch its
    landmark regions nonuniformly, lifting the upper orbital plane, dropping
    the mouth and extending the jaw and chin. Keep crown height, neck, body,
    and headdress fixed, and compress the facial width slightly. This bends
    the same closed surface without disconnected sculptural add-ons.
    """
    def ease(value: float) -> float:
        t = max(0., min(1., value))
        return t*t*(3.-2.*t)

    vertices = []
    for px, py, pz in mesh.vertices:
        yy = py/scale_y
        zz = pz/scale_z
        orbital_lift = .125*math.exp(-((zz-4.06)/.335)**2)
        jaw_drop = .270*ease((3.82-zz)/.84)
        # A 92% facial width makes world-space face height / cheek width
        # genuinely elongated. Full headcloth/ear widths are not rescaled.
        vertices.append((px, yy*.920*scale_y,
                         (zz+orbital_lift-jaw_drop)*scale_z))
    return Mesh(tuple(vertices), mesh.faces)


def _anatomical_sphinx_face(*, scale_x: float, scale_y: float,
                           scale_z: float) -> Mesh:
    """Closed landmark-shaped facial surface, not an elliptical loft mask.

    Front-face vertices are sampled across a height-dependent trapezoidal
    silhouette. Brow ridges, orbital cavities, paired eyelids, nasal bridge,
    cheekbone planes, mouth and chin are continuous *surface geometry* in the
    same mesh. A joined rear and capped sides give Blender a watertight solid
    for the existing voxel sculpt workflow. The landmarks are interpretive
    proportions informed by frontal/profile photographs, not survey points.
    """
    profile = (
        (2.69, .30), (2.81, .36), (3.02, .41), (3.31, .465),
        (3.61, .525), (3.91, .559), (4.14, .551),
        (4.34, .519), (4.51, .473),
    )

    def half_width(z: float) -> float:
        for (low_z, low_w), (high_z, high_w) in zip(profile, profile[1:]):
            if z <= high_z:
                t = max(0., min(1., (z-low_z)/(high_z-low_z)))
                return low_w+(high_w-low_w)*t
        return profile[-1][1]

    def blob(y: float, z: float, cy: float, cz: float,
             ry: float, rz: float, *, exponent: float = 2.) -> float:
        return math.exp(-(abs((y-cy)/ry)**exponent)
                        -(abs((z-cz)/rz)**exponent))

    def anterior_depth(y: float, z: float, width: float) -> float:
        lateral = abs(y)
        u = lateral/width
        # Broad cheek and forehead planes slope gently back at the temples;
        # the actual front is not an elliptical cross-section.
        xx = -4.235+.206*u*u+.025*((z-3.75)/.8)**2
        interior = (1.-min(1., u)**6)**2
        # Bony orbital rims are projected ahead of the tapered almond eye
        # cavities. The two rim curves converge at each eye corner instead
        # of forming the old flat, dark, super-Gaussian horizontal slots.
        for sign in (-1, 1):
            cy = sign*.292
            aperture_u = (y-cy)/.177
            if abs(aperture_u) < 1.:
                half_open = .040*(max(0., 1.-aperture_u**2))**.72
                eye_center = 3.965+.013*aperture_u*sign
                xx += .040*blob(y, z, cy, eye_center,
                                 .154, max(.017, half_open), exponent=2.5)
                xx -= .055*blob(y, z, cy, eye_center+half_open+.022,
                                 .167, .027, exponent=2.)
                xx -= .030*blob(y, z, cy, eye_center-half_open-.022,
                                 .158, .030, exponent=2.)
            xx -= .071*blob(y, z, sign*.305, 4.088,
                             .191, .065, exponent=2.)
            xx -= .063*blob(y, z, sign*.397, 3.725,
                             .169, .203, exponent=2.)
            # Subtle nasolabial transition separates the lower cheek from
            # the surviving nasal scar without pasted-on spheres.
            xx += .026*blob(y, z, sign*.198, 3.620,
                             .049, .169, exponent=2.)
        # Vertical nasal root/bridge and interrupted broken tip. Leave a
        # proud surviving bridge, NOT a triangular hole at the nose's end.
        xx -= .122*blob(y, z, 0., 3.870, .112, .158, exponent=2.)
        xx -= .028*blob(y, z, 0., 3.719, .124, .078, exponent=2.)
        xx += .016*blob(y, z, -.055, 3.703, .075, .052)
        # Central philtrum, irregular bowed mouth cut, paired lip volumes.
        mouth_z = 3.275+.020*(y/.24)**2+.005*y
        xx -= .035*(blob(y, z, -.062, 3.450, .033, .097) +
                    blob(y, z, .068, 3.449, .033, .097))
        xx -= .039*blob(y, z, 0., 3.339, .240, .063, exponent=4.)
        xx += .024*blob(y, z, 0., mouth_z, .233, .019, exponent=4.)
        xx -= .031*blob(y, z, 0., 3.223, .216, .051, exponent=4.)
        xx -= .087*blob(y, z, 0., 2.934, .302, .145, exponent=3.)
        xx += .036*blob(y, z, 0., 2.725, .272, .069, exponent=2.)
        # Physically incised crown pleats and differential mineral wear.
        fold_mask = max(0., min(1., (z-4.29)/.055, (4.54-z)/.075))
        xx += .036*fold_mask*max(0., math.cos(y*28.))**12
        wear = rock_height_field(xx*3.7, y*4.6, z*5.1,
                                 seed=319, strata_spacing=.31)
        xx += .019*wear
        xx += .037*blob(y, z, -.332, 4.183, .099, .086)*max(0., .65+.25*wear)
        xx += .024*blob(y, z, .426, 3.784, .119, .128)*max(0., .65+.25*wear)
        # At the border, facial relief smoothly vanishes so the thick side
        # surface joins the hood without floating anatomy or hard seams.
        return (-4.235+.206*u*u+.025*((z-3.75)/.8)**2 +
                (xx-(-4.235+.206*u*u+.025*((z-3.75)/.8)**2))*interior)

    rows, cols = 129, 97
    front_count = rows*cols
    vertices: list[tuple[float, float, float]] = []
    for front in (True, False):
        for row in range(rows):
            z = profile[0][0]+(profile[-1][0]-profile[0][0])*row/(rows-1)
            width = half_width(z)
            for col in range(cols):
                y = width*(2.*col/(cols-1)-1.)
                x = (anterior_depth(y, z, width) if front
                     else -3.39+.065*(abs(y)/width)**2)
                vertices.append((x*scale_x, y*scale_y, z*scale_z))

    def index(row: int, col: int, front: bool) -> int:
        return (0 if front else front_count)+row*cols+col

    faces: list[tuple[int, ...]] = []
    for row in range(rows-1):
        for col in range(cols-1):
            a, b = index(row, col, True), index(row, col+1, True)
            c, d = index(row+1, col+1, True), index(row+1, col, True)
            faces.append((d, c, b, a))
            a, b = index(row, col, False), index(row, col+1, False)
            c, d = index(row+1, col+1, False), index(row+1, col, False)
            faces.append((a, b, c, d))
    # Stitch all four perimeter edges. Separate front/back grids remain
    # editable in one watertight solid, with no unowned boundary vertices.
    border = ([(0, col) for col in range(cols)] +
              [(row, cols-1) for row in range(1, rows)] +
              [(rows-1, col) for col in range(cols-2, -1, -1)] +
              [(row, 0) for row in range(rows-2, 0, -1)])
    for i, (row, col) in enumerate(border):
        next_row, next_col = border[(i+1) % len(border)]
        a, b = index(row, col, True), index(next_row, next_col, True)
        c, d = index(next_row, next_col, False), index(row, col, False)
        faces.append((a, b, c, d))
    return Mesh(tuple(vertices), tuple(faces))


def _sculpt_ear_relief(mesh: Mesh, *, side: int, scale_x: float,
                      scale_y: float, scale_z: float) -> Mesh:
    """Carve a raised outer helix and recessed concha in each embedded ear.

    The ears in the reference photographs are recognizably human, with a
    cupped interior and rounded rim, not smooth teardrop blocks dangling in
    front of the nemes. Preserve the source mesh's closed manifold topology.
    """
    if side not in (-1, 1) or type(side) is not int:
        raise GeometryError("Ear side must be -1 or 1")
    vertices = []
    for px, py, pz in mesh.vertices:
        xx, yy, zz = px/scale_x, py/scale_y, pz/scale_z
        outward = abs(yy)
        anterior = max(0., min(1., (-xx-4.015)/.125))
        anterior = anterior*anterior*(3.-2.*anterior)
        # A symmetric circular ring became a hollow doughnut after voxel
        # union. Real ears have an offset inner bowl and an upper curved rim,
        # with softer lower cartilage: avoid using |z-z_center| for the helix.
        concha = math.exp(-((zz-3.995)/.070)**2
                          - ((outward-.735)/.052)**2)
        upper_helix = math.exp(-((zz-4.155)/.052)**2
                               - ((outward-.749)/.081)**2)
        lower_cartilage = math.exp(-((zz-3.919)/.060)**2
                                   - ((outward-.746)/.064)**2)
        curved_rim = math.exp(-((zz-4.074)/.115)**2
                              - ((outward-.816)/.043)**2)
        displacement = anterior*(.064*concha-.038*upper_helix
                                 -.014*lower_cartilage-.014*curved_rim)
        vertices.append((px+displacement*scale_x, py, pz))
    return Mesh(tuple(vertices), mesh.faces)


def _nemes_drapery_wing(*, side: int, scale_x: float, scale_y: float,
                       scale_z: float) -> Mesh:
    """Watertight, swept royal headcloth wing based on frontal elevations.

    In the actual Sphinx the nemes spreads *outward and backward* from the
    temple, ending in a broad, almost horizontal shoulder-level hem. It is
    not a vertical cylindrical lappet. A two-sided subdivided strip creates
    its continuous triangular silhouette and shallow carved cloth/stone
    banding without an image texture or zero-thickness geometry.

    The input photographs inform the silhouette, not absolute survey points.
    Dimensions below describe the normalized 10.5 m educational statue.
    """
    if side not in (-1, 1) or type(side) is not int:
        raise GeometryError("Nemes wing side must be -1 or 1")
    # Z, near-face inner Y, outside Y. The outer edge sweeps to the flank;
    # the bottom hem is broad and squared off rather than pinching to a tip.
    stations = (
        (3.10, .75, 1.51),
        (3.21, .76, 1.57),
        (3.36, .77, 1.57),
        (3.61, .74, 1.49),
        (3.95, .68, 1.18),
        (4.23, .58, .87),
        (4.49, .49, .67),
    )
    divisions_z, divisions_y = 8, 20
    samples = []
    for index in range(len(stations)-1):
        low, high = stations[index], stations[index+1]
        for step in range(divisions_z):
            u = step/divisions_z
            samples.append(tuple(a+(b-a)*u for a, b in zip(low, high)))
    samples.append(stations[-1])
    rows, cols = len(samples), divisions_y+1
    vertices: list[tuple[float, float, float]] = []
    for front in (True, False):
        for zz, inner, outer in samples:
            for column in range(cols):
                t = column/divisions_y
                yy = inner+(outer-inner)*t
                taper = min(1., (zz-3.10)/.15, (4.49-zz)/.16)
                # Front pleats go across the width of the triangular stone
                # sheet. Their real relief survives the fine head voxel union.
                strata = .007*math.sin(zz*62 + t*1.7)
                grain = .014*rock_height_field(-3.9, yy, zz,
                                               seed=197, strata_spacing=.22)
                relief = max(0., taper)*(strata+grain)
                # The outer hem sweeps backward onto the lion shoulder.
                # Its X/Z diagonal is essential in profile; a constant-X
                # sheet reads like a plank attached to the skull.
                front_x = (-3.99+.08*t +
                           .62*t*(4.49-zz)/1.39 + relief)
                xx = front_x if front else front_x+.54-.04*t-.70*relief
                vertices.append((xx*scale_x, side*yy*scale_y, zz*scale_z))

    def idx(row: int, col: int, front: bool) -> int:
        return (0 if front else rows*cols)+row*cols+col

    faces: list[tuple[int, ...]] = []
    for row in range(rows-1):
        for col in range(cols-1):
            for front in (True, False):
                a, b = idx(row, col, front), idx(row, col+1, front)
                c, d = idx(row+1, col+1, front), idx(row+1, col, front)
                quad = (a, b, c, d)
                faces.append(tuple(reversed(quad)) if (front == (side == 1)) else quad)
    # Top, bottom and inner/outer edges close the physically thick cloth.
    border = ([(0, col) for col in range(cols)] +
              [(row, cols-1) for row in range(1, rows)] +
              [(rows-1, col) for col in range(cols-2, -1, -1)] +
              [(row, 0) for row in range(rows-2, 0, -1)])
    for index, (row, col) in enumerate(border):
        next_row, next_col = border[(index+1) % len(border)]
        a, b = idx(row, col, True), idx(next_row, next_col, True)
        c, d = idx(next_row, next_col, False), idx(row, col, False)
        faces.append((a, b, c, d) if side == 1 else (b, a, d, c))
    return Mesh(tuple(vertices), tuple(faces))


def _carve_nemes_crown(mesh: Mesh, *, scale_x: float, scale_y: float,
                       scale_z: float) -> Mesh:
    """Incise surviving linen-fold channels into the broad forehead/crown.

    Royal nemes folds run roughly crown-to-brow over the front and then turn
    into lateral bands on the wings. Recessed channels preserve a continuous
    carved volume; no floating decorative stripes or shader-only grooves.
    """
    vertices = []
    for px, py, pz in mesh.vertices:
        xx, yy, zz = px/scale_x, py/scale_y, pz/scale_z
        crown = max(0., min(1., (zz-4.17)/.22))
        crown *= max(0., min(1., (4.60-zz)/.065))
        front = max(0., min(1., (-xx-3.87)/.23))
        front = front*front*(3-2*front)
        channel = max(0., math.cos(yy*21.0+.34*math.sin(zz*7)))**8
        fractured_band = math.exp(-((zz-4.31)/.025)**2)
        relief = crown*front*(.030*channel + .014*fractured_band)
        vertices.append((px+relief*scale_x, py, pz))
    return Mesh(tuple(vertices), mesh.faces)


def sphinx_assembly(*, prefix: str, length: float = 10.5,
                    width: float | None = None, height: float | None = None,
                    restored_nose: bool = False) -> tuple[MeshInstance, ...]:
    """Lion-bodied royal figure, assembled from shaped closed stone solids.

    -X is the front of its face and forepaws. This is an interpretive
    silhouette study, not a scan or archaeological restoration. A nose can
    be toggled explicitly; it is absent on the default present-day study.
    """
    length = _finite_number(length, "sphinx length", 3, 150)
    width = _finite_number(width if width is not None else length*3.4/10.5,
                           "sphinx width", .9, 150)
    height = _finite_number(height if height is not None else length*3.6/10.5,
                            "sphinx height", 1.0, 150)
    if type(restored_nose) is not bool:
        raise GeometryError("restored_nose must be boolean")
    x, y, z = length/10.5, width/3.4, height/4.6

    def along_x(points: tuple[tuple[float, float, float, float, float], ...], *,
                roughness: float = .045, seed: int = 17,
                exponent: float = 3.0, bedding: float = .045,
                pitting: float = .027) -> Mesh:
        mesh = loft_elliptical_sections(tuple((p*x, a*y, b*z, ra*y, rb*z)
                                              for p, a, b, ra, rb in points), axis="x",
                                        radial_segments=96, subdivisions=5,
                                        cross_section_exponent=exponent)
        return erode_loft(mesh, axis="x", radial_segments=96,
                          strength=roughness*min(y, z), seed=seed,
                          strata_spacing=.31*z,
                          bedding_strength=bedding*min(y, z),
                          pitting_strength=pitting*min(y, z))

    def along_z(points: tuple[tuple[float, float, float, float, float], ...], *,
                roughness: float = .025, seed: int = 19,
                exponent: float = 2.6, subdivisions: int = 5,
                bedding: float = .009, pitting: float = .009) -> Mesh:
        mesh = loft_elliptical_sections(tuple((p*z, a*x, b*y, ra*x, rb*y)
                                              for p, a, b, ra, rb in points),
                                        radial_segments=96, subdivisions=subdivisions,
                                        cross_section_exponent=exponent)
        return erode_loft(mesh, axis="z", radial_segments=96,
                          strength=roughness*min(x, y), seed=seed,
                          strata_spacing=.29*z,
                          bedding_strength=bedding*min(x, y),
                          pitting_strength=pitting*min(x, y))

    items: list[MeshInstance] = []

    def part(name: str, mesh: Mesh, material: str = "sphinx-limestone") -> None:
        items.append(place_mesh(mesh, name=f"{prefix}-{name}", material=material))

    # The statue is hewn from a bedrock ridge, not perched on a tall plinth.
    # A low irregular terrain seam keeps the paws grounded in site material.
    part("bedrock", rough_bedrock_slab(length*1.065, width*1.37, height*.058,
                                       roughness=.10*min(x, y, z), seed=271),
         "bedrock-limestone")
    # Long recumbent lion back, tapering rump and forward shoulder masses.
    part("lion-torso", along_x(((-3.30, 0, 1.57, .68, .76),
                                (-2.86, 0, 1.80, 1.00, 1.00),
                                (-2.28, 0, 1.64, 1.16, 1.00),
                                (-1.20, 0, 1.28, 1.22, .94),
                                (0.25, 0, 1.17, 1.29, .90),
                                (1.65, 0, 1.18, 1.29, .89),
                                (2.85, 0, 1.28, 1.13, .90),
                                (3.67, 0, 1.10, .78, .69),
                                (4.16, 0, .99, .22, .29)),
                               roughness=.17, seed=211, exponent=2.25,
                               bedding=.083, pitting=.061),
         "sphinx-limestone")
    # A forward shoulder mass merges with the lion torso; the narrower neck
    # connects it to the headdress. The old single upstanding spheroid made the
    # figure read like a bust standing on a pedestal.
    part("chest", along_x(((-3.91, 0, 1.49, .54, .64),
                            (-3.60, 0, 1.49, .72, .82),
                            (-3.25, 0, 1.45, .92, .94),
                            (-2.80, 0, 1.36, 1.07, 1.02),
                            (-2.14, 0, 1.30, 1.21, .99)),
                          roughness=.14, seed=43, exponent=2.38,
                          bedding=.077, pitting=.051))
    # Broad, sloping royal breast/shoulder, not the former straight cylinder
    # carrying a disproportionately oversized helmet above a toy-like body.
    part("carved-neck", along_z(((1.68, -2.94, 0, 1.04, 1.00),
                                  (2.03, -3.08, 0, .94, .93),
                                  (2.43, -3.29, 0, .75, .81),
                                  (2.78, -3.45, 0, .61, .72),
                                  (3.13, -3.50, 0, .50, .64)),
                                 roughness=.100, seed=49, exponent=2.55,
                                 bedding=.042, pitting=.028))
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
                                        roughness=.145, seed=61, exponent=2.42,
                                        bedding=.047, pitting=.031))
        # Four shallow, closely spaced digits are recessed into each forepaw.
        # Keeping them as individually editable relief sources retains the
        # inter-toe shadow grooves at cinematic close-up distances.
        for toe in range(4):
            toe_y = side_y + (toe-1.5)*.177
            part(f"{side}-toe-{toe}", along_x(((-5.51, toe_y, .225, .058, .085),
                                                (-5.39, toe_y, .24, .105, .125),
                                                (-5.17, toe_y, .25, .126, .139),
                                                (-4.93, toe_y, .265, .129, .143)),
                                               roughness=.015, seed=73+toe,
                                               bedding=.004, pitting=.006),
                 "weathered-limestone")
        part(f"{side}-haunch", along_x(((1.90, sign*1.03, 1.03, .37, .56),
                                       (2.40, sign*1.04, 1.07, .62, .81),
                                       (3.14, sign*1.02, 1.03, .67, .84),
                                       (3.75, sign*.98, .84, .50, .65),
                                       (4.08, sign*.94, .60, .18, .31)),
                                      roughness=.155, seed=151, exponent=2.19,
                                      bedding=.086, pitting=.057))
    part("curved-tail", along_x(((2.1, -1.04, .61, .14, .15),
                                (2.7, -1.26, .58, .16, .17),
                                (3.4, -1.38, .65, .15, .15),
                                (3.86, -1.27, .74, .13, .12),
                                (4.13, -1.12, .82, .09, .09)),
                               roughness=.025, seed=156, bedding=.012,
                               pitting=.009))
    # The surviving forehead/crown is closely fitted to a human face, while
    # the wider triangular side flaps are distinct behind it. This avoids the
    # huge egg-shaped helmet silhouette of the original parametric sculpt.
    hood = along_z(((2.70, -3.40, 0, .58, .67),
                    (3.20, -3.49, 0, .66, .76),
                    (3.74, -3.51, 0, .70, .83),
                    (4.14, -3.47, 0, .68, .82),
                    (4.43, -3.48, 0, .64, .78),
                    (4.56, -3.50, 0, .53, .69),
                    (4.60, -3.50, 0, .43, .58)),
                   roughness=.026, seed=89, exponent=3.65)
    part("nemes-hood", _carve_nemes_crown(hood, scale_x=x, scale_y=y, scale_z=z),
         "casing-limestone")
    # Face landmarks must span a much longer vertical elevation than the
    # previous squashed block. The new lower-chin profile is undercut from
    # the jaw, while cheek width grows gradually to the brow and then tapers
    # into the narrower royal crown. This is source-geometry anatomy, not
    # shader-only contouring or scaling the whole head (which also squashes
    # the neck and headdress).
    face = along_z(((2.96, -3.87, 0, .25, .35),
                    (3.07, -3.94, 0, .33, .41),
                    (3.22, -3.96, 0, .39, .46),
                    (3.43, -3.97, 0, .43, .51),
                    (3.70, -3.94, 0, .45, .57),
                    (3.90, -3.92, 0, .47, .605),
                    (4.14, -3.90, 0, .465, .60),
                    (4.31, -3.89, 0, .425, .555),
                    (4.49, -3.87, 0, .35, .48)),
                   roughness=.025, seed=97, exponent=4.35,
                   subdivisions=8, bedding=.006, pitting=.012)
    # Keep the original loft sculpt helper for backward compatibility and
    # geometry tests, but production uses the landmark-shaped closed patch.
    part("face", _anatomical_sphinx_face(scale_x=x, scale_y=y, scale_z=z))
    for side, sign in (("left", -1), ("right", 1)):
        part(f"nemes-{side}-lappet", _nemes_drapery_wing(
             side=sign, scale_x=x, scale_y=y, scale_z=z),
             "casing-limestone")
        ear = along_z(((3.81, -4.00, sign*.66, .052, .057),
                       (3.89, -4.02, sign*.706, .096, .086),
                       (4.02, -4.03, sign*.729, .120, .109),
                       (4.13, -4.012, sign*.714, .101, .094),
                       (4.22, -3.99, sign*.680, .059, .058)),
                      roughness=.018, seed=118, subdivisions=8)
        part(f"{side}-ear", _sculpt_ear_relief(
             ear, side=sign, scale_x=x, scale_y=y, scale_z=z))
        # Ear relief is a small face-side feature; wing and crown grooves are
        # directly carved in the same limestone before the Blender union.
    if restored_nose:
        part("optional-restored-nose", along_z(((3.52, -4.43, 0, .13, .17),
                                                (3.82, -4.52, 0, .19, .13),
                                                (4.18, -4.45, 0, .09, .12))))
    return tuple(items)


def make_monument_study(*, kind: str = "giza-study",
                        pyramid_base: float = 36.0, pyramid_height: float = 23.0,
                        pyramid_courses: int = 18, casing_fraction: float = 0.04,
                        sphinx_length: float = 10.5, restored_nose: bool = False,
                        pyramid_preset: str = "illustrative") -> MonumentStudy:
    """Compose a pyramid, a lion-bodied royal figure or a staged pair."""
    if kind not in ("pyramid", "sphinx", "giza-study"):
        raise MonumentError("kind must be pyramid, sphinx or giza-study")
    if pyramid_preset not in ("illustrative", "khufu-petrie"):
        raise MonumentError("pyramid_preset must be illustrative or khufu-petrie")
    if pyramid_preset == "khufu-petrie":
        if kind != "pyramid":
            raise MonumentError("khufu-petrie currently requires kind=pyramid: the combined Giza site layout is not surveyed")
        if (pyramid_base != 36.0 or pyramid_height != 23.0 or
                pyramid_courses != 18 or casing_fraction != 0.04):
            raise MonumentError("khufu-petrie uses fixed historic geometry; omit the illustrative pyramid dimension/course/casing options")
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
        if pyramid_preset == "khufu-petrie":
            pyramid_base = KHUFU_ORIGINAL_BASE_M
            pyramid_height = KHUFU_ORIGINAL_HEIGHT_M
            pyramid_courses = PETRIE_LAST_PRESERVED_COURSE
            casing_fraction = 0.0
            pyramid_parts = khufu_petrie_assembly(prefix="pyramid")
        else:
            pyramid_parts = pyramid_assembly(
                prefix="pyramid", base_width=pyramid_base,
                height=pyramid_height, courses=pyramid_courses,
                casing_fraction=casing_fraction)
        for part in pyramid_parts:
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
        asset_id=(f"egyptian-{kind}-khufu-petrie-course-study" if pyramid_preset == "khufu-petrie"
                  else f"egyptian-{kind}-detail-study"), kind=kind,
        instances=tuple(parts),
        part_counts=(("pyramid_parts", sum(p.name.startswith("pyramid-") for p in parts)),
                     ("sphinx_parts", sum(p.name.startswith("sphinx-") for p in parts))),
        bounds_min=tuple(mins), bounds_max=tuple(maxs),
        parameters=(("pyramid_base_m", pyramid_base), ("pyramid_height_m", pyramid_height),
                    ("pyramid_courses", pyramid_courses), ("casing_fraction", casing_fraction),
                    ("sphinx_length_m", sphinx_length), ("restored_nose", restored_nose),
                    ("pyramid_preset", pyramid_preset)),
    )
