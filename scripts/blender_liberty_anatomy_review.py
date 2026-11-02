"""Neutral-clay anatomy checkpoint for the licensed Liberty/head-mesh hybrid.

Run with Blender 4.2+::

    blender -b -t 4 --python scripts/blender_liberty_anatomy_review.py -- \
      SOURCE.glb BLENDER_HUMAN_BASE_MESHES.blend OUTPUT_DIRECTORY

The script is an inspection tool: it imports only the CC0 realistic head and
eyes from Blender Studio's Human Base Meshes v1.2 bundle, not its entire scene.
It deliberately renders front, three-quarter and profile views before
integration into the production Statue of Liberty generator.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kumiki_blender.liberty_anatomy_source import verify_anatomy_source
from kumiki_blender.liberty_scan import verify_scan_source
from kumiki_blender.liberty_face_region import (
    conform_depth, face_jawline_z, face_radius, keep_face_vertex,
    sculpt_classical_face, keep_wrap_vertex, wrap_donor_weight,
)


HEAD_NAMES = (
    "GEO-head_sculpting_realistic",
    "GEO-head_sculpting_realistic.sclera.L",
    "GEO-head_sculpting_realistic.sclera.R",
    "GEO-head_sculpting_realistic.iris.L",
    "GEO-head_sculpting_realistic.iris.R",
)


def _bounds(objects):
    points = (obj.matrix_world @ vertex.co
              for obj in objects for vertex in obj.data.vertices)
    low, high = [math.inf] * 3, [-math.inf] * 3
    for point in points:
        for axis in range(3):
            low[axis] = min(low[axis], point[axis])
            high[axis] = max(high[axis], point[axis])
    return Vector(low), Vector(high)


def _smoothstep(value):
    t = min(1.0, max(0.0, value))
    return t * t * (3.0 - 2.0 * t)


def load_head(donor_blend: Path, *, target_origin=(-.30, .0, 7.42), scale=4.8,
              patch=False, conform=False):
    """Transform anatomically based CC0 head to scan-aligned coordinates."""
    with bpy.data.libraries.load(str(donor_blend), link=False) as (source, destination):
        missing = set(HEAD_NAMES) - set(source.objects)
        if missing:
            raise RuntimeError(f"Missing CC0 head objects: {sorted(missing)}")
        destination.objects = list(HEAD_NAMES)
    objects = list(destination.objects)
    transform_head = next(obj for obj in objects if obj.name == HEAD_NAMES[0])
    mapping = (Matrix.Translation(Vector(target_origin))
               @ Matrix.Scale(scale, 4) @ transform_head.matrix_world.inverted())
    for obj in objects:
        source_world = obj.matrix_world.copy()
        obj.parent = None
        bpy.context.scene.collection.objects.link(obj)
        obj.matrix_world = mapping @ source_world
        obj.name = f"Anatomical study / {obj.name}"
        obj["license"] = "CC0 1.0 / Blender Studio Human Base Meshes 1.2"
        obj["source_url"] = (
            "https://download.blender.org/demo/asset-bundles/human-base-meshes/"
            "human-base-meshes-bundle-v1.2.0.zip")
        if obj.type == "MESH":
            if obj is transform_head:
                # A *transitional anatomical face surface*, not an extra bald
                # head floating in front of the bronze hair and shoulders.
                # Push scalp, ears and neck behind the existing capture; keep
                # the high-topology central face and features at full depth.
                world_to_local = obj.matrix_world.inverted()
                for vertex in obj.data.vertices:
                    point = obj.matrix_world @ vertex.co
                    # The sculpture has a substantially lower, heavier gaze
                    # than this generic human base. Shift lids *and their
                    # surrounding socket topology* before subdivision rather
                    # than drawing fake rings on the low-detail bronze mesh.
                    eye = math.exp(-((point.z - 8.76) / .19) ** 2)
                    eye *= _smoothstep((.53 - abs(point.x + .30)) / .14)
                    point.z -= .095 * eye
                    # Classical mandible: widen the lower facial planes but
                    # avoid a uniform squash that also widens the eye line.
                    jaw = math.exp(-((point.z - 8.29) / .26) ** 2)
                    point.x = -.30 + (point.x + .30) * (1.0 + .095 * jaw)
                    top = _smoothstep((9.29 - point.z) / .23)
                    bottom = _smoothstep((point.z - 7.69) / .37)
                    sides = _smoothstep((.58 - abs(point.x + .30)) / .20)
                    front = _smoothstep((.13 - point.y) / .28)
                    keep = top * bottom * sides * front
                    if not patch and not conform:
                        point.y += (1.0 - keep) * .87
                    if conform:
                        point = Vector(sculpt_classical_face(*point))
                    vertex.co = world_to_local @ point
                obj.data.update()
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
        if obj is transform_head:
            subdivision = obj.modifiers.new("High-resolution clay preview", "SUBSURF")
            subdivision.levels = 2
            subdivision.render_levels = 2
        elif obj.name.endswith((".sclera.L", ".sclera.R", ".iris.L", ".iris.R")):
            # Eye surfaces follow the downward-moving socket and lid rings.
            obj.location.z -= .087
            if conform:
                inverse = obj.matrix_world.inverted()
                for vertex in obj.data.vertices:
                    point = obj.matrix_world @ vertex.co
                    vertex.co = inverse @ Vector(sculpt_classical_face(*point))
                obj.data.update()
    return objects


def _scan_depth_sampler(source_meshes):
    """Return a closest-front-depth sampler for the verified bronze scan."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    targets = []
    for obj in source_meshes:
        evaluated = obj.evaluated_get(depsgraph)
        targets.append((BVHTree.FromObject(evaluated, depsgraph),
                        obj.matrix_world.inverted(), obj.matrix_world.copy()))

    def sampled_depth(x, z, original_y):
        origin = Vector((x, -2.2, z))
        direction = Vector((0, 1, 0))
        candidates = []
        for tree, inverse, world in targets:
            hit, _, _, _ = tree.ray_cast(
                inverse @ origin, (inverse.to_3x3() @ direction).normalized(), 5.0)
            if hit is not None:
                value = (world @ hit).y
                if -1.2 < value < .10:
                    candidates.append(value)
        return min(candidates, key=lambda y: abs(y-original_y)) if candidates else None

    return sampled_depth


def conform_anatomical_face(source_meshes, donor_head):
    """Remove the CC0 donor's slab-like scalp/neck and register its border.

    Unlike the radial heightfield experiment, the original anatomical mesh's
    eyelids and philtrum stay intact. The outer 30% is eased toward the actual
    scan front, at matching image-space X/Z coordinates. Only the front facial
    surface is retained; the source scan provides the temples, hair and neck.
    This is a non-destructive *review* of two separate meshes, not a welded
    production retopology.
    """
    import bmesh

    sampled_depth = _scan_depth_sampler(source_meshes)
    world = donor_head.matrix_world
    inverse = world.inverted()
    bm = bmesh.new()
    try:
        bm.from_mesh(donor_head.data)
        keep = set()
        fitted = 0
        for vertex in bm.verts:
            point = world @ vertex.co
            if keep_face_vertex(point.x, point.y, point.z):
                keep.add(vertex)
                radius = face_radius(point.x, point.z)
                if radius > .70:
                    scanned = sampled_depth(point.x, point.z, point.y)
                    if scanned is not None:
                        point.y = conform_depth(point.y, scanned, radius)
                        vertex.co = inverse @ point
                        fitted += 1
        outside = [face for face in bm.faces if any(v not in keep for v in face.verts)]
        bmesh.ops.delete(bm, geom=outside, context="FACES")
        orphaned = [vertex for vertex in bm.verts if not vertex.link_faces]
        bmesh.ops.delete(bm, geom=orphaned, context="VERTS")
        bm.to_mesh(donor_head.data)
        donor_head.data.update()
        donor_head["facial_border_registration"] = "image-aligned sampled scan depth, smooth outer feather"
        donor_head["source_registration_samples"] = fitted
        return {"deleted_faces": len(outside), "conformed_vertices": fitted,
                "remaining_vertices": len(donor_head.data.vertices)}
    finally:
        bm.free()


def wrap_anatomical_face(source_meshes, donor_head):
    """Retain the donor cheek through the temple with a cropped neck.

    This is a separate candidate from the front-oval crop. Both experiments
    preserve the donor's real eyelid, philtrum, cheek and jaw topology. The
    broader crop aims to move the lateral border under captured hair and
    feathers lower-neck vertices to the bronze surface before clipping.
    """
    import bmesh

    sampled_depth = _scan_depth_sampler(source_meshes)
    world = donor_head.matrix_world
    inverse = world.inverted()
    bm = bmesh.new()
    try:
        bm.from_mesh(donor_head.data)
        keep = set()
        fitted = 0
        for vertex in bm.verts:
            point = world @ vertex.co
            if not keep_wrap_vertex(point.x, point.y, point.z):
                continue
            keep.add(vertex)
            if point.y < -.10:
                captured_depth = sampled_depth(point.x, point.z, point.y)
                if captured_depth is not None:
                    weight = wrap_donor_weight(point.x, point.z)
                    point.y = point.y*weight + (captured_depth+.035)*(1.-weight)
                    vertex.co = inverse @ point
                    fitted += 1
        outside = [face for face in bm.faces if any(v not in keep for v in face.verts)]
        bmesh.ops.delete(bm, geom=outside, context="FACES")
        orphans = [v for v in bm.verts if not v.link_faces]
        bmesh.ops.delete(bm, geom=orphans, context="VERTS")
        bm.to_mesh(donor_head.data)
        donor_head.data.update()
        donor_head["facial_border_registration"] = "expanded temple and neck surface wrap"
        donor_head["source_registration_samples"] = fitted
        return {"deleted_faces": len(outside), "conformed_vertices": fitted,
                "remaining_vertices": len(donor_head.data.vertices)}
    finally:
        bm.free()


def graft_clay_face(scan_meshes, head_meshes, *, rings=150, segments=320):
    """Surface-registered facial topology instead of a floating face mask.

    Build a single smooth oval whose external ring follows the captured face
    and whose center carries the CC0 anatomically modeled eyelid/nose/lip
    geometry. Ray tests follow the -Y camera (the source's canonical front).
    The facial transition is spatially blended across 20% of its radius.
    This is an independent high-density surface, not tubes over the capture.
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    def trees(objects):
        output = []
        for obj in objects:
            evaluated = obj.evaluated_get(depsgraph)
            output.append((obj, BVHTree.FromObject(evaluated, depsgraph),
                           obj.matrix_world.inverted(), obj.matrix_world.copy()))
        return output
    source = trees(scan_meshes)
    donor = trees(head_meshes)

    def front_at(x, z, collection):
        origin = Vector((x, -2.5, z))
        direction = Vector((0, 1, 0))
        closest = None
        for _, tree, inverse, transform in collection:
            local_o = inverse @ origin
            local_direction = (inverse.to_3x3() @ direction).normalized()
            point, normal, face, distance = tree.ray_cast(local_o, local_direction, 4.5)
            if point is None:
                continue
            y = (transform @ point).y
            if -.95 < y < .15 and (closest is None or y < closest):
                closest = y
        return closest

    cx, cz = -.30, 8.58
    rx, rz = .47, .57
    verts, faces = [], []
    last_scan = -.37
    for ring in range(rings + 1):
        radial = ring / rings
        for index in range(segments):
            angle = 2 * math.pi * index / segments
            x = cx + rx * radial * math.cos(angle)
            z = cz + rz * radial * math.sin(angle)
            captured = front_at(x, z, source)
            anatomical = front_at(x, z, donor)
            if captured is None:
                captured = last_scan
            last_scan = captured
            if anatomical is None:
                anatomical = captured
            blend = _smoothstep((.96 - radial) / .28)
            # Do not take min(scan, donor) in the central face: the noisy
            # bronze miniature can intermittently overtake anatomically
            # correct eyelid/lip surfaces, creating a sawtooth silhouette.
            front = anatomical
            y = captured * (1 - blend) + front * blend - .004
            verts.append((x, y, z))
            if ring:
                before = (ring - 1) * segments
                current = ring * segments
                nxt = (index + 1) % segments
                faces.append((before+index, current+index, current+nxt, before+nxt))
    # A single pole is adequate; duplicate first ring positions are discarded.
    mesh = bpy.data.meshes.new("registered face / eyelid nose mouth patch")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    graft = bpy.data.objects.new("Liberty / continuous facial-surface graft", mesh)
    scene.collection.objects.link(graft)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    subdiv = graft.modifiers.new("Smooth reconstructed facial transitions", "SUBSURF")
    subdiv.levels = 1
    subdiv.render_levels = 1
    graft["modeling_method"] = "radial high-density anatomical/survey surface registration"
    graft["status"] = "prototype; not retopologized into source scan"
    return graft


def carve_blurred_scan_face(meshes, *, conform=False, preserve_neck=False):
    """Remove the miniature's featureless skin, preserving crown/robe/hair.

    This separate review approach replaces anatomical topology instead of
    allowing two overlapping face surfaces to compete in depth testing.
    """
    import bmesh

    count = 0
    for obj in meshes:
        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            transform = obj.matrix_world
            reject = []
            for face in bm.faces:
                point = transform @ face.calc_center_median()
                if conform:
                    radius = face_radius(point.x, point.z)
                    remove = radius < .69 and point.z > face_jawline_z(point.x) - .015
                    if preserve_neck:
                        # The scan's neck bridges beneath the high-definition
                        # donor chin. Removing its upper triangles leaves an
                        # exposed, serrated gap in 3/4 and frontal views.
                        remove = remove and point.z > 8.28
                else:
                    radius = (((point.x + .30) / .42) ** 2
                              + ((point.z - 8.56) / .50) ** 2)
                    remove = radius < .68
                if remove and -.95 < point.y < -.12:
                    reject.append(face)
            if reject:
                bmesh.ops.delete(bm, geom=reject, context="FACES")
                bm.to_mesh(obj.data)
                obj.data.update()
                count += len(reject)
        finally:
            bm.free()
    return count


def _aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def build(source: Path, donor: Path, directory: Path):
    scan_record = verify_scan_source(source)
    donor_record = verify_anatomy_source(donor)
    if directory.is_symlink() or (directory.exists() and any(directory.iterdir())):
        raise FileExistsError(f"Anatomy review revisions are immutable: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(source))
    source_meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo, hi = _bounds(source_meshes)
    source_center = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    for obj in source_meshes:
        obj.matrix_world = Matrix.Translation(-source_center) @ obj.matrix_world
        obj["source_url"] = scan_record["original_url"]
        obj["source_license"] = scan_record["license"]
        obj["source_creator"] = scan_record["author"]
    patch_mode = "--patch" in sys.argv
    carve_mode = "--carve" in sys.argv
    conform_mode = "--conform" in sys.argv
    wrap_mode = "--wrap" in sys.argv
    if sum((patch_mode, carve_mode, conform_mode, wrap_mode)) > 1:
        raise ValueError("Choose exactly one anatomy review mode")
    donors = load_head(donor, patch=patch_mode, conform=conform_mode or wrap_mode)
    fitted = None
    if conform_mode:
        fitted = conform_anatomical_face(source_meshes, donors[0])
        print("LIBERTY_FITTED_FACE", fitted)
    if wrap_mode:
        fitted = wrap_anatomical_face(source_meshes, donors[0])
        print("LIBERTY_WRAPPED_FACE", fitted)
    if patch_mode:
        graft = graft_clay_face(source_meshes, donors)
        for obj in donors:
            obj.hide_render = True
    if carve_mode or conform_mode or wrap_mode:
        print("LIBERTY_CARVED_SCAN_FACES", carve_blurred_scan_face(
            source_meshes, conform=conform_mode or wrap_mode,
            preserve_neck=wrap_mode))

    # The mesh's normal glTF skin material is intentionally hidden: single-
    # color cavity shading makes undercuts and silhouette defects apparent.
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (.61, .75, .69)
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.curvature_ridge_factor = 1.3
    scene.display.shading.curvature_valley_factor = 1.4
    scene.world.color = (.09, .10, .13)
    camera_data = bpy.data.cameras.new("Liberty anatomy review / orthographic")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    target = Vector((-.30, -.38, 8.65))
    angles = (("front", 0), ("threequarter", 42), ("profile", 88))
    camera_data.ortho_scale = 1.85
    quick = "--quick" in sys.argv
    for scan_visible in ((True,) if quick else (True, False)):
        for obj in source_meshes:
            obj.hide_render = not scan_visible
        name = "combined" if scan_visible else "anatomy-only"
        for label, degrees in (angles[:2] if quick else angles):
            rad = math.radians(degrees)
            camera.location = (target.x + 7 * math.sin(rad),
                               target.y - 7 * math.cos(rad), target.z + .14)
            _aim(camera, target)
            scene.render.filepath = str(directory / f"{name}-{label}.png")
            bpy.ops.render.render(write_still=True)
    for obj in source_meshes:
        obj.hide_render = False

    # Portable, neutral patinated material shared across both CC BY bronze
    # and CC0 donor rather than retaining unrelated original skin textures.
    copper = bpy.data.materials.new("Liberty / neutral copper-grey clay")
    copper.diffuse_color = (.36, .55, .49, 1.0)
    copper.use_nodes = True
    shader = next(n for n in copper.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (.27, .47, .40, 1.0)
    shader.inputs["Metallic"].default_value = .31
    shader.inputs["Roughness"].default_value = .82
    exported = source_meshes + ([graft] if patch_mode else donors)
    for obj in exported:
        obj.data.materials.clear()
        obj.data.materials.append(copper)
    if "--export" in sys.argv:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in exported:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = exported[0]
        bpy.ops.export_scene.gltf(
            filepath=str(directory / "anatomy-checkpoint.glb"),
            export_format="GLB", use_selection=True, export_apply=True,
            export_extras=True, export_materials="EXPORT")
    bpy.ops.wm.save_as_mainfile(filepath=str(directory / "anatomy-checkpoint.blend"))
    metadata = {
        "source_scan": str(source),
        "anatomy_source": str(donor),
        "donor_license": "CC0 1.0",
        "donor_sha256": donor_record["sha256"],
        "scan_sha256": scan_record["sha256"],
        "sculpt_basis": HEAD_NAMES,
        "transform": {"target_origin": [-.30, .0, 7.42], "scale": 4.8},
        "donor_objects": [(obj.name, len(obj.data.vertices)) for obj in donors],
        "acceptance": "Unapproved anatomy prototype: compare all three angles with original photos",
        "quality_gate": "FAIL / distinct eyes and lips, but facial likeness and seams are not approved",
        "fitted_face": fitted,
        "method": ("expanded temple and neck wrap prototype" if wrap_mode else
                   "source-registered anatomically retained face" if conform_mode else
                   "surface-registered facial graft" if patch_mode else
                   "source face excavation and anatomical insertion" if carve_mode else
                   "transitional head overlap"),
    }
    (directory / "anatomy-review.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    start = sys.argv.index("--") + 1
    build(Path(sys.argv[start]), Path(sys.argv[start+1]), Path(sys.argv[start+2]))
