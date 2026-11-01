"""Licensed photogrammetry-based Liberty facial-sculpt inspection revision.

This is deliberately a *checkpoint*, not a claim that importing a miniature
scan resolves the photographed monument's eye, lip and hair differences. The
mesh comes from a verified CC BY 4.0 source and is visibly more sculptural than
the primitive-only prototype. Tight multi-angle face renders expose remaining
changes that must be approved before the full statue is replaced.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .blender_output import aim_at, create_model_scene, export_revision
from .blender_scene import _create_box, _material
from .blender_textures import ensure_metric_uvs
from .render_profiles import select_render_profile
from .liberty_face_sculpt import sculpt_relief_offset
from .shots import QualityShot
from .surfaces import get_surface


def _world_bounds(objects: list[Any], vector_type: Any) -> tuple[Any, Any]:
    low = [math.inf, math.inf, math.inf]
    high = [-math.inf, -math.inf, -math.inf]
    for obj in objects:
        matrix = obj.matrix_world
        for vertex in obj.data.vertices:
            value = matrix @ vertex.co
            for axis in range(3):
                low[axis] = min(low[axis], value[axis])
                high[axis] = max(high[axis], value[axis])
    return vector_type(low), vector_type(high)


def _refine_captured_face_topology(obj: Any) -> int:
    """Add sculpt resolution only where captured forelock striae can survive.

    The face scan is too noisy to extract authentic lids by tessellation. Leave
    its broad facial skin untouched and add resolution only to the captured
    hair mass. BMesh splits connected edges, without disconnected overlays.
    """
    import bmesh

    mesh = obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        transform = obj.matrix_world
        eligible = set()
        for vert in bm.verts:
            p = transform @ vert.co
            if (-.95 < p.x < .24 and -.93 < p.y < -.15
                    and 8.96 < p.z < 9.38):
                eligible.add(vert)
        edges = [edge for edge in bm.edges
                 if edge.verts[0] in eligible and edge.verts[1] in eligible]
        if not edges:
            return 0
        bmesh.ops.subdivide_edges(bm, edges=edges, cuts=2,
                                  use_grid_fill=True, smooth=0.0)
        bm.to_mesh(mesh)
        mesh.update()
        return len(edges)
    finally:
        bm.free()


def plan_liberty_scan_face_shots(height: float) -> tuple[QualityShot, ...]:
    """Full study cameras, plus two anatomy-only near-macro quality views."""
    if not math.isfinite(height) or height <= 0:
        raise ValueError("Scan height must be positive")
    # The photogrammetry source is a standalone bronze miniature, Z-up. The
    # chin/forehead span lies near 0.63–0.75 of its total (torch-inclusive)
    # height; values are ratios so imported miniature units remain untouched.
    h = height
    face_x = -h * .025
    face_z = h * .70
    return (
        QualityShot("hero", (h*.72, -h*2.0, h*.96), (0, 0, h*.54), h*1.20),
        QualityShot("face-front", (face_x, -h*1.3, face_z),
                    (face_x, 0, face_z), h*.17),
        QualityShot("eyes-macro", (face_x, -h*1.05, h*.705),
                    (face_x, 0, h*.705), h*.078),
        QualityShot("nose-mouth-macro", (face_x, -h*1.05, h*.665),
                    (face_x, 0, h*.665), h*.072),
        QualityShot("face-left-threequarter", (face_x-h*.72, -h*.98, face_z+h*.03),
                    (face_x, 0, face_z), h*.20),
        QualityShot("face-right-threequarter", (face_x+h*.72, -h*.98, face_z+h*.03),
                    (face_x, 0, face_z), h*.20),
        QualityShot("face-left-profile", (face_x-h*1.3, 0, face_z+h*.01),
                    (face_x, 0, face_z), h*.20),
        QualityShot("face-right-profile", (face_x+h*1.3, 0, face_z+h*.01),
                    (face_x, 0, face_z), h*.20),
        QualityShot("face-low-angle", (face_x+h*.45, -h*1.4, h*.49),
                    (face_x, 0, face_z), h*.21),
        QualityShot("hair-rear", (0, h*1.25, h*.70),
                    (face_x, 0, h*.70), h*.25),
        QualityShot("crown-top", (h*.28, -h*.7, h*1.05),
                    (face_x, 0, h*.74), h*.28),
    )


def build_liberty_scan_face_scene(source: Path, destination: Path,
                                  provenance: dict[str, Any], *,
                                  bpy_module: Any = None) -> dict[str, Any]:
    """Create a reviewable high-detail scan-driven source.blend and GLB.

    Does not import the scan until the caller verifies its full SHA-256 against
    the CC BY source record. The source scan is not tracked in git.
    """
    if bpy_module is None:
        raise RuntimeError("Blender bpy is required for the Liberty face checkpoint")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError("Blender 4.2+ is required")
    from mathutils import Matrix, Vector

    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Immutable revision already exists: {destination}")
    profile = select_render_profile(1360, 1360, 40)
    scene = create_model_scene(bpy, "statue-of-liberty-scan-face-checkpoint")
    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = profile.width
    scene.render.resolution_y = profile.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"

    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and not obj.hide_render]
    if not meshes:
        raise RuntimeError("The licensed photogrammetry scan contains no mesh")
    low, high = _world_bounds(meshes, Vector)
    height = high.z - low.z
    if height <= 0:
        raise RuntimeError("Scanned monument has invalid vertical extent")
    center = Vector(((low.x + high.x) / 2, (low.y + high.y) / 2, low.z))

    model = bpy.data.collections.new("Liberty / jerryfisher CC BY photogrammetry")
    scene.collection.children.link(model)
    material = _material(bpy, get_surface("verdigris-copper"))
    for number, obj in enumerate(meshes):
        world_matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = Matrix.Translation(-center) @ world_matrix
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        model.objects.link(obj)
        obj.name = f"liberty-scan-sculpt-{number:02d}"
        obj["kumiki_part"] = "face-study-scan-sculpture"
        obj["kumiki_source_author"] = provenance["author"]
        obj["kumiki_source_url"] = provenance["original_url"]
        obj["kumiki_license"] = provenance["license"]
        obj["kumiki_modifications"] = "Recentered, forelock-only retessellation, restrained eye/nose/lip/hair mesh relief, copper study material"
        _refine_captured_face_topology(obj)
        ensure_metric_uvs(obj.data, tile_size_m=.56)
        obj.data.materials.clear()
        obj.data.materials.append(material)
        # Deform existing scan vertices instead of overlaying decorative tubes.
        # Apply in normalized world coordinates so import node transforms do not
        # change which facial region is touched.
        to_local = obj.matrix_world.inverted()
        for vertex in obj.data.vertices:
            point = obj.matrix_world @ vertex.co
            displacement = sculpt_relief_offset(point.x, point.y, point.z)
            if displacement:
                point.y += displacement
                vertex.co = to_local @ point
        obj.data.update()
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    # No invented eyelid or noodle hair objects: shallow anatomy corrections
    # deform the captured surface, while the remaining scan stays observable.
    stage = bpy.data.collections.new("Liberty / dark neutral inspection stage")
    scene.collection.children.link(stage)
    floor = _material(bpy, get_surface("studio-matte", color="#262D31", roughness=.96))
    _create_box(bpy, stage, "Inspection ground", (height*5, height*5, .2),
                (0, 0, -.12), floor)

    world = bpy.data.worlds.new("Liberty / soft blue-grey studio")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (.23, .28, .31, 1)
    background.inputs["Strength"].default_value = .32
    scene.world = world

    def area(name: str, position: tuple[float, float, float], power: float,
             color: tuple[float, float, float], diameter: float):
        lamp = bpy.data.lights.new(name, "AREA")
        lamp.energy = power
        lamp.shape = "DISK"
        lamp.size = diameter
        lamp.color = color
        obj = bpy.data.objects.new(name, lamp)
        stage.objects.link(obj)
        obj.location = position
        aim_at(obj, (0, 0, height*.68))

    area("Key / sculpt raking light", (-height*.65, -height*.76, height*1.01),
         3600, (.91, .97, 1), height*.44)
    area("Fill / face anatomy", (height*.34, -height*.90, height*.78),
         1600, (.80, .91, 1), height*.60)
    area("Edge / crown and hair", (height*.4, height*.7, height*.90),
         2900, (1, .83, .64), height*.37)

    camera_data = bpy.data.cameras.new("Liberty / eleven sculpt quality-review angles")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    stage.objects.link(camera)
    scene.camera = camera
    shots = plan_liberty_scan_face_shots(height)
    metadata = {
        "asset_id": "statue-of-liberty-scan-face-checkpoint",
        "generator": "kumiki-blender/licensed-scan-assisted-face-checkpoint",
        "status": "face likeness checkpoint / further sculpt and peer review required",
        "provenance": "CC BY 4.0 photogrammetry of nineteenth-century bronze miniature",
        "license": provenance["license"],
        "creator": provenance["author"],
        "source_title": provenance["title"],
        "source_url": provenance["original_url"],
        "source_sha256": provenance["sha256"],
        "attribution": provenance["attribution"],
        "changes": ["centered scan", "forelock-only mesh retessellation",
                    "carved almond eyelid edges and iris rings", "nostril, lips and philtrum relief",
                    "combed forelock striations", "copper material treatment",
                    "face-angle inspection cameras"],
        "source_limitations": provenance["limitations"],
        "render_profile": {"name": profile.name, "width": profile.width,
                           "height": profile.height, "samples": profile.samples},
        "units": "original scan miniature units; intentionally not historic metre scaling",
        "native_axes": "Blender Z-up; principal face toward -Y",
        "named_parts": {"scan-sculpt": len(meshes)},
        "bounds_m": [[round(v, 5) for v in low - center],
                     [round(v, 5) for v in high - center]],
        "detail_cameras": [shot.name for shot in shots],
    }
    return export_revision(bpy, scene, camera, meshes, shots, destination,
                           samples=profile.samples, metadata=metadata)
