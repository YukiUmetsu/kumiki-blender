"""Blender adapter for the modular portico architecture detail study."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from .blender_output import create_model_scene, export_revision
from .blender_scene import _create_box, _material
from .blender_textures import ensure_metric_uvs
from .portico import PorticoStudy
from .render_profiles import select_render_profile
from .shots import plan_portico_shots
from .surfaces import get_surface


def build_portico_scene(study: PorticoStudy, destination: Path, *, bpy_module: Any = None) -> dict[str, Any]:
    """Stage, render and export a reusable stone arcade with named components."""
    if bpy_module is None:
        raise RuntimeError("Blender bpy context is required for this build")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError("Blender 4.2 or newer is required")
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Output revision already exists: {destination}")

    profile = select_render_profile(1440, 900, 48)
    scene = create_model_scene(bpy, study.asset_id)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile.samples
    scene.render.resolution_x = profile.width
    scene.render.resolution_y = profile.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"

    model = bpy.data.collections.new("Architecture / portable GLB")
    scene.collection.children.link(model)
    stage = bpy.data.collections.new("Studio / excluded from GLB")
    scene.collection.children.link(stage)

    palette = {
        "limestone": _material(bpy, get_surface("natural-stone", color="#C1B7A4", roughness=0.76)),
        "limestone-warm": _material(bpy, get_surface("natural-stone", color="#B8AB95", roughness=0.79)),
        "limestone-light": _material(bpy, get_surface("natural-stone", color="#D1C5AE", roughness=0.70)),
        "keystone": _material(bpy, get_surface("natural-stone", color="#D7C7A9", roughness=0.72)),
        "stone-foundation": _material(bpy, get_surface("natural-stone", color="#8C877E", roughness=0.86)),
    }
    cache: dict[tuple[int, str], Any] = {}
    objects = []
    from mathutils import Matrix  # Blender-provided module, only loaded during builds.

    for instance in study.instances:
        cache_key = id(instance.mesh), instance.material
        if cache_key not in cache:
            mesh = bpy.data.meshes.new(f"Geometry / {instance.name}")
            mesh.from_pydata(instance.mesh.vertices, [], instance.mesh.faces)
            mesh.update()
            ensure_metric_uvs(mesh)
            mesh.materials.append(palette[instance.material])
            if instance.name.startswith(("column-", "arch-")):
                for polygon in mesh.polygons:
                    if len(polygon.vertices) == 4:
                        polygon.use_smooth = True
            cache[cache_key] = mesh
        obj = bpy.data.objects.new(instance.name, cache[cache_key])
        model.objects.link(obj)
        obj.matrix_world = Matrix((
            (instance.axis_x[0], instance.axis_y[0], instance.axis_z[0], instance.origin[0]),
            (instance.axis_x[1], instance.axis_y[1], instance.axis_z[1], instance.origin[1]),
            (instance.axis_x[2], instance.axis_y[2], instance.axis_z[2], instance.origin[2]),
            (0, 0, 0, 1),
        ))
        obj["kumiki_part"] = instance.name.split("-")[0]
        bevel = obj.modifiers.new("Light-catching masonry edges", "BEVEL")
        bevel.width = 0.006 if instance.name.startswith("column-") and "-flute-" in instance.name else 0.018
        bevel.segments = 2
        obj.modifiers.new("Masonry weighted normals", "WEIGHTED_NORMAL")
        objects.append(obj)

    floor_material = _material(bpy, get_surface("studio-matte", color="#737673", roughness=0.97))
    extent = study.bounds_max[0] - study.bounds_min[0]
    _create_box(bpy, stage, "Stage / matte ground", (extent * 1.8, 22, 0.4),
                (0, 0, -0.56), floor_material)
    world = bpy.data.worlds.new("Kumiki / limestone daylight")
    world.use_nodes = True
    bg = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.22, 0.26, 0.31, 1)
    bg.inputs["Strength"].default_value = 0.42
    scene.world = world

    sun = bpy.data.lights.new("Sun / angled masonry relief", type="SUN")
    sun.energy = 2.1
    sun.angle = math.radians(4)
    sun_obj = bpy.data.objects.new(sun.name, sun)
    stage.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(25), math.radians(-22), math.radians(-30))
    fill = bpy.data.lights.new("Area / soft fill", type="AREA")
    fill.energy = 1000
    fill.shape = "DISK"
    fill.size = 9
    fill_obj = bpy.data.objects.new(fill.name, fill)
    stage.objects.link(fill_obj)
    fill_obj.location = (-4, -8, 11)
    from .blender_output import aim_at

    aim_at(fill_obj, (0, 0, 3.5))

    camera_data = bpy.data.cameras.new("Camera / masonry study")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    stage.objects.link(camera)
    shots = plan_portico_shots(study.bounds_min, study.bounds_max)
    metadata = {
        "asset_id": study.asset_id,
        "generator": "kumiki-blender/modular-portico",
        "status": "architectural-detail-study / visual-review-pending",
        "provenance": "Original illustrative architecture; no reference landmark claimed",
        "recipe_sha256": hashlib.sha256(f"portico-study-v1:bays={study.bays}".encode()).hexdigest(),
        "render_profile": {"name": profile.name, "width": profile.width,
                           "height": profile.height, "samples": profile.samples},
        "parameters": {"bays": study.bays, "bay_pitch_m": 4.2},
        "units": "meters",
        "native_axes": "Blender Z-up; GLB exported with glTF-compatible axes",
        "named_parts": dict(study.part_counts),
        "mesh_data_blocks": len(cache),
        "instances": len(study.instances),
        "bounds_m": [study.bounds_min, study.bounds_max],
    }
    return export_revision(bpy, scene, camera, objects, shots, destination,
                           samples=profile.samples, metadata=metadata)
