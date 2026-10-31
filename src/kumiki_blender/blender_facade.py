"""Blender 4.2+ adapter for the reusable architectural facade detail kit."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from .blender_output import aim_at, create_model_scene, export_revision
from .blender_lighting import add_architectural_reflections
from .blender_scene import _create_box, _material
from .blender_textures import ensure_metric_uvs
from .facade import FacadeStudy
from .render_profiles import select_render_profile
from .shots import QualityShot
from .surfaces import get_surface


def build_facade_scene(study: FacadeStudy, destination: Path, *, bpy_module: Any = None) -> dict[str, Any]:
    """Generate portable geometry, editable source and cinematic inspection views."""
    if bpy_module is None:
        raise RuntimeError("Blender bpy context is required for facade rendering")
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
    model = bpy.data.collections.new("Architectural facade / portable GLB")
    scene.collection.children.link(model)
    stage = bpy.data.collections.new("Studio / excluded from GLB")
    scene.collection.children.link(stage)

    materials = {
        "wall-stone": get_surface("weathered-stone", color="#A9A296", roughness=0.83),
        "wall-stone-light": get_surface("weathered-stone", color="#B9B1A4", roughness=0.78),
        "trim-stone": get_surface("natural-stone", color="#C9C0AF", roughness=0.73),
        "foundation": get_surface("weathered-stone", color="#77736D", roughness=0.89),
        "ornament-stone": get_surface("natural-stone", color="#C8BEAC", roughness=0.72),
        "ornament-highlight": get_surface("natural-stone", color="#D4CAB6", roughness=0.67),
        "window-metal": get_surface("painted-steel", color="#39454B", roughness=0.38),
        "window-glass": get_surface("optical-glass", color="#9BB7C4", roughness=0.07),
        "recess-dark": get_surface("painted-steel", color="#344753", roughness=0.84),
        "door-timber": get_surface("timber", color="#684936", roughness=0.59),
        "door-timber-light": get_surface("timber", color="#90694F", roughness=0.54),
        "bronze-metal": get_surface("galvanized-steel", color="#B4945D", roughness=0.31),
        "roof-slate": get_surface("natural-stone", color="#51565D", roughness=0.82),
        "roof-slate-light": get_surface("natural-stone", color="#5E6267", roughness=0.78),
        "rain-metal": get_surface("galvanized-steel", color="#525B60", roughness=0.44),
    }
    palette = {name: _material(bpy, spec) for name, spec in materials.items()}
    from mathutils import Matrix  # Blender-only dependency.

    cache: dict[tuple[int, str], Any] = {}
    objects: list[Any] = []
    for part in study.instances:
        key = (id(part.mesh), part.material)
        if key not in cache:
            mesh = bpy.data.meshes.new(f"Geometry / {part.name}")
            mesh.from_pydata(part.mesh.vertices, [], part.mesh.faces)
            mesh.update()
            ensure_metric_uvs(mesh)
            mesh.materials.append(palette[part.material])
            cache[key] = mesh
        obj = bpy.data.objects.new(part.name, cache[key])
        model.objects.link(obj)
        obj.matrix_world = Matrix((
            (part.axis_x[0], part.axis_y[0], part.axis_z[0], part.origin[0]),
            (part.axis_x[1], part.axis_y[1], part.axis_z[1], part.origin[1]),
            (part.axis_x[2], part.axis_y[2], part.axis_z[2], part.origin[2]),
            (0, 0, 0, 1),
        ))
        obj["kumiki_part"] = part.name.split("-")[0]
        # Subtle bevels catch glancing light; the source .blend keeps them as
        # editable modifiers, while export_revision evaluates them for GLB.
        if part.name.startswith(("masonry-", "cornice-", "foundation-", "door-", "window-", "ornament-")):
            bevel = obj.modifiers.new("Architectural edge soften", "BEVEL")
            bevel.width = 0.008 if part.name.startswith("masonry-") else 0.004
            bevel.segments = 2
            obj.modifiers.new("Weighted architectural normals", "WEIGHTED_NORMAL")
        objects.append(obj)

    minx, miny, minz = study.bounds_min
    maxx, maxy, maxz = study.bounds_max
    width, depth, height = maxx-minx, maxy-miny, maxz-minz
    cx, cy = (minx+maxx)/2, (miny+maxy)/2
    ground_mat = _material(bpy, get_surface("studio-matte", color="#6C706F", roughness=0.97))
    _create_box(bpy, stage, "Stage / neutral ground", (width*2.6, depth*3.0, 0.25),
                (cx, cy, minz-0.13), ground_mat)
    world = bpy.data.worlds.new("Kumiki / architectural soft daylight")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.27, 0.32, 0.37, 1)
    background.inputs["Strength"].default_value = 0.48
    scene.world = world
    sun_data = bpy.data.lights.new("Sun / facade relief", type="SUN")
    sun_data.energy = 2.2
    sun_data.angle = math.radians(4.5)
    sun = bpy.data.objects.new(sun_data.name, sun_data)
    stage.objects.link(sun)
    sun.rotation_euler = (math.radians(28), math.radians(-16), math.radians(-35))
    area_data = bpy.data.lights.new("Area / entrance fill", type="AREA")
    area_data.energy = 1400
    area_data.shape = "DISK"
    area_data.size = max(6, width*0.45)
    area = bpy.data.objects.new(area_data.name, area_data)
    stage.objects.link(area)
    area.location = (cx-5, miny-7, maxz+5)
    aim_at(area, (cx, miny, height*0.40))
    add_architectural_reflections(bpy, stage, center_x=cx, facade_y=miny,
                                 center_z=minz+height*0.49, span=width)

    camera_data = bpy.data.cameras.new("Camera / architectural detail review")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    stage.objects.link(camera)
    target_z = (minz + maxz)/2
    # Blender's orthographic scale is horizontal on landscape previews.
    frontal_scale = max(width*1.42, height*1.86)
    hero_scale = max(width*1.8, height*2.12)
    shots = (
        QualityShot("hero", (cx+width*0.55, miny-width*1.6, maxz*1.28),
                    (cx, cy, target_z), hero_scale),
        QualityShot("front", (cx, miny-width*1.85, target_z+0.5),
                    (cx, cy, target_z), frontal_scale),
        QualityShot("side", (maxx+width*1.2, cy, maxz*1.05),
                    (cx, cy, target_z), max(width*1.6, height*2.0)),
        QualityShot("detail", (cx+3.0, miny-7, 7.9),
                    (cx+1.4, miny, 6.4), 7.6),
    )
    metadata = {
        "asset_id": study.asset_id,
        "generator": "kumiki-blender/reusable-architectural-facade",
        "status": "architectural-craft-study / visual-review-pending",
        "provenance": "Original decorative building study; not a measured world landmark",
        "recipe_sha256": hashlib.sha256(f"facade-study-v1:bays={study.bays}".encode()).hexdigest(),
        "render_profile": {"name": profile.name, "width": profile.width,
                           "height": profile.height, "samples": profile.samples},
        "parameters": {"bays": study.bays, "bay_pitch_m": 3.8},
        "units": "meters",
        "native_axes": "Blender Z-up; GLB exported with glTF-compatible axes",
        "named_parts": dict(study.part_counts),
        "mesh_data_blocks": len(cache),
        "instances": len(study.instances),
        "bounds_m": [study.bounds_min, study.bounds_max],
    }
    return export_revision(bpy, scene, camera, objects, shots, destination,
                           samples=profile.samples, metadata=metadata)
