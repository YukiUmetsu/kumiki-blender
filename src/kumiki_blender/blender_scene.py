"""Blender-only scene construction and GLB/preview export.

Importing this module in ordinary Python is safe. Use a fresh background Blender
process to build an immutable output revision; no existing scene is deleted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .export_audit import audit_export_files
from .glb_audit import audit_glb
from .lattice import Beam, GussetPlate, TowerPlan
from .recipe import TowerRecipe
from .shots import plan_quality_shots
from .surfaces import PBRSurface, get_surface


def _srgb_channel_to_linear(value: int) -> float:
    x = value / 255
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def _color(hex_color: str) -> tuple[float, float, float, float]:
    rgb = bytes.fromhex(hex_color.lstrip("#"))
    return (*(_srgb_channel_to_linear(channel) for channel in rgb), 1.0)


def _material(bpy: Any, spec: PBRSurface) -> Any:
    material = bpy.data.materials.new(spec.name)
    material.use_nodes = True
    shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = _color(spec.color)
    shader.inputs["Roughness"].default_value = spec.roughness
    shader.inputs["Metallic"].default_value = spec.metallic
    if spec.transmission:
        shader.inputs["Transmission Weight"].default_value = spec.transmission
        shader.inputs["IOR"].default_value = spec.ior
    return material


def _attach_mesh(bpy: Any, collection: Any, name: str, vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]], material: Any) -> Any:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _beam_geometry(member: Beam) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    """Mesh a capped 16-sided structural tube between absolute 3D points."""
    ax, ay, az = member.start
    bx, by, bz = member.end
    direction = (bx - ax, by - ay, bz - az)
    distance = math.sqrt(sum(value * value for value in direction))
    if distance <= 0.000001:
        raise RuntimeError(f"Zero-length structural member: {member.name}")
    e = tuple(value / distance for value in direction)
    reference = (0, 1, 0) if abs(e[2]) > 0.9 else (0, 0, 1)

    def cross(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    across = cross(e, reference)
    length = math.sqrt(sum(value * value for value in across))
    u = tuple(value / length for value in across)
    v = cross(e, u)
    radial_segments = 16 if member.group in ("leg", "antenna") else 12
    vertices = []
    for origin in (member.start, member.end):
        for index in range(radial_segments):
            theta = math.tau * index / radial_segments
            vertices.append(tuple(origin[j] + member.radius * (math.cos(theta) * u[j] + math.sin(theta) * v[j]) for j in range(3)))
    faces = [
        (i, (i + 1) % radial_segments, radial_segments + (i + 1) % radial_segments, radial_segments + i)
        for i in range(radial_segments)
    ]
    faces.extend([tuple(reversed(range(radial_segments))), tuple(range(radial_segments, 2 * radial_segments))])
    return vertices, faces


def _create_beam(bpy: Any, collection: Any, member: Beam, material: Any) -> Any:
    vertices, faces = _beam_geometry(member)
    radial_segments = len(faces) - 2
    obj = _attach_mesh(bpy, collection, member.name, vertices, faces, material)
    obj["kumiki_part"] = member.group
    for face in obj.data.polygons[:radial_segments]:
        face.use_smooth = True
    if member.group == "leg":
        bevel = obj.modifiers.new("Manufactured edge soften", "BEVEL")
        bevel.width = min(member.radius * 0.08, 0.08)
        bevel.segments = 2
    return obj


def _box_geometry(dimensions: tuple[float, float, float], center: tuple[float, float, float]) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    x, y, z = (half / 2 for half in dimensions)
    cx, cy, cz = center
    vertices = [(cx + dx * x, cy + dy * y, cz + dz * z) for dz in (-1, 1) for dy in (-1, 1) for dx in (-1, 1)]
    faces = [(2, 3, 1, 0), (5, 7, 6, 4), (1, 5, 4, 0), (6, 7, 3, 2), (4, 6, 2, 0), (3, 7, 5, 1)]
    return vertices, faces


def _create_box(bpy: Any, collection: Any, name: str, dimensions: tuple[float, float, float], center: tuple[float, float, float], material: Any) -> Any:
    vertices, faces = _box_geometry(dimensions, center)
    obj = _attach_mesh(bpy, collection, name, vertices, faces, material)
    bevel = obj.modifiers.new("Edge rounding", "BEVEL")
    bevel.width = min(dimensions) * 0.07
    bevel.segments = 2
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return obj


def _gusset_geometry(plate: GussetPlate) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    """Orient a physical gusset plate in the face plane of an arbitrary bay."""
    local, faces = _box_geometry((plate.width, plate.height, plate.thickness), (0, 0, 0))
    axes = (plate.axis_u, plate.axis_v, plate.normal)
    vertices = [
        tuple(plate.center[i] + sum(coords[axis] * axes[axis][i] for axis in range(3)) for i in range(3))
        for coords in local
    ]
    return vertices, faces


def _create_gusset(bpy: Any, collection: Any, plate: GussetPlate, material: Any) -> Any:
    vertices, faces = _gusset_geometry(plate)
    obj = _attach_mesh(bpy, collection, plate.name, vertices, faces, material)
    obj["kumiki_part"] = "structural_gusset"
    bevel = obj.modifiers.new("Steel plate deburr", "BEVEL")
    bevel.width = min(plate.thickness * 0.3, 0.03)
    bevel.segments = 2
    obj.modifiers.new("Weighted plate normals", "WEIGHTED_NORMAL")
    return obj


def _aim_at(obj: Any, target: tuple[float, float, float]) -> None:
    from mathutils import Vector  # Blender-provided module, imported only when building.

    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def build_scene(recipe: TowerRecipe, plan: TowerPlan, destination: Path, *, bpy_module: Any = None) -> dict[str, Any]:
    """Create a new scene, preserve pre-existing scenes and export a revision.

    This function must be invoked inside Blender. A fresh factory-started
    background process is required for reproducible renders.
    """
    if bpy_module is None:
        raise RuntimeError("Blender bpy context is required for this build")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError("Blender 4.2 or newer is required")
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Output revision already exists: {destination}")

    scene = bpy.data.scenes.new(f"Kumiki / {recipe.asset_id}")
    # Background Blender builds may have no window; scene overrides work without
    # relying on a visible UI. Never remove or replace an existing scene.
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = recipe.samples
    scene.render.resolution_x = recipe.preview_width
    scene.render.resolution_y = recipe.preview_height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.render.film_transparent = False

    model_collection = bpy.data.collections.new("Model / portable GLB")
    scene.collection.children.link(model_collection)
    stage_collection = bpy.data.collections.new("Studio / excluded from GLB")
    scene.collection.children.link(stage_collection)

    steel = _material(bpy, get_surface("painted-steel", color=recipe.steel_color, roughness=recipe.roughness))
    gusset_steel = _material(bpy, get_surface("painted-steel", color=recipe.steel_color,
                                            roughness=min(1.0, recipe.roughness + 0.13)))
    fastener_steel = _material(bpy, get_surface("galvanized-steel"))
    trim = _material(bpy, get_surface("galvanized-steel", color="#9EA8AD", roughness=0.35))
    platform = _material(bpy, get_surface("cast-concrete", color="#70767B", roughness=0.78))
    ground_mat = _material(bpy, get_surface("natural-stone", color="#77736A", roughness=0.93))

    objects = []
    for beam in plan.beams:
        material = trim if beam.group == "railing" else steel
        objects.append(_create_beam(bpy, model_collection, beam, material))
    for plate in plan.gussets:
        objects.append(_create_gusset(bpy, model_collection, plate, gusset_steel))
    for fastener in plan.fasteners:
        objects.append(_create_beam(bpy, model_collection, fastener, fastener_steel))
    for deck in plan.decks:
        obj = _create_box(bpy, model_collection, deck.name, (deck.span_m, deck.span_m, deck.thickness), (0, 0, deck.z - deck.thickness / 2), platform)
        obj["kumiki_part"] = "observation_deck"
        objects.append(obj)

    ground_extent = recipe.base_span_m * 3
    _create_box(bpy, stage_collection, "Stage / ground", (ground_extent, ground_extent, 0.5), (0, 0, -0.4), ground_mat)
    world = bpy.data.worlds.new("Kumiki / neutral sky")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.28, 0.34, 0.4, 1)
    background.inputs["Strength"].default_value = 0.45
    scene.world = world

    sun = bpy.data.lights.new("Sun / soft daylight", type="SUN")
    sun.energy = 2.0
    sun.angle = math.radians(5)
    sun_obj = bpy.data.objects.new(sun.name, sun)
    stage_collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(30), math.radians(-18), math.radians(-32))

    camera_data = bpy.data.cameras.new("Camera / architecture hero")
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    stage_collection.objects.link(camera)
    shots = plan_quality_shots(recipe)
    hero = shots[0]
    camera.location = hero.position
    _aim_at(camera, hero.target)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = hero.ortho_scale
    scene.camera = camera

    destination.mkdir(parents=True, exist_ok=False)
    scene.render.filepath = str(destination / "preview.png")
    with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
        bpy.ops.wm.save_as_mainfile(filepath=str(destination / "source.blend"))

        # Model objects alone become the portable GLB. Never export lights/ground.
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True, view_layer=scene.view_layers[0])
        bpy.ops.export_scene.gltf(
            filepath=str(destination / "model.glb"), export_format="GLB",
            use_selection=True, export_materials="EXPORT", export_animations=False,
            export_cameras=False, export_lights=False, export_extras=True,
        )
        glb_report = audit_glb(destination / "model.glb")
        bpy.ops.object.select_all(action="DESELECT")
        preview_dir = destination / "previews"
        preview_dir.mkdir(exist_ok=True)
        for shot in shots:
            camera.location = shot.position
            _aim_at(camera, shot.target)
            camera_data.ortho_scale = shot.ortho_scale
            scene.cycles.samples = recipe.samples if shot.name == "hero" else max(8, recipe.samples // 2)
            scene.render.filepath = str(destination / "preview.png" if shot.name == "hero" else preview_dir / f"{shot.name}.png")
            bpy.ops.render.render(write_still=True, scene=scene.name)

    views = ["preview.png", *[f"previews/{shot.name}.png" for shot in shots[1:]]]
    exported_sizes = audit_export_files(destination, ["source.blend", *views])

    manifest = {
        "asset_id": recipe.asset_id,
        "schema_version": 1,
        "generator": "kumiki-blender/lattice-tower",
        "status": "structural-prototype / visual-review-pending",
        "provenance": "Original procedural demonstration; NOT a verified replica of a landmark",
        "blender_version": bpy.app.version_string,
        "recipe_sha256": recipe.recipe_sha256,
        "units": "meters",
        "native_axes": "Blender Z-up; exported GLB has glTF-compatible axes",
        "glb_audit": glb_report,
        "output_bytes": {**exported_sizes, "model.glb": glb_report["bytes"]},
        "named_parts": {"beams": len(plan.beams), "decks": len(plan.decks),
                        "gussets": len(plan.gussets), "fasteners": len(plan.fasteners)},
        "bounds_m": [plan.bounds_min, plan.bounds_max],
        "files": {"source": "source.blend", "model": "model.glb", "preview": "preview.png",
                  "quality_views": views},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
