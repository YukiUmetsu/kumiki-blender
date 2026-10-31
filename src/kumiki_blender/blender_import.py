"""Blender adapter for license-aware, source-agnostic static GLB imports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .asset_import import AssetImportError, AssetProvenance, inspect_source_glb
from .blender_output import aim_at, create_model_scene, export_revision
from .blender_scene import _create_box, _material
from .render_profiles import select_render_profile
from .shots import QualityShot
from .surfaces import get_surface


def imported_bounds(objects: list[Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute world-space bounds, including GLB hierarchy transforms."""
    from mathutils import Vector  # Blender-only dependency.

    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not corners or any(not math.isfinite(value) for point in corners for value in point):
        raise AssetImportError("Imported meshes have no finite world-space bounds")
    return (tuple(min(point[axis] for point in corners) for axis in range(3)),
            tuple(max(point[axis] for point in corners) for axis in range(3)))


def build_imported_scene(
    source: str | Path,
    provenance: AssetProvenance,
    destination: str | Path,
    *,
    bpy_module: Any = None,
) -> dict[str, Any]:
    """Import one reviewed GLB, preserve its materials and publish a QA revision."""
    if bpy_module is None:
        raise RuntimeError("Blender bpy context is required for asset import")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError("Blender 4.2 or newer is required")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise AssetImportError(f"Output revision already exists: {destination}")
    report = inspect_source_glb(source, provenance)
    profile = select_render_profile(1440, 900, 48)
    scene = create_model_scene(bpy, provenance.asset_id)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile.samples
    scene.render.resolution_x = profile.width
    scene.render.resolution_y = profile.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"

    known_object_ids = {obj.as_pointer() for obj in bpy.data.objects}
    status = bpy.ops.import_scene.gltf(
        filepath=str(source),
        import_pack_images=True,
        import_scene_extras=True,
        import_scene_as_collection=True,
        import_select_created_objects=True,
    )
    if "FINISHED" not in status:
        raise AssetImportError("Blender failed to import the supplied GLB")
    objects = [obj for obj in scene.objects if obj.as_pointer() not in known_object_ids and obj.type == "MESH"]
    if not objects:
        raise AssetImportError("GLB imported without any usable mesh objects")
    preserved_parts = sum(1 for obj in objects if isinstance(obj.get("kumiki_part"), str)
                          and obj["kumiki_part"].strip())
    expected_parts = report["glb_audit"]["semantic_nodes"]
    if expected_parts and preserved_parts < min(expected_parts, len(objects)):
        raise AssetImportError("Blender lost original GLB semantic part tags during import")
    for obj in objects:
        if len(obj.data.vertices) == 0 or len(obj.data.polygons) == 0:
            raise AssetImportError(f"Imported mesh is empty: {obj.name}")
        if not isinstance(obj.get("kumiki_part"), str) or not obj["kumiki_part"].strip():
            obj["kumiki_part"] = "imported_mesh"
    imported_materials = {slot.material.name for obj in objects for slot in obj.material_slots
                          if slot.material is not None}
    if report["glb_audit"]["materials"] and not imported_materials:
        raise AssetImportError("Blender did not preserve any source GLB materials")
    # Packed images make both source.blend and exported GLB portable even if
    # the original GLB was downloaded to a short-lived local path.
    for image in bpy.data.images:
        if image.source == "FILE" and image.filepath and not image.packed_file:
            raise AssetImportError(f"Imported texture was not packed: {image.name}")

    bpy.context.view_layer.update()
    bounds_min, bounds_max = imported_bounds(objects)
    dx, dy, dz = (bounds_max[index] - bounds_min[index] for index in range(3))
    cx, cy, cz = ((bounds_min[index] + bounds_max[index]) / 2 for index in range(3))
    extent = max(dx, dy, dz, 3.0)
    # Wide-safe orthographic composition, honoring glTF's meter-based units.
    frame = max(dx * 1.55, dy * 1.7, dz * 1.85 * (16 / 9), 5.0)

    studio = bpy.data.collections.new("Studio / excluded from portable GLB")
    scene.collection.children.link(studio)
    floor = _material(bpy, get_surface("cast-concrete", color="#787978", roughness=0.91))
    floor_extent = max(10, dx * 3, dy * 3)
    _create_box(bpy, studio, "Stage / matte ground",
                (floor_extent, floor_extent, 0.12), (cx, cy, bounds_min[2] - 0.13), floor)
    world = bpy.data.worlds.new("Kumiki / import studio sky")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.29, 0.34, 0.39, 1.0)
    background.inputs["Strength"].default_value = 0.45
    scene.world = world

    sun_data = bpy.data.lights.new("Sun / soft architectural relief", type="SUN")
    sun_data.energy = 2.0
    sun_data.angle = math.radians(4)
    sun = bpy.data.objects.new(sun_data.name, sun_data)
    studio.objects.link(sun)
    sun.rotation_euler = (math.radians(35), math.radians(-18), math.radians(-35))
    area_data = bpy.data.lights.new("Area / gentle fill", type="AREA")
    area_data.energy = 900
    area_data.shape = "DISK"
    area_data.size = max(5.0, extent / 6)
    area = bpy.data.objects.new(area_data.name, area_data)
    studio.objects.link(area)
    area.location = (cx - extent * 0.5, cy - extent * 0.6, cz + extent * 0.8)
    aim_at(area, (cx, cy, cz))

    camera_data = bpy.data.cameras.new("Camera / imported architecture")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    studio.objects.link(camera)
    shots = (
        QualityShot("hero", (cx + extent * 0.7, cy - extent * 1.4, cz + extent * 0.48),
                    (cx, cy, cz), frame),
        QualityShot("front", (cx, cy - extent * 1.6, cz + extent * 0.12),
                    (cx, cy, cz), frame * 0.95),
        QualityShot("detail", (cx + extent * 0.32, cy - extent * 0.7, cz + extent * 0.35),
                    (cx, cy, cz + dz * 0.2), max(2.0, frame * 0.34)),
    )
    category_counts: dict[str, int] = {}
    for obj in objects:
        part = obj["kumiki_part"]
        category_counts[part] = category_counts.get(part, 0) + 1
    metadata = {
        "asset_id": provenance.asset_id,
        "generator": "kumiki-blender/static-glb-import",
        "status": "reviewed-source-import / visual-review-pending",
        "provenance": provenance.to_manifest(report["input_sha256"]),
        "source_glb_bytes": report["input_bytes"],
        "imported_materials": len(imported_materials),
        "render_profile": {"name": profile.name, "width": profile.width,
                           "height": profile.height, "samples": profile.samples},
        "units": "meters",
        "native_axes": "Blender Z-up; GLB exported with glTF-compatible axes",
        "named_parts": category_counts,
        "bounds_m": [bounds_min, bounds_max],
    }
    return export_revision(bpy, scene, camera, objects, shots, destination,
                           samples=profile.samples, metadata=metadata)
