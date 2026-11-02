"""High-scan -> decimated, UV-unique, PBR-baked statue LOD in real Blender.

The high-resolution licensed GLB remains editable inside the private .blend;
only the low-resolution proxy plus baked maps is exported to GLB. This cannot
recover missing high-resolution facial geometry or authenticate a scan.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from kumiki_blender.asset_import import AssetImportError, AssetProvenance
from kumiki_blender.blender_import import imported_bounds
from kumiki_blender.blender_output import aim_at
from kumiki_blender.glb_audit import audit_glb

from .lods import ScanLODSettings, plan_scan_lods


def _deselect(bpy: Any) -> None:
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def _new_bake_material(bpy: Any, name: str, images: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    shader = nodes.get("Principled BSDF")
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 0.8
    textures = {}
    for kind in ("normal", "color", "roughness"):
        tex = nodes.new("ShaderNodeTexImage")
        tex.name = f"Baked {kind} / {name}"
        tex.image = images[kind]
        textures[kind] = tex
    return material, textures


def _connect_baked_pbr(material: Any, textures: dict[str, Any]) -> None:
    nodes = material.node_tree.nodes
    shader = nodes.get("Principled BSDF")
    links = material.node_tree.links
    normal = nodes.new("ShaderNodeNormalMap")
    normal.space = "TANGENT"
    links.new(textures["normal"].outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    links.new(textures["color"].outputs["Color"], shader.inputs["Base Color"])
    links.new(textures["roughness"].outputs["Color"], shader.inputs["Roughness"])


def _save_image(image: Any, path: Path) -> None:
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    image.pack()


def _bake_one(bpy: Any, high: Any, low: Any, target: Path,
              index: int, settings: ScanLODSettings) -> dict[str, Any]:
    """Make one unique UV atlas and transfer source-shape/material evidence."""
    original_faces = len(high.data.polygons)
    low_faces = len(low.data.polygons)
    _deselect(bpy)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    if low.data.uv_layers:
        low.data.uv_layers.clear()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    result = bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    if "FINISHED" not in result or not low.data.uv_layers:
        raise AssetImportError(f"Could not unwrap unique UVs for {low.name}")
    images = {}
    names = {}
    for kind in ("normal", "color", "roughness"):
        name = f"lod-{index:02d}-{kind}.png"
        image = bpy.data.images.new("Statue LOD / " + name, width=settings.map_size,
                                    height=settings.map_size, alpha=True, float_buffer=False)
        image.colorspace_settings.name = "Non-Color" if kind != "color" else "sRGB"
        images[kind], names[kind] = image, "maps/" + name
    material, textures = _new_bake_material(bpy, f"Scan LOD / {index:02d}", images)
    low.data.materials.clear()
    low.data.materials.append(material)
    for kind, bake_type, filter_set in (("normal", "NORMAL", None),
                                        ("color", "DIFFUSE", {"COLOR"}),
                                        ("roughness", "ROUGHNESS", None)):
        for tex in textures.values():
            tex.select = False
        tex = textures[kind]
        tex.select = True
        material.node_tree.nodes.active = tex
        _deselect(bpy)
        high.select_set(True)
        low.select_set(True)
        bpy.context.view_layer.objects.active = low
        kwargs = {
            "type": bake_type, "use_selected_to_active": True,
            "max_ray_distance": settings.ray_distance_m,
            "cage_extrusion": settings.ray_distance_m,
            "margin": 12, "use_clear": True,
        }
        if filter_set is not None:
            kwargs["pass_filter"] = filter_set
        status = bpy.ops.object.bake(**kwargs)
        if "FINISHED" not in status:
            raise AssetImportError(f"Blender failed to bake {kind} for {high.name}")
        _save_image(images[kind], target / names[kind])
    _connect_baked_pbr(material, textures)
    low["kumiki_part"] = str(high.get("kumiki_part", high.name))
    return {"source_name": high.name, "low_name": low.name,
            "high_faces": original_faces, "low_faces": low_faces,
            "uv_layers": len(low.data.uv_layers), "maps": names}


def _preview_comparison(bpy: Any, source: list[Any], simplified: list[Any],
                        bounds_min: tuple[float, ...], bounds_max: tuple[float, ...],
                        target: Path) -> tuple[str, str]:
    scene = bpy.context.scene
    world = bpy.data.worlds.new("LOD QA / neutral studio")
    world.use_nodes = True
    scene.world = world
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.42, 0.42, 0.42, 1)
    background.inputs["Strength"].default_value = 0.8
    size = max(bounds_max[i]-bounds_min[i] for i in range(3))
    center = tuple((bounds_min[i]+bounds_max[i])/2 for i in range(3))
    light = bpy.data.lights.new("LOD QA / raking sun", "SUN")
    light.energy = 2.0
    sun = bpy.data.objects.new(light.name, light)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.65, -0.45, -0.35)
    camera_data = bpy.data.cameras.new("LOD QA / same-camera comparison")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(size*1.8, 0.5)
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    camera.location = (center[0]+size*1.1, center[1]-size*1.7, center[2]+size*0.9)
    aim_at(camera, center)
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.render.resolution_x = scene.render.resolution_y = 320
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    frames = []
    for use_low, name in ((False, "source-high.png"), (True, "baked-low.png")):
        for obj in source:
            obj.hide_render = use_low
        for obj in simplified:
            obj.hide_render = not use_low
        scene.render.filepath = str(target / name)
        bpy.ops.render.render(write_still=True)
        frames.append(name)
    return tuple(frames)


def bake_statue_lods(source_glb: str | Path, provenance: AssetProvenance,
                     output: str | Path, settings: ScanLODSettings, *,
                     bpy_module: Any = None) -> dict[str, Any]:
    """Build a reviewable LOD mesh with real high->low Cycles texture bakes."""
    if bpy_module is None:
        raise RuntimeError("Scan LOD baking requires Blender bpy")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise AssetImportError("Scan LOD baking needs Blender 4.2+")
    preflight = plan_scan_lods(source_glb, provenance, output, settings)
    target = Path(output).resolve()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    status = bpy.ops.import_scene.gltf(filepath=preflight["source_glb"],
                                       import_pack_images=True)
    if "FINISHED" not in status:
        raise AssetImportError("Blender could not import the reviewed GLB")
    highs = [obj for obj in scene.objects if obj.type == "MESH"]
    if not 1 <= len(highs) <= 12:
        raise AssetImportError("Scan LOD requires 1–12 mesh parts; join/split the source scan by detail region first")
    if 3*len(highs)*settings.map_size*settings.map_size > 96_000_000:
        raise AssetImportError("Aggregate bake atlas exceeds the 96-million-texel budget; reduce map size or mesh parts")
    if any(not obj.data.polygons or not obj.data.vertices for obj in highs):
        raise AssetImportError("Cannot bake empty scan meshes")
    high_faces = sum(len(obj.data.polygons) for obj in highs)
    if high_faces > 10_000_000:
        raise AssetImportError("Source scan exceeds 10 million faces; prepare separate regions")
    if settings.target_triangles < 32*len(highs):
        raise AssetImportError("Triangle budget cannot preserve 32 faces per source part")
    bounds_min, bounds_max = imported_bounds(highs)
    if any(not math.isfinite(value) for value in (*bounds_min, *bounds_max)):
        raise AssetImportError("Source bounds are not finite")

    # Imported source meshes keep original materials, UVs and object hierarchy.
    # Low duplicates are flattened to identical world matrices before baking.
    lows = []
    for index, high in enumerate(highs):
        low = high.copy()
        low.data = high.data.copy()
        scene.collection.objects.link(low)
        world_matrix = high.matrix_world.copy()
        low.parent = None
        low.matrix_world = world_matrix
        low.name = f"LOD / {index:02d} / {high.name}"
        target_faces = max(32, int(settings.target_triangles*len(high.data.polygons)/high_faces))
        if len(low.data.polygons) > target_faces:
            decimate = low.modifiers.new("LOD / measured silhouette reduction", "DECIMATE")
            decimate.ratio = max(0.0001, target_faces/len(low.data.polygons))
            _deselect(bpy)
            low.select_set(True)
            bpy.context.view_layer.objects.active = low
            bpy.ops.object.modifier_apply(modifier=decimate.name)
        if not low.data.polygons:
            raise AssetImportError(f"Decimation removed all faces from {high.name}")
        lows.append(low)

    target.mkdir(parents=True, exist_ok=False)
    (target / "maps").mkdir()
    record = {**preflight, "status": "review-required", "high_faces": high_faces,
              "low_faces": 0, "parts": [], "failed_stage": None, "historical_fidelity_verified": False}

    def checkpoint() -> None:
        (target / "lod-report.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        for index, (high, low) in enumerate(zip(highs, lows)):
            part = _bake_one(bpy, high, low, target, index, settings)
            record["parts"].append(part)
            checkpoint()
        record["low_faces"] = sum(part["low_faces"] for part in record["parts"])
        record["face_reduction_fraction"] = round(1-record["low_faces"]/high_faces, 6)
        for high in highs:
            high.hide_render = True
        for low in lows:
            low.hide_render = False
        _deselect(bpy)
        for low in lows:
            low.select_set(True)
        bpy.context.view_layer.objects.active = lows[0]
        result = bpy.ops.export_scene.gltf(
            filepath=str(target / "model-lod.glb"), export_format="GLB",
            use_selection=True, export_extras=True, export_normals=True,
            export_texcoords=True, export_materials="EXPORT", export_apply=False,
            export_animations=False,
        )
        if "FINISHED" not in result:
            raise AssetImportError("Blender GLB export failed")
        record["glb_audit"] = audit_glb(target / "model-lod.glb", require_pbr_textures=True)
        if settings.render_previews:
            record["previews"] = _preview_comparison(bpy, highs, lows, bounds_min, bounds_max, target)
        # Keep the high scan in the editable .blend, but only export the decimated proxy.
        for high in highs:
            high.hide_render = True
        for low in lows:
            low.hide_render = False
        bpy.ops.wm.save_as_mainfile(filepath=str(target / "scan-lod.blend"))
        record["status"] = "lod-bake-candidate / artistic-review-required"
    except (AssetImportError, RuntimeError, OSError, ValueError) as error:
        record["failed_stage"] = "bake-or-export"
        record["error"] = str(error)
    checkpoint()
    return record
