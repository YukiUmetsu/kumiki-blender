"""Blender Cycles/GLB adapter for original Egyptian monument detail studies."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from .blender_output import aim_at, create_model_scene, export_revision
from .blender_import import imported_bounds
from .blender_scene import _create_box, _material
from .blender_textures import ensure_metric_uvs
from .egyptian_monuments import MonumentStudy
from .landmark_scan import (
    SphinxScan, fit_sphinx_scan, sphinx_proportion_review,
    sphinx_source_readiness_review,
)
from .asset_import import AssetImportError
from .render_profiles import select_render_profile
from .shots import QualityShot
from .sphinx_sculpt import combine_sculpture_sources, sphinx_head_parts, sphinx_integral_parts
from .surfaces import get_surface


def plan_monument_shots(study: MonumentStudy) -> tuple[QualityShot, ...]:
    """Framing adapts to a pyramid, a sphinx closeup, or both on one site."""
    lo, hi = study.bounds_min, study.bounds_max
    span = hi[0]-lo[0]
    depth = hi[1]-lo[1]
    height = hi[2]-lo[2]
    cx, cy = (lo[0]+hi[0])/2, (lo[1]+hi[1])/2
    cz = (lo[2]+hi[2])/2
    scale = max(span*1.75, depth*1.65, height*2.3, 15)
    shots = [
        QualityShot("hero", (cx-span*1.65, cy-depth*1.7, hi[2]*1.4),
                    (cx, cy, cz), scale),
        QualityShot("site", (cx, cy-depth*2.0, hi[2]*0.9),
                    (cx, cy, cz), scale*1.06),
    ]
    if study.kind in ("pyramid", "giza-study"):
        py = 8 if study.kind == "giza-study" else 0
        base = dict(study.parameters)["pyramid_base_m"]
        shots.append(QualityShot("masonry", (-base*.45, py-base*.72, base*.25),
                                 (-base*.32, py-base*.42, base*.13), base*.37))
    if study.kind in ("sphinx", "giza-study"):
        sy = -(dict(study.parameters)["pyramid_base_m"]/2+8) if study.kind == "giza-study" else 0
        length = dict(study.parameters)["sphinx_length_m"]
        sphinx_height = 3.6 * length/10.5
        # A side-biased three-quarter view keeps the whole recumbent lion and
        # front paws in frame; the former head-centered framing cropped its rump.
        shots.append(QualityShot("sphinx", (-length*.63, sy-length*.97, sphinx_height*1.28),
                                 (-length*.025, sy, sphinx_height*.51), length*1.48))
        shots.append(QualityShot("sphinx-profile", (length*.15, sy-length*1.6, sphinx_height*1.04),
                                 (-length*.005, sy, sphinx_height*.45), length*1.33))
        shots.append(QualityShot("sphinx-face", (-length*.88, sy-length*.42, sphinx_height*1.11),
                                 (-length*.36, sy, sphinx_height*.80), length*.46))
        # True frontal elevation is essential for comparing the triangular
        # nemes wings, squared jaw, eye spacing and ear placement against
        # archaeological photographs; the oblique face view hides asymmetry.
        shots.append(QualityShot("sphinx-frontal", (-length*.96, sy, sphinx_height*1.04),
                                 (-length*.36, sy, sphinx_height*.835), length*.33))
    return tuple(shots)


def _integrate_sphinx_group(bpy: Any, scene: Any, model: Any,
                            sources: tuple[Any, ...], objects: list[Any],
                            limestone: Any, *, label: str,
                            voxel: float) -> dict[str, int | float]:
    """Voxel-union one anatomy group, retaining hidden sources in .blend.

    The fused solid alone appears in portable GLB/renders; separately named
    source solids remain editable, including the face and nemes headdress.
    """
    if not sources:
        return {"sources": 0, "voxel_size_m": 0, "unified_vertices": 0}
    by_name = {obj.name: obj for obj in objects}
    if any(part.name not in by_name for part in sources):
        raise RuntimeError(f"Missing Blender source meshes for Sphinx {label} union")

    merged = combine_sculpture_sources(sources)
    union_mesh = bpy.data.meshes.new(f"Sphinx {label} / intersecting source volumes")
    union_mesh.from_pydata(merged.vertices, [], merged.faces)
    union_mesh.update()
    union_mesh.materials.append(limestone)
    sculpt = bpy.data.objects.new(f"sphinx-unified-carved-{label}", union_mesh)
    model.objects.link(sculpt)

    modifier = sculpt.modifiers.new(f"Manifold {label} / voxel union", "REMESH")
    modifier.mode = "VOXEL"
    modifier.voxel_size = voxel
    modifier.use_remove_disconnected = False
    modifier.use_smooth_shade = True
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = sculpt.evaluated_get(depsgraph)
    skin = bpy.data.meshes.new_from_object(
        evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
    if skin is None or len(skin.vertices) < 1000 or not skin.polygons:
        raise RuntimeError(f"Blender voxel union did not create a useful Sphinx {label} surface")
    if len(skin.vertices) > 800_000:
        raise RuntimeError(f"Blender Sphinx {label} union exceeded 800k vertices")
    # A voxel operation can change material slots and UVs. Assign both on its
    # evaluated geometry so GLB export and Cycles agree on the same sculpture.
    if not skin.materials:
        skin.materials.append(limestone)
    for poly in skin.polygons:
        poly.use_smooth = True
    ensure_metric_uvs(skin, tile_size_m=.95)
    sculpt.modifiers.clear()
    sculpt.data = skin
    bpy.data.meshes.remove(union_mesh)

    import bmesh
    check = bmesh.new()
    try:
        check.from_mesh(skin)
        bad_edges = sum(not edge.is_manifold for edge in check.edges)
        if bad_edges:
            raise RuntimeError(f"Unified Sphinx {label} has {bad_edges} non-manifold edges")
    finally:
        check.free()

    source_collection = bpy.data.collections.new(f"Sphinx / hidden editable {label} sources")
    scene.collection.children.link(source_collection)
    source_names = {part.name for part in sources}
    for obj in tuple(objects):
        if obj.name in source_names:
            model.objects.unlink(obj)
            source_collection.objects.link(obj)
            obj.hide_render = True
    source_collection.hide_render = True
    objects[:] = [obj for obj in objects if obj.name not in source_names]
    objects.append(sculpt)
    sculpt["kumiki_part"] = "sphinx"
    sculpt["kumiki_source_names"] = ",".join(sorted(source_names))
    return {"sources": len(sources), "voxel_size_m": voxel,
            "unified_vertices": len(skin.vertices)}


def _place_sphinx_scan(bpy: Any, scene: Any, model: Any, scan: SphinxScan,
                       study: MonumentStudy) -> tuple[list[Any], dict[str, Any]]:
    """Import and normalize a licensed scan, keeping original PBR and hierarchy."""
    from mathutils import Matrix

    known = {obj.as_pointer() for obj in bpy.data.objects}
    result = bpy.ops.import_scene.gltf(
        filepath=str(scan.source), import_pack_images=True,
        import_scene_extras=True, import_scene_as_collection=True,
        import_select_created_objects=True,
    )
    if "FINISHED" not in result:
        raise AssetImportError("Blender could not import the Sphinx scan")
    imported = [obj for obj in bpy.data.objects if obj.as_pointer() not in known]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes or any(not obj.data.vertices or not obj.data.polygons for obj in meshes):
        raise AssetImportError("Sphinx scan imported without usable mesh surfaces")
    if any(image.source == "FILE" and image.filepath and not image.packed_file
           for image in bpy.data.images):
        raise AssetImportError("Sphinx scan texture was not packed into the scene")
    source_materials = {slot.material.name: slot.material
                        for obj in meshes for slot in obj.material_slots
                        if slot.material is not None}
    if scan.inspection["glb_audit"]["materials"] and not source_materials:
        raise AssetImportError("Blender lost the Sphinx scan's PBR materials")
    # Move newly imported meshes and hierarchy empties into the intended model
    # collection. Parent transforms remain intact and are applied just once.
    for obj in imported:
        if model not in obj.users_collection:
            model.objects.link(obj)
        for prior in tuple(obj.users_collection):
            if prior != model:
                prior.objects.unlink(obj)
    bpy.context.view_layer.update()
    source_bounds = imported_bounds(meshes)
    params = dict(study.parameters)
    sy = -(params["pyramid_base_m"]/2+8) if study.kind == "giza-study" else 0.0
    alignment = fit_sphinx_scan(
        *source_bounds, front_axis=scan.front_axis,
        target_length_m=params["sphinx_length_m"], target_center_xy=(0.0, sy))
    transform = (Matrix.Translation(alignment.translation_m)
                 @ Matrix.Rotation(alignment.rotation_z_rad, 4, "Z")
                 @ Matrix.Scale(alignment.uniform_scale, 4))
    imported_ids = {obj.as_pointer() for obj in imported}
    for obj in imported:
        if obj.parent is None or obj.parent.as_pointer() not in imported_ids:
            obj.matrix_world = transform @ obj.matrix_world
    for obj in meshes:
        obj["kumiki_part"] = "sphinx-scan"
        obj["kumiki_source_asset"] = scan.provenance.asset_id
    bpy.context.view_layer.update()
    actual = imported_bounds(meshes)
    if any(abs(actual[side][axis] - alignment.aligned_bounds_m[side][axis]) > 1e-3
           for side in range(2) for axis in range(3)):
        raise AssetImportError("Sphinx scan hierarchy did not preserve its fitted bounds")
    source_readiness = sphinx_source_readiness_review(
        alignment,
        mesh_objects=len(meshes),
        vertex_count=sum(len(obj.data.vertices) for obj in meshes),
        triangle_count=sum(sum(len(poly.vertices)-2 for poly in obj.data.polygons)
                           for obj in meshes),
        uv_mesh_objects=sum(bool(obj.data.uv_layers) for obj in meshes),
        image_textures=sum(
            node.type == "TEX_IMAGE" and node.image is not None
            for material in source_materials.values() if material.use_nodes
            for node in material.node_tree.nodes),
    )
    return meshes, {
        "provenance": scan.provenance.to_manifest(scan.inspection["input_sha256"]),
        "source_glb_bytes": scan.inspection["input_bytes"],
        "source_front_axis": scan.front_axis,
        "mesh_objects": len(meshes), "source_materials": len(source_materials),
        "placement": alignment.to_manifest(),
        "verified_bounds_m": actual,
        "source_geometry_review": sphinx_proportion_review(alignment),
        "source_readiness_review": source_readiness,
        "geometry_origin": "external scan / visually and historically unverified",
    }


def build_monument_scene(study: MonumentStudy, destination: Path, *,
                         sphinx_scan: SphinxScan | None = None,
                         bpy_module: Any = None) -> dict[str, Any]:
    """Stage warm limestone lighting, save editable scene and audit portable GLB."""
    if bpy_module is None:
        raise RuntimeError("Building monuments requires Blender bpy")
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
    scene.view_settings.view_transform = "AgX"

    model = bpy.data.collections.new("Egyptian monuments / portable GLB")
    scene.collection.children.link(model)
    hidden_sphinx = None
    if sphinx_scan:
        if study.kind == "pyramid":
            raise AssetImportError("Cannot replace a Sphinx in a pyramid-only scene")
        hidden_sphinx = bpy.data.collections.new("Sphinx / hidden parametric authoring alternative")
        scene.collection.children.link(hidden_sphinx)
        hidden_sphinx.hide_render = True
    stage = bpy.data.collections.new("Warm limestone studio / excluded from GLB")
    scene.collection.children.link(stage)
    materials = {
        "core-limestone": get_surface("weathered-stone", color="#A89A7B", roughness=0.88,
                                      weathering_strength=0.69),
        "weathered-limestone": get_surface("weathered-stone", color="#B9A888", roughness=0.84,
                                           weathering_strength=0.55),
        "casing-limestone": get_surface("natural-stone", color="#DFD5BC", roughness=0.68),
        "sphinx-limestone": get_surface("weathered-stone", color="#B9A98A", roughness=0.79,
                                        weathering_strength=0.62),
        "sphinx-head-limestone": get_surface("weathered-stone", color="#C4B596", roughness=0.84,
                                             weathering_strength=0.27),
        "bedrock-limestone": get_surface("weathered-stone", color="#A89576", roughness=0.91,
                                         weathering_strength=0.78),
    }
    # Staggered courses should not all have precisely the same quarry colour
    # and pitting pattern. Stable per-block variants avoid arbitrary RNG state.
    for index, color in enumerate(("#A9987B", "#B9A789", "#C0AD8D", "#A38F75")):
        materials[f"quarried-limestone-{index}"] = get_surface(
            "weathered-stone", color=color, roughness=.93,
            weathering_strength=.85)
    palette = {name: _material(bpy, spec) for name, spec in materials.items()}
    # The packed maps remain portable glTF normal textures. Their shallow
    # relief needs stronger raking-light response on the weathered statue than
    # on generic polished architectural stone.
    for name in ("sphinx-limestone", "bedrock-limestone", "weathered-limestone",
                 *(f"quarried-limestone-{i}" for i in range(4))):
        for node in palette[name].node_tree.nodes:
            if node.type == "NORMAL_MAP":
                node.inputs["Strength"].default_value = (1.12 if name.startswith("quarried-") else .82)
    from mathutils import Matrix

    cache: dict[tuple[int, str], Any] = {}
    objects: list[Any] = []
    for part in study.instances:
        palette_key = part.material
        if (part.name.startswith("pyramid-course-") and
                part.material in ("weathered-limestone", "core-limestone")):
            variant = int.from_bytes(hashlib.sha256(part.name.encode()).digest()[:2], "big") % 4
            palette_key = f"quarried-limestone-{variant}"
        key = (id(part.mesh), palette_key)
        if key not in cache:
            mesh = bpy.data.meshes.new(f"Geometry / {part.name}")
            mesh.from_pydata(part.mesh.vertices, [], part.mesh.faces)
            mesh.update()
            ensure_metric_uvs(mesh, tile_size_m=1.2)
            mesh.materials.append(palette[palette_key])
            if part.name.startswith("sphinx-"):
                for poly in mesh.polygons:
                    # Preserve rock's low-frequency chipped facets on the
                    # torso and site while smoothing the head and small detail.
                    if len(poly.vertices) == 4 and not any(
                        role in part.name for role in
                        ("lion-torso", "chest", "forepaw", "haunch", "bedrock")):
                        poly.use_smooth = True
            cache[key] = mesh
        obj = bpy.data.objects.new(part.name, cache[key])
        (hidden_sphinx if part.name.startswith("sphinx-") and hidden_sphinx else model).objects.link(obj)
        obj.matrix_world = Matrix((
            (part.axis_x[0], part.axis_y[0], part.axis_z[0], part.origin[0]),
            (part.axis_x[1], part.axis_y[1], part.axis_z[1], part.origin[1]),
            (part.axis_x[2], part.axis_y[2], part.axis_z[2], part.origin[2]),
            (0, 0, 0, 1),
        ))
        obj["kumiki_part"] = "pyramid" if part.name.startswith("pyramid-") else "sphinx"
        if hidden_sphinx and part.name.startswith("sphinx-"):
            obj.hide_render = True
        # Organic silhouette lofts are already smooth; small bevels are only
        # needed on cut/exposed stone blocks and the terrain edge.
        if part.name.startswith("pyramid-") and "-core" not in part.name:
            bevel = obj.modifiers.new("Worn stone arris", "BEVEL")
            bevel.width = 0.016
            bevel.segments = 2
            obj.modifiers.new("Stone weighted normals", "WEIGHTED_NORMAL")
        elif "bedrock" in part.name:
            bevel = obj.modifiers.new("Carved detail arris", "BEVEL")
            bevel.width = 0.012
            bevel.segments = 2
            obj.modifiers.new("Stone weighted normals", "WEIGHTED_NORMAL")
        if not (hidden_sphinx and part.name.startswith("sphinx-")):
            objects.append(obj)

    if sphinx_scan:
        imported_meshes, scan_metadata = _place_sphinx_scan(bpy, scene, model, sphinx_scan, study)
        objects.extend(imported_meshes)
        sculpt_report: dict[str, Any] = {"mode": "external-scan", "source_parts_hidden":
                                        sum(p.name.startswith("sphinx-") for p in study.instances)}
    else:
        scan_metadata = None
        sculpt_scale = dict(study.parameters)["sphinx_length_m"] / 10.5
        body_sources = sphinx_integral_parts(study.instances)
        head_sources = sphinx_head_parts(study.instances)
        sculpt_report = {
            "body": _integrate_sphinx_group(
                bpy, scene, model, body_sources, objects, palette["sphinx-limestone"],
                label="body", voxel=max(.034, min(.08, .052*sculpt_scale))),
            "head": _integrate_sphinx_group(
                bpy, scene, model, head_sources, objects, palette["sphinx-head-limestone"],
                label="head", voxel=max(.016, min(.035, .019*sculpt_scale))),
        }

    span = study.bounds_max[0] - study.bounds_min[0]
    depth = study.bounds_max[1] - study.bounds_min[1]
    ground = _material(bpy, get_surface("studio-matte", color="#8D7D68", roughness=0.95))
    _create_box(bpy, stage, "Stage / sand-colored ground",
                (max(50, span*2.7), max(50, depth*2.2), 0.45),
                ((study.bounds_min[0]+study.bounds_max[0])/2,
                 (study.bounds_min[1]+study.bounds_max[1])/2, -0.245), ground)
    world = bpy.data.worlds.new("Kumiki / warm desert daylight")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.34, 0.28, 0.22, 1)
    background.inputs["Strength"].default_value = 0.55
    scene.world = world
    sun = bpy.data.lights.new("Sun / raking limestone relief", type="SUN")
    sun.energy = 2.1
    sun.angle = math.radians(3.0)
    sun_obj = bpy.data.objects.new(sun.name, sun)
    stage.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(31), math.radians(-18), math.radians(-40))
    fill = bpy.data.lights.new("Area / shadow detail", type="AREA")
    fill.energy = 1100
    fill.shape = "DISK"
    fill.size = 16
    fill_obj = bpy.data.objects.new(fill.name, fill)
    stage.objects.link(fill_obj)
    fill_obj.location = (-20, -28, 31)
    aim_at(fill_obj, (0, 0, 8))

    camera_data = bpy.data.cameras.new("Camera / monument inspection")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    stage.objects.link(camera)
    params = dict(study.parameters)
    measured_course_preset = params["pyramid_preset"] == "khufu-petrie"
    fidelity_label = ("historic Petrie course elevations / masonry and surface review pending"
                      if measured_course_preset else
                      "illustrative architecture / artistic review pending")
    metadata = {
        "asset_id": study.asset_id,
        "generator": "kumiki-blender/egyptian-monument-detail-kit",
        "status": ("scan-assisted / anatomical review pending" if sphinx_scan else fidelity_label),
        "provenance": ("Petrie 1883 measured NE course elevations (Ronald Birdsall 2003 transcription); "
                       "approximate pyramid width and block positions, no surveyed stone damage"
                       if measured_course_preset else
                       "Original parametric pyramid/site; external Sphinx scan with separate declared provenance"
                       if sphinx_scan else
                       "Original parametric study; not a surveyed reconstruction of Giza"),
        "recipe_sha256": hashlib.sha256(repr((study.kind, study.parameters)).encode()).hexdigest(),
        "render_profile": {"name": profile.name, "width": profile.width,
                           "height": profile.height, "samples": profile.samples},
        "parameters": {"kind": study.kind, **params},
        "units": "meters",
        "named_parts": dict(study.part_counts),
        "instances": len(study.instances),
        "mesh_data_blocks": len(cache),
        "visible_mesh_objects": len(objects),
        "sphinx_sculpt_union": sculpt_report,
        "sphinx_scan": scan_metadata,
        "bounds_m": [study.bounds_min, study.bounds_max],
    }
    return export_revision(bpy, scene, camera, objects, plan_monument_shots(study),
                           destination, samples=profile.samples, metadata=metadata)
