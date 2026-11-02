"""Blender-only preparation of a true MakeHuman head/neck sculpt base."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .blender_output import aim_at, create_model_scene, export_revision
from .blender_scene import _material
from .head_sources.makehuman import (MakeHumanSourceError, MeshSource, SOURCE_COMMIT,
                                     load_body_mesh)
from .render_profiles import select_render_profile
from .shots import QualityShot
from .surfaces import get_surface


# MakeHuman's base.obj uses decimeters, Y-up and +Z toward the face.
# The output is meters, Blender Z-up and -Y toward the face.
NECK_CUT_SOURCE_Y = 6.05
HEAD_VERTICAL_ORIGIN = 7.15
SOURCE_DECIMETERS_TO_METERS = 0.1


def _extract_closed_head(bpy: Any, source: MeshSource) -> tuple[Any, dict[str, Any]]:
    import bmesh  # type: ignore[import-not-found]

    source_vertices, faces = load_body_mesh(source)
    if len(source_vertices) < 10000 or len(faces) < 10000:
        raise MakeHumanSourceError("The official body source has unexpectedly little real topology")
    converted = [(x * SOURCE_DECIMETERS_TO_METERS, -z * SOURCE_DECIMETERS_TO_METERS,
                  (y - HEAD_VERTICAL_ORIGIN) * SOURCE_DECIMETERS_TO_METERS)
                 for x, y, z in source_vertices]
    mesh = bpy.data.meshes.new("MakeHuman / original CC0 facial topology")
    mesh.from_pydata(converted, [], faces)
    mesh.update()
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        cut_z = (NECK_CUT_SOURCE_Y - HEAD_VERTICAL_ORIGIN) * SOURCE_DECIMETERS_TO_METERS
        bmesh.ops.bisect_plane(bm, geom=[*bm.verts, *bm.edges, *bm.faces], dist=0.000001,
                               plane_co=(0, 0, cut_z), plane_no=(0, 0, 1),
                               clear_inner=True, clear_outer=False)
        bm.verts.ensure_lookup_table()
        if not bm.verts or any(v.co.z < cut_z - 0.00001 for v in bm.verts):
            raise MakeHumanSourceError("Anatomical neck cut kept invalid torso geometry")
        neck_boundary = [edge for edge in bm.edges if edge.is_boundary and
                         all(abs(vertex.co.z - cut_z) < 0.00001 for vertex in edge.verts)]
        if len(neck_boundary) < 8:
            raise MakeHumanSourceError("No continuous neck boundary found to cap")
        filled = bmesh.ops.holes_fill(bm, edges=neck_boundary, sides=0)
        if not filled["faces"]:
            raise MakeHumanSourceError("The anatomical neck could not be capped")
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        boundary_edges = sum(edge.is_boundary for edge in bm.edges)
        nonmanifold_edges = sum(not edge.is_manifold for edge in bm.edges)
        if boundary_edges or nonmanifold_edges:
            raise MakeHumanSourceError(
                f"Prepared head is not closed: {boundary_edges} boundary and "
                f"{nonmanifold_edges} nonmanifold edges")
        bm.to_mesh(mesh)
        mesh.update()
        diagnostic = {"vertices": len(mesh.vertices), "polygons": len(mesh.polygons),
                      "neck_cut_source_y_dm": NECK_CUT_SOURCE_Y,
                      "neck_cap_faces": len(filled["faces"]),
                      "boundary_edges": boundary_edges, "nonmanifold_edges": nonmanifold_edges}
    finally:
        bm.free()
    obj = bpy.data.objects.new("Head and neck / editable original surface", mesh)
    obj["kumiki_part"] = "anatomical_head"
    obj["source_asset"] = "makehuman-neutral-head"
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj, diagnostic


def build_makehuman_head(source: MeshSource, output: str | Path, *, bpy_module: Any) -> dict[str, Any]:
    """Build an immutable, source-editable head revision with front/profile QA."""
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError("Blender 4.2 or newer is required")
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise MakeHumanSourceError(f"Output revision already exists: {output}")
    # Verify immediately before constructing geometry, including in-process callers.
    from .head_sources.makehuman import verify_makehuman_obj
    source = verify_makehuman_obj(source.path)
    head, geometry = _extract_closed_head(bpy, source)
    profile = select_render_profile(1280, 960, 32)
    scene = create_model_scene(bpy, "makehuman-neutral-head")
    model_collection = bpy.data.collections.new("Anatomy / CC0 human source")
    scene.collection.children.link(model_collection)
    model_collection.objects.link(head)
    head.data.materials.append(_material(bpy, get_surface("studio-matte", color="#bcb8ae", roughness=0.84)))

    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile.samples
    scene.render.resolution_x = profile.width
    scene.render.resolution_y = profile.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.render.film_transparent = False
    studio = bpy.data.collections.new("Studio / excluded from portable GLB")
    scene.collection.children.link(studio)
    world = bpy.data.worlds.new("MakeHuman / portrait studio")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.25, 0.27, 0.31, 1.0)
    background.inputs["Strength"].default_value = 0.45
    scene.world = world

    area_data = bpy.data.lights.new("Portrait / large softbox", type="AREA")
    area_data.energy = 130
    area_data.shape = "DISK"
    area_data.size = 0.55
    area = bpy.data.objects.new(area_data.name, area_data)
    studio.objects.link(area)
    area.location = (-0.25, -0.35, 0.35)
    aim_at(area, (0, 0, 0))
    fill_data = bpy.data.lights.new("Portrait / fill", type="AREA")
    fill_data.energy = 90
    fill_data.size = 0.45
    fill = bpy.data.objects.new(fill_data.name, fill_data)
    studio.objects.link(fill)
    fill.location = (0.24, 0.16, 0.15)
    aim_at(fill, (0, 0, 0))
    camera_data = bpy.data.cameras.new("Camera / head anatomy")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    studio.objects.link(camera)

    corners = [vertex.co for vertex in head.data.vertices]
    bounds_min = tuple(min(point[i] for point in corners) for i in range(3))
    bounds_max = tuple(max(point[i] for point in corners) for i in range(3))
    width, depth, height = (bounds_max[i] - bounds_min[i] for i in range(3))
    if not (0.15 < width < 0.23 and 0.18 < height < 0.28 and 0.13 < depth < 0.27):
        raise MakeHumanSourceError(f"Implausible human-head metric bounds: {bounds_min} .. {bounds_max}")
    cx, cy, cz = ((bounds_min[i] + bounds_max[i]) / 2 for i in range(3))
    shots = (
        QualityShot("hero", (cx + 0.24, cy - 0.36, cz + 0.08), (cx, cy, cz), 0.44),
        QualityShot("front", (cx, cy - 0.45, cz), (cx, cy, cz), 0.43),
        QualityShot("profile", (cx + 0.45, cy, cz), (cx, cy, cz), 0.43),
        QualityShot("three-quarter", (cx - 0.26, cy - 0.35, cz + 0.03), (cx, cy, cz), 0.43),
    )
    metadata = {
        "asset_id": "makehuman-neutral-head",
        "generator": "kumiki-blender/verified-makehuman-cc0-head",
        "status": "anatomical-base / artistic-review-required / no-statue-likeness-claim",
        "provenance": {"source_url": source.upstream_url,
                       "source_sha256": source.sha256, "license": source.license_id,
                       "upstream_commit": SOURCE_COMMIT,
                       "modifications": "body-only extraction; neck clip and cap; dm Y-up to m Z-up"},
        "mesh_diagnostic": geometry,
        "units": "meters", "native_axes": "Blender Z-up; face towards -Y",
        "bounds_m": [bounds_min, bounds_max],
        "named_parts": {"anatomical_head": 1},
        "render_profile": {"name": profile.name, "width": profile.width,
                           "height": profile.height, "samples": profile.samples},
    }
    result = export_revision(bpy, scene, camera, (head,), shots, output,
                             samples=profile.samples, metadata=metadata)
    # The landmark-fit importer intentionally requires a separate rights record.
    # Supply one with the *exported GLB* digest, not the upstream OBJ digest.
    exported_hash = hashlib.sha256((output / "model.glb").read_bytes()).hexdigest()
    handoff = {
        "schema_version": 1, "asset_id": "makehuman-neutral-head",
        "creator": "MakeHuman Community",
        "source_url": source.upstream_url,
        "license": "CC0-1.0",
        "attribution": "Official MakeHuman core graphical base mesh; prepared by Kumiki Blender",
        "redistribution_allowed": True,
        "commercial_use_allowed": True,
        "expected_sha256": exported_hash,
    }
    (output / "head-provenance.json").write_text(json.dumps(handoff, indent=2) + "\n",
                                                encoding="utf-8")
    result["files"]["head_provenance"] = "head-provenance.json"
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n",
                                          encoding="utf-8")
    return result
