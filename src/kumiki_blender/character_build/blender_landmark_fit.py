"""Native Blender surface-geodesic landmark fitting on an unrigged detailed part.

Changes only one explicitly identified mesh in a new revision; fully source
locked, bounded by edge strain, face flips, vertex displacement and protected
anchors. No UV remapping, voxel union or nonlocal mesh edits.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, export_interchange_obj, export_registered_obj, write_report  # noqa: E402
from kumiki_blender.character_build.geodesic_fit import LandmarkPin, fit_landmark_surface  # noqa: E402


def _digest_mesh(obj):
    h = hashlib.sha256()
    for row in obj.matrix_world:
        h.update(struct.pack("<4d", *row))
    for vertex in obj.data.vertices:
        h.update(struct.pack("<3d", *vertex.co))
        for group in vertex.groups:
            h.update(struct.pack("<Id", group.group, group.weight))
    for poly in obj.data.polygons:
        h.update(struct.pack("<I", len(poly.vertices)))
        for vid in poly.vertices:
            h.update(struct.pack("<I", vid))
        h.update(struct.pack("<I", poly.material_index))
    for uvmap in obj.data.uv_layers:
        h.update(uvmap.name.encode())
        for loop in uvmap.data:
            h.update(struct.pack("<2d", *loop.uv))
    for mat in obj.data.materials:
        h.update((mat.name if mat else "<none>").encode())
    for modifier in obj.modifiers:
        h.update((modifier.name+":"+modifier.type).encode())
    if obj.data.shape_keys:
        for key in obj.data.shape_keys.key_blocks:
            h.update(key.name.encode())
            for vertex in key.data:
                h.update(struct.pack("<3d", *vertex.co))
    return h.hexdigest()


def _transform(obj, positions):
    return np.asarray([tuple(obj.matrix_world @ v.co) for v in positions], dtype=float)


def main():
    args = arguments()
    plan, output = args["plan"], Path(args["output"]).resolve()
    source = Path(plan["source_blend"]).resolve()
    if output.exists() or output == source.parent or source.is_relative_to(output):
        raise FileExistsError("Fit output revision must be new and outside its source .blend folder")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Open precisely the SHA-pinned original Blender source file")
    source_digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            source_digest.update(chunk)
    if source_digest.hexdigest() != plan["source_sha256"]:
        raise ValueError("Source Blender file changed after the fit plan was created")
    obj = bpy.data.objects.get(plan["mesh"])
    if obj is None or obj.type != "MESH":
        raise ValueError("Could not find explicitly named editable mesh")
    if obj.library or obj.data.library or obj.data.shape_keys:
        raise ValueError("Linked or shape-key mesh is not safe to pre-rig fit")
    if obj.parent is not None and obj.parent.type == "ARMATURE":
        raise ValueError("Pre-rig landmark fitting cannot modify bone-parented meshes")
    if obj.modifiers or obj.vertex_groups:
        raise ValueError("This tool fits unrigged, unskinned geometry only")
    if len(obj.data.vertices) != plan["expected_vertex_count"]:
        raise ValueError("Vertex indices stale: mesh resolution differs from reviewed plan")
    if obj.matrix_world.determinant() <= 1e-10:
        raise ValueError("Refuse mirrored, degenerate or singular object transforms")
    if len(obj.data.vertices) > 300_000 or len(obj.data.polygons) > 500_000:
        raise ValueError("Dense source geometry needs a coarser editable cage")
    before_other = {o.name: _digest_mesh(o) for o in bpy.data.objects
                    if o.type == "MESH" and o is not obj}
    before_target = _digest_mesh(obj)
    before_uvs = [
        (uvmap.name, [tuple(loop.uv) for loop in uvmap.data])
        for uvmap in obj.data.uv_layers]
    source_materials = [
        (mat.name if mat else None) for mat in obj.data.materials]
    source_world = _transform(obj, obj.data.vertices)
    faces = [list(poly.vertices) for poly in obj.data.polygons]
    pins = [LandmarkPin(
        id=pin["id"], vertex=pin["source_vertex"],
        target_world_m=tuple(pin["target_world_m"]),
        radius_m=pin["radius_m"]) for pin in plan["pins"]]
    fitted, measurements = fit_landmark_surface(
        source_world, faces, pins, tuple(plan["protected_vertices"]),
        max_displacement_m=plan["max_displacement_m"],
        max_relative_edge_change=plan["max_relative_edge_change"])
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    inverse = obj.matrix_world.inverted()
    for vertex, position in zip(obj.data.vertices, fitted):
        vertex.co = inverse @ Vector(position)
    obj.data.update()
    bpy.context.view_layer.update()
    after_world = _transform(obj, obj.data.vertices)
    if not np.allclose(after_world, fitted, rtol=0, atol=1e-7):
        raise RuntimeError("Blender roundtrip changed fitted landmark coordinates")
    if [mat.name if mat else None for mat in obj.data.materials] != source_materials:
        raise ValueError("Target material slots changed unexpectedly")
    if [(uvmap.name, [tuple(loop.uv) for loop in uvmap.data]) for uvmap in obj.data.uv_layers] != before_uvs:
        raise ValueError("Target UV corner coordinates changed")
    if {name: _digest_mesh(bpy.data.objects[name]) for name in before_other} != before_other:
        raise RuntimeError("Unrelated original character geometry changed")
    folder = output
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "landmark-fitted.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # Source-view QA uses Blender metres, +Z up and -Y forward. The OBJ
    # operator otherwise defaults to Y-up and silently breaks registration.
    export_interchange_obj(folder / "landmark-fitted.obj")
    export_registered_obj(folder / "landmark-fitted-source.obj")
    write_report(folder, {
        "schema_version": 1,
        "character_id": plan["character_id"],
        "source_blend": str(source), "source_blend_sha256": plan["source_sha256"],
        "landmark_report_sha256": plan["landmark_report_sha256"],
        "fit_spec_sha256": plan["fit_spec_sha256"],
        "mesh": obj.name, "original_mesh_hash": before_target,
        "fitted_mesh_hash": _digest_mesh(obj),
        **measurements,
        "status": "geodesic-landmark-candidate-visual-qa-required",
        "unrelated_meshes_unchanged": True,
        "original_materials_and_uvs_unchanged": True,
        "files": ["landmark-fitted.blend", "landmark-fitted.obj", "landmark-fitted-source.obj"],
        "limitations": [
            "World-space source positions use manually annotated and registered original image landmarks.",
            "Blender is not solving unknown anatomical shape or vertex-to-landmark correspondence.",
            "No source albedo, silhouette, face expression or posed deformation has been artist-approved.",
            "This is a pre-rig cage correction: regenerate skinning/morphs only after reviewing the candidate.",
        ],
    })


main()
