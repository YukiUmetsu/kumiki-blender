"""Blender 4.5 pre-rig hinge-loop densification of existing Quad cage meshes.

BMesh edge subdivision works only where reviewed joint planes cross quad
strips. It must not touch a rigged/shape-key mesh, UV ownership, or source
files. This is a local edge-density candidate, *not* semantic retopology
certification or an automatic skinning fix.
"""
from __future__ import annotations

import hashlib
import math
import struct
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402


def _hash(obj):
    h = hashlib.sha256()
    for vertex in obj.data.vertices:
        h.update(struct.pack("<3d", *vertex.co))
    for face in obj.data.polygons:
        for vi in face.vertices:
            h.update(struct.pack("<I", vi))
    for loop_uv in obj.data.uv_layers:
        h.update(loop_uv.name.encode())
        for loop in loop_uv.data:
            h.update(struct.pack("<2d", *loop.uv))
    for mat in obj.data.materials:
        h.update((mat.name if mat else "missing").encode())
    return h.hexdigest()


def _stats(bm):
    return {
        "vertices": len(bm.verts),
        "edges": len(bm.edges),
        "faces": len(bm.faces),
        "quads": sum(len(face.verts) == 4 for face in bm.faces),
        "boundary_edges": sum(edge.is_boundary for edge in bm.edges),
        "nonmanifold_edges": sum(not edge.is_manifold and not edge.is_boundary
                                 for edge in bm.edges),
        "signed_volume_m3": bm.calc_volume(signed=True),
    }


def _choose(bm, obj, zone, threshold):
    center = Vector(zone["center_m"])
    axis = Vector(zone["axis"]).normalized()
    radius = zone["search_radius_m"]
    # A registered hinge point must be inside the axial span of the chosen
    # edge. The edge itself must travel mostly along the joint bone.
    found = []
    for edge in bm.edges:
        if not edge.is_manifold or any(len(face.verts) != 4 for face in edge.link_faces):
            continue
        a, b = (obj.matrix_world @ vertex.co for vertex in edge.verts)
        da, db = (a-center).dot(axis), (b-center).dot(axis)
        if not min(da, db) < 0 < max(da, db):
            continue
        midpoint = (a + b) * .5
        radial = midpoint - center - axis * ((midpoint-center).dot(axis))
        if radial.length > radius:
            continue
        travel = b - a
        if travel.length < 1e-9 or abs(travel.normalized().dot(axis)) < threshold:
            continue
        found.append(edge)
    return found


def main():
    args = arguments()
    plan, topo = args["plan"], args["topology"]
    output = Path(args["output"]).resolve()
    source = Path(plan["source_blend"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output revision already exists: {output}")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Load precisely the source Blender revision from the plan")
    obj = bpy.data.objects.get(plan["mesh"])
    if obj is None or obj.type != "MESH":
        raise ValueError(f"Retopology target {plan['mesh']} must be a mesh object")
    if obj.library or obj.data.library or obj.data.shape_keys is not None:
        raise ValueError("Linked or shape-key meshes cannot be topology-edited")
    if obj.modifiers or len(obj.vertex_groups):
        raise ValueError("Refine hinge topology before skin weights and deformation modifiers")
    other_before = {o.name: _hash(o) for o in bpy.data.objects
                    if o.type == "MESH" and o is not obj}
    source_materials = [mat.name if mat else None for mat in obj.data.materials]
    source_uv_names = [uv.name for uv in obj.data.uv_layers]
    source_positions = {
        tuple(round(float(x), 8) for x in vertex.co)
        for vertex in obj.data.vertices
    }
    zones_by_id = {z["id"]: z for z in topo["zones"]}
    if not set(plan["zones"]) <= set(zones_by_id):
        raise ValueError("Selected joint is not in the source-verified topology plan")
    if len(obj.data.vertices) > 300_000 or len(obj.data.polygons) > 500_000:
        raise ValueError("Only pre-rig retopology cage meshes are supported")

    # Copy the mesh datablock if other objects in the scene reference it.
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        before = _stats(bm)
        if before["nonmanifold_edges"] or before["boundary_edges"] or before["quads"] < .65 * before["faces"]:
            raise ValueError("Requires a closed, mostly quad-based, manifold pre-rig cage")
        if before["signed_volume_m3"] <= 0:
            raise ValueError("Input cage must have consistent outward normals")
        bands = []
        for id in plan["zones"]:
            zone = zones_by_id[id]
            targets = _choose(bm, obj, zone, plan["axial_alignment_min"])
            if len(targets) < 4:
                raise ValueError(f"{id}: fewer than 4 genuine axial quad edges cross the reviewed hinge")
            size_prior = len(bm.verts)
            bmesh.ops.subdivide_edges(
                bm, edges=targets, cuts=plan["cuts"], use_grid_fill=True,
                smooth=0.0, quad_corner_type="STRAIGHT_CUT")
            bm.verts.index_update()
            bm.edges.index_update()
            added = len(bm.verts) - size_prior
            if added <= 0:
                raise RuntimeError(f"{id}: Blender did not create additional deformation geometry")
            if len(bm.verts) - before["vertices"] > plan["max_added_vertices"]:
                raise ValueError("Joint-local topology exceeded the new vertex budget")
            bands.append({"joint": id, "crossing_axial_edges": len(targets),
                          "new_vertices": added, "cuts_per_edge": plan["cuts"]})
        after = _stats(bm)
        if (after["boundary_edges"] != before["boundary_edges"] or
                after["nonmanifold_edges"] != before["nonmanifold_edges"]):
            raise ValueError("Local retopology introduced open or nonmanifold edges")
        volume_ratio = after["signed_volume_m3"] / max(before["signed_volume_m3"], 1e-9)
        if not .98 <= volume_ratio <= 1.02:
            raise ValueError(f"Local topology changed the sculpt volume by {volume_ratio:.4f}")
        bm.to_mesh(obj.data)
        obj.data.update()
    finally:
        bm.free()
    if [mat.name if mat else None for mat in obj.data.materials] != source_materials:
        raise ValueError("Original target material assignments changed")
    if [uv.name for uv in obj.data.uv_layers] != source_uv_names:
        raise ValueError("Original UV map names changed")
    # Existing source vertices must still exist after local linear subdivisions.
    created_positions = {tuple(round(float(x), 8) for x in vertex.co)
                         for vertex in obj.data.vertices}
    if not source_positions <= created_positions:
        raise ValueError("Local topology moved or removed an original approved vertex")
    if {name: _hash(bpy.data.objects[name]) for name in other_before} != other_before:
        raise ValueError("A non-target character mesh changed unexpectedly")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "hinge-retopo.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(filepath=str(output / "hinge-retopo.obj"),
                          export_selected_objects=True, export_materials=False)
    write_report(output, {
        "schema_version": 1,
        "status": "local-hinge-topology-candidate-review-required",
        "source_blend": str(source),
        "source_mesh": obj.name,
        "reviewed_joint_bands": bands,
        "before": before, "after": after,
        "new_vertices": after["vertices"] - before["vertices"],
        "volume_ratio": round(volume_ratio, 7),
        "original_source_vertices_preserved": True,
        "original_materials_and_uv_names_preserved": True,
        "unrelated_character_meshes_preserved": True,
        "files": ["hinge-retopo.blend", "hinge-retopo.obj"],
        "limitations": [
            "Added axial contour loops are a controlled pre-rig density aid, not proven semantic edge-flow.",
            "Existing shape keys, weights and modifiers are intentionally unsupported; regenerate them after topology approval.",
            "Source surface normals, UV interpolation and deformed mesh silhouette still need independent visual QA.",
            "This operation never edits source .blend and does not reconstruct unseen reference geometry.",
        ],
    })


main()
