"""Blender worker: non-destructive region-local geometry correction.

Runs on an existing .blend, modifies only the named mesh in memory, and writes
a *different* .blend and an OBJ candidate. No voxel remeshing, UV loss, or
automatic changes to unrelated parts. Sculpt outcomes need independent QA.
"""
from __future__ import annotations

import hashlib
import math
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402


def mesh_digest(obj) -> str:
    sha = hashlib.sha256()
    for v in obj.data.vertices:
        sha.update(struct.pack("<3d", *v.co))
    for polygon in obj.data.polygons:
        sha.update(struct.pack("<I", len(polygon.vertices)))
        for vertex_index in polygon.vertices:
            sha.update(struct.pack("<I", vertex_index))
    return sha.hexdigest()


def main() -> None:
    args = arguments()
    plan, out = args["plan"], Path(args["output"]).resolve()
    if out.exists():
        raise FileExistsError(f"Output revision already exists: {out}")
    if not bpy.data.filepath:
        raise ValueError("refine-region must load a source .blend, not factory startup")
    if Path(bpy.data.filepath).resolve().is_relative_to(out):
        raise ValueError("Source .blend cannot be inside the output revision")
    obj = bpy.data.objects.get(plan["object"])
    if obj is None or obj.type != "MESH":
        raise ValueError(f"Expected named Blender mesh object: {plan['object']}")
    if obj.data.shape_keys is not None:
        raise ValueError("Vertex-level refinement on shape-key meshes requires a dedicated shape-key-aware workflow")
    if obj.library is not None or obj.data.library is not None:
        raise ValueError("Refuse to modify linked library mesh data")
    if obj.data.users > 1:
        # Avoid changing all objects that share the same mesh datablock.
        obj.data = obj.data.copy()
    if plan.get("expected_vertex_count") is not None and len(obj.data.vertices) != plan["expected_vertex_count"]:
        raise ValueError("Mesh changed since the refinement plan was authored")
    before_other = {o.name: mesh_digest(o) for o in bpy.data.objects
                    if o.type == "MESH" and o != obj}
    before_mesh = mesh_digest(obj)
    before_normals = [polygon.normal.copy() for polygon in obj.data.polygons]
    transform = obj.matrix_world
    inverse_basis = transform.inverted_safe().to_3x3()
    initial = [v.co.copy() for v in obj.data.vertices]
    changed = set()
    for operation in plan["operations"]:
        center = Vector(operation["center_m"])
        delta = Vector(operation["move_m"])
        radius = float(operation["radius_m"])
        for vertex in obj.data.vertices:
            position_world = transform @ vertex.co
            distance = (position_world - center).length
            if distance >= radius:
                continue
            alpha = max(0.0, 1 - distance / radius)
            if operation["falloff"] == "smoothstep":
                alpha = alpha * alpha * (3 - 2 * alpha)
            vertex.co += inverse_basis @ (delta * alpha)
            if alpha > 0:
                changed.add(vertex.index)
    if not changed:
        raise ValueError("No vertex fell inside the requested sculpt radius")
    obj.data.update()
    # Guard against collapsing or flipping faces. This checks local geometry, not
    # proof of correct silhouette or reference appearance.
    flipped = 0
    for original, polygon in zip(before_normals, obj.data.polygons):
        if original.length > 1e-8 and polygon.normal.length > 1e-8 and original.dot(polygon.normal) <= 0:
            flipped += 1
    if flipped:
        raise ValueError(f"Sculpt rejected: {flipped} faces flipped orientation")
    displacement = [((transform @ vertex.co) - (transform @ initial[vertex.index])).length
                    for vertex in obj.data.vertices]
    maximum = max(displacement)
    if maximum > plan.get("max_displacement_m", .05) + 1e-6:
        raise ValueError(f"Displacement {maximum:.6f} exceeds the plan budget")
    untouched = [i for i in range(len(displacement)) if i not in changed]
    if any(displacement[i] > 1e-8 for i in untouched):
        raise ValueError("Out-of-region vertices changed unexpectedly")
    after_other = {o.name: mesh_digest(o) for o in bpy.data.objects
                   if o.type == "MESH" and o != obj}
    if before_other != after_other:
        raise ValueError("A non-target mesh changed during region refinement")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "refined.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(filepath=str(out / "region.obj"), export_selected_objects=True,
                          export_materials=False)
    write_report(out, {
        "schema_version": 1, "status": "geometric-candidate-art-review-required",
        "source_blend": str(Path(args["source_blend"]).resolve()),
        "object": obj.name,
        "source_mesh_sha256": before_mesh,
        "refined_mesh_sha256": mesh_digest(obj),
        "untouched_mesh_sha256": before_other,
        "edited_vertices": len(changed),
        "untouched_vertices": len(untouched),
        "total_vertices": len(obj.data.vertices),
        "maximum_displacement_m": round(maximum, 7),
        "flipped_faces": flipped,
        "files": ["refined.blend", "region.obj"],
        "limitations": [
            "This is a bounded positional sculpt adjustment, not semantic anatomy synthesis.",
            "Evaluate full-frame verified source masks and animated deformation before accepting the candidate.",
            "Source .blend is never saved; the generated refined.blend is an independent revision."
        ],
    })


main()
