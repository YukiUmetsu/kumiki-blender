"""Blender worker: place parts on the body, join the union parts, measure the seams.

Arguments (JSON): plan (from ``assembly.load_assembly``), output.
For each part the report gives the distance from its vertices to the body
surface and the fraction of its vertices inside the body. A separate part
(glasses, a button) should touch the body: a small minimum distance and few
vertices inside. Union parts are merged by voxel remesh, so their seams close.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Matrix
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import (activate, arguments, clear_startup_scene,  # noqa: E402
                                                           import_mesh, mesh_stats, write_report)


def seam_metrics(part, body_tree) -> dict:
    distances, inside = [], 0
    for vertex in part.data.vertices:
        point = part.matrix_world @ vertex.co
        location, normal, _index, distance = body_tree.find_nearest(point)
        if location is None:
            continue
        distances.append(distance)
        inside += (point - location).dot(normal) < 0
    distances.sort()
    count = max(1, len(distances))
    return {"min_distance_m": round(distances[0], 6) if distances else None,
            "p05_distance_m": round(distances[int(0.05 * (count - 1))], 6) if distances else None,
            "inside_fraction": round(inside / count, 4)}


def main() -> None:
    args = arguments()
    plan, output = args["plan"], Path(args["output"]).resolve()
    clear_startup_scene()
    body = import_mesh(plan["body"], "Body")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(body, depsgraph)
    parts, union = [], []
    for part in plan["parts"]:
        obj = import_mesh(part["file"], f"Part_{part['id']}")
        obj.matrix_world = Matrix(part["matrix"]) @ obj.matrix_world
        activate(obj)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        metrics = seam_metrics(obj, tree)
        parts.append({"id": part["id"], "join": part["join"], "anchor_rms_m": round(part["rms_m"], 6),
                      "scale": round(part["scale"], 6), **metrics, **mesh_stats(obj)})
        if part["join"] == "union":
            union.append(obj)
    union_names = [o.name for o in union]  # join() deletes the joined objects
    if union:
        activate(body)
        for obj in union:
            obj.select_set(True)
        bpy.ops.object.join()
        remesh = body.modifiers.new("Join", "REMESH")
        remesh.mode, remesh.voxel_size = "VOXEL", plan["join_voxel_m"]
        activate(body)
        bpy.ops.object.modifier_apply(modifier=remesh.name)
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "assembled.blend"))
    bpy.ops.export_scene.gltf(filepath=str(output / "assembled.glb"), export_format="GLB")
    activate(body)
    bpy.ops.wm.obj_export(filepath=str(output / "assembled_body.obj"), export_selected_objects=True,
                          export_materials=False)
    write_report(output, {"body": mesh_stats(body), "parts": parts, "union_parts": union_names,
                          "files": ["assembled.blend", "assembled.glb", "assembled_body.obj"]})


main()
