"""Blender worker: compare a rig at rest against controlled joint poses.

Reads an existing .blend; does NOT save or overwrite it. Structural failures
are flagged. Volume is only measured for closed manifolds, not for open cloth.
Contact checks are advisory surface intersections, not a physics simulation.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402


def skin_weights(mesh) -> dict:
    count, missing, excessive, bad_sum = 0, 0, 0, 0
    for vertex in mesh.data.vertices:
        influences = [g.weight for g in vertex.groups if g.weight > 1e-4]
        count += 1
        missing += not influences
        excessive += len(influences) > 4
        bad_sum += bool(influences and abs(sum(influences) - 1.0) > .02)
    return {"vertices": count, "unweighted": missing,
            "over_four_influences": excessive, "non_normalized": bad_sum}


def snapshot(obj, depsgraph) -> dict:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        coords = [tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices]
        edges = [tuple(edge.vertices) for edge in mesh.edges]
        faces = [tuple(face.vertices) for face in mesh.polygons]
        areas = [face.area for face in mesh.polygons]
        bm = bmesh.new()
        bm.from_mesh(mesh)
        closed = bool(bm.faces) and all(edge.is_manifold for edge in bm.edges)
        volume = abs(bm.calc_volume()) if closed else None
        bm.free()
        return {"coords": coords, "edges": edges, "faces": faces,
                "areas": areas, "volume_m3": volume, "closed": closed}
    finally:
        evaluated.to_mesh_clear()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    samples = sorted(values)
    return round(samples[min(len(samples) - 1, math.ceil(fraction * len(samples)) - 1)], 6)


def deformation(rest: dict, current: dict) -> dict:
    if (len(rest["coords"]) != len(current["coords"]) or
        rest["faces"] != current["faces"]):
        return {"topology_changed": True, "status": "fail"}
    rest_points, points = rest["coords"], current["coords"]
    strains, degenerate = [], 0
    for a, b in rest["edges"]:
        original = math.dist(rest_points[a], rest_points[b])
        posed = math.dist(points[a], points[b])
        if original <= 1e-9:
            degenerate += 1
        else:
            strains.append(abs(posed / original - 1))
    surface_ratios = [abs(a / b - 1)
                      for a, b in zip(current["areas"], rest["areas"]) if b > 1e-12]
    rest_volume, pose_volume = rest["volume_m3"], current["volume_m3"]
    return {
        "topology_changed": False,
        "degenerate_rest_edges": degenerate,
        "p95_relative_edge_stretch": percentile(strains, .95),
        "max_relative_edge_stretch": percentile(strains, 1),
        "p95_relative_face_area_change": percentile(surface_ratios, .95),
        "volume_change_fraction": round((pose_volume / rest_volume) - 1, 6)
            if rest_volume is not None and pose_volume is not None and rest_volume > 1e-12 else None,
        "closed_manifold": rest["closed"] and current["closed"],
    }


def main() -> None:
    args = arguments()
    output = Path(args["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"Revision exists: {output}")
    plan = args["plan"]
    rig = bpy.data.objects.get(plan["armature"])
    if rig is None or rig.type != "ARMATURE":
        raise ValueError(f"Armature object not found: {plan['armature']}")
    bound = [obj for obj in bpy.data.objects if obj.type == "MESH" and any(
        mod.type == "ARMATURE" and mod.object == rig for mod in obj.modifiers)]
    if not bound:
        raise ValueError("No mesh is skinned to the specified armature")
    for pose in plan["poses"]:
        for bone in pose["bones"]:
            if bone not in rig.pose.bones:
                raise ValueError(f"Pose {pose['name']}: missing bone {bone}")
    for first, second in plan.get("collision_pairs", []):
        if any(bpy.data.objects.get(name) is None or bpy.data.objects[name].type != "MESH"
               for name in (first, second)):
            raise ValueError(f"Collision pair meshes missing: {first}, {second}")
    if plan.get("render") and bpy.context.scene.camera is None:
        raise ValueError("Pose renders require a camera in the .blend")

    # Ensure the actual saved neutral pose is the baseline; never modify the
    # source .blend on disk. Restore local in-memory pose at the end.
    original = {}
    for bone in rig.pose.bones:
        original[bone.name] = (bone.rotation_mode, bone.rotation_euler.copy(),
                               bone.rotation_quaternion.copy(), bone.rotation_axis_angle[:],
                               bone.location.copy(), bone.scale.copy())
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (0, 0, 0)
        bone.location = (0, 0, 0)
        bone.scale = (1, 1, 1)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rest = {o.name: snapshot(o, depsgraph) for o in bound}
    weights = {o.name: skin_weights(o) for o in bound}
    problems: list[str] = []
    for name, result in weights.items():
        if any(result[key] for key in ("unweighted", "over_four_influences", "non_normalized")):
            problems.append(f"{name}: incomplete/invalid deform weights")
    poses = []
    try:
        for item in plan["poses"]:
            for bone in rig.pose.bones:
                bone.rotation_euler = (0, 0, 0)
            for bone_name, rotation in item["bones"].items():
                vector = [0.0, 0.0, 0.0]
                vector["XYZ".index(rotation["axis"])] = math.radians(rotation["degrees"])
                rig.pose.bones[bone_name].rotation_euler = vector
            bpy.context.view_layer.update()
            entry = {"name": item["name"], "meshes": {}, "collision_candidates": []}
            for obj in bound:
                metrics = deformation(rest[obj.name], snapshot(obj, depsgraph))
                entry["meshes"][obj.name] = metrics
                if metrics.get("topology_changed"):
                    problems.append(f"{item['name']}/{obj.name}: topology changed during deformation")
                limit = plan.get("max_p95_edge_stretch")
                if limit is not None and metrics.get("p95_relative_edge_stretch") is not None and (
                    metrics["p95_relative_edge_stretch"] > limit):
                    problems.append(f"{item['name']}/{obj.name}: edge stretch exceeded {limit}")
            for first, second in plan.get("collision_pairs", []):
                a, b = bpy.data.objects[first], bpy.data.objects[second]
                tree_a = BVHTree.FromObject(a, depsgraph)
                tree_b = BVHTree.FromObject(b, depsgraph)
                overlap = len(tree_a.overlap(tree_b))
                if overlap:
                    entry["collision_candidates"].append({
                        "objects": [first, second], "overlapping_triangle_pairs": overlap,
                        "status": "requires-contact-review"})
            if plan.get("render"):
                output.mkdir(parents=True, exist_ok=True)
                bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
                bpy.context.scene.render.resolution_x = 480
                bpy.context.scene.render.resolution_y = 480
                bpy.context.scene.render.resolution_percentage = 100
                bpy.context.scene.render.filepath = str(output / f"{item['name']}.png")
                bpy.ops.render.render(write_still=True)
                entry["image"] = f"{item['name']}.png"
            poses.append(entry)
    finally:
        for bone in rig.pose.bones:
            mode, euler, quaternion, axis_angle, location, scale = original[bone.name]
            bone.rotation_mode = mode
            bone.rotation_euler = euler
            bone.rotation_quaternion = quaternion
            bone.rotation_axis_angle = axis_angle
            bone.location = location
            bone.scale = scale
        bpy.context.view_layer.update()

    result = {
        "schema_version": 1,
        "status": "structural-fail" if problems else "deformation-review-required",
        "armature": rig.name,
        "bound_meshes": sorted(o.name for o in bound),
        "rest_skin_weights": weights,
        "poses": poses,
        "problems": problems,
        "limitations": [
            "Surface triangle intersection counts are advisory, not penetration volumes.",
            "Closed-manifold volume changes are diagnostic, not sufficient to certify deformation.",
            "Edge stretch and area change do not certify aesthetic quality or joint motion.",
            "A neutral pose is reconstructed by zeroing controller pose transforms; verify the intended rest state.",
            "No source .blend is written or overwritten.",
        ],
    }
    write_report(output, result)


main()
