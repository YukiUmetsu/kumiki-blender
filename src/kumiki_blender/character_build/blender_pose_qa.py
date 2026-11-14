"""Blender worker: compare a rig at rest against controlled joint poses.

Reads an existing .blend; does NOT save or overwrite it. Structural failures
are flagged. Volume is only measured for closed manifolds, not for open cloth.
Contact checks are advisory surface intersections, not a physics simulation.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402
from kumiki_blender.character_build.deformation_metrics import (  # noqa: E402
    collision_delta,
    pose_motion,
)
from kumiki_blender.character_build.weight_zone_metrics import audit_weight_zone  # noqa: E402


def skin_weights(mesh, deform_bones: set[str]) -> dict:
    count, missing, excessive, bad_sum = 0, 0, 0, 0
    for vertex in mesh.data.vertices:
        influences = [g.weight for g in vertex.groups
                      if g.weight > 1e-4 and mesh.vertex_groups[g.group].name in deform_bones]
        count += 1
        missing += not influences
        excessive += len(influences) > 4
        bad_sum += bool(influences and abs(sum(influences) - 1.0) > .02)
    return {"vertices": count, "unweighted": missing,
            "over_four_influences": excessive, "non_normalized": bad_sum}


def skin_weight_zone(mesh, zone: dict, deform_bones: set[str]) -> dict:
    """Audit explicit semantic part regions before playing the pose battery."""
    group = mesh.vertex_groups.get(zone["region_group"])
    if group is None:
        raise ValueError(f"{mesh.name}: missing semantic vertex group {zone['region_group']}")
    allowed = set(zone["allowed_bones"])
    if not allowed <= deform_bones:
        raise ValueError(f"{mesh.name}/{zone['region_group']}: allowed_bones contain "
                         f"missing or non-deforming bones: {sorted(allowed - deform_bones)}")
    if zone["region_group"] in deform_bones:
        raise ValueError("Semantic zone must use a separate non-deforming vertex group")

    def samples():
        for vertex in mesh.data.vertices:
            member = 0.0
            influences = {}
            for entry in vertex.groups:
                bone_name = mesh.vertex_groups[entry.group].name
                if bone_name == zone["region_group"]:
                    member = entry.weight
                if bone_name in deform_bones and entry.weight > 0:
                    influences[bone_name] = entry.weight
            yield vertex.index, member, influences

    return audit_weight_zone(
        samples(), allowed,
        max_outside_weight=zone["max_outside_weight"],
        min_vertices=zone["min_vertices"])


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



def world_bvh(obj, depsgraph):
    """Build collision geometry in WORLD coordinates, not Blender object-local BVH space."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        vertices = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        polygons = [tuple(poly.vertices) for poly in mesh.polygons]
        # Polygon connectivity is part of the collision identity. BVH overlaps
        # cannot be compared by triangle index if a modifier retopologized the mesh.
        return BVHTree.FromPolygons(vertices, polygons), (len(vertices), tuple(polygons))
    finally:
        evaluated.to_mesh_clear()


def main() -> None:
    args = arguments()
    output = Path(args["output"]).resolve()
    # A pose report without the exact saved source .blend SHA cannot be
    # trusted by downstream M0 benchmarks after the rig has been edited.
    if not bpy.data.filepath:
        raise ValueError("Pose QA requires an existing, saved source .blend")
    source_blend = Path(bpy.data.filepath).resolve()
    if not source_blend.is_file() or source_blend.suffix.lower() != ".blend":
        raise ValueError("Pose QA source .blend is missing or unsaved")
    def original_hash():
        digest = hashlib.sha256()
        with source_blend.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    source_sha = original_hash()
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
    collision_baselines = {}
    for first, second in plan.get("collision_pairs", []):
        a, topology_a = world_bvh(bpy.data.objects[first], depsgraph)
        b, topology_b = world_bvh(bpy.data.objects[second], depsgraph)
        collision_baselines[(first, second)] = (
            set(a.overlap(b)), (topology_a, topology_b))
    deform_bones = {bone.name for bone in rig.data.bones if bone.use_deform}
    weights = {o.name: skin_weights(o, deform_bones) for o in bound}
    problems: list[str] = []
    for name, result in weights.items():
        if any(result[key] for key in ("unweighted", "over_four_influences", "non_normalized")):
            problems.append(f"{name}: incomplete/invalid deform weights")
    named_meshes = {obj.name: obj for obj in bound}
    weight_zones = []
    for zone in plan.get("weight_zones", []):
        mesh = named_meshes.get(zone["mesh"])
        if mesh is None:
            raise ValueError(f"Weight zone mesh {zone['mesh']} is not bound to the armature")
        result = skin_weight_zone(mesh, zone, deform_bones)
        weight_zones.append({
            "mesh": mesh.name, "region_group": zone["region_group"],
            "allowed_bones": zone["allowed_bones"], **result,
        })
        if result["status"] != "pass":
            problems.append(
                f"{mesh.name}/{zone['region_group']}: {result['violating_vertices']} "
                f"vertices exceed outside-weight limit and "
                f"{result['selected_vertices']} belong to the marked region "
                f"(minimum {result['minimum_required_vertices']})")
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
            entry = {"name": item["name"], "meshes": {},
                     "collision_candidates": [], "contact_deltas": []}
            for obj in bound:
                current = snapshot(obj, depsgraph)
                try:
                    motion = pose_motion(rest[obj.name]["coords"], current["coords"])
                except ValueError:
                    motion = {"finite": False, "max_vertex_displacement_m": None,
                              "p95_vertex_displacement_m": None, "moving_vertices": None}
                if not motion["finite"]:
                    metrics = {"status": "fail", "nonfinite_or_changed_vertices": True,
                               "topology_changed": len(rest[obj.name]["coords"]) != len(current["coords"])}
                    problems.append(f"{item['name']}/{obj.name}: invalid evaluated vertex coordinates")
                else:
                    metrics = deformation(rest[obj.name], current)
                metrics["motion"] = motion
                entry["meshes"][obj.name] = metrics
                if metrics.get("topology_changed"):
                    problems.append(f"{item['name']}/{obj.name}: topology changed during deformation")
                limit = plan.get("max_p95_edge_stretch")
                if limit is not None and metrics.get("p95_relative_edge_stretch") is not None and (
                    metrics["p95_relative_edge_stretch"] > limit):
                    problems.append(f"{item['name']}/{obj.name}: edge stretch exceeded {limit}")
            entry["max_vertex_displacement_m"] = max(
                (metrics["motion"]["max_vertex_displacement_m"] or 0)
                for metrics in entry["meshes"].values())
            min_motion = plan.get("min_pose_displacement_m")
            if min_motion is not None and entry["max_vertex_displacement_m"] < min_motion:
                problems.append(
                    f"{item['name']}: evaluated meshes moved less than {min_motion} m; "
                    "check bone influence, constraints and Action overrides")
            for first, second in plan.get("collision_pairs", []):
                a, topology_a = world_bvh(bpy.data.objects[first], depsgraph)
                b, topology_b = world_bvh(bpy.data.objects[second], depsgraph)
                baseline, topology = collision_baselines[(first, second)]
                if (topology_a, topology_b) != topology:
                    problems.append(
                        f"{item['name']}/{first}-{second}: collision mesh topology changed; "
                        "triangle-index contact deltas are invalid")
                    entry["contact_deltas"].append({
                        "objects": [first, second], "status": "topology-changed"})
                    entry["collision_candidates"].append({
                        "objects": [first, second], "status": "topology-changed"})
                    continue
                delta = collision_delta(baseline, a.overlap(b))
                entry["contact_deltas"].append({"objects": [first, second], **delta})
                if delta["posed_overlapping_triangle_pairs"]:
                    entry["collision_candidates"].append({
                        "objects": [first, second],
                        "overlapping_triangle_pairs": delta["posed_overlapping_triangle_pairs"],
                        **delta})
                max_new = plan.get("max_new_collision_pairs")
                if max_new is not None and delta["new_overlapping_triangle_pairs"] > max_new:
                    problems.append(
                        f"{item['name']}/{first}-{second}: {delta['new_overlapping_triangle_pairs']} "
                        f"new triangle overlap pairs exceed {max_new}")
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
        "source_blend": str(source_blend),
        "source_blend_sha256": source_sha,
        "status": "structural-fail" if problems else "deformation-review-required",
        "armature": rig.name,
        "bound_meshes": sorted(o.name for o in bound),
        "rest_skin_weights": weights,
        "rest_weight_zones": weight_zones,
        "poses": poses,
        "problems": problems,
        "limitations": [
            "Neutral-relative triangle contacts are advisory, not penetration volumes; "
            "only explicitly named pairs are checked.",
            "Triangle indices are compared only when evaluated polygon topology is unchanged.",
            "Motion displacement includes root movement and cannot certify local skin weights.",
            "Weight-zone checks cover only explicitly marked non-deforming semantic groups; "
            "unmarked anatomy is not certified.",
            "Weight-zone violations are conservative bone-influence checks, not deformation quality scores.",
            "Closed-manifold volume changes are diagnostic, not sufficient to certify deformation.",
            "Edge stretch and area change do not certify aesthetic quality or joint motion.",
            "A neutral pose is reconstructed by zeroing controller pose transforms; verify the intended rest state.",
            "No source .blend is written or overwritten.",
        ],
    }
    if original_hash() != source_sha:
        raise ValueError("Source Blender rig changed while running the pose battery")
    write_report(output, result)


main()
