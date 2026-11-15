"""Blender worker: armature from joints.json, automatic weights, range-of-motion test, skinned GLB.

Arguments (JSON): mesh, joints (validated dict), output, rom_deg, render_rom (bool).

A character that is joined from separate parts (a body, a tail, eyes, a bag) is
not one closed surface, and Blender's heat weighting then finds no solution. The
weights then come from a proxy: a voxel remesh of the whole character, which is
one closed surface, gets the heat weights, and each vertex of the character takes
the weights of the nearest place on the proxy.

Range of motion: each test bone turns by +rom_deg and -rom_deg about its own X
axis. The report gives the mesh volume ratio against the rest pose (a joint
that collapses loses volume) and whether every vertex keeps a weight.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import (activate, arguments, clear_startup_scene,  # noqa: E402
                                                           import_mesh, mesh_stats, write_report)

MAX_INFLUENCES = 4  # glTF skinning and most engines


def evaluated_volume(obj) -> float:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    bm.from_object(obj, depsgraph)
    volume = abs(bm.calc_volume())
    bm.free()
    return volume


def unweighted_fraction(obj) -> float:
    count = sum(1 for vertex in obj.data.vertices if not any(g.weight > 1e-4 for g in vertex.groups))
    return count / max(1, len(obj.data.vertices))


def keep_largest_island(obj) -> int:
    """Remove every connected piece of the mesh but the largest; returns the number removed.

    Where parts overlap, a voxel remesh leaves small closed pockets inside the character.
    No bone is visible from inside a pocket, and the heat weighting then has no solution
    for the whole mesh.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    island = [-1] * len(bm.verts)
    sizes = []
    for start in bm.verts:
        if island[start.index] >= 0:
            continue
        stack, size = [start], 0
        island[start.index] = len(sizes)
        while stack:
            vertex = stack.pop()
            size += 1
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if island[other.index] < 0:
                    island[other.index] = len(sizes)
                    stack.append(other)
        sizes.append(size)
    if len(sizes) > 1:
        largest = sizes.index(max(sizes))
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if island[v.index] != largest], context="VERTS")
        bm.to_mesh(obj.data)
        obj.data.update()
    bm.free()
    return len(sizes) - 1


def weights_from_proxy(body, rig, voxel_m: float, work_size_m: float = 3.0) -> dict:
    """Heat weights on a closed voxel remesh of ``body``, moved to ``body``; returns what happened.

    The heat weighting also fails on a small mesh with many faces: a 0.3 m figure with 2 mm
    faces has no solution, and the same mesh ten times larger has one. So the proxy and a
    copy of the armature are weighted at ``work_size_m`` (largest dimension), and the proxy
    goes back to its size for the transfer.
    """
    proxy = body.copy()
    proxy.data = body.data.copy()
    proxy.name = "WeightProxy"
    bpy.context.scene.collection.objects.link(proxy)
    proxy.parent = None
    proxy.matrix_world = body.matrix_world.copy()
    proxy.vertex_groups.clear()
    for modifier in list(proxy.modifiers):
        proxy.modifiers.remove(modifier)
    activate(proxy)
    proxy.data.remesh_voxel_size = voxel_m
    bpy.ops.object.voxel_remesh()
    removed = keep_largest_island(proxy)
    helper = rig.copy()
    helper.data = rig.data.copy()
    helper.name = "WeightProxyRig"
    bpy.context.scene.collection.objects.link(helper)
    factor = max(1.0, work_size_m / max(max(body.dimensions), 1e-9))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in (proxy, helper):
        obj.matrix_world = Matrix.Scale(factor, 4) @ obj.matrix_world
        obj.select_set(True)
    bpy.context.view_layer.objects.active = helper
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    failed = unweighted_fraction(proxy)
    report = {"voxel_m": voxel_m, "proxy_vertices": len(proxy.data.vertices), "proxy_islands_removed": removed,
              "weighted_at_scale": round(factor, 3), "proxy_unweighted_fraction": round(failed, 4), "used": failed < 0.5}
    if report["used"]:
        for modifier in list(proxy.modifiers):
            proxy.modifiers.remove(modifier)
        proxy.parent = None
        proxy.matrix_world = Matrix.Scale(1.0 / factor, 4)       # its vertices were scaled; back to the size of the body
        bpy.context.view_layer.update()
        body.vertex_groups.clear()
        for group in proxy.vertex_groups:                # the same names; the modifier fills them
            body.vertex_groups.new(name=group.name)
        activate(body)
        transfer = body.modifiers.new("WeightTransfer", "DATA_TRANSFER")
        transfer.object = proxy
        transfer.use_object_transform = True
        transfer.use_vert_data = True
        transfer.data_types_verts = {"VGROUP_WEIGHTS"}
        transfer.vert_mapping = "POLYINTERP_NEAREST"
        transfer.layers_vgroup_select_src = "ALL"
        transfer.layers_vgroup_select_dst = "NAME"
        bpy.ops.object.modifier_move_to_index(modifier=transfer.name, index=0)
        bpy.ops.object.modifier_apply(modifier=transfer.name)
    helper_data = helper.data
    bpy.data.objects.remove(proxy, do_unlink=True)
    bpy.data.objects.remove(helper, do_unlink=True)
    bpy.data.armatures.remove(helper_data)
    return report


def main() -> None:
    args = arguments()
    output = Path(args["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    joints, bones = args["joints"]["joints"], args["joints"]["bones"]
    clear_startup_scene()
    body = import_mesh(args["mesh"], "Character")
    armature_data = bpy.data.armatures.new("CharacterRig")
    rig = bpy.data.objects.new("CharacterRig", armature_data)
    bpy.context.scene.collection.objects.link(rig)
    activate(rig)
    bpy.ops.object.mode_set(mode="EDIT")
    for bone in bones:
        edit = armature_data.edit_bones.new(bone["name"])
        edit.head, edit.tail = Vector(joints[bone["head"]]), Vector(joints[bone["tail"]])
        if bone["parent"]:
            edit.parent = armature_data.edit_bones[bone["parent"]]
            edit.use_connect = (edit.parent.tail - edit.head).length < 1e-5
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    # Dual-quaternion "Preserve Volume" is an opt-in native Blender deformation
    # setting. Keep the old behavior by default to preserve baseline comparisons.
    modifiers = [modifier for modifier in body.modifiers if modifier.type == "ARMATURE"]
    if not modifiers:
        raise RuntimeError("Automatic armature parenting did not install a skinning modifier")
    for modifier in modifiers:
        modifier.use_deform_preserve_volume = bool(args.get("preserve_volume", False))

    proxy = None
    heat_failed = unweighted_fraction(body)
    if heat_failed > 0.01 and args.get("proxy_weights", True):
        proxy = weights_from_proxy(body, rig, float(args.get("proxy_voxel_m") or max(body.dimensions) / 150))
        proxy["heat_unweighted_fraction"] = round(heat_failed, 4)

    activate(body)
    bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=MAX_INFLUENCES)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
    # Heat weighting leaves some vertices without weights and does not say so. Give each one to its nearest bone.
    segments = [(b["name"], Vector(joints[b["head"]]), Vector(joints[b["tail"]])) for b in bones]
    fixed = 0
    for vertex in body.data.vertices:
        if any(g.weight > 1e-4 for g in vertex.groups):
            continue
        point = body.matrix_world @ vertex.co

        def distance(segment):
            _name, head, tail = segment
            axis = tail - head
            t = max(0.0, min(1.0, (point - head).dot(axis) / axis.length_squared))
            return (point - (head + axis * t)).length

        name = min(segments, key=distance)[0]
        group = body.vertex_groups.get(name) or body.vertex_groups.new(name=name)
        group.add([vertex.index], 1.0, "REPLACE")
        fixed += 1
    used_groups = {g.group for v in body.data.vertices for g in v.groups if g.weight > 1e-4}
    empty_bones = [b["name"] for b in bones
                   if body.vertex_groups.get(b["name"]) is None or body.vertex_groups[b["name"]].index not in used_groups]

    rest_volume = evaluated_volume(body)
    angle = math.radians(float(args.get("rom_deg", 45)))
    tests = [b["name"] for b in bones if b["name"].split(".")[0] in
             ("head", "spine", "thigh", "shin", "upper_arm", "fore_arm", "upper_wing", "fore_wing", "tail1")]
    scene = bpy.context.scene
    if args.get("render_rom"):
        (output / "rom").mkdir(exist_ok=True)
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.resolution_x = scene.render.resolution_y = 512
        scene.render.film_transparent = True
        cam = bpy.data.objects.new("ROM_Camera", bpy.data.cameras.new("ROM_Camera"))
        scene.collection.objects.link(cam)
        height = max(body.dimensions)
        cam.data.type, cam.data.ortho_scale = "ORTHO", 1.3 * height
        cam.location, cam.rotation_euler = (0, -3 * height, 0.5 * height), (math.pi / 2, 0, 0)
        scene.camera = cam
    rom = []
    for name in tests:
        pose = rig.pose.bones[name]
        for sign in (1, -1):
            pose.rotation_mode = "XYZ"
            pose.rotation_euler = (sign * angle, 0, 0)
            bpy.context.view_layer.update()
            ratio = evaluated_volume(body) / max(rest_volume, 1e-12)
            entry = {"bone": name, "angle_deg": round(math.degrees(sign * angle)), "volume_ratio": round(ratio, 4)}
            if args.get("render_rom"):
                scene.render.filepath = str(output / "rom" / f"{name}_{entry['angle_deg']:+d}.png")
                bpy.ops.render.render(write_still=True)
            rom.append(entry)
        pose.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "rigged.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    rig.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(output / "rigged.glb"), export_format="GLB", use_selection=True,
                              export_skins=True)
    worst = max(rom, key=lambda r: abs(1 - r["volume_ratio"])) if rom else None
    fallback = fixed / max(1, len(body.data.vertices))
    warnings = []
    if fallback > 0.01:
        warnings.append(f"Heat weighting failed for {fallback:.0%} of vertices; they use nearest-bone weights. "
                        "Check that the mesh is one closed surface.")
    if proxy and proxy["used"]:
        warnings.append("Heat weighting had no solution on the mesh itself (separate parts, or small faces). The "
                        f"weights come from a closed voxel proxy ({proxy['voxel_m'] * 1000:.1f} mm). Parts that touch "
                        "(a tail on a back) share weights there.")
    write_report(output, {"mesh": mesh_stats(body), "bones": len(bones), "vertices_given_nearest_bone": fixed,
                          "proxy_weights": proxy,
                          "nearest_bone_fraction": round(fallback, 4), "warnings": warnings,
                          "bones_without_weights": empty_bones, "max_influences": MAX_INFLUENCES, "preserve_volume": bool(args.get("preserve_volume", False)),
                          "rom_deg": math.degrees(angle), "range_of_motion": rom, "worst_volume_change": worst,
                          "files": ["rigged.blend", "rigged.glb"]})


main()
