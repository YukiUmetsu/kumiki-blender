"""Blender worker: armature from joints.json, automatic weights, range-of-motion test, skinned GLB.

Arguments (JSON): mesh, joints (validated dict), output, rom_deg, render_rom (bool).

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
from mathutils import Vector

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
    write_report(output, {"mesh": mesh_stats(body), "bones": len(bones), "vertices_given_nearest_bone": fixed,
                          "nearest_bone_fraction": round(fallback, 4), "warnings": warnings,
                          "bones_without_weights": empty_bones, "max_influences": MAX_INFLUENCES, "preserve_volume": bool(args.get("preserve_volume", False)),
                          "rom_deg": math.degrees(angle), "range_of_motion": rom, "worst_volume_change": worst,
                          "files": ["rigged.blend", "rigged.glb"]})


main()
