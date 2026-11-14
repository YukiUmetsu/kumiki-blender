"""Blender worker: visually bake native IK/FK controls to portable deform skeleton.

Never updates the source .blend. Produces editable baked.blend plus a skinned
GLB with one sampled clip. This initial worker bakes skeletal motion only;
facial shape-key action tracks require a separate explicit clip contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402
from kumiki_blender.character_build.glb_animation import rewrite_single_animation_name  # noqa: E402
from kumiki_blender.character_build.glb_morph_parity import verify_glb_morph_samples  # noqa: E402
from kumiki_blender.character_build.blender_corrective_bake import (  # noqa: E402
    collect_corrective_drivers, sample_correctives, compare_samples, keyframe_correctives,
)
from kumiki_blender.character_build.blender_viseme_bake import (  # noqa: E402
    collect_viseme_actions, merge_morph_tracks, verify_facial_samples,
    guard_untracked_shape_weights,
)


def _sample(rig, bones, frames):
    scene = bpy.context.scene
    sample = {}
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        sample[frame] = {name: rig.pose.bones[name].matrix.copy() for name in bones}
    return sample


def _error(before, after, bones, frames):
    max_position, max_angle = 0.0, 0.0
    worst = None
    for frame in frames:
        for name in bones:
            a, b = before[frame][name], after[frame][name]
            position = (a.translation - b.translation).length
            angle = math.degrees(a.to_quaternion().rotation_difference(b.to_quaternion()).angle)
            if position > max_position or angle > max_angle:
                worst = {"frame": frame, "bone": name,
                         "position_error_m": round(position, 7),
                         "rotation_error_deg": round(angle, 5)}
            max_position = max(max_position, position)
            max_angle = max(max_angle, angle)
    return {"max_position_error_m": round(max_position, 7),
            "max_rotation_error_deg": round(max_angle, 5), "worst_observed": worst}


def _glb_document(path):
    blob = path.read_bytes()
    if len(blob) < 20 or blob[:4] != b"glTF":
        raise ValueError("Exporter did not produce a binary GLB")
    declared_size = struct.unpack_from("<I", blob, 8)[0]
    if declared_size != len(blob):
        raise ValueError("GLB header length mismatch")
    pos, doc = 12, None
    while pos + 8 <= len(blob):
        chunk_len, chunk_type = struct.unpack_from("<II", blob, pos)
        if pos + 8 + chunk_len > len(blob):
            raise ValueError("GLB truncated chunk")
        if chunk_type == 0x4E4F534A:
            doc = json.loads(blob[pos + 8:pos + 8 + chunk_len])
        pos += chunk_len + 8
    if doc is None:
        raise ValueError("GLB JSON chunk is missing")
    return doc


def main():
    args = arguments()
    plan = args["plan"]
    source = Path(plan["source_blend"]).resolve()
    output = Path(args["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"Animation revision exists: {output}")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Open the exact source controlled.blend named in the bake plan")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != plan["source_sha256"]:
        raise ValueError("Animated rig source changed since source-locked bake plan")
    rig = bpy.data.objects.get(plan["armature"])
    if rig is None or rig.type != "ARMATURE":
        raise ValueError(f"Armature not found: {plan['armature']}")
    if rig.animation_data is None or rig.animation_data.action is None:
        raise ValueError("The control rig requires an active keyed animation Action")
    if len(rig.animation_data.nla_tracks):
        raise ValueError("Mixed NLA tracks need an explicit multi-clip workflow; not yet supported")
    controls = [bone.name for bone in rig.data.bones
                if bone.name.startswith("CTRL_") and not bone.use_deform]
    if not controls:
        raise ValueError("No approved non-deforming CTRL_ bones to bake")
    deform = [bone.name for bone in rig.data.bones if bone.use_deform]
    if not deform:
        raise ValueError("No deform bones found for GLB skinning")
    bound = [obj for obj in bpy.data.objects if obj.type == "MESH" and any(
        mod.type == "ARMATURE" and mod.object == rig for mod in obj.modifiers)]
    if not bound:
        raise ValueError("No skinned meshes available for GLB export")
    bake_correctives = bool(plan.get("bake_correctives", False))
    bake_visemes = bool(plan.get("bake_visemes", False))
    if any(ob.library or ob.data.library for ob in [rig] + bound):
        raise ValueError("Linked armatures/meshes cannot be baked in this workflow")
    scene = bpy.context.scene
    start, end, step = plan["start_frame"], plan["end_frame"], plan["step"]
    frames = list(range(start, end + 1, step))
    if frames[-1] != end:
        frames.append(end)
    if (bake_correctives or bake_visemes) and step != 1:
        raise ValueError("Unified morph animation baking requires step=1")
    correctives = collect_corrective_drivers(
        bound, rig, bake_correctives, allow_facial_actions=bake_visemes)
    faces = collect_viseme_actions(bound, bake_visemes)
    tracked = merge_morph_tracks(correctives, faces)
    guard_untracked_shape_weights(bound, tracked, frames)
    scene.render.fps = plan["fps"]
    scene.frame_start, scene.frame_end = start, end
    original_action = rig.animation_data.action
    before_morph = sample_correctives(tracked, frames) if tracked else {}
    if faces:
        verify_facial_samples(before_morph, faces)
    before = _sample(rig, deform, frames)

    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="DESELECT")
    for bone in rig.pose.bones:
        bone.bone.select = bone.name in deform
    rig.data.bones.active = rig.data.bones[deform[0]]
    result = bpy.ops.nla.bake(frame_start=start, frame_end=end, step=step,
                              only_selected=True, visual_keying=True,
                              clear_constraints=True, clear_parents=False,
                              use_current_action=False, bake_types={"POSE"})
    bpy.ops.object.mode_set(mode="OBJECT")
    if "FINISHED" not in result:
        raise RuntimeError(f"Native visual-keying bake refused: {result}")
    baked = rig.animation_data.action
    if baked is None or baked == original_action:
        raise RuntimeError("Blender did not create a new baked deform-bone Action")
    baked.name = plan["clip"]
    # Controller inputs can no longer affect the bake: the active Action now
    # has sampled keys for the deform skeleton and constraints are cleared.
    for name in deform:
        if len(rig.pose.bones[name].constraints):
            raise RuntimeError(f"Unbaked control constraint remains on {name}")
    after = _sample(rig, deform, frames)
    drift = _error(before, after, deform, frames)
    if (drift["max_position_error_m"] > plan["max_position_error_m"]
            or drift["max_rotation_error_deg"] > plan["max_rotation_error_deg"]):
        raise ValueError(f"Visual bake differs from controlled source: {drift}")

    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for name in controls:
        edit = rig.data.edit_bones.get(name)
        if edit:
            rig.data.edit_bones.remove(edit)
    bpy.ops.object.mode_set(mode="OBJECT")
    if any(bone.name.startswith("CTRL_") for bone in rig.data.bones):
        raise RuntimeError("Unbaked control bones remain")
    if rig.animation_data.action != baked:
        raise RuntimeError("Baked action was lost after controller removal")
    post_removal = _error(before, _sample(rig, deform, frames), deform, frames)
    if (post_removal["max_position_error_m"] > plan["max_position_error_m"]
            or post_removal["max_rotation_error_deg"] > plan["max_rotation_error_deg"]):
        raise ValueError(f"Removing controllers changed baked deformation: {post_removal}")
    morph_pose_drift = None
    if tracked:
        morph_pose_drift = compare_samples(
            before_morph, sample_correctives(tracked, frames))
        if correctives:
            keyframe_correctives(correctives, before_morph, frames, plan["clip"])
        compare_samples(
            before_morph, sample_correctives(tracked, frames), tolerance=.005)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    scene.frame_set(start)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "baked.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bound:
        obj.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    export = {
        "filepath": str(output / "animated.glb"),
        "export_format": "GLB",
        "use_selection": True,
        "export_skins": True,
        "export_animations": True,
    }
    params = bpy.ops.export_scene.gltf.get_rna_type().properties
    if "export_animation_mode" in params:
        modes = {e.identifier for e in params["export_animation_mode"].enum_items}
        if "ACTIVE_ACTIONS" not in modes:
            raise ValueError(f"glTF exporter lacks ACTIVE_ACTIONS mode: {sorted(modes)}")
        export["export_animation_mode"] = "ACTIVE_ACTIONS"
    if tracked:
        for option in ("export_morph", "export_morph_animation", "export_force_sampling"):
            if option not in params:
                raise ValueError(f"glTF exporter missing corrective animation feature {option}")
            export[option] = True
    result = bpy.ops.export_scene.gltf(**export)
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender failed to export animated GLB: {result}")
    clip_receipt = rewrite_single_animation_name(output / "animated.glb", plan["clip"])
    gltf = _glb_document(output / "animated.glb")
    animations = gltf.get("animations", [])
    if not gltf.get("skins") or not animations or not any(a.get("channels") for a in animations):
        raise ValueError("GLB missing skinned, keyed animation data")
    if len(animations) != 1 or animations[0].get("name") != plan["clip"]:
        raise ValueError("Exported GLB clip count/name does not match the requested action")
    skin_joint_nodes = {idx for skin in gltf["skins"] for idx in skin.get("joints", [])}
    deform_channels = sum(
        1 for animation in animations for channel in animation.get("channels", [])
        if channel.get("target", {}).get("node") in skin_joint_nodes
    )
    if deform_channels == 0:
        raise ValueError("Exported animation has no channels targeting deform skeleton joints")
    control_nodes = [n.get("name") for n in gltf.get("nodes", [])
                     if n.get("name", "").startswith("CTRL_")]
    if control_nodes:
        raise ValueError(f"Control bones leaked into portable GLB: {control_nodes}")
    morph_receipt = (verify_glb_morph_samples(
        output / "animated.glb", before_morph, fps=plan["fps"])
        if tracked else None)
    if hashlib.sha256(source.read_bytes()).hexdigest() != plan["source_sha256"]:
        raise ValueError("Original animated rig .blend changed during GLB export")
    write_report(output, {
        "schema_version": 1,
        "status": "exported-visual-parity-review-required",
        "source_blend": str(source),
        "clip": plan["clip"], "fps": plan["fps"],
        "frames": [start, end], "samples": len(frames),
        "armature": rig.name,
        "deform_bones": len(deform),
        "removed_control_bones": controls,
        "pose_parity": drift,
        "pose_parity_after_controller_removal": post_removal,
        "morph_correctives_requested": bake_correctives,
        "morph_visemes_requested": bake_visemes,
        "morph_viseme_objects": sorted(faces),
        "morph_driver_to_baked_pose_max_error": morph_pose_drift,
        "morph_parity": morph_receipt,
        "clip_rename_receipt": clip_receipt,
        "glb": {
            "skins": len(gltf["skins"]),
            "animations": len(animations),
            "channels": sum(len(a.get("channels", [])) for a in animations),
            "deform_joint_channels": deform_channels,
            "clip_names": [a.get("name") for a in animations],
            "joint_count": len(gltf["skins"][0].get("joints", [])),
        },
        "files": ["baked.blend", "animated.glb"],
        "limitations": [
            "One skeletal clip with opt-in authored facial Actions and joint corrective morphs; NLA tracks and multiple animation clips are unsupported.",
            "Face and joint morph weights are sampled per frame and compared with exported GLB binary animation alongside the skeleton.",
            "Inspect Blender/WebGL pose, silhouette and skin fidelity before production approval.",
        ],
    })


main()
