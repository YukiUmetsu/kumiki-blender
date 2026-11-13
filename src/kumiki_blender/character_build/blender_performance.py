"""Blender worker: export a single synchronized skeletal + facial GLB clip.

Blender remains the animation authoring source. Both the rig's deform Action
and the facial mesh's shape-key Action must be active in one saved .blend.
The output is an independent .blend and a GLB with a single combined clip.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402
from kumiki_blender.character_build.glb_performance import finalize_performance_glb  # noqa: E402


def _check_animated(rig, shape_keys, frames):
    scene = bpy.context.scene
    original = scene.frame_current
    rig_bones = [b.name for b in rig.data.bones if b.use_deform]
    if not rig_bones:
        raise ValueError("Rig has no exportable deform bones")
    morph_names = [b.name for b in shape_keys.key_blocks][1:]
    if not morph_names:
        raise ValueError("Facial mesh needs at least one authored morph shape")
    bone_snapshots = []
    morph_snapshots = []
    try:
        for frame in frames:
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            bone_snapshots.append([
                rig.pose.bones[name].matrix.copy() for name in rig_bones
            ])
            morph_snapshots.append([
                shape_keys.key_blocks[name].value for name in morph_names
            ])
    finally:
        scene.frame_set(original)
    bone_motion = any(
        (a.translation - b.translation).length > 1e-6 or
        a.to_quaternion().rotation_difference(b.to_quaternion()).angle > 1e-5
        for other in bone_snapshots[1:]
        for a, b in zip(bone_snapshots[0], other)
    )
    morph_motion = any(
        abs(a - b) > 1e-4
        for other in morph_snapshots[1:]
        for a, b in zip(morph_snapshots[0], other)
    )
    if not bone_motion or not morph_motion:
        raise ValueError(f"Performance frame range lacks measurable body/face motion: "
                         f"body={bone_motion}, face={morph_motion}")
    return {
        "deform_bones": len(rig_bones),
        "shape_keys": morph_names,
        "nontrivial_body_motion": True,
        "nontrivial_morph_motion": True,
    }



def _apply_offline_cues(face, viseme, performance):
    """Key existing, neutral facial geometry in memory without mutating base mesh vertices."""
    keys = face.data.shape_keys
    if keys is None or not keys.use_relative:
        raise ValueError("Offline mouth cues need existing relative shape keys")
    if keys.animation_data and (keys.animation_data.action or keys.animation_data.nla_tracks):
        raise ValueError("Facial mesh already has animation; refuse to overwrite it")
    if face.data.users != 1 or face.library or face.data.library:
        raise ValueError("Authored mouth data must be local, unique and editable")
    if any(abs(key.value) > 1e-6 for key in keys.key_blocks[1:]):
        raise ValueError("Source face must start with all authored mouth shapes neutral")
    if (viseme["source_blend"] != performance["source_blend"] or
            viseme["source_blend_sha256"] != performance["source_sha256"] or
            viseme["mouth_object"] != performance["mouth_object"] or
            viseme["clip"] != performance["clip"] or
            viseme["fps"] != performance["fps"] or
            viseme["first_frame"] != performance["start_frame"]):
        raise ValueError("Body and viseme plans must reference the same source, clip and timeline")
    start, end = performance["start_frame"], performance["end_frame"]
    if start + viseme["cues"][-1]["end"] * viseme["fps"] > end + 1e-6:
        raise ValueError("Face cue track exceeds the body clip timeline")
    mapping = viseme["shape_keys"]
    if any(name not in keys.key_blocks for name in mapping.values()):
        raise ValueError("Face is missing an authored cue shape key")
    blocks = {code: keys.key_blocks[name] for code, name in mapping.items()}
    used = {cue["value"] for cue in viseme["cues"]} - {"X"}
    if not used:
        raise ValueError("No non-silent facial cue to animate")
    basis = keys.key_blocks[0]
    moved = {}
    for code, block in blocks.items():
        changed = sum(1 for a, b in zip(basis.data, block.data)
                      if (a.co - b.co).length > 1e-6)
        if code in used and changed == 0:
            raise ValueError(f"Shape key {block.name} contains no authored geometry")
        moved[code] = changed
    # World geometry and armature bone motions remain untouched. Frame bounds
    # intentionally match the body Action, including a neutral key at its end.
    def key(block, frame, weight):
        block.value = float(weight)
        if not block.keyframe_insert(data_path="value", frame=frame, group="KumikiVisemes"):
            raise RuntimeError(f"Could not key authored shape {block.name} at frame {frame}")

    ordered = list(blocks.values())
    for block in ordered:
        key(block, start, 0.0)
    for cue in viseme["cues"]:
        first = start + cue["start"] * viseme["fps"]
        last = start + cue["end"] * viseme["fps"]
        ramp = min(float(viseme["ramp_frames"]), .35 * (last - first))
        target = blocks.get(cue["value"])
        for block in ordered:
            key(block, first, 0)
        if target:
            key(target, first + ramp, 1)
            key(target, last - ramp, 1)
        for block in ordered:
            key(block, last, 0)
    for block in ordered:
        key(block, end, 0)
    if not keys.animation_data or not keys.animation_data.action:
        raise RuntimeError("Offline cues did not create a facial Blender Action")
    keys.animation_data.action.name = performance["clip"] + "_face"
    return {
        "cue_count": len(viseme["cues"]),
        "source_cues_sha256": viseme["source_cues_sha256"],
        "moved_vertices_by_viseme": moved,
        "timeline_frames": [start, end],
    }


def main():
    args = arguments()
    plan = args["plan"]
    output = Path(args["output"]).resolve()
    source = Path(plan["source_blend"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output revision exists: {output}")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Open the exact source .blend from the performance plan")
    rig = bpy.data.objects.get(plan["armature"])
    face = bpy.data.objects.get(plan["mouth_object"])
    if rig is None or rig.type != "ARMATURE":
        raise ValueError(f"Character rig not found: {plan['armature']}")
    if face is None or face.type != "MESH" or not face.data.shape_keys:
        raise ValueError(f"Authored face mesh/shape keys not found: {plan['mouth_object']}")
    keys = face.data.shape_keys
    automatic_cues = None
    if args.get("viseme_plan") is not None:
        automatic_cues = _apply_offline_cues(face, args["viseme_plan"], plan)
    if (rig.library or rig.data.library or face.library or face.data.library):
        raise ValueError("Linked character meshes/armatures are unsupported for export")
    if (rig.animation_data is None or rig.animation_data.action is None or
            keys.animation_data is None or keys.animation_data.action is None):
        raise ValueError("Both deform skeleton and facial shape keys need active Actions")
    if rig.animation_data.nla_tracks or keys.animation_data.nla_tracks:
        raise ValueError("Mixed NLA timelines need separate baking before combined performance")
    if any(b.name.startswith("CTRL_") for b in rig.data.bones):
        raise ValueError("Bake CTRL_ IK/FK authoring controls onto deform bones before performance export")
    deform = {bone.name for bone in rig.data.bones if bone.use_deform}
    if any(len(rig.pose.bones[name].constraints) for name in deform):
        raise ValueError("Deform bone constraints must be visually baked first")
    bound = [ob for ob in bpy.data.objects if ob.type == "MESH"
             and any(mod.type == "ARMATURE" and mod.object == rig
                     for mod in ob.modifiers)]
    if not bound:
        raise ValueError("A skinned body mesh is required to produce portable GLB")
    start, end = plan["start_frame"], plan["end_frame"]
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = start, end
    scene.render.fps = plan["fps"]
    frames = sorted(set([start, end] + [
        start + round((end - start) * fraction)
        for fraction in (.25, .5, .75)
    ]))
    motion = _check_animated(rig, keys, frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    scene.frame_set(start)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "combined.blend"))
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    included = set(bound + [face])
    for ob in included:
        ob.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    export = {
        "filepath": str(output / "performance.glb"),
        "export_format": "GLB", "use_selection": True,
        "export_skins": True, "export_morph": True,
        "export_animations": True,
    }
    props = bpy.ops.export_scene.gltf.get_rna_type().properties
    if "export_animation_mode" in props:
        mode_list = {mode.identifier for mode in props["export_animation_mode"].enum_items}
        if "ACTIVE_ACTIONS" not in mode_list:
            raise ValueError("This Blender exporter lacks active-action animation export")
        export["export_animation_mode"] = "ACTIVE_ACTIONS"
    if "FINISHED" not in bpy.ops.export_scene.gltf(**export):
        raise RuntimeError("Failed to export combined facial/skeletal GLB")
    combined = finalize_performance_glb(output / "performance.glb", plan["clip"],
              time_alignment_tolerance_s=plan["time_alignment_tolerance_s"])
    write_report(output, {
        "schema_version": 1,
        "status": "combined-performance-review-required",
        "source_blend": str(source),
        "clip": plan["clip"],
        "armature": rig.name,
        "mouth_object": face.name,
        "frames": [start, end],
        "fps": plan["fps"],
        "motion_verified": motion,
        "auto_applied_offline_cues": automatic_cues,
        "exported_skinned_meshes": sorted(ob.name for ob in bound),
        "glb": combined,
        "files": ["combined.blend", "performance.glb"],
        "limitations": [
            "Only one active deform skeleton Action plus one active facial shape-key Action.",
            "The two actions must use the same frame origin and end; they are not retimed automatically.",
            "Blender CTRL_ authoring bones need prior constraint baking; NLA layering not supported.",
            "GLB binary tracks and animation targets are validated, but browser render parity remains unverified.",
        ],
    })


main()
