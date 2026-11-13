"""Blender worker: animate authored mouth shape keys from offline phoneme cues.

The worker does not invent geometry or overwrite the source .blend. Exported
GLB must contain a named animation with actual morph-weight channels.
"""
from __future__ import annotations

import hashlib
import math
import struct
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402
from kumiki_blender.character_build.glb_animation import (  # noqa: E402
    read_single_animation, rewrite_single_animation_name,
)


def _geometry_hash(obj):
    digest = hashlib.sha256()
    for v in obj.data.vertices:
        digest.update(struct.pack("<3d", *v.co))
    for face in obj.data.polygons:
        digest.update(struct.pack("<I", len(face.vertices)))
        for index in face.vertices:
            digest.update(struct.pack("<I", index))
    return digest.hexdigest()


def _keyframe(block, frame, value):
    block.value = float(value)
    if not block.keyframe_insert(data_path="value", frame=frame, group="KumikiVisemes"):
        raise RuntimeError(f"Failed keyframe {block.name} at frame {frame}")


def _keyframe_cues(obj, plan):
    keys = obj.data.shape_keys
    mapping = plan["shape_keys"]
    blocks = {code: keys.key_blocks[name] for code, name in mapping.items()}
    fps, origin = plan["fps"], plan["first_frame"]
    ordered_blocks = list(blocks.values())

    for block in ordered_blocks:
        _keyframe(block, origin - 0.5, 0)
    for cue in plan["cues"]:
        start = origin + cue["start"] * fps
        end = origin + cue["end"] * fps
        ramp = min(float(plan["ramp_frames"]), .35 * (end - start))
        target = blocks.get(cue["value"])
        for block in ordered_blocks:
            _keyframe(block, start, 0)
        if target:
            _keyframe(target, start + ramp, 1)
            _keyframe(target, end - ramp, 1)
        for block in ordered_blocks:
            _keyframe(block, end, 0)
    final = origin + plan["cues"][-1]["end"] * fps
    for block in ordered_blocks:
        _keyframe(block, final + .5, 0)

    # Each actively mapped shape must visibly deform at least one vertex
    # versus Basis; a renamed but unchanged shape key is not a real viseme.
    basis = keys.key_blocks[0]
    changed = {}
    for code, block in blocks.items():
        changes = sum(
            1 for a, b in zip(basis.data, block.data)
            if (a.co - b.co).length > 1e-6
        )
        if code in {c["value"] for c in plan["cues"]} and changes == 0:
            raise ValueError(f"Viseme {code} / {block.name} has no authored geometry")
        changed[code] = changes
    return changed


def main():
    args = arguments()
    plan = args["plan"]
    output = Path(args["output"]).resolve()
    source = Path(plan["source_blend"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output revision exists: {output}")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Open the exact character .blend from the viseme plan")
    obj = bpy.data.objects.get(plan["mouth_object"])
    if obj is None or obj.type != "MESH":
        raise ValueError(f"Authored mouth object not found: {plan['mouth_object']}")
    if obj.library or obj.data.library:
        raise ValueError("Linked library geometry cannot be modified")
    keys = obj.data.shape_keys
    if keys is None:
        raise ValueError("Mouth mesh needs authored Basis and viseme shape keys")
    if keys.animation_data and (keys.animation_data.action or keys.animation_data.nla_tracks):
        raise ValueError("Source shape keys already have animation; merge with explicit NLA first")
    if any(name not in keys.key_blocks for name in plan["shape_keys"].values()):
        raise ValueError("Missing authored shape key required by the cue mapping")
    if any(k != keys.key_blocks[0] and abs(k.value) > 1e-6 for k in keys.key_blocks):
        raise ValueError("Save source .blend with all face shape keys at neutral zero")
    if keys.use_relative is False:
        raise ValueError("Absolute shape keys are not supported")
    if obj.data.users != 1:
        raise ValueError("Shared mouth mesh datablock must be made unique before animating")
    rigs = {mod.object for mod in obj.modifiers if mod.type == "ARMATURE" and mod.object}
    if len(rigs) > 1:
        raise ValueError("Face mesh must have at most one controlling armature")
    rig = next(iter(rigs)) if rigs else None
    if rig and rig.animation_data and (rig.animation_data.action or rig.animation_data.nla_tracks):
        raise ValueError("Combined skeletal+facial actions need the next multi-track export stage")
    geometry_sha = _geometry_hash(obj)
    changes = _keyframe_cues(obj, plan)
    if keys.animation_data is None or keys.animation_data.action is None:
        raise RuntimeError("No Blender action was generated for the shape keys")
    keys.animation_data.action.name = plan["clip"]

    scene = bpy.context.scene
    scene.render.fps = plan["fps"]
    scene.frame_start = max(0, plan["first_frame"] - 1)
    scene.frame_end = math.ceil(plan["first_frame"] + plan["cues"][-1]["end"] * plan["fps"] + 1)
    scene.frame_set(plan["first_frame"])
    if geometry_sha != _geometry_hash(obj):
        raise RuntimeError("Face geometry changed during animation-key creation")

    # Verify a non-silent middle frame actually activates its authored key.
    active = next((cue for cue in plan["cues"] if cue["value"] != "X"), None)
    if active is None:
        raise ValueError("Only-silence cues do not produce meaningful morph animation")
    frame = plan["first_frame"] + (active["start"] + active["end"]) * .5 * plan["fps"]
    scene.frame_set(int(round(frame)))
    key = keys.key_blocks[plan["shape_keys"][active["value"]]]
    if key.value < 0.05:
        raise ValueError("Animated mouth shape key did not evaluate at the expected cue")
    scene.frame_set(plan["first_frame"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "visemes.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    if rig:
        rig.select_set(True)
        for part in bpy.data.objects:
            if part.type == "MESH" and part is not obj and any(
                mod.type == "ARMATURE" and mod.object == rig for mod in part.modifiers
            ):
                part.select_set(True)
    bpy.context.view_layer.objects.active = obj
    export = {
        "filepath": str(output / "face-animated.glb"),
        "export_format": "GLB",
        "use_selection": True,
        "export_animations": True,
        "export_morph": True,
    }
    properties = bpy.ops.export_scene.gltf.get_rna_type().properties
    if "export_animation_mode" in properties:
        modes = {item.identifier for item in properties["export_animation_mode"].enum_items}
        if "ACTIVE_ACTIONS" in modes:
            export["export_animation_mode"] = "ACTIVE_ACTIONS"
    if "FINISHED" not in bpy.ops.export_scene.gltf(**export):
        raise RuntimeError("Blender failed to export facial GLB")
    receipt = rewrite_single_animation_name(output / "face-animated.glb", plan["clip"])
    gltf = read_single_animation(output / "face-animated.glb")
    morph_channels = [
        c for c in gltf["animations"][0].get("channels", [])
        if c.get("target", {}).get("path") == "weights"
    ]
    if not morph_channels:
        raise ValueError("GLB contains no animated morph-weight channel")
    targets = sum(len(p.get("targets", []))
                  for m in gltf.get("meshes", []) for p in m.get("primitives", []))
    if targets == 0:
        raise ValueError("GLB contains no morph target geometry")
    write_report(output, {
        "schema_version": 1,
        "status": "facial-performance-review-required",
        "source_blend": str(source),
        "clip": plan["clip"], "mouth_object": obj.name,
        "cues": len(plan["cues"]), "duration_s": plan["cues"][-1]["end"],
        "fps": plan["fps"], "shape_key_moved_vertices": changes,
        "rest_geometry_sha256": geometry_sha,
        "glb": {"animations": len(gltf["animations"]),
                "morph_weight_channels": len(morph_channels),
                "morph_targets": targets,
                "clip_names": [a.get("name") for a in gltf["animations"]]},
        "clip_receipt": receipt,
        "files": ["visemes.blend", "face-animated.glb"],
        "limitations": [
            "No mouth geometry is generated; artist-authored shape keys must exist.",
            "Cue timing is external; Rhubarb may not resolve all Japanese phonemes accurately.",
            "Only a standalone facial clip is exported; combination with skeletal NLA actions remains separate work.",
            "Visual appearance and actual spoken intelligibility require review in Blender and Kumiki-UI.",
        ],
    })


main()
