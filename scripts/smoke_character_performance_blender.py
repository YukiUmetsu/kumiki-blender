"""Native Blender integration test for synchronized skeletal + facial glTF clips."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kumiki_blender.character_build.performance_plan import save_performance_plan  # noqa: E402
from kumiki_blender.character_build.glb_animation import read_single_animation  # noqa: E402


def make_character():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, .18))
    rig = bpy.context.object
    rig.name = "CharacterRig"
    rig.data.bones[0].name = "root"
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12,
                                         radius=.045, location=(0, -.05, .21))
    face = bpy.context.object
    face.name = "Mouth"
    face.shape_key_add(name="Basis", from_mix=False)
    key = face.shape_key_add(name="viseme_A", from_mix=False)
    for i, vert in enumerate(face.data.vertices):
        if vert.co.z >= 0:
            key.data[i].co.z += .014
    weights = face.vertex_groups.new(name="root")
    weights.add(list(range(len(face.data.vertices))), 1., "REPLACE")
    mod = face.modifiers.new("Character Skin", "ARMATURE")
    mod.object = rig

    pose = rig.pose.bones["root"]
    pose.rotation_mode = "XYZ"
    for frame, angle in ((1, 0), (12, .35)):
        bpy.context.scene.frame_set(frame)
        pose.rotation_euler = (0, 0, angle)
        pose.keyframe_insert(data_path="rotation_euler", frame=frame)
    for frame, value in ((1, 0), (6, 1), (12, 0)):
        bpy.context.scene.frame_set(frame)
        key.value = value
        key.keyframe_insert(data_path="value", frame=frame)
    bpy.context.scene.frame_set(1)
    bpy.context.scene.render.fps = 24
    bpy.context.view_layer.update()


def main():
    with tempfile.TemporaryDirectory(prefix="kumiki-combined-") as temp:
        root = Path(temp)
        make_character()
        source = root / "body-and-face.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        plan = save_performance_plan(source, "talk_wave", 1, 12, root / "plan")
        out = root / "out"
        args = root / "worker.json"
        args.write_text(json.dumps({"plan": plan, "output": str(out)}))
        script = ROOT / "src/kumiki_blender/character_build/blender_performance.py"
        process = subprocess.run([
            bpy.app.binary_path, "--background", "--python-exit-code", "1",
            str(source), "--python", str(script), "--", str(args)
        ], capture_output=True, text=True, timeout=100)
        if process.returncode or not (out / "report.json").is_file():
            print(process.stdout[-9500:], file=sys.stderr)
            print(process.stderr[-9500:], file=sys.stderr)
            raise RuntimeError(f"Combined performance worker failed: {process.returncode}")
        report = json.loads((out / "report.json").read_text())
        gltf = read_single_animation(out / "performance.glb")
        clips = gltf["animations"]
        paths = {channel["target"]["path"] for channel in clips[0]["channels"]}
        assert (out / "combined.blend").is_file()
        assert report["status"] == "combined-performance-review-required"
        assert report["glb"]["output_tracks"] == 1
        assert report["glb"]["clip"] == "talk_wave"
        assert report["glb"]["deform_joint_channels"] >= 1
        assert report["glb"]["morph_weight_channels"] >= 1
        assert "weights" in paths
        assert paths & {"translation", "rotation", "scale"}
        assert clips[0]["name"] == "talk_wave"
        print(json.dumps({
            "status": "native-combined-performance-verified",
            "blender_version": bpy.app.version_string,
            "input_tracks": report["glb"]["input_tracks"],
            "clip": report["clip"],
            "skinned_joint_channels": report["glb"]["deform_joint_channels"],
            "facial_morph_channels": report["glb"]["morph_weight_channels"],
        }, indent=2))


main()
