"""End-to-end native Blender smoke for authored facial morph animation.

Uses procedural mouth geometry and offline Rhubarb-style JSON; no audio model,
private character assets, or external Python installation needed in Blender.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kumiki_blender.character_build.viseme_plan import save_viseme_plan  # noqa: E402
from kumiki_blender.character_build.glb_animation import read_single_animation  # noqa: E402


def authored_face():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12,
                                         radius=.035, location=(0, -.045, .22))
    obj = bpy.context.object
    obj.name = "Mouth"
    obj.shape_key_add(name="Basis", from_mix=False)
    a = obj.shape_key_add(name="viseme_A", from_mix=False)
    c = obj.shape_key_add(name="viseme_C", from_mix=False)
    for i, vertex in enumerate(obj.data.vertices):
        if vertex.co.z > 0:
            a.data[i].co.z = vertex.co.z + .012
        if vertex.co.x > 0:
            c.data[i].co.x = vertex.co.x + .014
    bpy.context.view_layer.update()
    return obj


def main():
    with tempfile.TemporaryDirectory(prefix="kumiki-face-morph-") as tmp:
        folder = Path(tmp)
        authored_face()
        source = folder / "authored-face.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        cues = folder / "rhubarb.json"
        cues.write_text(json.dumps({
            "metadata": {"soundFile": "not-required-for-this-fixture.wav", "duration": .85},
            "mouthCues": [
                {"start": 0, "end": .10, "value": "X"},
                {"start": .10, "end": .38, "value": "A"},
                {"start": .38, "end": .75, "value": "C"},
                {"start": .75, "end": .85, "value": "X"},
            ]
        }), encoding="utf-8")
        plan = save_viseme_plan(source, cues, "Mouth", "hello",
                                folder / "plan", fps=24)
        output = folder / "animated"
        args = folder / "args.json"
        args.write_text(json.dumps({"plan": plan, "output": str(output)}))
        worker = ROOT / "src/kumiki_blender/character_build/blender_visemes.py"
        cmd = [bpy.app.binary_path, "--background", "--python-exit-code", "1",
               str(source), "--python", str(worker), "--", str(args)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=100)
        if result.returncode != 0 or not (output / "report.json").is_file():
            print(result.stdout[-8000:], file=sys.stderr)
            print(result.stderr[-8000:], file=sys.stderr)
            raise RuntimeError(f"Facial morph worker failed: {result.returncode}")
        report = json.loads((output / "report.json").read_text())
        glb = read_single_animation(output / "face-animated.glb")
        assert (output / "visemes.blend").is_file()
        assert report["status"] == "facial-performance-review-required"
        assert report["glb"]["morph_weight_channels"] >= 1
        assert report["glb"]["morph_targets"] >= 2
        assert report["glb"]["clip_names"] == ["hello"]
        assert glb["animations"][0]["name"] == "hello"
        assert report["shape_key_moved_vertices"]["A"] > 0
        assert report["shape_key_moved_vertices"]["C"] > 0
        assert any(ch["target"]["path"] == "weights"
                   for ch in glb["animations"][0]["channels"])
        print(json.dumps({
            "status": "native-facial-morph-export-verified",
            "blender_version": bpy.app.version_string,
            "face_clip": report["clip"],
            "cues": report["cues"],
            "morph_channels": report["glb"]["morph_weight_channels"],
            "morph_targets": report["glb"]["morph_targets"],
        }, indent=2))


main()
