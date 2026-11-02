"""Offline Blender glTF animation → Kumiki PM2 contract, no bpy needed."""

from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import struct
import tempfile
import unittest

from kumiki_blender.glb_motion import MotionExportError, export_glb_motion, write_glb_motion


def sample_glb(path: Path, *, animation_changes: dict | None = None,
               untagged: bool = False, corrupted: bool = False,
               second_role: str = "ground", reversed_roots: bool = False,
               extra_clips: list[dict] | None = None) -> Path:
    times = struct.pack("<3f", 0, .5, 1)
    positions = struct.pack("<9f", 0, 1, 0, 0, 0, 0, 0, .3, 0)
    rotations = struct.pack("<12f", 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1)
    binary = times + positions + rotations
    animation = {
        "name": "rigid", "samplers": [
            {"input": 0, "output": 1, "interpolation": "LINEAR"},
            {"input": 0, "output": 2, "interpolation": "LINEAR"},
        ], "channels": [
            {"sampler": 0, "target": {"node": 0, "path": "translation"}},
            {"sampler": 1, "target": {"node": 0, "path": "rotation"}},
        ],
    }
    if animation_changes:
        animation.update(animation_changes)
    node = {"name": "fastener", "extras": {"kumiki_part": "hardware"}, "translation": [0, 1, 0]}
    if untagged:
        node.pop("extras")
    doc = {
        "asset": {"version": "2.0"},
        "nodes": [node, {"name": "floor", "extras": {"kumiki_part": second_role}}],
        "scenes": [{"nodes": [1, 0] if reversed_roots else [0, 1]}], "scene": 0,
        "meshes": [{"primitives": [{}]}],
        "animations": [animation, *(extra_clips or [])],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(times)},
            {"buffer": 0, "byteOffset": len(times), "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(times)+len(positions), "byteLength": len(rotations)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
    }
    payload = json.dumps(doc).encode("utf-8")
    payload += b" "*(-len(payload)%4)
    content = struct.pack("<4sII", b"glTF" if not corrupted else b"BAD!", 2,
                          12+8+len(payload)+8+len(binary))
    content += struct.pack("<I4s", len(payload), b"JSON") + payload
    content += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    path.write_bytes(content)
    return path


class GlbMotionTests(unittest.TestCase):
    def test_actual_glb_animation_becomes_sampled_local_y_up_pm2(self):
        with tempfile.TemporaryDirectory() as folder:
            path = sample_glb(Path(folder)/"model.glb")
            [action] = export_glb_motion(path, contacts=[
                {"frame": 20.5, "target": "kumiki:hardware:1", "speedMps": .8},
                {"frame": 15.25, "target": "kumiki:hardware:1", "speedMps": 3.1, "material": "metal"},
            ])
            self.assertEqual(action["id"], "rigid-baked")
            self.assertEqual(action["schemaVersion"], 1)
            self.assertEqual(action["durationFrames"], 30)
            self.assertEqual(len(action["tracks"]), 1)
            track = action["tracks"][0]
            self.assertEqual(track["target"], "kumiki:hardware:1")
            self.assertEqual(track["keys"][0]["position"], [0, 1, 0])
            self.assertAlmostEqual(track["keys"][7]["position"][1], 1-7/15, places=5)
            self.assertAlmostEqual(track["keys"][15]["position"][1], 0)
            self.assertAlmostEqual(track["keys"][30]["position"][1], .3, places=5)
            self.assertAlmostEqual(sum(v*v for v in track["keys"][7]["rotation"]), 1, places=5)
            self.assertEqual([c["frame"] for c in action["contacts"]], [15.25, 20.5])

    def test_export_writes_new_sidecar_without_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            model = sample_glb(base/"model.glb")
            cues = base/"cues.json"
            cues.write_text(json.dumps([{"frame": 15, "target": "kumiki:hardware:1",
                                         "speedMps": 2.5, "material": "metal"}]), encoding="utf-8")
            output = base/"generated"/"baked-actions.json"
            report = write_glb_motion(model, output, clip_name="rigid", contacts_path=cues)
            self.assertEqual(report["tracks"], 1)
            self.assertEqual(report["contacts"], 1)
            self.assertEqual(json.loads(output.read_text())[0]["contacts"][0]["target"], "kumiki:hardware:1")
            with self.assertRaisesRegex(MotionExportError, "already exists"):
                write_glb_motion(model, output, clip_name="rigid")

    def test_cli_exports_one_clip_without_blender_or_optional_dependencies(self):
        from kumiki_blender.cli import main

        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            model = sample_glb(base/"model.glb")
            output = base/"bake.json"
            result = io.StringIO()
            with redirect_stdout(result):
                code = main(["export-motion", "--model", str(model), "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(result.getvalue())["action"], "rigid-baked")
            self.assertEqual(json.loads(output.read_text())[0]["tracks"][0]["target"], "kumiki:hardware:1")

    def test_rejects_unsupported_interpolation_untagged_motion_and_missing_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            model = sample_glb(base/"model.glb")
            with self.assertRaisesRegex(MotionExportError, "exactly one"):
                export_glb_motion(model, clip_name="unknown")
            cubic = sample_glb(base/"cubic.glb", animation_changes={"samplers": [
                {"input": 0, "output": 1, "interpolation": "CUBICSPLINE"},
                {"input": 0, "output": 2},
            ]})
            with self.assertRaisesRegex(MotionExportError, "LINEAR and STEP"):
                export_glb_motion(cubic, clip_name="rigid")
            missing = sample_glb(base/"untagged.glb", untagged=True)
            with self.assertRaisesRegex(MotionExportError, "kumiki_part"):
                export_glb_motion(missing, clip_name="rigid")
            invalid = sample_glb(base/"invalid.glb", corrupted=True)
            with self.assertRaisesRegex(MotionExportError, "glTF 2.0"):
                export_glb_motion(invalid, clip_name="rigid")

    def test_does_not_guess_contact_physics_and_rejects_invalid_external_cues(self):
        with tempfile.TemporaryDirectory() as folder:
            model = sample_glb(Path(folder)/"model.glb")
            self.assertEqual(export_glb_motion(model, clip_name="rigid")[0]["contacts"], [])
            for invalid in (
                [{"frame": 15, "target": "kumiki:unknown:1", "speedMps": 3}],
                [{"frame": -1, "target": "kumiki:hardware:1", "speedMps": 3}],
                [{"frame": 3, "target": "kumiki:hardware:1", "speedMps": float("inf")}],
            ):
                with self.assertRaises(MotionExportError):
                    export_glb_motion(model, clip_name="rigid", contacts=invalid)

    def test_step_timing_and_duplicate_channels_are_not_silently_misrepresented(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            step_samplers = [
                {"input": 0, "output": 1, "interpolation": "STEP"},
                {"input": 0, "output": 2, "interpolation": "STEP"},
            ]
            step = sample_glb(base/"step.glb", animation_changes={"samplers": step_samplers})
            track = export_glb_motion(step, clip_name="rigid")[0]["tracks"][0]
            self.assertEqual(track["interpolation"], "step")
            self.assertEqual(track["keys"][14]["position"], [0, 1, 0])
            self.assertEqual(track["keys"][15]["position"], [0, 0, 0])
            original = {
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}]*2,
            }
            duplicate = sample_glb(base/"duplicate.glb", animation_changes=original)
            with self.assertRaisesRegex(MotionExportError, "repeats"):
                export_glb_motion(duplicate, clip_name="rigid")

    def test_canonical_instance_ordinals_match_gltf_scene_dfs_not_json_node_order(self):
        with tempfile.TemporaryDirectory() as folder:
            model = sample_glb(Path(folder)/"reordered.glb", second_role="hardware", reversed_roots=True)
            [action] = export_glb_motion(model)
            self.assertEqual(action["tracks"][0]["target"], "kumiki:hardware:2")
            with self.assertRaisesRegex(MotionExportError, "canonical semantic part"):
                export_glb_motion(model, contacts=[{"frame": 10,
                    "target": "kumiki:hardware:3", "speedMps": 1}])

    def test_merges_disjoint_per_object_blender_clips_but_rejects_channel_collisions(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            second = {"name": "second-object", "samplers": [{"input": 0, "output": 1}],
                      "channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}]}
            model = sample_glb(base/"disjoint.glb", extra_clips=[second])
            with self.assertRaisesRegex(MotionExportError, "exactly one"):
                export_glb_motion(model)
            [action] = export_glb_motion(model, merge_clips=True, action_id="two-body-hit")
            self.assertEqual(action["id"], "two-body-hit")
            self.assertEqual({t["target"] for t in action["tracks"]},
                             {"kumiki:hardware:1", "kumiki:ground:1"})
            duplicate = {**second, "channels": [{"sampler": 0,
                         "target": {"node": 0, "path": "translation"}}]}
            bad = sample_glb(base/"overlap.glb", extra_clips=[duplicate])
            with self.assertRaisesRegex(MotionExportError, "repeats"):
                export_glb_motion(bad, merge_clips=True)


if __name__ == "__main__":
    unittest.main()
