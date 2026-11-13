"""Lossless simultaneous morph+skeletal GLB action merger contracts."""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.glb_performance import finalize_performance_glb
from kumiki_blender.character_build.glb_animation import read_single_animation


def make_glb(*, combined=False, mismatch=False, duplicate=False, missing_morph=False):
    skeleton = {"name": "BodyAction", "samplers": [{"input": 0, "output": 2}],
                "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}]}
    morph = {"name": "FaceAction", "samplers": [{"input": 1, "output": 3}],
             "channels": [{"sampler": 0, "target": {
                 "node": 1 if duplicate else 2,
                 "path": "rotation" if duplicate else ("scale" if missing_morph else "weights"),
             }}]}
    if combined:
        skeleton["samplers"].extend(morph["samplers"])
        skeleton["channels"].append({**morph["channels"][0], "sampler": 1})
    inputs = [
        {"componentType": 5126, "type": "SCALAR", "min": [0], "max": [1], "count": 2},
        {"componentType": 5126, "type": "SCALAR",
         "min": [.3 if mismatch else 0], "max": [1.3 if mismatch else 1], "count": 2},
        {"componentType": 5126, "type": "VEC4", "count": 2},
        {"componentType": 5126, "type": "SCALAR", "count": 2},
    ]
    doc = {
        "asset": {"version": "2.0"},
        "skins": [{"joints": [1]}],
        "nodes": [{"name": "CharacterRoot"}, {"name": "Bone"}, {"name": "Mouth", "mesh": 0}],
        "meshes": [{"primitives": [{"targets": [{"POSITION": 0}]}]}],
        "accessors": inputs,
        "animations": [skeleton] if combined else [skeleton, morph],
    }
    payload = json.dumps(doc, separators=(",", ":")).encode()
    payload += b" " * ((-len(payload)) % 4)
    binary = b"deadbeef"
    chunks = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
    return struct.pack("<4sII", b"glTF", 2, len(chunks) + 12) + chunks


class PerformanceGlbTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "performance.glb"

    def test_merge_two_tracks_without_changing_binary(self):
        initial = make_glb()
        self.path.write_bytes(initial)
        result = finalize_performance_glb(self.path, "talk_wave")
        gltf = read_single_animation(self.path)
        self.assertEqual(result["input_tracks"], 2)
        self.assertEqual(result["deform_joint_channels"], 1)
        self.assertEqual(result["morph_weight_channels"], 1)
        self.assertEqual(gltf["animations"][0]["name"], "talk_wave")
        self.assertEqual(len(gltf["animations"][0]["channels"]), 2)
        self.assertEqual([c["sampler"] for c in gltf["animations"][0]["channels"]], [0, 1])
        self.assertEqual(self.path.read_bytes()[-16:], initial[-16:])
        self.assertFalse(self.path.with_suffix(".glb.performance.tmp").exists())

    def test_existing_combined_track_can_be_renamed(self):
        self.path.write_bytes(make_glb(combined=True))
        result = finalize_performance_glb(self.path, "talk_and_look")
        self.assertEqual(result["input_tracks"], 1)
        self.assertEqual(read_single_animation(self.path)["animations"][0]["name"], "talk_and_look")

    def test_prevent_conflicting_or_drifting_timelines(self):
        for option, fragment in (({"mismatch": True}, "not time aligned"),
                                 ({"duplicate": True}, "Overlapping"),
                                 ({"missing_morph": True}, "morph")):
            with self.subTest(option=option):
                self.path.write_bytes(make_glb(**option))
                with self.assertRaisesRegex(ValueError, fragment):
                    finalize_performance_glb(self.path, "combined")
                # Input GLB must remain unchanged when refusing invalid merge.
                self.assertEqual(self.path.read_bytes(), make_glb(**option))

    def test_invalid_name_or_asset(self):
        self.path.write_bytes(make_glb())
        with self.assertRaisesRegex(ValueError, "clip"):
            finalize_performance_glb(self.path, "../unsafe")
        self.path.write_bytes(b"not a GLB")
        with self.assertRaisesRegex(ValueError, "glTF"):
            finalize_performance_glb(self.path, "talk")


if __name__ == "__main__":
    unittest.main()
