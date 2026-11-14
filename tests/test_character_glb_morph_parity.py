"""Binary GLB morph-channel parity tests, no Blender or private character data."""
from __future__ import annotations

import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.glb_morph_parity import verify_glb_morph_samples


def fake_glb(tamper=None):
    times = [1 / 24, 2 / 24, 3 / 24]
    # Two morph targets per frame. The first is an unrelated neutral facial
    # shape; the second is a real joint corrective.
    weights = [0, 0, 0, .5, 0, 1]
    binary = struct.pack("<3f6f", *times, *weights)
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 12},
            {"buffer": 0, "byteOffset": 12, "byteLength": 24},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126,
             "count": 3, "type": "SCALAR", "min": [times[0]], "max": [times[-1]]},
            {"bufferView": 1, "componentType": 5126, "count": 6, "type": "SCALAR"},
        ],
        "nodes": [
            {"name": "Wing.L", "mesh": 0, "skin": 0},
            {"name": "CharacterRig"},
        ],
        "meshes": [{
            "extras": {"targetNames": ["eye_blink", "CORR_WingRaise.L"]},
            "primitives": [{"targets": [{"POSITION": 2}, {"POSITION": 3}]}],
            "weights": [0, 0],
        }],
        "animations": [{
            "name": "wing_fold_corrected",
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}],
            "samplers": [{"input": 0, "output": 1, "interpolation": "LINEAR"}],
        }],
    }
    if tamper:
        tamper(doc)
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    chunks = struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
    chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


class MorphParityTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "animated.glb"
        self.path.write_bytes(fake_glb())
        self.expected = {
            "Wing.L": {"CORR_WingRaise.L": [(1, 0), (2, .5), (3, 1)]},
        }

    def verify(self):
        return verify_glb_morph_samples(self.path, self.expected, fps=24)

    def test_real_binary_samples_and_target_order(self):
        result = self.verify()
        self.assertEqual(result["checked_correctives"], 1)
        self.assertEqual(result["target_meshes"], 1)
        self.assertLess(result["max_weight_error"], 1e-5)
        self.assertEqual(result["meshes"][0]["targets"],
                         ["eye_blink", "CORR_WingRaise.L"])
        # Source intermediate frame absent as a glTF key must interpolate.
        self.expected["Wing.L"]["CORR_WingRaise.L"] = [
            (1, 0), (2, .5), (3, 1)]
        self.assertLess(self.verify()["max_weight_error"], 1e-5)

    def test_rejects_missing_morph_target(self):
        self.path.write_bytes(fake_glb(
            lambda d: d["meshes"][0]["extras"].update(
                targetNames=["eye_blink", "WrongKey"])))
        with self.assertRaisesRegex(ValueError, "missing"):
            self.verify()

    def test_rejects_missing_or_duplicate_animation_channels(self):
        for channels in ([], [
            {"sampler": 0, "target": {"node": 0, "path": "weights"}},
            {"sampler": 0, "target": {"node": 0, "path": "weights"}},
        ]):
            self.path.write_bytes(fake_glb(
                lambda d: d["animations"][0].update(channels=channels)))
            with self.assertRaisesRegex(ValueError, "channel"):
                self.verify()

    def test_rejects_morph_weight_drift_even_with_named_channels(self):
        self.expected["Wing.L"]["CORR_WingRaise.L"] = [(1, 0), (2, .25), (3, 1)]
        with self.assertRaisesRegex(ValueError, "drift"):
            self.verify()

    def test_rejects_wrong_accessor_count_and_out_of_bounds_binary(self):
        for count, expected_message in ((5, "stream"), (600, "bounds")):
            self.path.write_bytes(fake_glb(
                lambda d: d["accessors"][1].update(count=count)))
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, expected_message):
                self.verify()

    def test_rejects_ambiguous_glb_node_and_target_names(self):
        def duplicate_node(doc):
            doc["nodes"].append(copy.deepcopy(doc["nodes"][0]))
        self.path.write_bytes(fake_glb(duplicate_node))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.verify()
        self.path.write_bytes(fake_glb(
            lambda d: d["meshes"][0]["extras"].update(
                targetNames=["CORR_WingRaise.L", "CORR_WingRaise.L"])))
        with self.assertRaisesRegex(ValueError, "unique"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
