"""Lossless JSON-only GLB animation clip renaming with binary preservation."""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.glb_animation import (
    read_single_animation, rewrite_single_animation_name,
)


def glb_bytes(clips: int = 1):
    doc = {"asset": {"version": "2.0"},
           "animations": [{"name": "Animation", "channels": [
               {"sampler": 0, "target": {"node": 0, "path": "rotation"}}],
               "samplers": [{"input": 0, "output": 1}]} for _ in range(clips)]}
    payload = json.dumps(doc, separators=(",", ":")).encode()
    payload += b" " * ((-len(payload)) % 4)
    binary = b"ABCD"
    chunks = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    chunks += struct.pack("<II", 4, 0x004E4942) + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


class ClipNameTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "character.glb"
        self.path.write_bytes(glb_bytes())

    def test_rename_does_not_touch_binary_chunk(self):
        original = self.path.read_bytes()
        result = rewrite_single_animation_name(self.path, "wing_fold")
        self.assertEqual(result["clip_name"], "wing_fold")
        self.assertTrue(result["binary_chunks_unchanged"])
        self.assertEqual(read_single_animation(self.path)["animations"][0]["name"], "wing_fold")
        self.assertEqual(self.path.read_bytes()[-12:], original[-12:])
        self.assertEqual(struct.unpack_from("<I", self.path.read_bytes(), 8)[0],
                         self.path.stat().st_size)
        self.assertFalse(self.path.with_suffix(".glb.tmp").exists())

    def test_rejects_bad_names_bad_assets_and_multiple_clips(self):
        with self.assertRaises(ValueError):
            rewrite_single_animation_name(self.path, "../unsafe")
        self.path.write_bytes(glb_bytes(2))
        with self.assertRaisesRegex(ValueError, "single"):
            rewrite_single_animation_name(self.path, "good")
        self.path.write_bytes(b"not a glb")
        with self.assertRaisesRegex(ValueError, "glTF"):
            rewrite_single_animation_name(self.path, "good")

    def test_rejects_truncated_chunk(self):
        data = bytearray(glb_bytes())
        struct.pack_into("<I", data, 12, 9999)
        self.path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "chunk"):
            rewrite_single_animation_name(self.path, "walk")


if __name__ == "__main__":
    unittest.main()
