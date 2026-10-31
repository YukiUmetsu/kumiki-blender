import json
import struct
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.glb_audit import AuditError, audit_glb


def write_glb(path, document, binary=b"\x00\x00\x00\x00"):
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    data = struct.pack("<I4s", len(payload), b"JSON") + payload
    data += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(data)) + data)


class GlbAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "model.glb"
        self.model = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0]}],
            "nodes": [{"name": "leg-0-00", "mesh": 0, "extras": {"kumiki_part": "leg"}}],
            "meshes": [{"name": "leg-0-00", "primitives": [{"attributes": {"POSITION": 0}}]}],
            "materials": [{"name": "steel"}],
        }

    def test_valid_embedded_glb_reports_counts(self):
        write_glb(self.path, self.model)
        summary = audit_glb(self.path)
        self.assertEqual(summary["meshes"], 1)
        self.assertEqual(summary["semantic_nodes"], 1)
        self.assertEqual(summary["external_references"], 0)

    def test_truncated_glb_is_rejected(self):
        write_glb(self.path, self.model)
        self.path.write_bytes(self.path.read_bytes()[:-3])
        with self.assertRaisesRegex(AuditError, "length"):
            audit_glb(self.path)

    def test_external_texture_reference_is_rejected(self):
        self.model["images"] = [{"uri": "../../private/texture.png"}]
        write_glb(self.path, self.model)
        with self.assertRaisesRegex(AuditError, "external"):
            audit_glb(self.path)

    def test_missing_meshes_is_rejected(self):
        self.model["meshes"] = []
        write_glb(self.path, self.model)
        with self.assertRaisesRegex(AuditError, "mesh"):
            audit_glb(self.path)

    def test_malformed_image_collection_is_rejected_as_audit_error(self):
        self.model["images"] = None
        write_glb(self.path, self.model)
        with self.assertRaisesRegex(AuditError, "image"):
            audit_glb(self.path)

    def test_malformed_mesh_primitives_are_rejected(self):
        self.model["meshes"][0]["primitives"] = "pretend-mesh"
        write_glb(self.path, self.model)
        with self.assertRaisesRegex(AuditError, "primitives"):
            audit_glb(self.path)

    def test_malformed_glb_magic_is_rejected(self):
        self.path.write_bytes(b"not-a-glb")
        with self.assertRaises(AuditError):
            audit_glb(self.path)


if __name__ == "__main__":
    unittest.main()
