import json
import struct
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.glb_audit import audit_glb
from kumiki_blender.revision_audit import RevisionAuditError, audit_revision


def create_revision(folder: Path) -> dict:
    (folder / "previews").mkdir()
    files = {
        "source.blend": b"BLENDER-native-source",
        "preview.png": b"\x89PNG\r\n\x1a\nhero",
        "previews/detail.png": b"\x89PNG\r\n\x1a\ndetail",
    }
    for name, data in files.items():
        (folder / name).write_bytes(data)
    gltf = {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}]}],
            "nodes": [{"name": "column", "extras": {"kumiki_part": "column"}}]}
    json_bytes = json.dumps(gltf).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)
    binary = struct.pack("<4sII", b"glTF", 2, 20 + len(json_bytes))
    binary += struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    (folder / "model.glb").write_bytes(binary)
    sizes = {name: len(data) for name, data in files.items()}
    sizes["model.glb"] = len(binary)
    manifest = {
        "schema_version": 1,
        "asset_id": "portico-study",
        "files": {"source": "source.blend", "model": "model.glb", "preview": "preview.png",
                  "quality_views": ["preview.png", "previews/detail.png"]},
        "output_bytes": sizes,
        "glb_audit": audit_glb(folder / "model.glb"),
        "named_parts": {"columns": 1},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class RevisionAuditTests(unittest.TestCase):
    def test_complete_revision_is_self_contained_and_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            create_revision(folder)
            report = audit_revision(folder)
            self.assertTrue(report["verified"])
            self.assertEqual(report["asset_id"], "portico-study")
            self.assertEqual(report["glb_audit"]["semantic_nodes"], 1)
            self.assertEqual(report["output_count"], 4)

    def test_tampering_after_manifest_generation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            create_revision(folder)
            (folder / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\nchanged")
            with self.assertRaisesRegex(RevisionAuditError, "size mismatch"):
                audit_revision(folder)

    def test_malformed_or_missing_manifest_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(RevisionAuditError):
                audit_revision(folder)
            (folder / "manifest.json").write_text("not JSON")
            with self.assertRaises(RevisionAuditError):
                audit_revision(folder)

    def test_manifest_path_traversal_or_missing_semantic_tags_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = create_revision(folder)
            manifest["files"]["model"] = "../model.glb"
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(RevisionAuditError):
                audit_revision(folder)
            manifest["files"]["model"] = "model.glb"
            manifest["glb_audit"]["semantic_nodes"] = 0
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RevisionAuditError, "GLB audit"):
                audit_revision(folder)

    def test_detailed_multi_angle_reviews_allow_32_views_but_not_unbounded_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = create_revision(folder)
            for number in range(2, 32):
                name = f"previews/angle-{number:02d}.png"
                data = b"\x89PNG\r\n\x1a\nview" + bytes([number])
                (folder / name).write_bytes(data)
                manifest["files"]["quality_views"].append(name)
                manifest["output_bytes"][name] = len(data)
            (folder / "manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(audit_revision(folder)["verified"])
            manifest["files"]["quality_views"].append("previews/angle-32.png")
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RevisionAuditError, "invalid source/model/preview"):
                audit_revision(folder)


if __name__ == "__main__":
    unittest.main()
