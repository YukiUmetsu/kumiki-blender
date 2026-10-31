import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.asset_import import (
    AssetImportError, inspect_source_glb, load_asset_provenance,
)


def fixture_glb(path, *, animations=False, meshes=True):
    data = {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}]}],
            "nodes": [{"mesh": 0, "name": "Stone"}] if meshes else [{"name": "Empty"}]}
    if animations:
        data["animations"] = [{"channels": [], "samplers": []}]
    chunk = json.dumps(data).encode("utf-8")
    chunk += b" " * (-len(chunk) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 20 + len(chunk))
                     + struct.pack("<I4s", len(chunk), b"JSON") + chunk)


def provenance_json(**changes):
    data = {
        "schema_version": 1,
        "asset_id": "sample-portico",
        "creator": "Kumiki author",
        "source_url": "https://github.com/YukiUmetsu/kumiki-blender",
        "license": "MIT",
        "attribution": "Original procedural study",
        "redistribution_allowed": True,
    }
    data.update(changes)
    return data


class AssetImportTests(unittest.TestCase):
    def test_accepts_explicit_provenance_and_checks_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.glb"
            fixture_glb(model)
            sha = hashlib.sha256(model.read_bytes()).hexdigest()
            recipe = root / "asset.json"
            recipe.write_text(json.dumps(provenance_json(expected_sha256=sha)))
            provenance = load_asset_provenance(recipe)
            report = inspect_source_glb(model, provenance)
            self.assertEqual(report["input_sha256"], sha)
            self.assertEqual(report["glb_audit"]["meshes"], 1)
            self.assertTrue(report["static_mesh_only"])
            self.assertEqual(provenance.to_manifest(sha)["license"], "MIT")

    def test_missing_unexpected_or_malformed_provenance_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "asset.json"
            for content in (
                "not json", "[]", json.dumps(provenance_json(schema_version=True)),
                json.dumps(provenance_json(typo="unexpected")),
                json.dumps({key: value for key, value in provenance_json().items() if key != "creator"}),
            ):
                recipe.write_text(content)
                with self.subTest(content=content), self.assertRaises(AssetImportError):
                    load_asset_provenance(recipe)

    def test_rights_and_attribution_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "asset.json"
            cases = (
                {"redistribution_allowed": False},
                {"redistribution_allowed": "true"},
                {"license": "CC-BY-4.0", "attribution": ""},
                {"source_url": "http://example.com/asset"},
                {"source_url": "https://user:secret@example.com/asset"},
                {"source_url": "https://example.com/asset#fragment"},
                {"source_url": "https://example.com/asset?token=secret"},
                {"asset_id": "../../outside"},
                {"expected_sha256": "xyz"},
            )
            for changes in cases:
                recipe.write_text(json.dumps(provenance_json(**changes)))
                with self.subTest(changes=changes), self.assertRaises(AssetImportError):
                    load_asset_provenance(recipe)

    def test_rejects_unpinned_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.glb"
            fixture_glb(model)
            recipe = root / "asset.json"
            recipe.write_text(json.dumps(provenance_json(expected_sha256="0" * 64)))
            with self.assertRaisesRegex(AssetImportError, "SHA-256"):
                inspect_source_glb(model, load_asset_provenance(recipe))

    def test_rejects_animated_models_missing_mesh_nodes_and_wrong_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.glb"
            fixture_glb(model, animations=True)
            with self.assertRaisesRegex(AssetImportError, "Animated"):
                inspect_source_glb(model)
            fixture_glb(model, meshes=False)
            with self.assertRaisesRegex(AssetImportError, "no instanced mesh"):
                inspect_source_glb(model)
            model.rename(root / "model.obj")
            with self.assertRaisesRegex(AssetImportError, r"\.glb"):
                inspect_source_glb(root / "model.obj")


if __name__ == "__main__":
    unittest.main()
