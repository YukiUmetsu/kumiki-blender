import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.asset_import import AssetImportError
from kumiki_blender.material_overrides import load_material_overrides, verify_material_overrides


def overrides_json(**changes):
    data = {
        "schema_version": 1,
        "overrides": [{"material_name": "Limestone", "preset": "natural-stone",
                       "color": "#C8BCA9", "roughness": 0.83}],
    }
    data.update(changes)
    return data


class MaterialOverrideTests(unittest.TestCase):
    def test_loads_and_normalizes_pbr_specs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overrides.json"
            path.write_text(json.dumps(overrides_json()))
            entries = load_material_overrides(path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].surface.texture_kind, "natural-stone")
            self.assertEqual(entries[0].to_manifest(), {
                "source_material": "Limestone", "preset": "natural-stone",
                "color": "#C8BCA9", "roughness": 0.83,
            })
            verify_material_overrides(["Limestone", "Copper"], entries)

    def test_rejects_unknown_ambiguous_and_missing_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overrides.json"
            path.write_text(json.dumps(overrides_json()))
            entries = load_material_overrides(path)
            for available in (["Sandstone"], [], ["Limestone", "Limestone"]):
                with self.subTest(available=available), self.assertRaisesRegex(AssetImportError, "matches"):
                    verify_material_overrides(available, entries)

    def test_rejects_malformed_duplicate_and_unsafe_specs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overrides.json"
            bad_specs = (
                "not json", "[]", json.dumps(overrides_json(schema_version=True)),
                json.dumps(overrides_json(overrides=[])),
                json.dumps(overrides_json(overrides=[{"material_name": "x", "preset": "missing"}])),
                json.dumps(overrides_json(overrides=[{"material_name": "Limestone", "preset": "natural-stone",
                                                             "roughness": 2.0}])),
                json.dumps(overrides_json(overrides=[{"material_name": " Limestone ", "preset": "natural-stone"}])),
                json.dumps(overrides_json(overrides=[{"material_name": "Limestone", "preset": "natural-stone"},
                                                             {"material_name": "Limestone", "preset": "natural-stone"}])),
                json.dumps(overrides_json(overrides=[{"material_name": "Limestone", "preset": "natural-stone",
                                                             "external_url": "https://example.com"}])),
            )
            for content in bad_specs:
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(AssetImportError):
                    load_material_overrides(path)

    def test_optional_weathering_strength_is_explicitly_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weathering.json"
            path.write_text(json.dumps(overrides_json(overrides=[
                {"material_name": "Stone", "preset": "weathered-stone",
                 "weathering_strength": 0.7},
            ])))
            entries = load_material_overrides(path)
            self.assertEqual(entries[0].to_manifest()["weathering_strength"], 0.7)
            path.write_text(json.dumps(overrides_json(overrides=[
                {"material_name": "Glass", "preset": "optical-glass",
                 "weathering_strength": 0.6},
            ])))
            with self.assertRaises(AssetImportError):
                load_material_overrides(path)


if __name__ == "__main__":
    unittest.main()
