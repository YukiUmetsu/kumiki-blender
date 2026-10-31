import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.asset_import import AssetImportError
from kumiki_blender.material_packs import load_texture_packs, verify_texture_targets
from kumiki_blender.procedural_pbr import make_pbr_maps


def sample(directory: Path) -> tuple[Path, dict]:
    tiles = directory / "textures"
    tiles.mkdir()
    maps = make_pbr_maps("natural-stone", "#BBB3A5", 0.77, resolution=16)
    files = {"base_color": maps.base_color_png, "roughness": maps.roughness_png,
             "normal": maps.normal_png}
    for channel, payload in files.items():
        (tiles / f"{channel}.png").write_bytes(payload)
    data = {
        "schema_version": 1,
        "packs": [{
            "id": "limestone", "creator": "Example Author",
            "source_url": "https://example.com/materials/limestone",
            "license": "CC-BY-4.0", "attribution": "Example Author / Limestone",
            "redistribution_allowed": True, "tile_size_m": 1.7,
            "maps": {name: f"textures/{name}.png" for name in files},
            "sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
        }],
        "assignments": [{"target": "wall-stone", "pack": "limestone"}],
    }
    path = directory / "pack.json"
    path.write_text(json.dumps(data))
    return path, data


class MaterialPackTests(unittest.TestCase):
    def test_loads_verified_pack_with_provenance_and_export_manifest(self):
        with tempfile.TemporaryDirectory() as dirname:
            path, _ = sample(Path(dirname))
            assignments = load_texture_packs(path)
            self.assertEqual(len(assignments), 1)
            self.assertEqual(assignments[0].pack.tile_size_m, 1.7)
            self.assertEqual(len(assignments[0].pack.maps), 3)
            manifest = assignments[0].to_manifest()
            self.assertEqual(manifest["pack"]["license"], "CC-BY-4.0")
            self.assertEqual(len(manifest["pack"]["maps"]["normal"]["sha256"]), 64)
            self.assertNotIn(dirname, json.dumps(manifest))
            verify_texture_targets(["wall-stone", "trim-stone"], assignments)

    def test_rejects_changed_or_unavailable_files(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            path, data = sample(root)
            (root / "textures/base_color.png").write_bytes(b"broken png")
            with self.assertRaisesRegex(AssetImportError, "bytes|PNG|pin"):
                load_texture_packs(path)
            fresh = make_pbr_maps("natural-stone", "#BBB3A5", 0.77, resolution=16)
            (root / "textures/base_color.png").write_bytes(fresh.base_color_png)
            path.write_text(json.dumps(data))
            (root / "textures/normal.png").unlink()
            with self.assertRaisesRegex(AssetImportError, "Cannot read"):
                load_texture_packs(path)

    def test_rejects_unsafe_paths_and_symlinks(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            path, data = sample(root)
            for candidate in ("../secret.png", "/tmp/stolen.png", "a\\b.png"):
                data["packs"][0]["maps"]["normal"] = candidate
                path.write_text(json.dumps(data))
                with self.subTest(candidate=candidate), self.assertRaisesRegex(AssetImportError, "relative path"):
                    load_texture_packs(path)
            data["packs"][0]["maps"]["normal"] = "textures/alias.png"
            (root / "textures/alias.png").symlink_to(root / "textures/normal.png")
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(AssetImportError, "symlink"):
                load_texture_packs(path)

    def test_rejects_bogus_rights_and_formats(self):
        with tempfile.TemporaryDirectory() as dirname:
            path, original = sample(Path(dirname))
            mutations = (
                ("redistribution_allowed", False),
                ("source_url", "http://example.com/stone"),
                ("source_url", "https://example.com/stone?token=secret"),
                ("attribution", ""),
                ("normal_convention", "directx"),
                ("tile_size_m", 0),
                ("metallic", 2),
                ("normal_strength", float("nan")),
            )
            for key, value in mutations:
                data = json.loads(json.dumps(original))
                data["packs"][0][key] = value
                path.write_text(json.dumps(data))
                with self.subTest(key=key), self.assertRaises(AssetImportError):
                    load_texture_packs(path)

    def test_rejects_duplicate_unresolved_and_ambiguous_targets(self):
        with tempfile.TemporaryDirectory() as dirname:
            path, data = sample(Path(dirname))
            assignment = load_texture_packs(path)
            for names in ([], ["wall-stone", "wall-stone"]):
                with self.assertRaisesRegex(AssetImportError, "matches"):
                    verify_texture_targets(names, assignment)
            data["assignments"].append(data["assignments"][0])
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(AssetImportError, "Duplicate"):
                load_texture_packs(path)

    def test_rejects_surplus_unknown_fields_and_partial_maps(self):
        with tempfile.TemporaryDirectory() as dirname:
            path, original = sample(Path(dirname))
            cases = (
                {"packs": [dict(original["packs"][0], token="secret")]},
                {"packs": [dict(original["packs"][0], maps={"base_color": "textures/base_color.png"})]},
                {"packs": [dict(original["packs"][0], sha256={"base_color": "bad"})]},
                {"assignments": [{"target": "wall-stone", "pack": "missing"}]},
                {"assignments": [{"target": "wall-stone", "pack": "limestone", "foo": "bar"}]},
            )
            for change in cases:
                data = dict(original, **change)
                path.write_text(json.dumps(data))
                with self.subTest(change=change), self.assertRaises(AssetImportError):
                    load_texture_packs(path)


if __name__ == "__main__":
    unittest.main()
