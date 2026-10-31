import contextlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kumiki_blender.blender_output import export_revision
from kumiki_blender.shots import QualityShot


class BlenderOutputTests(unittest.TestCase):
    def test_shared_exporter_preserves_native_source_and_embeds_modifiers_in_glb(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "r01"
            render = SimpleNamespace(filepath="")
            scene = SimpleNamespace(render=render, cycles=SimpleNamespace(samples=32),
                                    view_layers=[object()], name="Test scene", camera=None)
            camera = SimpleNamespace(location=None, data=SimpleNamespace(ortho_scale=None))
            model = SimpleNamespace(select_set=Mock())
            export_options = {}

            def save_as_mainfile(*, filepath):
                Path(filepath).write_bytes(b"BLENDER-FAKE-CONTENT")

            def export_gltf(**options):
                export_options.update(options)
                data = {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}]}],
                        "nodes": [{"extras": {"kumiki_part": "arch-stone"}}]}
                chunk = json.dumps(data).encode()
                chunk += b" " * (-len(chunk) % 4)
                Path(options["filepath"]).write_bytes(
                    struct.pack("<4sII", b"glTF", 2, 20 + len(chunk)) +
                    struct.pack("<I4s", len(chunk), b"JSON") + chunk)

            def write_render(**_):
                Path(render.filepath).write_bytes(b"\x89PNG\r\n\x1a\n" + b"FAKE-IMAGE")

            bpy = SimpleNamespace(
                app=SimpleNamespace(version_string="4.2-test"),
                context=SimpleNamespace(temp_override=lambda **_: contextlib.nullcontext()),
                ops=SimpleNamespace(
                    wm=SimpleNamespace(save_as_mainfile=save_as_mainfile),
                    object=SimpleNamespace(select_all=Mock()),
                    export_scene=SimpleNamespace(gltf=export_gltf),
                    render=SimpleNamespace(render=write_render),
                ),
            )
            shots = (QualityShot("hero", (8, -12, 9), (0, 0, 3), 12),
                     QualityShot("front", (0, -20, 4), (0, 0, 3), 10))
            with patch("kumiki_blender.blender_output.aim_at"):
                manifest = export_revision(bpy, scene, camera, [model], shots, output,
                                           samples=32, metadata={"asset_id": "test-study"})
            self.assertEqual(manifest["files"]["quality_views"], ["preview.png", "previews/front.png"])
            self.assertEqual(manifest["glb_audit"]["semantic_nodes"], 1)
            self.assertTrue(export_options["export_apply"], "Must bake bevel/weighted-normal modifiers to GLB")
            self.assertTrue(export_options["export_extras"])
            self.assertEqual(render.filepath, str(output / "previews/front.png"))
            self.assertTrue((output / "manifest.json").is_file())
            model.select_set.assert_called_once()

    def test_existing_output_or_invalid_shots_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                export_revision(None, None, None, [object()],
                                [QualityShot("hero", (0, 0, 0), (1, 0, 0), 2)],
                                existing, samples=8, metadata={})
            fresh = root / "fresh"
            with self.assertRaisesRegex(RuntimeError, "hero"):
                export_revision(None, None, None, [object()],
                                [QualityShot("detail", (0, 0, 0), (1, 0, 0), 2)],
                                fresh, samples=8, metadata={})
            self.assertFalse(fresh.exists())


if __name__ == "__main__":
    unittest.main()
