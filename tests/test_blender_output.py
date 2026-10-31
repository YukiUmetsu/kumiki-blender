import contextlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kumiki_blender.blender_output import create_model_scene, export_revision
from kumiki_blender.shots import QualityShot


class BlenderOutputTests(unittest.TestCase):
    def test_background_uses_active_factory_scene_and_hides_startup_objects(self):
        starter = SimpleNamespace(hide_render=False, hide_set=Mock())
        scene = SimpleNamespace(name="Scene", objects=[starter])
        bpy = SimpleNamespace(
            app=SimpleNamespace(background=True),
            data=SimpleNamespace(filepath="", scenes=SimpleNamespace(new=Mock())),
            context=SimpleNamespace(scene=scene),
        )
        self.assertIs(create_model_scene(bpy, "study"), scene)
        self.assertEqual(scene.name, "Kumiki / study")
        self.assertTrue(starter.hide_render)
        starter.hide_set.assert_called_once_with(True)
        bpy.data.scenes.new.assert_not_called()

    def test_background_rejects_loaded_user_file_before_mutating_scene(self):
        scene = SimpleNamespace(name="User work", objects=[])
        bpy = SimpleNamespace(
            app=SimpleNamespace(background=True),
            data=SimpleNamespace(filepath="user.blend"),
            context=SimpleNamespace(scene=scene),
        )
        with self.assertRaisesRegex(RuntimeError, "factory-startup"):
            create_model_scene(bpy, "study")
        self.assertEqual(scene.name, "User work")

    def test_interactive_mode_creates_separate_scene(self):
        scene = object()
        window = SimpleNamespace(scene=None)
        bpy = SimpleNamespace(
            app=SimpleNamespace(background=False),
            data=SimpleNamespace(scenes=SimpleNamespace(new=Mock(return_value=scene))),
            context=SimpleNamespace(window=window),
        )
        self.assertIs(create_model_scene(bpy, "study"), scene)
        bpy.data.scenes.new.assert_called_once_with("Kumiki / study")
        self.assertIs(window.scene, scene)

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
            with patch("kumiki_blender.blender_output.aim_at"), \
                    patch("kumiki_blender.blender_output.bake_export_meshes") as bake:
                manifest = export_revision(bpy, scene, camera, [model], shots, output,
                                           samples=32, metadata={"asset_id": "test-study"})
            self.assertEqual(manifest["files"]["quality_views"], ["preview.png", "previews/front.png"])
            self.assertEqual(manifest["glb_audit"]["semantic_nodes"], 1)
            bake.assert_called_once_with(bpy, [model])
            self.assertFalse(export_options["export_apply"], "Modifiers were baked into temporary export geometry")
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
