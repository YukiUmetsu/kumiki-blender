"""Non-Blender guardrails for the reversible whole-head sculpt integration."""

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_liberty_sculpt_integration.py"
spec = importlib.util.spec_from_file_location("liberty_sculpt_integration", SCRIPT)
integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integration)


class LibertySculptIntegrationTests(unittest.TestCase):
    def test_sculpt_objects_are_existing_head_and_eyes(self):
        self.assertEqual(len(integration.SOURCE_OBJECTS), 5)
        self.assertEqual(len(set(integration.SOURCE_OBJECTS)), 5)
        self.assertEqual(integration.HEAD_NAME, integration.SOURCE_OBJECTS[0])
        self.assertTrue(all("sculpt candidate /" in name for name in integration.SOURCE_OBJECTS))

    def test_target_photo_boards_are_included_without_camera_registration(self):
        self.assertEqual(integration.REFERENCE_SCENES,
                         ("02 HAER POSES / UNREGISTERED", "03 HAER ANATOMY / UNREGISTERED"))
        source = SCRIPT.read_text(encoding="utf8")
        self.assertIn("imported.scenes = list(REFERENCE_SCENES)", source)
        self.assertIn("photo contact board / pose unknown", source)
        self.assertIn("Photo board provenance or unregistered status was lost", source)

    def test_uniform_head_envelope_preserves_depth_ratio(self):
        self.assertAlmostEqual(integration.uniform_head_scale(3.011, 5.677, 3.05, 5.26),
                               3.05/3.011)
        with self.assertRaisesRegex(ValueError, "nonuniform anatomical warp prohibited"):
            integration.uniform_head_scale(2.05, 3.53, 3.05, 2.20)
        with self.assertRaisesRegex(ValueError, "positive metres"):
            integration.uniform_head_scale(0., 3.53, 3.05, 5.26)

    def test_existing_sculpt_only_no_procedural_anatomy_or_scan_reuse(self):
        source = SCRIPT.read_text(encoding="utf8")
        self.assertIn("bpy.data.libraries.load(source_filepath, link=False)", source)
        self.assertIn("obj.hide_render = True", source)
        self.assertIn("obj.hide_set(True)", source)
        self.assertIn("obj.matrix_world = placement @ obj.matrix_world", source)
        self.assertIn("m.type == \"MULTIRES\"", source)
        self.assertIn("HOLD / CC0 anatomical donor", source)
        self.assertNotIn("Gaussian", source)
        self.assertNotIn("sculpt_classical_face", source)
        self.assertNotIn("mesh.from_pydata", source)


if __name__ == "__main__":
    unittest.main()
