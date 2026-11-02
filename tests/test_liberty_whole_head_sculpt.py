"""Offline contracts for the distinct, continuous Liberty sculpt-base trial."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "blender_liberty_whole_head_sculpt.py"
spec = importlib.util.spec_from_file_location("liberty_whole_head_sculpt", SCRIPT)
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


class LibertyWholeHeadSculptTests(unittest.TestCase):
    def test_source_is_one_complete_anatomical_head_with_both_eyes(self):
        self.assertEqual(trial.SOURCE_HEAD, "GEO-head_sculpting_realistic")
        self.assertEqual(len(trial.SOURCE_EYES), 4)
        self.assertEqual(len(set(trial.SOURCE_EYES)), 4)
        self.assertGreater(trial.SOURCE_CHIN_Z, 0)
        self.assertLess(trial.SOURCE_CRANIAL_WIDTH_Z[0],
                        trial.SOURCE_CRANIAL_WIDTH_Z[1])

    def test_front_threequarter_and_opposing_profiles_are_distinct(self):
        shots = dict(trial.SHOTS)
        self.assertEqual(list(shots), [
            "front", "threequarter", "profile", "opposite-profile"])
        self.assertEqual(len(set(shots.values())), 4)
        self.assertEqual(shots["profile"][0], -shots["opposite-profile"][0])
        self.assertEqual(shots["profile"][1:], shots["opposite-profile"][1:])

    def test_source_and_parent_transform_are_checked_and_preserved(self):
        source = SCRIPT.read_text(encoding="utf8")
        self.assertIn("verify_anatomy_source(cfg.source)", source)
        self.assertIn("source_matrices = {obj: obj.matrix_world.copy()", source)
        self.assertIn("obj.matrix_world = target_from_source @ source_matrices[obj]", source)
        self.assertIn("bpy.context.view_layer.update()", source)
        self.assertIn("cranium_width = max(cranial_section) - min(cranial_section)", source)
        self.assertIn("sz = target.height / (high.z - SOURCE_CHIN_Z)", source)
        self.assertIn("if cfg.output.exists() and any(cfg.output.iterdir())", source)

    def test_trial_does_not_claim_ancient_monument_scan_or_production_release(self):
        source = SCRIPT.read_text(encoding="utf8")
        self.assertIn('head["source_authenticity"] = "CC0 generic human anatomy;', source)
        self.assertIn('scene["candidate_status"] = "HOLD / not photo-faithful"', source)
        self.assertIn('head["depth_evidence"] = "UNKNOWN', source)
        self.assertNotIn("sculpt_classical_face", source)
        self.assertNotIn("conform_anatomical_face", source)


if __name__ == "__main__":
    unittest.main()
