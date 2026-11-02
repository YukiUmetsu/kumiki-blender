"""The final reversible spatial-sculpt experiment is bounded and honest."""

import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "blender_liberty_3d_sculpt_study.py"
spec = importlib.util.spec_from_file_location("liberty_3d_sculpt_study", SCRIPT)
study = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = study
spec.loader.exec_module(study)


class Liberty3dSculptStudyTests(unittest.TestCase):
    def test_sculpt_falloff_is_smooth_and_bounded(self):
        falloff = study.smooth_falloff
        self.assertAlmostEqual(falloff(0, 1), 1)
        self.assertAlmostEqual(falloff(1, 1), 0)
        self.assertAlmostEqual(falloff(2, 1), 0)
        self.assertAlmostEqual(falloff(.5, 1), .5)
        self.assertGreater(falloff(.2, 1), falloff(.8, 1))
        for distance, radius in ((.2, 0), (math.nan, .7), (.3, math.inf)):
            with self.assertRaises(ValueError):
                falloff(distance, radius)

    def test_bilateral_strokes_are_gently_symmetric(self):
        strokes = study.STROKES
        self.assertEqual(len(strokes), 8)
        for left, right in ((strokes[0], strokes[1]),
                            (strokes[2], strokes[3]),
                            (strokes[6], strokes[7])):
            self.assertLess(left.x, 0)
            self.assertAlmostEqual(left.x, -right.x)
            self.assertAlmostEqual(left.z, right.z)
            self.assertAlmostEqual(left.dx, -right.dx)
            self.assertAlmostEqual(left.dy, right.dy)
            self.assertAlmostEqual(left.dz, right.dz)
            self.assertAlmostEqual(left.radius, right.radius)
        self.assertLessEqual(max(abs(stroke.dy) for stroke in strokes), .11)
        self.assertGreater(min(stroke.radius for stroke in strokes), .5)

    def test_model_keeps_original_source_as_reversible_shape_key(self):
        source = SCRIPT.read_text(encoding="utf8")
        self.assertIn('head.shape_key_add(name="r59 donor base / recoverable")', source)
        self.assertIn('head.shape_key_add(name="r60 inferred 3D anatomy volume study")', source)
        self.assertIn('scene["candidate_status"] = "HOLD', source)
        self.assertIn('"CC0 generic human anatomy; NOT a real Liberty scan"', source)
        self.assertIn('BVHTree.FromObject(head, depsgraph)', source)
        self.assertIn('if cfg.output.exists() and any(cfg.output.iterdir())', source)


if __name__ == "__main__":
    unittest.main()
