"""Head sculpt inspection cameras expose both depth-bearing profiles."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "blender_liberty_face_clay.py"
spec = importlib.util.spec_from_file_location("liberty_face_clay_views", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LibertyClayCameraTests(unittest.TestCase):
    def test_all_four_important_views_are_persistent_and_unique(self):
        shots = dict(module.FIXED_CLAY_SHOTS)
        self.assertEqual(tuple(shots), (
            "front", "threequarter", "profile", "opposite-profile"))
        self.assertEqual(len(set(shots.values())), 4)

    def test_profiles_are_opposite_not_duplicate_views(self):
        shots = dict(module.FIXED_CLAY_SHOTS)
        left = shots["profile"]
        right = shots["opposite-profile"]
        self.assertGreater(left[0], 0)
        self.assertLess(right[0], 0)
        self.assertEqual(left[0], -right[0])
        self.assertEqual(left[1:], right[1:])


if __name__ == "__main__":
    unittest.main()
