"""Pure-Python validation for Blender region-local sculpt operation plans."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.region_sculpt_plan import load_region_sculpt_plan


class SculptPlanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sculpt.json"
        self.data = {
            "schema_version": 1,
            "object": "Eye.L",
            "max_displacement_m": 0.01,
            "expected_vertex_count": 100,
            "operations": [{"center_m": [0, -0.05, .2], "radius_m": .02,
                            "move_m": [0, -.005, .002], "falloff": "smoothstep"}],
        }

    def load(self, data=None):
        self.path.write_text(json.dumps(data or self.data))
        return load_region_sculpt_plan(self.path)

    def test_valid_plan_and_boundaries(self):
        self.assertEqual(self.load()["object"], "Eye.L")

    def test_reject_unsafe_and_nonfinite(self):
        for field, value in [("object", "../../unsafe"), ("expected_vertex_count", 0),
                             ("max_displacement_m", float("inf")), ("operations", [])]:
            case = copy.deepcopy(self.data)
            case[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(case)

    def test_reject_too_large_motion(self):
        case = copy.deepcopy(self.data)
        case["operations"][0]["move_m"] = [0, 0, .04]
        with self.assertRaisesRegex(ValueError, "budget"):
            self.load(case)

    def test_reject_bad_falloff_and_radius(self):
        for field, value in [("falloff", "arbitrary-python"), ("radius_m", -1),
                             ("move_m", [float("nan"), 0, 0])]:
            case = copy.deepcopy(self.data)
            case["operations"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(case)


if __name__ == "__main__":
    unittest.main()
