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

    def test_source_locked_geodesic_plan_with_protected_vertices(self):
        case = copy.deepcopy(self.data)
        case.update({
            "expected_mesh_sha256": "a" * 64,
            "protected_vertex_indices": [5, 6],
        })
        case["operations"][0].update({
            "distance_mode": "geodesic", "seed_vertex": 12,
            "max_seed_distance_m": .003,
        })
        validated = self.load(case)
        self.assertEqual(validated["operations"][0]["seed_vertex"], 12)
        self.assertEqual(validated["protected_vertex_indices"], [5, 6])

    def test_geodesic_requires_stable_source_geometry_and_seed(self):
        case = copy.deepcopy(self.data)
        case["operations"][0].update({
            "distance_mode": "geodesic", "seed_vertex": 12,
            "max_seed_distance_m": .003,
        })
        with self.assertRaisesRegex(ValueError, "expected_mesh_sha256"):
            self.load(case)
        case["expected_mesh_sha256"] = "b" * 64
        for changes, expected in [
            ({"seed_vertex": True}, "seed_vertex"),
            ({"seed_vertex": -1}, "seed_vertex"),
            ({"seed_vertex": 100}, "seed_vertex"),
            ({"max_seed_distance_m": .04}, "max_seed_distance_m"),
            ({"max_seed_distance_m": float("nan")}, "max_seed_distance_m"),
            ({"distance_mode": "invalid"}, "distance_mode"),
        ]:
            with self.subTest(changes=changes):
                invalid = copy.deepcopy(case)
                invalid["operations"][0].update(changes)
                with self.assertRaisesRegex(ValueError, expected):
                    self.load(invalid)
        invalid = copy.deepcopy(case)
        invalid["protected_vertex_indices"] = [12]
        with self.assertRaisesRegex(ValueError, "protected"):
            self.load(invalid)
        invalid = copy.deepcopy(case)
        invalid["expected_mesh_sha256"] = "untrusted"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.load(invalid)

    def test_legacy_euclidean_plan_remains_compatible(self):
        self.assertEqual(self.load(), self.data)
        case = copy.deepcopy(self.data)
        case["operations"][0]["distance_mode"] = "euclidean"
        self.assertEqual(self.load(case)["operations"][0]["distance_mode"], "euclidean")
        case["operations"][0]["seed_vertex"] = 12
        with self.assertRaisesRegex(ValueError, "require distance_mode"):
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
