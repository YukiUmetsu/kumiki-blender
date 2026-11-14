"""No Blender needed: validate explicit, bounded deformation test plans."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.pose_spec import load_pose_plan


class PoseSpecTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "poses.json"
        self.data = {
            "schema_version": 1,
            "armature": "CharacterRig",
            "poses": [
                {"name": "head-turn", "bones": {"head": {"axis": "Z", "degrees": 30}}},
                {"name": "tail-flex", "bones": {"tail1": {"axis": "X", "degrees": -45}}},
            ],
            "max_p95_edge_stretch": 0.5,
            "collision_pairs": [["Tail", "Body"]],
            "render": False,
        }

    def load(self, data=None):
        self.path.write_text(json.dumps(data if data is not None else self.data))
        return load_pose_plan(self.path)

    def test_accepts_bounded_named_pose_battery(self):
        self.assertEqual(len(self.load()["poses"]), 2)

    def test_rejects_unknown_or_malformed_plans(self):
        for key, value in [("schema_version", 2), ("armature", "../unsafe"),
                           ("poses", []), ("render", 1),
                           ("max_p95_edge_stretch", float("nan"))]:
            with self.subTest(key=key):
                case = copy.deepcopy(self.data)
                case[key] = value
                with self.assertRaises((ValueError, TypeError)):
                    self.load(case)

    def test_duplicate_poses_invalid_axis_and_angle(self):
        case = copy.deepcopy(self.data)
        case["poses"][1]["name"] = "head-turn"
        with self.assertRaisesRegex(ValueError, "unique"):
            self.load(case)
        case = copy.deepcopy(self.data)
        case["poses"][0]["bones"]["head"]["axis"] = "P"
        with self.assertRaisesRegex(ValueError, "axis"):
            self.load(case)
        case = copy.deepcopy(self.data)
        case["poses"][0]["bones"]["head"]["degrees"] = 360
        with self.assertRaisesRegex(ValueError, "180"):
            self.load(case)

    def test_optional_evaluated_pose_and_new_contact_gates(self):
        case = copy.deepcopy(self.data)
        case["min_pose_displacement_m"] = .0005
        case["max_new_collision_pairs"] = 0
        self.assertEqual(self.load(case)["max_new_collision_pairs"], 0)
        for key, value in [
            ("min_pose_displacement_m", 0),
            ("min_pose_displacement_m", float("inf")),
            ("min_pose_displacement_m", True),
            ("max_new_collision_pairs", -1),
            ("max_new_collision_pairs", True),
            ("max_new_collision_pairs", 1.5),
        ]:
            with self.subTest(key=key, value=value):
                invalid = copy.deepcopy(case)
                invalid[key] = value
                with self.assertRaises(ValueError):
                    self.load(invalid)
        case["collision_pairs"] = []
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            self.load(case)

    def test_explicit_semantic_skin_weight_zones(self):
        base = copy.deepcopy(self.data)
        base["weight_zones"] = [{
            "mesh": "Wing.L",
            "region_group": "REGION_WingLeft",
            "allowed_bones": ["upper_wing.L", "fore_wing.L"],
            "max_outside_weight": 0.02,
            "min_vertices": 5,
        }]
        self.assertEqual(self.load(base)["weight_zones"][0]["min_vertices"], 5)
        variants = [
            ("duplicate zone", base["weight_zones"] * 2, "Duplicate"),
            ("empty allowed", [{**base["weight_zones"][0], "allowed_bones": []}], "allowed_bones"),
            ("unknown extra", [{**base["weight_zones"][0], "typo": True}], "weight_zones entries"),
            ("negative foreign", [{**base["weight_zones"][0], "max_outside_weight": -.1}], "max_outside_weight"),
            ("NaN foreign", [{**base["weight_zones"][0], "max_outside_weight": float("nan")}], "max_outside_weight"),
            ("invalid count", [{**base["weight_zones"][0], "min_vertices": True}], "min_vertices"),
            ("unsafe group", [{**base["weight_zones"][0], "region_group": "../unsafe"}], "region_group"),
        ]
        for label, values, error in variants:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, error):
                bad = copy.deepcopy(base)
                bad["weight_zones"] = values
                self.load(bad)

    def test_pair_requires_two_distinct_meshes(self):
        case = copy.deepcopy(self.data)
        case["collision_pairs"] = [["Body", "Body"]]
        with self.assertRaisesRegex(ValueError, "two distinct"):
            self.load(case)


if __name__ == "__main__":
    unittest.main()
