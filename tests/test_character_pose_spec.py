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

    def test_pair_requires_two_distinct_meshes(self):
        case = copy.deepcopy(self.data)
        case["collision_pairs"] = [["Body", "Body"]]
        with self.assertRaisesRegex(ValueError, "two distinct"):
            self.load(case)


if __name__ == "__main__":
    unittest.main()
