"""Topology review plan tests: registered joint planes and strict input limits."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_character_rig_controls import rig_fixture
from kumiki_blender.character_build.topology_zones import (
    load_topology_plan, make_topology_plan, save_topology_plan, validate_topology_plan,
)


class TopologyZonesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.joints = self.root / "joints.json"
        self.joints.write_text(json.dumps(rig_fixture(with_tail=True)), encoding="utf-8")

    def test_default_zones_and_immutable_plan(self):
        plan = save_topology_plan(self.joints, self.root / "topology")
        ids = {z["id"] for z in plan["zones"]}
        self.assertTrue({"neck", "hips", "elbow.L", "knee.R", "tail0"} <= ids)
        self.assertEqual(load_topology_plan(self.root / "topology" / "topology-plan.json"), plan)
        self.assertTrue(all(z["half_width_m"] > 0 for z in plan["zones"]))
        with self.assertRaises(FileExistsError):
            save_topology_plan(self.joints, self.root / "topology")

    def test_explicit_joint_selection(self):
        plan = make_topology_plan(self.joints, joints=["elbow.L", "tail0"],
                                  min_coverage=.75, min_edges=12, strict=True)
        self.assertEqual([z["id"] for z in plan["zones"]], ["elbow.L", "tail0"])
        self.assertTrue(plan["strict"])
        self.assertEqual(plan["min_crossing_edges"], 12)

    def test_missing_duplicate_and_invalid_inputs(self):
        for values in (["nose"], ["elbow.L", "elbow.L"], [], ["../unsafe"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                make_topology_plan(self.joints, joints=values)
        with self.assertRaises(ValueError):
            make_topology_plan(self.joints, min_coverage=float("nan"))
        with self.assertRaises(ValueError):
            make_topology_plan(self.joints, min_edges=1)

    def test_source_integrity_and_zone_unit_axes(self):
        save_topology_plan(self.joints, self.root / "plan")
        plan_file = self.root / "plan" / "topology-plan.json"
        modified = json.loads(plan_file.read_text())
        modified["zones"][0]["axis"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "unit"):
            validate_topology_plan(modified)
        self.joints.write_text(self.joints.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_topology_plan(plan_file)

    def test_refuses_nonnumeric_joint_positions(self):
        data = rig_fixture()
        data["joints"]["elbow.L"] = [float("nan"), 0, 0]
        self.joints.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "finite"):
            make_topology_plan(self.joints, joints=["elbow.L"])


if __name__ == "__main__":
    unittest.main()
