"""Pure Python tests of source-hashed pre-rig hinge topology planning."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.test_character_rig_controls import rig_fixture
from kumiki_blender.character_build.topology_zones import save_topology_plan
from kumiki_blender.character_build.joint_refine_plan import (
    load_joint_refine_plan, make_joint_refine_plan, save_joint_refine_plan,
    validate_joint_refine_plan,
)


class HingeRefinePlanTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source.blend"
        self.source.write_bytes(b"BLENDER-pre-rig")
        self.joints = self.root / "joints.json"
        self.joints.write_text(json.dumps(rig_fixture()))
        save_topology_plan(self.joints, self.root / "topology",
                           mesh="Neck", joints=["neck"])
        self.topo = self.root / "topology/topology-plan.json"

    def test_immutable_and_source_locked(self):
        saved = save_joint_refine_plan(self.source, self.topo, ["neck"],
                                       self.root / "plan", cuts=2)
        self.assertEqual(saved["zones"], ["neck"])
        self.assertEqual(saved["mesh"], "Neck")
        self.assertEqual(load_joint_refine_plan(self.root / "plan/joint-refine-plan.json"), saved)
        with self.assertRaises(FileExistsError):
            save_joint_refine_plan(self.source, self.topo, ["neck"], self.root / "plan")
        self.source.write_bytes(b"BLENDER-pre-rig-mutated")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_joint_refine_plan(self.root / "plan/joint-refine-plan.json")

    def test_rejects_unknown_zones_and_unbounded_values(self):
        with self.assertRaisesRegex(ValueError, "reviewed"):
            make_joint_refine_plan(self.source, self.topo, ["elbow.L"])
        with self.assertRaisesRegex(ValueError, "1 to 4"):
            make_joint_refine_plan(self.source, self.topo, ["neck"], cuts=8)
        with self.assertRaisesRegex(ValueError, "maximum|100000"):
            make_joint_refine_plan(self.source, self.topo, ["neck"], max_added_vertices=200000)
        with self.assertRaisesRegex(ValueError, "axial_alignment"):
            make_joint_refine_plan(self.source, self.topo, ["neck"], axial_alignment_min=.01)

    def test_rejects_mutated_plan(self):
        saved = save_joint_refine_plan(self.source, self.topo, ["neck"], self.root / "plan")
        saved["mesh"] = "OtherObject"
        path = self.root / "plan/joint-refine-plan.json"
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "changed"):
            load_joint_refine_plan(path)


if __name__ == "__main__":
    unittest.main()
