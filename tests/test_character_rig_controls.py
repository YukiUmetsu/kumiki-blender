"""Synthetic rig tests: profile-aware IK/FK plans with reviewed joint coordinates."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.rig_control_plan import (
    build_control_plan, load_control_plan, save_control_plan, validate_control_plan,
)


def rig_fixture(limbs="arms", with_tail=False):
    joints = {
        "root": [0, 0, 0.015], "hips": [0, 0, .11], "chest": [0, 0, .20],
        "neck": [0, 0, .245], "head_top": [0, 0, .33],
    }
    bones = [
        {"name": "root", "head": "root", "tail": "hips", "parent": None},
        {"name": "spine", "head": "hips", "tail": "chest", "parent": "root"},
        {"name": "head", "head": "neck", "tail": "head_top", "parent": "spine"},
    ]
    for side, sign in (("L", 1), ("R", -1)):
        joints.update({
            f"hip.{side}": [sign * .025, 0, .11],
            f"knee.{side}": [sign * .035, -.008, .065],
            f"ankle.{side}": [sign * .035, .003, .015],
            f"shoulder.{side}": [sign * .06, 0, .20],
            f"elbow.{side}": [sign * .11, 0, .16],
            f"hand.{side}": [sign * .16, 0, .13],
        })
        bones.extend([
            {"name": f"thigh.{side}", "head": f"hip.{side}",
             "tail": f"knee.{side}", "parent": "root"},
            {"name": f"shin.{side}", "head": f"knee.{side}",
             "tail": f"ankle.{side}", "parent": f"thigh.{side}"},
            {"name": f"upper_{limbs[:-1]}.{side}", "head": f"shoulder.{side}",
             "tail": f"elbow.{side}", "parent": "spine"},
            {"name": f"fore_{limbs[:-1]}.{side}", "head": f"elbow.{side}",
             "tail": f"hand.{side}", "parent": f"upper_{limbs[:-1]}.{side}"},
        ])
    if with_tail:
        joints.update({"tail0": [0, .015, .10], "tail1": [0, .04, .13],
                       "tail2": [0, .07, .16], "tail3": [0, .10, .18]})
        for i in range(1, 4):
            bones.append({"name": f"tail{i}", "head": f"tail{i-1}",
                          "tail": f"tail{i}", "parent": f"tail{i-1}" if i > 1 else "root"})
    return {"schema_version": 1, "height_m": .35, "joints": joints, "bones": bones}


class RigControlPlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "joints.json"

    def write(self, data):
        self.source.write_text(json.dumps(data), encoding="utf-8")

    def test_generic_biped_creates_non_deform_ik_and_fk(self):
        self.write(rig_fixture())
        spec = save_control_plan(self.source, "generic-biped", self.root / "controls")
        self.assertEqual(spec["profile"], "generic-biped")
        self.assertEqual(len(spec["controls"]), 8)
        self.assertEqual(len(spec["constraints"]), 6)
        self.assertEqual(len([c for c in spec["constraints"] if c["type"] == "IK"]), 2)
        self.assertEqual(len([c for c in spec["controls"] if c["kind"] == "pole"]), 2)
        self.assertEqual(load_control_plan(self.root / "controls" / "controls.json"), spec)
        with self.assertRaises(FileExistsError):
            save_control_plan(self.source, "generic-biped", self.root / "controls")

    def test_tail_profile_tracks_parent_control_chain(self):
        self.write(rig_fixture(with_tail=True))
        spec = build_control_plan(self.source, "stylized-biped-tail")
        tail = [c for c in spec["controls"] if c["name"].startswith("CTRL_tail")]
        self.assertEqual(len(tail), 3)
        self.assertEqual([c["parent"] for c in tail], ["root", "CTRL_tail1", "CTRL_tail2"])

    def test_bird_requires_reviewed_wing_chains(self):
        self.write(rig_fixture(limbs="wings"))
        spec = build_control_plan(self.source, "avian-biped-wings")
        self.assertTrue(any(c["name"] == "CTRL_upper_wing.L" for c in spec["controls"]))
        self.assertEqual(len(spec["constraints"]), 6)
        self.write(rig_fixture(limbs="arms"))
        with self.assertRaisesRegex(ValueError, "requires reviewed"):
            build_control_plan(self.source, "avian-biped-wings")

    def test_rejects_invalid_profiles_and_stale_inputs(self):
        self.write(rig_fixture())
        with self.assertRaisesRegex(ValueError, "Unknown"):
            build_control_plan(self.source, "magic-rig")
        with self.assertRaisesRegex(ValueError, "pole_angle"):
            build_control_plan(self.source, "generic-biped", pole_angle_deg=float("nan"))
        save_control_plan(self.source, "generic-biped", self.root / "v1")
        self.source.write_text(self.source.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_control_plan(self.root / "v1" / "controls.json")

    def test_rejects_control_cycles_and_malformed_constraints(self):
        self.write(rig_fixture())
        plan = build_control_plan(self.source, "generic-biped")
        plan["controls"][0]["parent"] = plan["controls"][1]["name"]
        plan["controls"][1]["parent"] = plan["controls"][0]["name"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_control_plan(plan)
        spec = build_control_plan(self.source, "generic-biped")
        spec["constraints"][0]["pole"] = "unknown"
        with self.assertRaisesRegex(ValueError, "IK"):
            validate_control_plan(spec)


if __name__ == "__main__":
    unittest.main()
