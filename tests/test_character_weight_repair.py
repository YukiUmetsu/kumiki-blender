"""Conservative wing/paw/tail weight repair and immutable CLI preflight tests."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.character_build.weight_repair import propose_weight_repair
from kumiki_blender.character_build.__main__ import main


class WeightRepairMathTest(unittest.TestCase):
    def test_removes_spine_leak_without_changing_original(self):
        source = {"upper_wing.L": .9, "spine": .1}
        immutable = copy.deepcopy(source)
        proposal = propose_weight_repair(source, {"upper_wing.L"},
                                         outside_tolerance=.02)
        self.assertEqual(source, immutable)
        self.assertEqual(proposal["after"], {"upper_wing.L": 1.0})
        self.assertEqual(proposal["removed_bones"], ["spine"])
        self.assertAlmostEqual(proposal["removed_outside_weight"], .1)
        self.assertAlmostEqual(sum(proposal["after"].values()), 1.0)

    def test_rebalances_only_existing_wing_joint_bones(self):
        proposal = propose_weight_repair(
            {"upper_wing.L": .72, "fore_wing.L": .18, "head": .10},
            {"upper_wing.L", "fore_wing.L"}, outside_tolerance=.01)
        self.assertAlmostEqual(proposal["after"]["upper_wing.L"], .8)
        self.assertAlmostEqual(proposal["after"]["fore_wing.L"], .2)
        self.assertEqual(len(proposal["after"]), 2)
        self.assertNotIn("head", proposal["after"])

    def test_does_not_touch_weights_within_explicit_zone_tolerance(self):
        self.assertIsNone(propose_weight_repair(
            {"tail1": .98, "spine": .02}, {"tail1"}, outside_tolerance=.02))
        self.assertIsNone(propose_weight_repair(
            {"tail1": 1.0}, {"tail1"}, outside_tolerance=.02))

    def test_large_or_unrelated_influences_require_manual_decision(self):
        for source, reason in [
            ({"upper_wing.L": .7, "head": .3}, "exceeds maximum"),
            ({"head": 1.0}, "No existing allowed"),
            ({"upper_wing.L": .85, "spine": .05}, "not normalized"),
        ]:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, reason):
                propose_weight_repair(source, {"upper_wing.L"}, outside_tolerance=.02)

    def test_no_allowed_weight_is_never_invented(self):
        with self.assertRaisesRegex(ValueError, "No existing allowed"):
            propose_weight_repair({"unrelated": 1.0}, {"upper_wing.L"},
                                  outside_tolerance=.02, max_removed_weight=.5)

    def test_excessive_existing_joint_influences_are_not_hidden(self):
        weights = {"b1": .18, "b2": .18, "b3": .18, "b4": .18, "b5": .18, "spine": .1}
        with self.assertRaisesRegex(ValueError, "more than four"):
            propose_weight_repair(weights, {"b1", "b2", "b3", "b4", "b5"},
                                  outside_tolerance=.01)

    def test_rejects_broken_fraction_and_policy(self):
        for weights in [
            {"wing": float("nan"), "spine": .1},
            {"wing": True},
            {"wing": -1},
            {},
        ]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                propose_weight_repair(weights, {"wing"}, outside_tolerance=.01)
        for kwargs in [
            {"outside_tolerance": .1, "max_removed_weight": .1},
            {"outside_tolerance": -.1},
            {"outside_tolerance": .01, "max_removed_weight": .9},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                propose_weight_repair({"wing": .9, "spine": .1}, {"wing"}, **kwargs)


class WeightRepairCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.blend = self.root / "wing.blend"
        self.blend.write_bytes(b"synthetic blend placeholder for CLI preflight")
        self.pose = self.root / "poses.json"
        self.plan = {
            "schema_version": 1,
            "armature": "CharacterRig",
            "poses": [{"name": "wing-up",
                       "bones": {"upper_wing.L": {"axis": "Z", "degrees": 30}}}],
            "weight_zones": [{
                "mesh": "Wing.L", "region_group": "REGION_WingLeft",
                "allowed_bones": ["upper_wing.L", "fore_wing.L"],
                "max_outside_weight": .02, "min_vertices": 20,
            }],
        }
        self.pose.write_text(json.dumps(self.plan), encoding="utf-8")

    def invoke(self, *extra, output="preview"):
        args = [
            "repair-rig-weights", "--blend", str(self.blend),
            "--pose-plan", str(self.pose),
            "--output", str(self.root / output),
            *extra,
        ]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(args)

    def test_blend_checksum_bound_and_preview_forwarded_to_worker(self):
        with patch("kumiki_blender.character_build.blender.run_worker") as worker:
            worker.return_value = {"status": "weight-repair-proposal-only", "changed_vertices": 1}
            self.assertEqual(self.invoke("--dry-run"), 0)
            called = worker.call_args
            self.assertEqual(called.args[0], "weight-repair")
            payload = called.args[1]
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["weight_zones"], self.plan["weight_zones"])
            self.assertEqual(payload["source_blend_sha256"],
                             hashlib.sha256(self.blend.read_bytes()).hexdigest())
            self.assertEqual(called.kwargs["blend_file"], self.blend.resolve())

    def test_no_marked_semantics_rejected_before_blender(self):
        self.plan["weight_zones"] = []
        self.pose.write_text(json.dumps(self.plan))
        with patch("kumiki_blender.character_build.blender.run_worker") as worker:
            self.assertEqual(self.invoke(), 2)
            worker.assert_not_called()

    def test_large_threshold_or_bad_budget_rejected_before_blender(self):
        with patch("kumiki_blender.character_build.blender.run_worker") as worker:
            self.assertEqual(self.invoke("--max-removed-weight", ".02"), 2)
            self.assertEqual(self.invoke("--max-removed-weight", "nan"), 2)
            self.assertEqual(self.invoke("--max-changed-vertices", "0"), 2)
            self.assertEqual(self.invoke("--max-changed-vertices", "10001"), 2)
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
