"""CLI integration of a Blender geodesic fit and required registered source QA.

Real source-art masks and geodesic math are covered separately. This suite
mocks the expensive Blender child process only, verifying fail-closed preflight,
non-destructive candidate retention and automatic numeric-gate exit codes.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.character_build.__main__ import main as character_cli


class AutomaticFitSourceQA(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.blend = self.root / "original.blend"
        self.blend.write_bytes(b"BLENDER-synthetic")
        self.fit_plan_file = self.root / "fit-plan.json"
        self.fit_plan_file.write_text("{}")
        (self.root / "mesh").mkdir()
        self.baseline = self.root / "mesh" / "eye.ply"
        self.baseline.write_text("ply\nformat ascii 1.0\n", encoding="utf-8")
        source_spec = self.root / "source-spec.json"
        source_spec.write_text("{}")
        self.source_plan_folder = self.root / "character-plan"
        self.source_plan_folder.mkdir()
        self.frozen_plan = {
            "schema_version": 1,
            "character_id": "fixture",
            "source_blend": str(self.blend),
            "landmark_report": str(self.root / "landmarks.json"),
        }
        self.character_plan = {
            "source_spec": str(source_spec),
            "source_sha256": "source-sha",
        }
        self.character_spec = {
            "character_id": "fixture", "height_m": .35,
            "regions": [{"id": "eye.L", "source_confidence": "verified",
                         "source_mesh": "mesh/eye.ply", "verified_views": ["front"]}],
        }
        self.fit_receipt = {
            "status": "geodesic-landmark-candidate-visual-qa-required",
            "edited_vertices": 12,
        }
        self.qa = {
            "numeric_gate_pass": True, "source_spec_sha256": "source-sha",
            "failed_views": [], "status": "no-registered-regression-artistic-review-required",
        }
        self.mock_plan = self.enterContext(patch(
            "kumiki_blender.character_build.landmark_fit_plan.load_landmark_fit_plan",
            return_value=self.frozen_plan))
        self.mock_character = self.enterContext(patch(
            "kumiki_blender.character_build.character_spec_v2.open_plan",
            return_value=(self.character_plan, self.character_spec)))
        self.mock_landmarks = self.enterContext(patch(
            "kumiki_blender.character_build.landmark_triangulation.load_triangulated_landmarks",
            return_value={"height_m": .35}))
        self.mock_worker = self.enterContext(patch(
            "kumiki_blender.character_build.blender.run_worker",
            side_effect=self.worker))
        self.mock_gate = self.enterContext(patch(
            "kumiki_blender.character_build.region_revision_gate.gate_region_revision",
            side_effect=lambda *_args, **_kw: self.qa.copy()))

    def worker(self, name, args, *, blender=None, blend_file=None):
        self.assertEqual(name, "geodesic-fit")
        self.assertEqual(blend_file, self.blend)
        target = Path(args["output"])
        target.mkdir(parents=True)
        (target/"report.json").write_text(json.dumps(self.fit_receipt))
        (target/"landmark-fitted.blend").write_bytes(b"BLENDER-model-candidate")
        (target/"landmark-fitted-source.obj").write_text("v 0 0 0\n")
        return self.fit_receipt

    def invoke(self, *extra):
        argv = [
            "fit-landmark-surface",
            "--plan", str(self.fit_plan_file),
            "--output", str(self.root/"revision"),
            *extra,
        ]
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = character_cli(argv)
        return code, out.getvalue(), err.getvalue()

    def gating_args(self):
        return ["--character", str(self.source_plan_folder),
                "--region", "eye.L"]

    def test_success_creates_independent_receipt_and_exit_zero(self):
        code, text, _err = self.invoke(*self.gating_args())
        self.assertEqual(code, 0)
        receipt = json.loads((self.root/"revision"/"source-fidelity-receipt.json").read_text())
        self.assertTrue(receipt["numeric_gate_pass"])
        self.assertEqual(receipt["region_id"], "eye.L")
        self.assertEqual(receipt["source_qa_report"], "source-fidelity/report.json")
        self.assertTrue(receipt["artistic_review_required"])
        self.assertTrue((self.root/"revision"/"landmark-fitted.blend").is_file())
        self.mock_gate.assert_called_once()
        args, kwargs = self.mock_gate.call_args
        self.assertEqual(args[1], "eye.L")
        self.assertEqual(args[2], self.root/"revision"/"landmark-fitted-source.obj")
        self.assertEqual(kwargs["before"], self.baseline.resolve())
        self.assertEqual(json.loads(text)["status"], "source-numeric-gate-artistic-review-required")

    def test_regression_returns_one_but_keeps_editable_mesh(self):
        self.qa["numeric_gate_pass"] = False
        self.qa["failed_views"] = ["front"]
        code, stdout, _err = self.invoke(*self.gating_args())
        self.assertEqual(code, 1)
        receipt = json.loads((self.root/"revision"/"source-fidelity-receipt.json").read_text())
        self.assertEqual(receipt["status"], "source-regression-rejected")
        self.assertEqual(receipt["failed_views"], ["front"])
        self.assertTrue((self.root/"revision"/"landmark-fitted.blend").is_file())
        self.assertFalse(json.loads(stdout)["numeric_gate_pass"])

    def test_legacy_no_gate_workflow_is_unchanged(self):
        code, stdout, _err = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["edited_vertices"], 12)
        self.assertFalse((self.root/"revision"/"source-fidelity-receipt.json").exists())
        self.mock_gate.assert_not_called()
        self.mock_character.assert_not_called()

    def test_incomplete_gate_args_fail_before_running_blender(self):
        for args in (
            ["--character", str(self.source_plan_folder)],
            ["--region", "eye.L"],
            ["--qa-before", str(self.baseline)],
            [*self.gating_args(), "--max-regression", "nan"],
            [*self.gating_args(), "--min-iou", "1.5"],
        ):
            with self.subTest(args=args):
                code, _stdout, stderr = self.invoke(*args)
                self.assertEqual(code, 2)
                self.assertIn("error:", stderr)
                self.mock_worker.assert_not_called()

    def test_unverified_source_and_wrong_character_id_are_fail_closed(self):
        self.character_spec["regions"][0]["verified_views"] = []
        code, _text, err = self.invoke(*self.gating_args())
        self.assertEqual(code, 2)
        self.assertIn("verified reference masks", err)
        self.mock_worker.assert_not_called()
        self.character_spec["regions"][0]["verified_views"] = ["front"]
        self.character_spec["character_id"] = "another"
        code, _text, err = self.invoke(*self.gating_args())
        self.assertEqual(code, 2)
        self.assertIn("same character ID", err)
        self.mock_worker.assert_not_called()

    def test_mismatched_scale_and_missing_baseline_refused_before_blender(self):
        self.mock_landmarks.return_value = {"height_m": .45}
        code, _stdout, err = self.invoke(*self.gating_args())
        self.assertEqual(code, 2)
        self.assertIn("reference scale differ", err)
        self.mock_worker.assert_not_called()
        self.mock_landmarks.return_value = {"height_m": .35}
        del self.character_spec["regions"][0]["source_mesh"]
        code, _stdout, err = self.invoke(*self.gating_args())
        self.assertEqual(code, 2)
        self.assertIn("--qa-before", err)
        self.mock_worker.assert_not_called()

    def test_explicit_baseline_accepted_if_no_region_source_mesh(self):
        del self.character_spec["regions"][0]["source_mesh"]
        code, _stdout, _err = self.invoke(
            *self.gating_args(), "--qa-before", str(self.baseline))
        self.assertEqual(code, 0)
        self.assertEqual(self.mock_gate.call_args.kwargs["before"], self.baseline.resolve())


if __name__ == "__main__":
    unittest.main()
