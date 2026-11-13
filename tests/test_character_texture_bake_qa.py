"""One-command texture-parts bake and source-fidelity gate orchestration."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from kumiki_blender.character_build.__main__ import main as cli
from kumiki_blender.character_build.mesh import Mesh


class TextureBakeQATests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.handoff = self.root / "handoff"
        (self.handoff / "aligned").mkdir(parents=True)
        (self.handoff / "spec.json").write_text(
            json.dumps({"views": ["front"], "metres_per_aligned_px": .00035}))
        Image.new("RGBA", (8, 8), (100, 70, 50, 255)).save(
            self.handoff / "aligned" / "front.png")
        self.parts = self.root / "reviewed-parts"
        self.parts.mkdir()
        self.mesh = self.root / "mesh.obj"
        self.mesh.write_text("v 0 0 0\n")
        self.before = self.root / "before.png"
        Image.new("RGB", (8, 8), (100, 70, 50)).save(self.before)
        self.out = self.root / "candidate.png"
        mesh = Mesh(np.asarray([[0, 0, .1], [.02, 0, .1],
                                [0, 0, .12]]), np.asarray([[0, 1, 2]]))
        uv = np.asarray([[[0,0], [1,0], [0,1]]], dtype=float)
        self.mesh_reader = self.enterContext(patch(
            "kumiki_blender.character_build.texture.read_obj_uv",
            return_value=(mesh, uv)))
        self.baker = self.enterContext(patch(
            "kumiki_blender.character_build.parts.bake_parts",
            return_value=(np.full((16, 16, 3), 120, np.uint8),
                          {"source_signature_sha256": "valid", "materials": []})))
        self.estimator = self.enterContext(patch(
            "kumiki_blender.character_build.texture.estimate_uv_projection_samples",
            return_value=4000))
        self.evidence = {
            "numeric_gate_pass": True,
            "failed_views": [],
            "input_sha256": {"handoff_spec": "src", "baseline_texture": "prior"},
        }
        self.gate = self.enterContext(patch(
            "kumiki_blender.character_build.texture_revision_gate.audit_texture_revision",
            side_effect=self.fake_gate))

    def fake_gate(self, *args, **kwargs):
        folder = Path(args[4])
        folder.mkdir()
        (folder / "report.json").write_text(json.dumps(self.evidence))
        return self.evidence.copy()

    def run_cli(self, *flags):
        args = [
            "texture-parts", str(self.handoff), str(self.mesh), str(self.parts),
            "--output", str(self.out), "--size", "16", *flags,
        ]
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli(args)
        return code, out.getvalue(), err.getvalue()

    def qa_flag(self):
        return ("--qa-baseline", str(self.before))

    def test_default_texture_bake_remains_backward_compatible(self):
        code, stdout, _stderr = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["materials"], [])
        self.assertTrue(self.out.is_file())
        self.assertTrue(self.out.with_suffix(".json").is_file())
        self.assertFalse(self.out.with_suffix(".source-fidelity.json").exists())
        self.gate.assert_not_called()
        self.estimator.assert_not_called()

    def test_success_bakes_a_candidate_and_records_unattended_gate_receipt(self):
        code, stdout, err = self.run_cli(*self.qa_flag())
        self.assertEqual(code, 0, err)
        receipt = json.loads(
            self.out.with_suffix(".source-fidelity.json").read_text())
        self.assertTrue(receipt["source_qa_numeric_pass"])
        self.assertTrue(receipt["artistic_review_required"])
        self.assertEqual(receipt["status"],
                         "texture-numeric-gate-artistic-review-required")
        self.assertTrue((self.root / "candidate-source-qa" / "report.json").is_file())
        self.assertTrue(self.out.is_file())
        self.assertEqual(json.loads(stdout), receipt)
        self.estimator.assert_called_once()
        self.gate.assert_called_once()
        self.assertEqual(self.gate.call_args.kwargs["max_projection_samples"], 2_000_000)

    def test_failed_gate_keeps_candidate_texture_for_visual_debug(self):
        self.evidence["numeric_gate_pass"] = False
        self.evidence["failed_views"] = ["front"]
        code, _stdout, err = self.run_cli(*self.qa_flag())
        self.assertEqual(code, 1, err)
        receipt = json.loads(
            self.out.with_suffix(".source-fidelity.json").read_text())
        self.assertEqual(receipt["failed_views"], ["front"])
        self.assertEqual(receipt["status"], "texture-source-regression-rejected")
        self.assertTrue(self.out.is_file())
        self.assertTrue(self.out.with_suffix(".json").is_file())
        self.assertTrue((self.root/"candidate-source-qa"/"report.json").is_file())

    def test_missing_source_art_or_baseline_refuses_bake_early(self):
        for kind in ("baseline", "aligned"):
            with self.subTest(kind=kind):
                path = (self.before if kind == "baseline" else
                        self.handoff / "aligned" / "front.png")
                original = path.read_bytes()
                path.unlink()
                code, _stdout, err = self.run_cli(*self.qa_flag())
                self.assertEqual(code, 2)
                self.assertIn("error:", err)
                self.baker.assert_not_called()
                self.assertFalse(self.out.exists())
                path.write_bytes(original)

    def test_budget_and_invalid_threshold_refuse_expensive_bake(self):
        for flags in (
            (*self.qa_flag(), "--qa-max-mean-increase", "nan"),
            (*self.qa_flag(), "--qa-max-p90-increase", "-0.5"),
            (*self.qa_flag(), "--qa-max-within-10-drop", "2"),
            (*self.qa_flag(), "--qa-max-projection-samples", "0"),
        ):
            with self.subTest(flags=flags):
                code, _stdout, err = self.run_cli(*flags)
                self.assertEqual(code, 2)
                self.assertIn("error:", err)
                self.baker.assert_not_called()
        self.estimator.return_value = 5_000_000
        code, _stdout, err = self.run_cli(*self.qa_flag())
        self.assertEqual(code, 2)
        self.assertIn("Projection sample budget exceeded", err)
        self.baker.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_existing_output_evidence_and_report_never_overwritten(self):
        for occupied in (self.out, self.out.with_suffix(".json"),
                         self.root / "candidate-source-qa",
                         self.out.with_suffix(".source-fidelity.json")):
            with self.subTest(occupied=occupied):
                if occupied.suffix:
                    occupied.write_text("human review")
                else:
                    occupied.mkdir()
                    (occupied / "human.txt").write_text("approved")
                code, _stdout, err = self.run_cli(*self.qa_flag())
                self.assertEqual(code, 2)
                self.assertIn("already exists", err)
                self.baker.assert_not_called()
                if occupied.suffix:
                    self.assertEqual(occupied.read_text(), "human review")
                    occupied.unlink()
                else:
                    self.assertEqual((occupied/"human.txt").read_text(), "approved")
                    (occupied/"human.txt").unlink()
                    occupied.rmdir()

    def test_qa_output_without_qa_baseline_is_invalid(self):
        code, _stdout, err = self.run_cli("--qa-output", str(self.root/"qa"))
        self.assertEqual(code, 2)
        self.assertIn("requires --qa-baseline", err)
        self.baker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
