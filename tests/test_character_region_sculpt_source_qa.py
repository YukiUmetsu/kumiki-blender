"""CLI integration: geometric sculpt output cannot silently bypass source masks."""
from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.character_build.__main__ import main


class RegionSculptSourceGateTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.blend = self.root / "source.blend"
        self.blend.write_bytes(b"BLENDER synthetic")
        self.plan = self.root / "sculpt.json"
        self.plan.write_text(json.dumps({
            "schema_version": 1, "object": "Eye.L",
            "expected_vertex_count": 100, "max_displacement_m": .01,
            "operations": [{
                "center_m": [0, 0, .2], "radius_m": .02,
                "move_m": [0, -.002, 0], "falloff": "smoothstep",
            }],
        }), encoding="utf-8")
        self.spec = self.root / "character.json"
        self.spec.write_text("{}")
        self.baseline = self.root / "baseline-source.obj"
        self.baseline.write_text("v 0 0 0\n")
        self.character = self.root / "character-plan"
        self.character.mkdir()

    def _argv(self, output: Path):
        return [
            "refine-region", "--blend", str(self.blend), "--plan", str(self.plan),
            "--output", str(output), "--character", str(self.character),
            "--region", "eye.L",
        ]

    def _run(self, output: Path, numeric_pass: bool):
        source_plan = {"source_spec": str(self.spec)}
        spec = {
            "character_id": "pip",
            "regions": [{
                "id": "eye.L", "verified_views": ["front"],
                "source_mesh": self.baseline.name,
            }],
        }

        def fake_worker(_name, arguments, **_kwargs):
            folder = Path(arguments["output"])
            folder.mkdir()
            (folder / "region-source.obj").write_text("v 0 0 0\n")
            (folder / "report.json").write_text("{}")
            return {"status": "geometric-candidate-art-review-required"}

        qa = {"numeric_gate_pass": numeric_pass,
              "failed_views": [] if numeric_pass else ["front"],
              "source_spec_sha256": "b" * 64}
        with (
            patch("kumiki_blender.character_build.blender.run_worker",
                  side_effect=fake_worker) as worker,
            patch("kumiki_blender.character_build.character_spec_v2.open_plan",
                  return_value=(source_plan, spec)),
            patch("kumiki_blender.character_build.region_revision_gate.gate_region_revision",
                  return_value=qa) as gate,
            redirect_stdout(StringIO()),
        ):
            code = main(self._argv(output))
        self.assertEqual(code, 0 if numeric_pass else 1)
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(gate.call_count, 1)
        receipt = json.loads((output / "source-fidelity-receipt.json").read_text())
        self.assertEqual(receipt["numeric_gate_pass"], numeric_pass)
        self.assertTrue(receipt["artistic_review_required"])
        self.assertEqual(receipt["failed_views"], qa["failed_views"])

    def test_pass_is_only_numeric_not_artistic_approval(self):
        self._run(self.root / "accepted", True)

    def test_source_regression_fails_cli_even_with_valid_geometry(self):
        self._run(self.root / "rejected", False)

    def test_malformed_source_gate_flags_rejected_before_blender(self):
        with patch("kumiki_blender.character_build.blender.run_worker") as worker:
            with self.assertRaisesRegex(ValueError, "both --character and --region"):
                main([
                    "refine-region", "--blend", str(self.blend),
                    "--plan", str(self.plan), "--output", str(self.root / "unused"),
                    "--region", "eye.L",
                ])
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
