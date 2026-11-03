"""Deforming-mesh authoring contract, testable without Blender/GPU."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from kumiki_blender.deformation_plan import DeformationPlanError, load_deformation_plan, parse_deformation_plan


VALID = {"schema_version": 1, "asset_id": "cloth-qa", "action_id": "cloth-sag",
         "target": "role:cloth-panel", "duration_frames": 24, "sample_stride": 2,
         "max_vertices": 128}


class DeformationPlanTests(unittest.TestCase):
    def test_bounded_forward_samples_include_nondivisible_final_frame(self):
        shot = parse_deformation_plan({**VALID, "duration_frames": 25})
        self.assertEqual(shot.frames, (*range(0, 25, 2), 25))
        self.assertEqual(shot.target, "role:cloth-panel")
        self.assertEqual(len(shot.frames) - 1, 13)

    def test_rejects_unknown_code_fields_unsafe_selectors_floats_and_overbudget(self):
        variants = [
            {**VALID, "execute": "import os"},
            {**VALID, "action_id": "../../escape"},
            {**VALID, "target": "name:../../escape"},
            {**VALID, "target": "asset:cloth-panel"},
            {**VALID, "duration_frames": 24.0},
            {**VALID, "sample_stride": 0},
            {**VALID, "max_vertices": 12_001},
            {**VALID, "max_vertices": True},
            {**VALID, "duration_frames": 240, "sample_stride": 1},
            {**VALID, "duration_frames": 120, "sample_stride": 2, "max_vertices": 10_000},
        ]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(DeformationPlanError):
                parse_deformation_plan(variant)

    def test_host_cli_does_not_import_bpy_and_reports_exact_morph_budget(self):
        from kumiki_blender.cli import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec = root / "shot.json"
            spec.write_text(json.dumps(VALID), encoding="utf-8")
            self.assertEqual(load_deformation_plan(spec).frames[-1], 24)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["plan-deformation", "--spec", str(spec)])
            self.assertEqual(code, 0)
            record = json.loads(output.getvalue())
            self.assertEqual(record["morph_targets"], 12)
            self.assertEqual(record["frames"][0], 0)
            self.assertEqual(record["frames"][-1], 24)
            self.assertEqual(record["target"], VALID["target"])
            with self.assertRaises(DeformationPlanError):
                load_deformation_plan(root / "missing.json")
            oversize = root / "oversized.json"
            oversize.write_text(" " * 40_000, encoding="utf-8")
            with self.assertRaisesRegex(DeformationPlanError, "32 KiB"):
                load_deformation_plan(oversize)


if __name__ == "__main__":
    unittest.main()
