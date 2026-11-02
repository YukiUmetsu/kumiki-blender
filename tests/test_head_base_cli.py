"""Real command behavior for the anatomical-head reconstruction handoff."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.cli import main
from tests.test_asset_import import fixture_glb, provenance_json
from tests.test_head_base import SOURCE, target


class HeadBaseCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture_glb(self.root / "head.glb", materials=["Stone"])
        (self.root / "rights.json").write_text(json.dumps(
            provenance_json(asset_id="licensed-anatomical-base", license="CC0-1.0")))
        self.plan = self.root / "fit.json"
        self.plan.write_text(json.dumps({
            "schema_version": 1, "asset_id": "redeemer-head-fit",
            "source_glb": "head.glb", "source_provenance": "rights.json",
            "mesh_name": "Head", "fit_mode": "rigid",
            "landmarks": [
                {"id": name, "region": "face", "source_xyz_m": list(point),
                 "target_xyz_m": list(target(point)), "radius_m": 0.12}
                for name, point in SOURCE.items()
            ],
        }))

    def test_planning_prints_auditable_alignment_without_creating_revision(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["plan-head-base", "--manifest", str(self.plan)])
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "head-fit-candidate / artistic-review-required")
        self.assertEqual(report["mesh_name"], "Head")
        self.assertEqual(report["fit_mode"], "rigid")
        self.assertFalse(report["historical_fidelity_verified"])
        self.assertAlmostEqual(report["rms_error_m"], 0.0, places=5)
        self.assertEqual(set(path.name for path in self.root.iterdir()),
                         {"head.glb", "rights.json", "fit.json"})

    def test_build_refuses_existing_output_before_launch(self):
        revision = self.root / "revision"
        revision.mkdir()
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            result = main(["build-head-base", "--manifest", str(self.plan),
                           "--output", str(revision), "--blender", "blender-not-installed"])
        self.assertEqual(result, 2)
        self.assertIn("already exists", error.getvalue())

    def test_build_reports_missing_blender_without_creating_output(self):
        revision = self.root / "revision"
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            result = main(["build-head-base", "--manifest", str(self.plan),
                           "--output", str(revision), "--blender", "blender-not-installed"])
        self.assertEqual(result, 2)
        self.assertIn("Blender executable", error.getvalue())
        self.assertFalse(revision.exists())

    def test_invalid_head_plan_is_reported_as_error(self):
        data = json.loads(self.plan.read_text())
        data["mesh_name"] = ""
        self.plan.write_text(json.dumps(data))
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            result = main(["plan-head-base", "--manifest", str(self.plan)])
        self.assertEqual(result, 2)
        self.assertIn("mesh_name", error.getvalue())


if __name__ == "__main__":
    unittest.main()
