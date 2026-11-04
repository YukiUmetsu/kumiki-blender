"""CLI safety checks for both completion modes and immutable build handoff."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.cli import main
from tests.test_asset_import import fixture_glb
from tests.test_completion import SOURCE


class CompletionCLITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        base = self.root / "base.glb"
        part = self.root / "part.glb"
        fixture_glb(base)
        fixture_glb(part)
        self.manifest = self.root / "completion.json"
        self.data = {
            "schema_version": 1, "asset_id": "demo-statue", "mode": "historical",
            "base_origin": "observed",
            "base_glb": "base.glb", "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
            "regions": [{
                "id": "missing-ear", "candidate_glb": "part.glb",
                "candidate_sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
                "origin": "generated", "generator": {"adapter": "supplied-glb", "model_id": "sample-model"},
                "max_error_m": 0.01,
                "anchors": [{"id": f"p-{index}", "source_xyz_m": list(point),
                             "target_xyz_m": list(point)} for index, point in enumerate(SOURCE)],
            }],
        }
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")

    def test_plan_excludes_historical_generation_without_side_effects(self):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = main(["plan-completion", "--manifest", str(self.manifest)])
        self.assertEqual(code, 0)
        report = json.loads(captured.getvalue())
        self.assertEqual(report["mode"], "historical")
        self.assertFalse(report["regions"][0]["included"])
        self.assertEqual(len(list(self.root.iterdir())), 3)

    def test_build_refuses_existing_destination_before_blender_launch(self):
        output = self.root / "result"
        output.mkdir()
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            code = main(["build-completion", "--manifest", str(self.manifest),
                         "--output", str(output), "--blender", "missing-blender"])
        self.assertEqual(code, 2)
        self.assertIn("already exists", captured.getvalue())
        self.assertEqual(list(output.iterdir()), [])

    def test_build_requires_blender_without_creating_output(self):
        output = self.root / "result"
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            code = main(["build-completion", "--manifest", str(self.manifest),
                         "--output", str(output), "--blender", "missing-blender"])
        self.assertEqual(code, 2)
        self.assertIn("Blender executable", captured.getvalue())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
