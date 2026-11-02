"""Preflight for a dense scan's private Blender review and CLI handoff."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.blender_dense import plan_dense_blender_review
from kumiki_blender.reconstruction.references import ReferenceError


class DenseBlenderReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dense = self.root / "dense-r01"
        self.dense.mkdir()
        self.ply = self.dense / "sculpt.ply"
        # Real imported geometry is exercised in the Blender 4.5 smoke.
        self.ply.write_bytes(
            b"ply\nformat ascii 1.0\nelement vertex 1100\n"
            b"property float x\nelement face 1800\nend_header\n" + b"body"
        )
        self.report = self.dense / "dense-report.json"
        self.report.write_text(json.dumps({
            "status": "dense-mesh-candidate / artistic-and-rights-review-required",
            "mesh": {"ply": "sculpt.ply", "sha256": hashlib.sha256(self.ply.read_bytes()).hexdigest(),
                     "vertices": 1100, "faces": 1800},
        }))

    def test_preflight_verifies_geometry_hash_and_does_not_create_output(self):
        output = self.root / "review"
        record = plan_dense_blender_review(self.dense, output)
        self.assertEqual(record["faces"], 1800)
        self.assertFalse(record["distribution_allowed"])
        self.assertFalse(output.exists())
        self.assertEqual(record["source_ply"], str(self.ply))

    def test_tampering_and_unsafe_path_are_rejected_before_output(self):
        output = self.root / "review"
        self.ply.write_bytes(self.ply.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ReferenceError, "changed after reconstruction"):
            plan_dense_blender_review(self.dense, output)
        self.assertFalse(output.exists())
        self.ply.write_bytes(self.ply.read_bytes()[:-8])
        modified = json.loads(self.report.read_text())
        modified["mesh"]["ply"] = "../outside.ply"
        self.report.write_text(json.dumps(modified))
        with self.assertRaisesRegex(ReferenceError, "safe relative"):
            plan_dense_blender_review(self.dense, output)
        self.assertFalse(output.exists())

    def test_cli_plan_and_headless_worker_handoff(self):
        output = self.root / "review"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["plan-dense-blender-review",
                                   "--dense-revision", str(self.dense),
                                   "--output", str(output)]), 0)
        self.assertFalse(output.exists())
        with patch("kumiki_blender.cli.shutil.which", return_value="/bin/blender"), \
             patch("kumiki_blender.cli.subprocess.run") as runner:
            runner.return_value.returncode = 0
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["render-dense-blender-review",
                                       "--dense-revision", str(self.dense),
                                       "--output", str(output), "--no-renders"]), 0)
        command = runner.call_args.args[0]
        self.assertIn("render-dense-blender-review", command)
        self.assertIn("--no-renders", command)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
