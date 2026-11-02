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
                                       "--output", str(output), "--no-renders",
                                       "--matched-views", "0"]), 0)
        command = runner.call_args.args[0]
        self.assertIn("render-dense-blender-review", command)
        self.assertIn("--no-renders", command)
        self.assertIn("--matched-views", command)
        self.assertIn("--matched-resolution-percent", command)
        self.assertFalse(output.exists())

    def _prepare_pinned_sparse_cameras(self):
        source = self.root / "sparse"
        model = source / "undistorted/text-model"
        model.mkdir(parents=True)
        images = source / "undistorted/images"
        images.mkdir()
        (model / "cameras.txt").write_text("1 PINHOLE 640 480 600 600 320 240\n")
        (model / "images.txt").write_text("".join(
            f"{i+1} 1 0 0 0 {i} 0 4 1 view-{i:02d}.jpg\n\n" for i in range(4)))
        entries = []
        for i in range(4):
            filename = f"view-{i:02d}.jpg"
            data = f"first-party-photo-{i}".encode()
            (images / filename).write_bytes(data)
            entries.append({"filename": filename,
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "dimensions": [640, 480]})
        (source / "undistorted-reference-index.json").write_text(
            json.dumps({"images": entries}))
        (source / "reconstruction-report.json").write_text(
            json.dumps({"status": "candidate / visual-review-required",
                        "registered_images": 4}))
        report = json.loads(self.report.read_text())
        report["source_revision"] = str(source)
        self.report.write_text(json.dumps(report))
        return images

    def test_matched_views_are_camera_calibrated_and_hash_pinned(self):
        images = self._prepare_pinned_sparse_cameras()
        record = plan_dense_blender_review(self.dense, self.root / "review", matched_views=3)
        self.assertEqual(len(record["matched_cameras"]), 3)
        first = record["matched_cameras"][0]
        self.assertEqual(first["location_colmap"], (0.0, 0.0, -4.0))
        self.assertEqual(first["rotation_3x3"], [[1.0, 0.0, 0.0],
                                                  [0.0, -1.0, 0.0],
                                                  [0.0, 0.0, -1.0]])
        self.assertEqual(first["intrinsics"]["fx_px"], 600.0)
        self.assertEqual(record["matched_resolution_percent"], 100)
        self.assertFalse((self.root / "review").exists())
        (images / "view-00.jpg").write_text("changed photo")
        with self.assertRaisesRegex(ReferenceError, "changed since reconstruction"):
            plan_dense_blender_review(self.dense, self.root / "review", matched_views=3)

    def test_matched_views_require_sparse_revision_and_valid_bounds(self):
        with self.assertRaisesRegex(ReferenceError, "source sparse"):
            plan_dense_blender_review(self.dense, self.root / "review", matched_views=1)
        for invalid in (-1, 13, True, 0.1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ReferenceError, "0–12"):
                plan_dense_blender_review(self.dense, self.root / "review",
                                          matched_views=invalid)
        for invalid in (0, 101, 99.5, False):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ReferenceError, "10–100"):
                plan_dense_blender_review(self.dense, self.root / "review",
                                          matched_resolution_percent=invalid)


if __name__ == "__main__":
    unittest.main()
