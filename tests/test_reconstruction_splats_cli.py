"""Mac-first Gaussian splat CLI: multiple images, manifests, registered cameras."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from kumiki_blender.cli import main


class SplatCLITests(unittest.TestCase):
    def setUp(self):
        from tests.test_reconstruction_splats import SplatDatasetTests

        self.fixture = SplatDatasetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.root = self.fixture.root
        self.photos = self.root / "photos"
        self.photos.mkdir()
        for index in range(8):
            Image.new("RGB", (128, 96), (index * 24, 65, 85)).save(
                self.photos / f"angle-{index:02d}.jpg")
        self.trainer = self.fixture._fake_opensplat()

    def _call(self, args):
        output = io.StringIO()
        error = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(args)
        return code, json.loads(output.getvalue()) if output.getvalue() else None, error.getvalue()

    def test_registered_camera_cli_plans_and_trains(self):
        base = ["--reconstruction", str(self.fixture.source),
                "--output", str(self.root / "splats-r01"),
                "--opensplat", str(self.trainer), "--iterations", "300"]
        code, report, _ = self._call(["plan-splats", *base])
        self.assertEqual(code, 0)
        self.assertEqual(report["registered_images"], 8)
        self.assertFalse((self.root / "splats-r01").exists())
        code, report, _ = self._call(["build-splats", *base])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertTrue((self.root / "splats-r01/splat.ply").is_file())

    def test_photo_folder_and_manifest_plan_camera_registration(self):
        code, report, _ = self._call([
            "plan-splats", "--photos", str(self.photos), "--asset-id", "old-statue",
            "--output", str(self.root / "r01"), "--opensplat", str(self.trainer)])
        self.assertEqual(code, 0)
        self.assertEqual(report["independent_images"], 8)
        self.assertEqual(report["status"], "candidate / camera-registration-required")
        self.assertFalse((self.root / "r01").exists())

        images = []
        for index, file in enumerate(sorted(self.photos.iterdir())):
            images.append({"id": f"angle-{index:02d}", "file": f"photos/{file.name}",
                           "source_url": f"https://example.com/angle-{index:02d}",
                           "view": "unknown", "regions": ["object"]})
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "asset_id": "old-statue",
                                        "subject": "Old statue",
                                        "required_views": ["unknown"],
                                        "required_regions": ["object"], "images": images}))
        code, report, error = self._call([
            "plan-splats", "--references", str(manifest),
            "--output", str(self.root / "r02"), "--opensplat", str(self.trainer)])
        self.assertEqual(code, 0, error)
        self.assertEqual(report["independent_images"], 8)

    def test_one_command_photos_keeps_camera_reconstruction_inside_splat_revision(self):
        def fake_camera_registration(refs, target, **kwargs):
            self.assertTrue(kwargs["undistort"])
            self.assertEqual(len(refs.images), 8)
            shutil.copytree(self.fixture.source, target)
            return {"status": "candidate / visual-review-required", "registered_images": 8,
                    "undistorted_images": 8}

        output = self.root / "all-in-one-r01"
        with patch("kumiki_blender.reconstruction.colmap.build_sparse_reconstruction",
                   side_effect=fake_camera_registration):
            code, report, error = self._call([
                "build-splats", "--photos", str(self.photos), "--asset-id", "old-statue",
                "--output", str(output), "--opensplat", str(self.trainer),
                "--iterations", "300", "--engine", "colmap", "--colmap", str(self.trainer)])
        self.assertEqual(code, 0, error)
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertEqual(report["camera_reconstruction"]["registered_images"], 8)
        self.assertTrue((output / "sparse-reconstruction/reconstruction-report.json").exists())
        self.assertTrue((output / "dataset/sparse/0/points3D.bin").exists())
        self.assertTrue((output / "splat.ply").exists())

    def test_registration_failure_never_claims_splat_success(self):
        with patch("kumiki_blender.reconstruction.colmap.build_sparse_reconstruction",
                   return_value={"status": "review-required", "failed_stage": "mapper"}):
            code, report, error = self._call([
                "build-splats", "--photos", str(self.photos), "--asset-id", "old-statue",
                "--output", str(self.root / "failed"), "--opensplat", str(self.trainer),
                "--engine", "colmap", "--colmap", str(self.trainer)])
        self.assertEqual(code, 3, error)
        self.assertEqual(report["failed_stage"], "camera-registration")
        self.assertFalse((self.root / "failed/splat.ply").exists())

    def test_invalid_camera_configuration_fails_before_revision_exists(self):
        output = self.root / "bad-camera-config"
        code, _, error = self._call([
            "build-splats", "--photos", str(self.photos), "--asset-id", "old-statue",
            "--output", str(output), "--opensplat", str(self.trainer),
            "--engine", "colmap", "--colmap", str(self.trainer),
            "--max-image-size", "100"])
        self.assertEqual(code, 2)
        self.assertIn("image size", error.lower())
        self.assertFalse(output.exists())

    def test_mutually_exclusive_inputs_and_missing_asset_id(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as captured:
                main(["plan-splats", "--photos", str(self.photos),
                      "--reconstruction", str(self.fixture.source),
                      "--output", str(self.root / "x")])
        self.assertEqual(captured.exception.code, 2)
        code, _, error = self._call([
            "plan-splats", "--photos", str(self.photos),
            "--output", str(self.root / "x")])
        self.assertEqual(code, 2)
        self.assertIn("asset-id", error)


if __name__ == "__main__":
    unittest.main()
