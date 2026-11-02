"""Gaussian-splat input and export behavior without requiring a GPU."""

from __future__ import annotations

import shutil
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.splat_input import (
    load_photo_folder, plan_splat_input, plan_splat_dataset, stage_splat_dataset)
from kumiki_blender.reconstruction.splats import (
    SplatSettings, audit_splat_artifact, build_splat_revision)


class SplatInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()

    def _images(self, count: int = 8):
        from PIL import Image

        for index in range(count):
            image = Image.new("RGB", (128, 96), (index * 19, 40 + index, 110))
            image.save(self.photos / f"image-{index:02d}.png")

    def test_photo_folder_builds_private_multiview_reference_set(self):
        self._images()
        references = load_photo_folder(self.photos, "old-statue")
        self.assertEqual(references.asset_id, "old-statue")
        self.assertEqual(len(references.images), 8)
        self.assertEqual({image.dimensions for image in references.images}, {(128, 96)})
        self.assertTrue(all(image.view == "unknown" for image in references.images))
        report = plan_splat_input(references)
        self.assertEqual(report["independent_images"], 8)
        self.assertFalse(report["distribution_allowed"])
        self.assertEqual(report["status"], "candidate / camera-registration-required")

    def test_duplicate_photos_do_not_meet_minimum_independent_count(self):
        self._images(7)
        shutil.copyfile(self.photos / "image-00.png", self.photos / "image-07.png")
        with self.assertRaisesRegex(ReferenceError, "independent"):
            load_photo_folder(self.photos, "old-statue")

    def test_rejects_symlinked_photo_and_input_directory(self):
        self._images()
        external = self.root / "other.png"
        shutil.copyfile(self.photos / "image-00.png", external)
        (self.photos / "linked.png").symlink_to(external)
        with self.assertRaisesRegex(ReferenceError, "symbolic link"):
            load_photo_folder(self.photos, "old-statue")
        link = self.root / "linked-dir"
        link.symlink_to(self.photos, target_is_directory=True)
        with self.assertRaisesRegex(ReferenceError, "symbolic link"):
            load_photo_folder(link, "old-statue")

    def test_rejects_too_few_photos_and_bad_asset_id(self):
        self._images(7)
        with self.assertRaisesRegex(ReferenceError, "8"):
            load_photo_folder(self.photos, "old-statue")
        with self.assertRaises(ReferenceError):
            load_photo_folder(self.photos, "Invalid ID")


class SplatDatasetTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "sparse-r01"
        images = self.source / "undistorted" / "images"
        images.mkdir(parents=True)
        sparse = self.source / "undistorted" / "sparse"
        sparse.mkdir()
        for name in ("cameras", "images", "points3D"):
            (sparse / f"{name}.bin").write_bytes((name + "-sample").encode())
        index = []
        for num in range(8):
            name = f"photo-{num:04d}.jpg"
            filename = images / name
            Image.new("RGB", (128, 96), (10 + num, 80, 90)).save(filename)
            index.append({"filename": name, "sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                          "dimensions": [128, 96]})
        (self.source / "undistorted-reference-index.json").write_text(
            json.dumps({"images": index}))
        (self.source / "reconstruction-report.json").write_text(json.dumps({
            "status": "candidate / visual-review-required", "asset_id": "old-statue",
            "manifest_sha256": "1" * 64, "registered_images": 8,
            "staged_independent_images": 8, "undistorted_images": 8}))

    def test_stages_exact_rectified_images_and_colmap_camera_files(self):
        output = self.root / "dataset"
        report = stage_splat_dataset(self.source, output)
        self.assertEqual(report["registered_images"], 8)
        self.assertEqual(report["images"], 8)
        self.assertFalse(report["distribution_allowed"])
        for num in range(8):
            name = f"photo-{num:04d}.jpg"
            self.assertEqual((output / "images" / name).read_bytes(),
                             (self.source / "undistorted/images" / name).read_bytes())
        for name in ("cameras", "images", "points3D"):
            self.assertEqual((output / "sparse/0" / f"{name}.bin").read_bytes(),
                             (self.source / "undistorted/sparse" / f"{name}.bin").read_bytes())

    def test_rejects_tampered_rectified_photo_before_staging(self):
        photo = self.source / "undistorted/images/photo-0000.jpg"
        photo.write_bytes(photo.read_bytes() + b"changed")
        with self.assertRaisesRegex(ReferenceError, "changed"):
            stage_splat_dataset(self.source, self.root / "dataset")
        self.assertFalse((self.root / "dataset").exists())

    def test_detects_photo_modified_after_preflight(self):
        def altered_plan(*args, **kwargs):
            valid_plan = plan_splat_dataset(*args, **kwargs)
            photo = self.source / "undistorted/images/photo-0000.jpg"
            photo.write_bytes(photo.read_bytes() + b"modified-after-preflight")
            return valid_plan

        with patch("kumiki_blender.reconstruction.splat_input.plan_splat_dataset",
                   side_effect=altered_plan):
            with self.assertRaisesRegex(ReferenceError, "changed during staging"):
                stage_splat_dataset(self.source, self.root / "dataset")

    def test_uses_rectified_masks_and_rejects_bad_dimensions(self):
        from PIL import Image

        masks = self.root / "rectified-masks"
        masks.mkdir()
        for num in range(8):
            mask = Image.new("L", (128, 96), 0)
            mask.paste(255, (20, 20, 100, 75))
            mask.save(masks / f"photo-{num:04d}.jpg.mask.png")
        output = self.root / "dataset"
        report = stage_splat_dataset(self.source, output, rectified_masks=masks)
        self.assertEqual(report["masks"], 8)
        self.assertTrue((output / "masks/photo-0000.png").exists())
        Image.new("L", (32, 32), 255).save(masks / "photo-0001.jpg.mask.png")
        with self.assertRaisesRegex(ReferenceError, "dimensions"):
            stage_splat_dataset(self.source, self.root / "another", rectified_masks=masks)

    def test_rejects_incomplete_registration_and_overwrite(self):
        report = json.loads((self.source / "reconstruction-report.json").read_text())
        report["registered_images"] = 7
        (self.source / "reconstruction-report.json").write_text(json.dumps(report))
        with self.assertRaises(ReferenceError):
            stage_splat_dataset(self.source, self.root / "dataset")
        report["registered_images"] = 8
        (self.source / "reconstruction-report.json").write_text(json.dumps(report))
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(ReferenceError, "exists"):
            stage_splat_dataset(self.source, existing)

    def _fake_opensplat(self, behavior="ok"):
        executable = self.root / f"opensplat-{behavior}"
        script = f'''#!{sys.executable}
import pathlib, struct, sys, time
folder = pathlib.Path(sys.argv[1]); assert (folder/'sparse/0/cameras.bin').is_file()
assert len(list((folder/'images').glob('*.jpg'))) == 8
if {behavior!r} == 'fail':
    sys.exit(7)
if {behavior!r} == 'timeout':
    time.sleep(2)
dest = pathlib.Path(sys.argv[sys.argv.index('-o')+1])
header = 'ply\\nformat binary_little_endian 1.0\\nelement vertex 2\\n'
for field in ('x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'):
    header += 'property float '+field+'\\n'
header += 'end_header\\n'
dest.write_bytes(header.encode()+struct.pack('<28f', *(float(x) for x in range(28))))
print('trained from 8 images')
'''
        executable.write_text(script)
        executable.chmod(0o755)
        return executable

    def test_executes_opensplat_and_audits_real_ply_contract(self):
        output = self.root / "splats-r01"
        report = build_splat_revision(self.source, output,
            SplatSettings(iterations=300, opensplat=str(self._fake_opensplat())))
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertEqual(report["artifact"]["gaussians"], 2)
        self.assertFalse(report["distribution_allowed"])
        self.assertTrue((output / "dataset/images/photo-0000.jpg").is_file())
        self.assertTrue((output / "logs/opensplat.log").read_text().find("trained from 8 images") >= 0)
        self.assertTrue((output / "splat.ply").is_file())
        self.assertTrue((output / "splat-report.json").is_file())

    def test_invalid_gaussian_ply_fails_audit(self):
        invalid = self.root / "fake.ply"
        invalid.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n")
        with self.assertRaisesRegex(ReferenceError, "Gaussian"):
            audit_splat_artifact(invalid)

    def test_nonzero_trainer_exit_preserves_failure_report(self):
        output = self.root / "failed-r01"
        report = build_splat_revision(self.source, output,
            SplatSettings(iterations=300, opensplat=str(self._fake_opensplat("fail"))))
        self.assertEqual(report["status"], "failed / review-required")
        self.assertEqual(report["exit_code"], 7)
        self.assertEqual(report["failed_stage"], "opensplat")
        self.assertFalse((output / "splat.ply").exists())

    def test_timeout_reports_failure_and_output_cannot_be_overwritten(self):
        output = self.root / "timeout-r01"
        report = build_splat_revision(self.source, output,
            SplatSettings(iterations=300, timeout_seconds=1,
                          opensplat=str(self._fake_opensplat("timeout"))))
        self.assertEqual(report["status"], "failed / review-required")
        self.assertEqual(report["failed_stage"], "opensplat-timeout")
        with self.assertRaisesRegex(ReferenceError, "exists"):
            build_splat_revision(self.source, output,
                SplatSettings(iterations=300, opensplat=str(self._fake_opensplat())))

    def test_rejects_unsafe_train_configuration_before_output(self):
        with self.assertRaises(ReferenceError):
            build_splat_revision(self.source, self.root / "invalid",
                SplatSettings(iterations=-1, opensplat=str(self._fake_opensplat())))
        self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
