"""End-to-end photo-folder-to-mesh orchestration; fake workers, no external GPUs."""

import contextlib
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.photo_model import (
    PhotoModelSettings, build_photo_model, plan_photo_model, plan_registered_photo_model,
)
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.splat_input import load_photo_folder


class PhotoModelTests(unittest.TestCase):
    @staticmethod
    def _minimal_glb():
        payload = json.dumps({"asset": {"version": "2.0"},
                              "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                              "nodes": [{"mesh": 0}]}).encode("utf-8")
        payload += b" " * (-len(payload) % 4)
        return (struct.pack("<4sII", b"glTF", 2, len(payload) + 20)
                + struct.pack("<I4s", len(payload), b"JSON") + payload)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        for index in range(8):
            with Image.new("RGB", (96, 96), (index * 25, 70, 90)) as photo:
                photo.save(self.photos / f"angle-{index:02}.jpg")
        self.references = load_photo_folder(self.photos, "sphinx-photo-test")
        self.settings = PhotoModelSettings(render_previews=False)
        self.dependencies = ({"pycolmap": "python-module",
                              "InterfaceCOLMAP": "/bin/true",
                              "DensifyPointCloud": "/bin/true",
                              "ReconstructMesh": "/bin/true",
                              "RefineMesh": "/bin/true",
                              "TextureMesh": "/bin/true",
                              "blender": "/bin/true"}, [])

    def test_preflight_does_not_mutate_output_or_input(self):
        target = self.root / "result"
        report = plan_photo_model(self.references, target, self.settings,
                                  dependencies=self.dependencies)
        self.assertEqual(report["status"], "ready-to-reconstruct")
        self.assertEqual(report["independent_images"], 8)
        self.assertTrue(report["missing_photo_rights"])
        self.assertFalse(target.exists())
        self.assertEqual(len(list(self.photos.iterdir())), 8)

    def test_complete_pipeline_emits_real_worker_result_not_proxy(self):
        output = self.root / "result"
        captured = []

        def sparse(refs, directory, **kwargs):
            captured.append("sparse")
            self.assertEqual(len(refs.images), 8)
            self.assertTrue(kwargs["undistort"])
            directory.mkdir()
            (directory / "reconstruction-report.json").write_text("{}")
            return {"status": "candidate / visual-review-required",
                    "registered_images": 8, "staged_independent_images": 8,
                    "undistorted_images": 8}

        def dense(revision, directory, settings):
            captured.append("dense")
            self.assertEqual(revision, output / "sparse")
            self.assertTrue(settings.texture)
            directory.mkdir()
            (directory / "dense-report.json").write_text("{}")
            return {"status": "dense-mesh-candidate / visual-review-required",
                    "mesh": {"ply": "refined.ply"}}

        def blender(command, log, timeout):
            captured.append("blender")
            self.assertIn("--textured", command)
            self.assertIn("--no-renders", command)
            review = output / "review"
            review.mkdir()
            (review / "private-textured-review.glb").write_bytes(self._minimal_glb())
            (review / "private-textured-review.blend").write_bytes(b"BLENDER")
            (review / "blender-review-report.json").write_text(json.dumps({
                "status": "private-Blender-mesh-candidate / artistic-and-rights-review-required",
                "review_glb": "private-textured-review.glb",
                "review_blend": "private-textured-review.blend"}))
            return 0

        report = build_photo_model(
            self.references, output, self.settings, dependencies=self.dependencies,
            sparse_builder=sparse, dense_builder=dense, blender_runner=blender)
        self.assertEqual(captured, ["sparse", "dense", "blender"])
        self.assertTrue(report["status"].startswith("private-GLB-candidate"))
        self.assertEqual((output / "model.glb").read_bytes(),
                         (output / "review/private-textured-review.glb").read_bytes())
        self.assertEqual(report["model_sha256"],
                         hashlib.sha256((output / "model.glb").read_bytes()).hexdigest())
        self.assertEqual(report["glb_audit"]["meshes"], 1)
        self.assertEqual(json.loads((output / "photo-model-report.json").read_text())["model_glb"],
                         "model.glb")

    def test_registration_failure_never_calls_mesher_or_exports_model(self):
        output = self.root / "failed"

        def sparse(refs, directory, **kwargs):
            directory.mkdir()
            return {"status": "review-required", "registered_images": 2,
                    "staged_independent_images": 8, "undistorted_images": 0,
                    "failed_stage": "mapper"}

        def unexpected(*args, **kwargs):
            self.fail("Downstream worker ran without valid overlapping-camera registration")

        report = build_photo_model(
            self.references, output, self.settings, dependencies=self.dependencies,
            sparse_builder=sparse, dense_builder=unexpected, blender_runner=unexpected)
        self.assertEqual(report["failed_stage"], "sparse-registration")
        self.assertEqual(report["status"], "review-required")
        self.assertFalse((output / "model.glb").exists())

    def test_blender_does_not_promote_invalid_glb(self):
        output = self.root / "bad-glb"

        def sparse(refs, directory, **kwargs):
            directory.mkdir()
            return {"status": "candidate / visual-review-required", "registered_images": 8,
                    "staged_independent_images": 8, "undistorted_images": 8}

        def dense(revision, directory, settings):
            directory.mkdir()
            return {"status": "dense-mesh-candidate / visual-review-required"}

        def blender(command, log, timeout):
            review = output / "review"
            review.mkdir()
            (review / "private-textured-review.glb").write_bytes(b"not a glb")
            (review / "blender-review-report.json").write_text(json.dumps({
                "status": "private-Blender-mesh-candidate / review-required",
                "review_glb": "private-textured-review.glb"}))
            return 0

        report = build_photo_model(self.references, output, self.settings,
                                   dependencies=self.dependencies, sparse_builder=sparse,
                                   dense_builder=dense, blender_runner=blender)
        self.assertEqual(report["failed_stage"], "blender-export")
        self.assertFalse((output / "model.glb").exists())

    def test_missing_dependencies_fail_before_creating_revision(self):
        target = self.root / "no-tools"
        deps = ({}, ["pycolmap", "InterfaceCOLMAP"])
        self.assertEqual(plan_photo_model(self.references, target, self.settings,
                                          dependencies=deps)["status"],
                         "blocked-missing-dependencies")
        with self.assertRaisesRegex(ReferenceError, "missing"):
            build_photo_model(self.references, target, self.settings, dependencies=deps)
        self.assertFalse(target.exists())

    def test_refuses_existing_output_or_inside_photo_folder(self):
        with self.assertRaisesRegex(ReferenceError, "outside"):
            plan_photo_model(self.references, self.photos / "result", self.settings,
                             dependencies=self.dependencies)
        with self.assertRaisesRegex(ReferenceError, "already exists"):
            plan_photo_model(self.references, self.root, self.settings,
                             dependencies=self.dependencies)

    def test_cli_plan_from_folder_and_reject_missing_asset_id(self):
        output = io.StringIO()
        target = self.root / "cli-r01"
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["plan-photo-model", "--photos", str(self.photos),
                                   "--asset-id", "test-sphinx", "--output", str(target)]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["source_images"], 8)
        self.assertIn(report["status"], ("blocked-missing-dependencies", "ready-to-reconstruct"))
        self.assertFalse(target.exists())
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            self.assertEqual(main(["build-photo-model", "--photos", str(self.photos),
                                   "--output", str(target)]), 2)
        self.assertIn("--asset-id", error.getvalue())
        self.assertFalse(target.exists())

    def test_unrelated_web_photos_are_not_treated_as_sufficient_overlap(self):
        output = self.root / "inadequate-overlap"

        def sparse(refs, directory, **kwargs):
            directory.mkdir()
            return {"status": "candidate / visual-review-required",
                    "registered_images": 3, "staged_independent_images": 8,
                    "undistorted_images": 3}

        report = build_photo_model(self.references, output, self.settings,
                                   dependencies=self.dependencies, sparse_builder=sparse)
        self.assertEqual(report["failed_stage"], "sparse-registration")
        self.assertFalse((output / "model.glb").exists())

    def _registered_fixture(self):
        from tests.test_reconstruction_splats import SplatDatasetTests

        fixture = SplatDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        return fixture.source

    def test_existing_registered_cameras_skip_sfm_and_produce_glb(self):
        source = self._registered_fixture()
        output = self.root / "reuse-existing"
        self.assertEqual(plan_registered_photo_model(
            source, output, self.settings, dependencies=self.dependencies)["registered_images"], 8)
        calls = []

        def should_not_register(*args, **kwargs):
            self.fail("COLMAP reran despite an existing registered camera revision")

        def dense(reconstruction, directory, settings):
            calls.append("dense")
            self.assertEqual(reconstruction, source.resolve())
            directory.mkdir()
            return {"status": "dense-mesh-candidate / visual-review-required"}

        def blender(command, log, timeout):
            calls.append("blender")
            review = output / "review"
            review.mkdir()
            (review / "private-textured-review.glb").write_bytes(self._minimal_glb())
            (review / "blender-review-report.json").write_text(json.dumps({
                "status": "private-Blender-mesh-candidate / artistic-review-required",
                "review_glb": "private-textured-review.glb",
                "review_blend": "private-textured-review.blend"}))
            return 0

        report = build_photo_model(
            None, output, self.settings, registered_reconstruction=source,
            dependencies=self.dependencies, sparse_builder=should_not_register,
            dense_builder=dense, blender_runner=blender)
        self.assertEqual(calls, ["dense", "blender"])
        self.assertEqual(report["source_mode"], "reuse-registered-cameras")
        self.assertEqual(report["source_revision"], str(source.resolve()))
        self.assertTrue(report["status"].startswith("private-GLB-candidate"))
        self.assertTrue((output / "model.glb").is_file())
        self.assertFalse((output / "sparse").exists())
        self.assertTrue((source / "reconstruction-report.json").exists())

    def test_existing_camera_masks_are_validated_and_passed_to_mesher(self):
        source = self._registered_fixture()
        masks = self.root / "reviewed-masks"
        masks.mkdir()
        for i in range(8):
            mask = Image.new("L", (128, 96), 0)
            mask.paste(255, (20, 20, 100, 75))
            mask.save(masks / f"photo-{i:04}.jpg.mask.png")
        target = self.root / "masked"
        plan = plan_registered_photo_model(
            source, target, self.settings, rectified_masks=masks,
            dependencies=self.dependencies)
        self.assertEqual(len(plan["rectified_masks"]), 8)
        self.assertEqual(plan["rectified_mask_source"], str(masks.resolve()))

        def dense(reconstruction, directory, settings, *, rectified_masks):
            self.assertEqual(reconstruction, source.resolve())
            self.assertEqual(Path(rectified_masks), masks)
            directory.mkdir()
            return {"status": "review-required"}  # No fabricated mesh.

        result = build_photo_model(
            None, target, self.settings, registered_reconstruction=source,
            rectified_masks=masks, dependencies=self.dependencies, dense_builder=dense)
        self.assertEqual(result["failed_stage"], "dense-mesh")
        self.assertFalse((target / "model.glb").exists())
        mask = masks / "photo-0001.jpg.mask.png"
        Image.new("L", (32, 32), 255).save(mask)
        with self.assertRaisesRegex(ReferenceError, "dimensions"):
            plan_registered_photo_model(source, self.root / "masked-next", self.settings,
                                        rectified_masks=masks, dependencies=self.dependencies)
        self.assertTrue(target.exists())  # Failed revisions remain inspectable.

    def test_registered_photo_plan_detects_source_changes_and_avoids_pycolmap(self):
        source = self._registered_fixture()
        target = self.root / "registered-plan"
        no_sparse_dependency = ({k: v for k, v in self.dependencies[0].items()
                                 if k != "pycolmap"}, [])
        plan = plan_registered_photo_model(
            source, target, self.settings, dependencies=no_sparse_dependency)
        self.assertEqual(plan["status"], "ready-to-reconstruct")
        self.assertFalse(target.exists())
        self.assertIn("undistorted/sparse/images.bin", plan["source_evidence_sha256"])
        with self.assertRaisesRegex(ReferenceError, "outside"):
            plan_registered_photo_model(source, source / "nested", self.settings,
                                        dependencies=no_sparse_dependency)
        photo = source / "undistorted/images/photo-0000.jpg"
        photo.write_bytes(photo.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ReferenceError, "changed"):
            plan_registered_photo_model(source, target, self.settings,
                                        dependencies=no_sparse_dependency)

    def test_cli_plan_registered_cameras_without_asset_id(self):
        source = self._registered_fixture()
        target = self.root / "cli-existing"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["plan-photo-model", "--reconstruction", str(source),
                                   "--output", str(target)]), 0)
        self.assertEqual(json.loads(output.getvalue())["source_mode"],
                         "reuse-registered-cameras")
        self.assertFalse(target.exists())
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            self.assertEqual(main(["plan-photo-model", "--reconstruction", str(source),
                                   "--asset-id", "not-applicable", "--output", str(target)]), 2)
        self.assertIn("--asset-id", error.getvalue())


if __name__ == "__main__":
    unittest.main()
