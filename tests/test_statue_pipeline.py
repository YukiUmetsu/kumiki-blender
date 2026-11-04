"""Stage orchestration regression tests without native Blender/COLMAP models."""

import contextlib
import io
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.statue_pipeline import (
    StatuePipelineSettings, _coverage_warnings, _mask_warnings,
    _infer_front_photo, _resolve_sam2_checkpoint, build_statue_glb,
)


class StatuePipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        for index in range(8):
            Image.new("RGB", (64, 64), (90, 70, 100 + index)).save(
                self.photos / f"image-{index}.png")
        self.output = self.root / "run"
        self.model = self.root / "model.pt"
        self.model.write_bytes(b"local-checkpoint")
        self.settings = StatuePipelineSettings(front_photo="image-0.png",
                                                sam2_checkpoint=self.model)

    @staticmethod
    def coverage():
        poses = [-110, -80, -30, 0, 30, 80, 110, 135]
        return {"angle_reference": {"kind": "annotated-front"},
                "existing_photo_priorities": [],
                "images": [{"registered_in_best_model": True,
                            "estimated_view": {"relative_azimuth_deg": angle,
                                               "relative_elevation_deg": (20 if index > 5 else 0)}}
                           for index, angle in enumerate(poses)]}

    def runner(self, *, mask_bad=False, coverage_bad=False):
        calls = []

        def run(argv, log, _environment):
            stage = log.stem
            calls.append(stage)
            if stage == "cameras":
                folder = self.output / "cameras"
                folder.mkdir()
                (folder / "reconstruction-report.json").write_text(json.dumps({
                    "status": "candidate / review-required", "registered_images": 8,
                    "staged_independent_images": 8, "undistorted_images": 8}))
            elif stage == "coverage":
                folder = self.output / "coverage"
                folder.mkdir()
                info = self.coverage()
                if coverage_bad:
                    info["images"] = info["images"][:2]
                (folder / "photo-advice.json").write_text(json.dumps(info))
            elif stage == "masks":
                folder = self.output / "masks"
                (folder / "masks").mkdir(parents=True)
                images = []
                for index in range(8):
                    filename = f"photo-{index:04}.png"
                    Image.new("L", (64, 64), 255).save(folder / "masks" / (filename + ".mask.png"))
                    images.append({"filename": filename, "foreground_fraction": 0.4,
                                   "warnings": (["No visible-anatomy landmarks"] if mask_bad else []),
                                   "missing_parts": (["head"] if mask_bad else []),
                                   "background_seed_violations": 0})
                (folder / "mask-report.json").write_text(json.dumps({"images": images}))
            elif stage.startswith("mask-audit-"):
                folder = self.output / stage
                folder.mkdir()
                (folder / "mask-consistency-report.json").write_text(json.dumps({
                    "mixed_label_fraction": 0.01,
                    "images": [{"filename": f"photo-{i:04}.png"} for i in range(8)]}))
            elif stage == "dense":
                folder = self.output / "dense"
                folder.mkdir()
                self.assertIn("--no-refine", argv)
                self.assertIn("--resolution-level", argv)
                self.assertIn("--rectified-masks", argv)
                (folder / "dense-report.json").write_text(json.dumps({
                    "status": "dense-mesh-candidate / review-required"}))
            elif stage == "review":
                folder = self.output / "review"
                folder.mkdir()
                (folder / "private-dense-review.glb").write_bytes(b"fake-audited-glb")
                (folder / "blender-review-report.json").write_text(json.dumps({
                    "status": "private-Blender-mesh-candidate / review-required",
                    "review_glb": "private-dense-review.glb"}))
            else:
                raise AssertionError(f"Unexpected stage: {stage}")
            return 0

        return run, calls

    def build(self, run, settings=None, *, resume=False):
        with patch("kumiki_blender.reconstruction.statue_pipeline.shutil.which", return_value="/bin/true"), \
             patch("kumiki_blender.reconstruction.statue_pipeline.importlib.util.find_spec", return_value=True), \
             patch("kumiki_blender.reconstruction.statue_pipeline._possible_blue_sky_leaks", return_value=[]), \
             patch("kumiki_blender.reconstruction.statue_pipeline.audit_glb", return_value={"meshes": 1}):
            return build_statue_glb(self.photos, "great-sphinx-test", self.output,
                                    settings or self.settings, resume=resume, runner=run)

    def test_one_command_runs_every_stage_and_copies_audited_glb(self):
        run, calls = self.runner()
        report = self.build(run)
        self.assertEqual(calls[:3], ["cameras", "coverage", "masks"])
        self.assertEqual(calls[-2:], ["dense", "review"])
        self.assertEqual((self.output / "model.glb").read_bytes(), b"fake-audited-glb")
        self.assertTrue(report["status"].startswith("private-GLB-candidate"))
        self.assertEqual(report["glb_audit"]["meshes"], 1)

    def test_missing_profiles_stop_before_masks_and_mesh(self):
        run, calls = self.runner(coverage_bad=True)
        report = self.build(run)
        self.assertEqual(calls, ["cameras", "coverage"])
        self.assertEqual(report["status"], "blocked / missing-photo-coverage")
        self.assertFalse((self.output / "model.glb").exists())

    def test_bad_masks_block_mesh_and_resume_with_corrected_revision(self):
        run, calls = self.runner(mask_bad=True)
        blocked = self.build(run)
        self.assertEqual(blocked["status"], "blocked / mask-review-required")
        self.assertNotIn("dense", calls)
        # A separate edited revision can be supplied on retry, without any
        # re-registration or mutation of the initial SAM2 mask candidates.
        corrected = self.root / "corrected"
        corrected.mkdir()
        for source in (self.output / "masks/masks").glob("*.mask.png"):
            (corrected / source.name).write_bytes(source.read_bytes())
        Image.new("L", (64, 64), 0).save(corrected / "photo-0000.png.mask.png")
        revised_images = []
        for path in corrected.glob("*.mask.png"):
            revised_images.append({"filename": path.name[:-len(".mask.png")],
                                   "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (self.root / "mask-report.json").write_text(json.dumps({
            "source_revision": str((self.output / "cameras").resolve()),
            "images": revised_images}))
        revised = StatuePipelineSettings(front_photo="image-0.png",
                                         sam2_checkpoint=self.model, masks_dir=corrected)
        run2, calls2 = self.runner()
        report = self.build(run2, revised, resume=True)
        self.assertEqual(len(calls2), 3)
        self.assertTrue(calls2[0].startswith("mask-audit-"))
        self.assertTrue(report["status"].startswith("private-GLB-candidate"))

    def test_source_changes_refuse_resume(self):
        run, _calls = self.runner(coverage_bad=True)
        self.build(run)
        Image.new("RGB", (64, 64), "red").save(self.photos / "new-view.png")
        with self.assertRaisesRegex(ReferenceError, "source photos"):
            self.build(run, resume=True)

    def test_changed_model_weights_refuse_resume_even_at_same_checkpoint_path(self):
        run, _calls = self.runner(coverage_bad=True)
        self.build(run)
        self.model.write_bytes(b"changed-local-checkpoint")
        with self.assertRaisesRegex(ReferenceError, "quality settings changed"):
            self.build(run, resume=True)

    def test_preflight_auto_checkpoint_discovery_and_offline_failure(self):
        model, source = _resolve_sam2_checkpoint(self.model, "small", allow_download=False)
        self.assertEqual((model, source), (self.model.resolve(), "explicit"))
        cached = self.root / "models"
        cached.mkdir()
        installed = cached / "sam2.1_hiera_small.pt"
        installed.write_bytes(b"cached official-model fixture")
        with patch.dict("os.environ", {"KUMIKI_SAM2_CHECKPOINT": "", "SAM2_CHECKPOINT": ""}):
            auto, source = _resolve_sam2_checkpoint(None, "small", allow_download=False,
                                                     cache_root=cached)
        self.assertEqual((auto, source), (installed.resolve(), "discovered"))
        settings = StatuePipelineSettings(front_photo="image-0.png", auto_download_model=False)
        with patch("kumiki_blender.reconstruction.statue_pipeline._resolve_sam2_checkpoint",
                   side_effect=ReferenceError("SAM2 model not found locally")):
            with self.assertRaisesRegex(ReferenceError, "not found locally"):
                self.build(self.runner()[0], settings)
        self.assertFalse(self.output.exists())

    def test_cli_fails_before_creating_anything_without_checkpoint(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), patch(
                "kumiki_blender.reconstruction.statue_pipeline.importlib.util.find_spec",
                return_value=None):
            code = main(["build-statue-glb", "--photos", str(self.photos),
                         "--asset-id", "great-sphinx-test", "--output", str(self.output)])
        self.assertEqual(code, 2)
        self.assertIn("SAM2 is not installed", stderr.getvalue())
        self.assertFalse(self.output.exists())

    def test_auto_front_chooses_registered_frontal_photo_and_records_uncertainty(self):
        sparse = self.root / "camera-revision"
        sparse.mkdir()
        (sparse / "staged-reference-index.json").write_text(json.dumps({"images": [
            {"filename": "photo-0001.png", "original_filename": "sphinx-left-side.jpg"},
            {"filename": "photo-0002.png", "original_filename": "1280px-front_view.jpeg"},
            {"filename": "photo-0003.png", "original_filename": "sphinx-top-view.jpg"}]}))
        result = _infer_front_photo(sparse, {"registered_filenames": ["photo-0001.png", "photo-0002.png"]})
        self.assertEqual(result["filename"], "1280px-front_view.jpeg")
        self.assertEqual(result["confidence"], "tentative")
        unknown = _infer_front_photo(sparse, {"registered_filenames": ["photo-0001.png"]})
        self.assertIsNone(unknown["filename"])

    def test_geometry_and_masks_gates_are_conservative(self):
        self.assertTrue(_coverage_warnings(self.coverage(), 0.75, 0.8))
        self.assertTrue(_mask_warnings({"images": [{"filename": "bad.png",
                    "foreground_fraction": 0.5,
                    "warnings": ["No visible-anatomy landmarks"],
                    "missing_parts": [], "background_seed_violations": 0}]},
                    {"mixed_label_fraction": 0.3}, 0.12))


if __name__ == "__main__":
    unittest.main()
