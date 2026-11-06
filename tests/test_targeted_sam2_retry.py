"""Offline targeted SAM2 retry tests, with no model downloads or source mutations."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from kumiki_blender.reconstruction.masking import _DEFAULT_BOX, _load_guides
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.targeted_sam2_retry import retry_sam2_masks


class TargetedSAM2RetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "camera-revision"
        (self.source / "undistorted/images").mkdir(parents=True)
        self.masks = self.root / "current-masks"
        self.masks.mkdir()
        self.images = []
        self.mask_records = []
        for name in ("view-1.png", "view-2.png", "view-3.png"):
            photo_path = self.source / "undistorted/images" / name
            mask_path = self.masks / (name + ".mask.png")
            Image.new("RGB", (100, 80), (100, 80, 50)).save(photo_path)
            Image.new("L", (100, 80), 0).save(mask_path)
            self.images.append({"filename": name,
                                "sha256": hashlib.sha256(photo_path.read_bytes()).hexdigest()})
            self.mask_records.append({"filename": name,
                                      "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest()})
        (self.source / "undistorted-reference-index.json").write_text(
            json.dumps({"images": self.images}))
        self.checkpoint = self.root / "sam2.pt"
        self.checkpoint.write_bytes(b"synthetic-test-only-model")
        self.guide_file = self.root / "guides.json"
        self.guide_file.write_text(json.dumps({"images": {"view-1.png": {
            "box": [0.1, 0.1, 0.9, 0.9],
            "part_points": {"head": [[0.5, 0.3]], "body": [[0.5, 0.7]]},
            "background_points": [[0.1, 0.1]],
        }}}))
        self.plan = {"source_revision": str(self.source),
                     "undistorted_images": [row["filename"] for row in self.images],
                     "rectified_masks": self.mask_records}

    def run_retry(self, *, targets=None, guides=None):
        with (patch("kumiki_blender.reconstruction.targeted_sam2_retry.plan_dense_reconstruction",
                    return_value=self.plan),
              patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    return_value=(object(), "cpu")),
              patch("kumiki_blender.reconstruction.masking._sam2_mask",
                    side_effect=self.fake_mask)):
            return retry_sam2_masks(
                self.source, self.masks, guides or self.guide_file,
                self.root / "output", targets=targets or ["view-1.png"],
                sam2_checkpoint=self.checkpoint, sam2_device="cpu")

    @staticmethod
    def fake_mask(photo, guide, predictor, max_side):
        result = np.zeros((photo.height, photo.width), dtype=np.uint8)
        for points in guide["part_points"].values():
            for x, y in points:
                cx, cy = round(x * (photo.width-1)), round(y * (photo.height-1))
                result[cy-3:cy+4, cx-3:cx+4] = 255
        return result, {"missing_parts": []}

    def test_outputs_only_private_candidates_and_keeps_original_masks(self):
        before = {row["filename"]: (self.masks / (row["filename"] + ".mask.png")).read_bytes()
                  for row in self.images}
        report = self.run_retry()
        self.assertEqual(report["status"], "review-only-sam2-candidates / manual-photo-qa-required")
        self.assertFalse(report["approved_for_reconstruction"])
        saved_guide = self.root / "output" / report["guides_copy"]
        self.assertEqual(saved_guide.read_bytes(), self.guide_file.read_bytes())
        self.assertEqual(hashlib.sha256(saved_guide.read_bytes()).hexdigest(), report["guides_sha256"])
        self.assertEqual(report["images"][0]["status"], "candidate / human-photo-review-required")
        self.assertEqual(report["images"][0]["prompt_qa"]["missing_positive_points"], 0)
        self.assertEqual(report["images"][0]["original_prompt_qa"]["missing_positive_points"], 2)
        self.assertEqual(report["images"][0]["prompt_qa"]["anatomical_parts"]["head"],
                         {"points": 1, "missed": 0})
        self.assertGreater(report["images"][0]["pixel_change_qa"]["newly_included_fraction"], 0)
        self.assertEqual(report["images"][0]["pixel_change_qa"]["newly_excluded_fraction"], 0)
        self.assertTrue((self.root / "output/candidates/view-1.png.mask.png").is_file())
        self.assertTrue((self.root / "output/review/view-1.png.jpg").is_file())
        self.assertFalse((self.root / "output/masks").exists())
        self.assertEqual(set(before), {row["filename"] for row in self.images})
        for name, raw in before.items():
            self.assertEqual((self.masks / (name + ".mask.png")).read_bytes(), raw)

    def test_missing_anatomical_points_or_bad_target_fails_without_writes(self):
        self.guide_file.write_text(json.dumps({"images": {"view-1.png": {
            "box": [0.1, 0.1, 0.9, 0.9], "background_points": [[0.1, 0.1]]}}}))
        with self.assertRaisesRegex(ReferenceError, "anatomical positive clicks"):
            self.run_retry()
        self.assertFalse((self.root / "output").exists())
        with self.assertRaisesRegex(ReferenceError, "explicit anatomical guide"):
            self.run_retry(targets=["view-2.png"])
        self.assertFalse((self.root / "output").exists())

    def test_missing_model_runtime_does_not_create_output(self):
        with (patch("kumiki_blender.reconstruction.targeted_sam2_retry.plan_dense_reconstruction",
                    return_value=self.plan),
              patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    side_effect=ReferenceError("SAM2 runtime unavailable"))):
            with self.assertRaisesRegex(ReferenceError, "runtime unavailable"):
                retry_sam2_masks(self.source, self.masks, self.guide_file,
                                 self.root / "output", targets=["view-1.png"],
                                 sam2_checkpoint=self.checkpoint)
        self.assertFalse((self.root / "output").exists())

    def test_uploaded_sphinx_review_guide_is_schema_valid(self):
        guide = Path(__file__).resolve().parents[1] / "examples/sphinx-sam2-retry-3-views.example.json"
        names = ["photo-0007.png", "photo-0031.png", "photo-0040.png"]
        resolved, sha = _load_guides(guide, names, _DEFAULT_BOX)
        self.assertEqual(len(sha), 64)
        for name in names:
            points = [point for rows in resolved[name]["part_points"].values() for point in rows]
            self.assertGreaterEqual(len(points), 8)
            self.assertGreaterEqual(len(resolved[name]["background_points"]), 3)
            self.assertTrue(any(abs(a[1] - b[1]) > 0.12 for a in points for b in points))
            self.assertTrue(all(math.dist(a, b) >= 0.035 for a in points
                                for b in resolved[name]["background_points"]))

    def test_refuses_existing_or_nested_destinations_without_mutating_source(self):
        with patch("kumiki_blender.reconstruction.targeted_sam2_retry.plan_dense_reconstruction",
                   return_value=self.plan):
            for output in (self.masks / "candidate", self.source / "new", self.root):
                with self.subTest(output=output), self.assertRaisesRegex(
                        ReferenceError, "new directory separate"):
                    retry_sam2_masks(self.source, self.masks, self.guide_file, output,
                                     targets=["view-1.png"], sam2_checkpoint=self.checkpoint)
        self.assertFalse((self.masks / "candidate").exists())
        self.assertFalse((self.source / "new").exists())


if __name__ == "__main__":
    unittest.main()
