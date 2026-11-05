"""Background removal QA on synthetic rectified statue views."""

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.dense import DenseSettings, plan_dense_reconstruction
from kumiki_blender.reconstruction.masking import (
    _ISNET_SHA256, _box, _load_guides, _verified_model_sha,
    generate_rectified_masks,
)
from kumiki_blender.reconstruction.references import ReferenceError


class DenseMaskTests(unittest.TestCase):
    def test_grounded_sam2_receipts_and_mask_cache_are_separate(self):
        weights = self.root / "sam2-small.pt"
        weights.write_bytes(b"unit-test-checkpoint")
        model_dir = self.root / "grounding"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake-grounding")
        cache = self.root / "shared-mask-cache"
        binary = np.zeros((256, 320), dtype=np.uint8)
        binary[20:220, 108:225] = 255
        from kumiki_blender.reconstruction.masking import _guide
        proposed = _guide({"box": [0.25, 0.1, 0.75, 0.93]}, default_box=(0.12, 0.04, 0.88, 0.96))
        with (patch("kumiki_blender.reconstruction.subject_grounding.resolve_grounding_model",
                    return_value=model_dir),
              patch("kumiki_blender.reconstruction.subject_grounding.grounding_identity",
                    return_value="pinned:test-sha"),
              patch("kumiki_blender.reconstruction.subject_grounding.grounded_guide",
                    return_value=(proposed, {"status": "candidate / text-grounded-box", "cache_hit": True})),
              patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    return_value=(object(), "mps")),
              patch("kumiki_blender.reconstruction.masking._sam2_mask",
                    return_value=(binary, {"components": ["subject"], "missing_parts": [],
                                           "background_seed_violations": 0})) as inference):
            first = generate_rectified_masks(
                self.sparse, self.root / "grounded-masks", segmenter="sam2",
                sam2_checkpoint=weights, sam2_device="mps", cache_dir=cache,
                subject_detector="grounding-dino", grounding_prompt="sphinx statue.",
                grounding_model_dir=model_dir)
        self.assertEqual(first["subject_detector"], "grounding-dino")
        self.assertEqual(first["grounding_identity"], "pinned:test-sha")
        self.assertEqual(first["images"][0]["guide"]["box"], proposed["box"])
        self.assertEqual(first["images"][0]["guide_source"], "grounding-dino / candidate")
        self.assertEqual(first["cache"]["hits"], 3)
        self.assertEqual(inference.call_count, 1)

    def test_sam2_checkpoint_path_survives_progress_reporting(self):
        weights = self.root / "sam2-small.pt"
        weights.write_bytes(b"unit-test-checkpoint")
        output = self.root / "sam2-candidates"
        binary = np.zeros((256, 320), dtype=np.uint8)
        binary[20:220, 108:225] = 255
        with (patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    return_value=(object(), "mps")) as loader,
              patch("kumiki_blender.reconstruction.masking._sam2_mask",
                    return_value=(binary, {"components": ["subject"],
                                           "missing_parts": [],
                                           "background_seed_violations": 0}))):
            report = generate_rectified_masks(
                self.sparse, output, segmenter="sam2", sam2_checkpoint=weights,
                sam2_size="small", sam2_device="mps")
        loader.assert_called_once_with(weights.resolve(), "small", "mps")
        self.assertEqual(report["segmentation_model_checkpoint"], str(weights.resolve()))
        self.assertEqual(len(report["images"]), 4)

    def test_resume_incomplete_sam2_masks_without_repeating_cached_inference(self):
        index_file = self.sparse / "undistorted-reference-index.json"
        index = json.loads(index_file.read_text())
        for count, entry in enumerate(index["images"]):
            photo = self.sparse / "undistorted/images" / entry["filename"]
            with Image.open(photo) as original:
                unique = original.copy()
            unique.putpixel((0, 0), (count * 30, 25, 50))
            unique.save(photo)
            entry["sha256"] = hashlib.sha256(photo.read_bytes()).hexdigest()
        index_file.write_text(json.dumps(index))
        weights = self.root / "sam2-small.pt"
        weights.write_bytes(b"unit-test-checkpoint")
        output = self.root / "interrupted-masks"
        cache = self.root / "cached-masks"
        binary = np.zeros((256, 320), dtype=np.uint8)
        binary[20:220, 108:225] = 255
        details = {"components": ["subject"], "missing_parts": [],
                   "background_seed_violations": 0}
        calls = []

        def interrupted(*_args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("simulated worker interruption")
            return binary, details

        with (patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    return_value=(object(), "mps")),
              patch("kumiki_blender.reconstruction.masking._sam2_mask",
                    side_effect=interrupted)):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                generate_rectified_masks(self.sparse, output, segmenter="sam2",
                                         sam2_checkpoint=weights, sam2_device="mps",
                                         cache_dir=cache)
        incomplete = json.loads((output / "mask-report.json").read_text())
        self.assertEqual(len(incomplete["images"]), 1)
        self.assertEqual(incomplete["status"], "incomplete / manual-mask-review-required")
        original = (output / "mask-report.json").read_bytes()
        with self.assertRaisesRegex(ReferenceError, "Cannot resume masks"):
            generate_rectified_masks(self.sparse, output, segmenter="sam2",
                                     sam2_checkpoint=weights, sam2_size="tiny",
                                     sam2_device="mps", cache_dir=cache, resume=True)
        self.assertEqual((output / "mask-report.json").read_bytes(), original)

        with (patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    return_value=(object(), "mps")) as loader,
              patch("kumiki_blender.reconstruction.masking._sam2_mask",
                    return_value=(binary, details)) as inference):
            completed = generate_rectified_masks(
                self.sparse, output, segmenter="sam2", sam2_checkpoint=weights,
                sam2_device="mps", cache_dir=cache, resume=True)
        self.assertEqual((completed["cache"]["hits"], completed["cache"]["misses"]), (1, 3))
        self.assertEqual(len(completed["images"]), 4)
        loader.assert_called_once()
        self.assertEqual(inference.call_count, 3)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sparse = self.root / "sparse"
        rectified = self.sparse / "undistorted"
        images = rectified / "images"
        images.mkdir(parents=True)
        model = rectified / "sparse"
        model.mkdir()
        for stem in ("cameras", "images", "points3D"):
            (model / (stem + ".bin")).write_bytes(b"synthetic COLMAP")
        entries = []
        for index in range(4):
            name = f"view-{index:02d}.png"
            img = Image.new("RGB", (320, 256), (93, 171, 62))
            img.paste((115, 117, 120), (110, 20, 220, 235))
            img.save(images / name)
            entries.append({"filename": name,
                            "sha256": hashlib.sha256((images / name).read_bytes()).hexdigest(),
                            "dimensions": [320, 256]})
        (self.sparse / "undistorted-reference-index.json").write_text(json.dumps({"images": entries}))
        (self.sparse / "reconstruction-report.json").write_text(json.dumps({
            "status": "candidate / visual-review-required", "registered_images": 4,
            "staged_independent_images": 4, "undistorted_images": 4,
        }))

    def test_invalid_normalized_boxes_and_unknown_guides_rejected(self):
        for value in ([0.2, 0.2, 0.1, 0.9], [0, 0, 0.01, 1],
                      [-0.1, 0, 0.8, 0.9], [0, float("nan"), 1, 1], [True, 0, 1, 1]):
            with self.subTest(value=value), self.assertRaises(ReferenceError):
                _box(value)
        guide = self.root / "guides.json"
        guide.write_text(json.dumps({"images": {"unknown.jpg": {"box": [0.1, 0.1, 0.9, 0.9]}}}))
        with self.assertRaisesRegex(ReferenceError, "filenames"):
            _load_guides(guide, ["view-00.png"], (0.1, 0.1, 0.9, 0.9))
        guide.write_text(json.dumps({"default": {"box": [0.2, 0.1, 0.8, 0.9]},
                                     "images": {"view-00.png": {"foreground_points": [[0.5, 0.5]]}}}))
        parsed, sha = _load_guides(guide, ["view-00.png", "view-01.png"],
                                   (0.1, 0.1, 0.9, 0.9))
        self.assertEqual(parsed["view-00.png"]["box"], [0.2, 0.1, 0.8, 0.9])
        self.assertEqual(parsed["view-00.png"]["foreground_points"], [[0.5, 0.5]])
        self.assertEqual(len(sha), 64)

    def test_all_masks_are_binary_pinned_and_usable_by_openmvs(self):
        # Fake only GrabCut inference; real Pillow encoding and dense mask audit run.
        image_mask = np.zeros((256, 320), dtype=np.uint8)
        image_mask[18:235, 109:221] = 255
        output = self.root / "masks-r01"
        with patch.dict(sys.modules, {"cv2": object()}), \
             patch("kumiki_blender.reconstruction.masking._grabcut", return_value=image_mask):
            report = generate_rectified_masks(self.sparse, output)
        self.assertEqual(report["status"], "candidate / inspect-every-mask-before-dense-reconstruction")
        self.assertFalse(report["approved_for_reconstruction"])
        self.assertEqual(len(report["images"]), 4)
        self.assertTrue((output / report["contact_sheet"]).is_file())
        self.assertIn("Unreviewed center-box guess; add per-photo guides for busy backgrounds",
                      report["images"][0]["warnings"])
        for entry in report["images"]:
            path = output / entry["mask"]
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["mask_sha256"])
            self.assertTrue((output / entry["preview"]).is_file())
        plan = plan_dense_reconstruction(self.sparse, self.root / "dense",
                                         DenseSettings(min_registered_images=3),
                                         rectified_masks=output / "masks", executables={})
        self.assertEqual(len(plan["rectified_masks"]), 4)
        self.assertFalse((self.root / "dense").exists())
        first = output / report["images"][0]["mask"]
        with Image.open(first) as mask:
            mask.putpixel((0, 0), 254)
            mask.save(first)
        with self.assertRaisesRegex(ReferenceError, "binary pixels"):
            plan_dense_reconstruction(self.sparse, self.root / "dense",
                                      DenseSettings(min_registered_images=3),
                                      rectified_masks=output / "masks", executables={})

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "optional OpenCV segmentation extra not installed")
    def test_real_opencv_grabcut_separates_a_synthetic_subject(self):
        output = self.root / "inferred"
        report = generate_rectified_masks(self.sparse, output, max_side=320)
        with Image.open(output / report["images"][0]["mask"]) as mask:
            self.assertEqual(mask.getpixel((160, 100)), 255)
            self.assertEqual(mask.getpixel((25, 25)), 0)
        self.assertGreater(report["images"][0]["foreground_fraction"], 0.15)
        self.assertLess(report["images"][0]["foreground_fraction"], 0.6)

    def test_bad_parameters_leave_no_partial_revision(self):
        for kwargs in ({"max_side": 32}, {"iterations": 20},
                       {"center_box": (0.0, 0.0, 0.01, 0.01)},
                       {"segmenter": "unknown"}):
            with self.assertRaises(ReferenceError):
                generate_rectified_masks(self.sparse, self.root / "output", **kwargs)
            self.assertFalse((self.root / "output").exists())

    def test_cli_masks_bad_guide_fails_before_output(self):
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(sys.modules, {"cv2": object()}):
            self.assertEqual(main(["prepare-dense-masks", "--reconstruction", str(self.sparse),
                                   "--output", str(self.root / "bad"),
                                   "--center-box", "0.1", "0.1", "0.1", "0.8"]), 2)
        self.assertFalse((self.root / "bad").exists())

    def test_semantic_model_reuses_one_session_and_still_requires_review(self):
        matte = Image.new("L", (320, 256), 0)
        matte.paste(240, (105, 18, 222, 236))
        rembg = Mock()
        rembg.new_session.return_value = object()
        rembg.remove.return_value = matte
        output = self.root / "semantic"
        with patch.dict(sys.modules, {"rembg": rembg}), \
             patch("kumiki_blender.reconstruction.masking._verified_model_sha",
                   return_value=_ISNET_SHA256):
            report = generate_rectified_masks(self.sparse, output, segmenter="rembg")
        rembg.new_session.assert_called_once_with("isnet-general-use")
        self.assertEqual(rembg.remove.call_count, 4)
        self.assertEqual(report["segmentation_model"], "isnet-general-use")
        self.assertEqual(report["segmentation_model_sha256"], _ISNET_SHA256)
        self.assertFalse(report["approved_for_reconstruction"])
        self.assertEqual(len(report["images"]), 4)
        self.assertTrue((output / report["contact_sheet"]).is_file())
        with self.assertRaisesRegex(ReferenceError, "use grabcut"):
            generate_rectified_masks(self.sparse, self.root / "bad",
                                     segmenter="rembg", guides=self.root / "unused.json")

    def test_semantic_model_weights_must_match_pinned_sha(self):
        model_dir = self.root / "model"
        model_dir.mkdir()
        (model_dir / "isnet-general-use.onnx").write_bytes(b"different untrusted weights")
        session = Mock()
        session.model_dir.return_value = model_dir
        with self.assertRaisesRegex(ReferenceError, "weights differ"):
            _verified_model_sha(session)


if __name__ == "__main__":
    unittest.main()
