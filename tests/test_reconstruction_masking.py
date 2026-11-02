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
from unittest.mock import patch

import numpy as np
from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.dense import DenseSettings, plan_dense_reconstruction
from kumiki_blender.reconstruction.masking import (
    _box, _load_guides, generate_rectified_masks,
)
from kumiki_blender.reconstruction.references import ReferenceError


class DenseMaskTests(unittest.TestCase):
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
                       {"center_box": (0.0, 0.0, 0.01, 0.01)}):
            with self.assertRaises(ReferenceError):
                generate_rectified_masks(self.sparse, self.root / "output", **kwargs)
            self.assertFalse((self.root / "output").exists())

    def test_cli_masks_bad_guide_fails_before_output(self):
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(sys.modules, {"cv2": object()}):
            self.assertEqual(main(["prepare-dense-masks", "--reconstruction", str(self.sparse),
                                   "--output", str(self.root / "bad"),
                                   "--center-box", "0.1", "0.1", "0.1", "0.8"]), 2)
        self.assertFalse((self.root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
