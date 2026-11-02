"""Pixel-accurate source-pinned subject-mask cleanup and polygon overrides."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.dense import DenseSettings, plan_dense_reconstruction
from kumiki_blender.reconstruction.mask_refinement import refine_rectified_masks
from kumiki_blender.reconstruction.references import ReferenceError


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaskRefinementTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sparse = self.root / "sparse"
        images = self.sparse / "undistorted/images"
        images.mkdir(parents=True)
        model = self.sparse / "undistorted/sparse"
        model.mkdir()
        for stem in ("cameras", "images", "points3D"):
            (model / (stem + ".bin")).write_bytes(b"first-party test SfM")
        self.source_masks = self.root / "masks-r01"
        (self.source_masks / "masks").mkdir(parents=True)
        sparse_entries, mask_entries = [], []
        for idx in range(3):
            filename = f"view-{idx:02d}.jpg"
            image = images / filename
            Image.new("RGB", (100, 80), (130, 130, 130)).save(image)
            sparse_entries.append({"filename": filename, "sha256": _hash(image),
                                   "dimensions": [100, 80]})
            mask = Image.new("L", (100, 80), 0)
            mask.paste(255, (20, 10, 80, 76))
            mask.paste(0, (35, 40, 37, 42))  # small internal hole
            mask.paste(255, (5, 5, 7, 7))  # detached background island
            mask_path = self.source_masks / "masks" / (filename + ".mask.png")
            mask.save(mask_path)
            mask_entries.append({"filename": filename, "source_photo_sha256": _hash(image),
                                 "mask": "masks/" + filename + ".mask.png",
                                 "mask_sha256": _hash(mask_path)})
        (self.sparse / "undistorted-reference-index.json").write_text(
            json.dumps({"images": sparse_entries}))
        (self.sparse / "reconstruction-report.json").write_text(json.dumps({
            "status": "candidate / visual-review-required", "registered_images": 3,
            "staged_independent_images": 3, "undistorted_images": 3,
        }))
        (self.source_masks / "mask-report.json").write_text(json.dumps({
            "status": "candidate / inspect-every-mask-before-dense-reconstruction",
            "source_revision": str(self.sparse), "images": mask_entries,
        }))
        self.edits = self.root / "edits.json"
        self.edits.write_text(json.dumps({"images": {"view-00.jpg": {
            "exclude_polygons": [[[0, .8], [1, .8], [1, 1], [0, 1]]],
            "include_polygons": [[[.34, .49], [.39, .49], [.39, .54], [.34, .54]]],
        }}}))

    def test_island_hole_and_polygon_cleanup_preserves_source_and_openmvs_compatibility(self):
        first_source = self.source_masks / "masks/view-00.jpg.mask.png"
        digest = _hash(first_source)
        target = self.root / "masks-r02"
        report = refine_rectified_masks(self.sparse, self.source_masks, target,
                                        edits=self.edits, min_island_pixels=10,
                                        max_hole_pixels=10)
        self.assertEqual(report["status"], "candidate / refined-mask-review-required")
        self.assertFalse(report["approved_for_reconstruction"])
        self.assertEqual(len(report["images"]), 3)
        self.assertTrue((target / report["contact_sheet"]).is_file())
        with Image.open(target / "masks/view-00.jpg.mask.png") as output:
            self.assertEqual(output.getpixel((5, 5)), 0)  # island removed
            self.assertEqual(output.getpixel((36, 41)), 255)  # hole filled/overridden
            self.assertEqual(output.getpixel((30, 72)), 0)  # unwanted ground/base removed
            self.assertEqual(output.getpixel((30, 30)), 255)  # sculpture retained
        self.assertEqual(_hash(first_source), digest)
        self.assertGreater(report["images"][0]["pixels_removed"], 10)
        dense = plan_dense_reconstruction(self.sparse, self.root / "dense",
                                          DenseSettings(min_registered_images=3),
                                          rectified_masks=target / "masks", executables={})
        self.assertEqual(len(dense["rectified_masks"]), 3)

    def test_tamper_invalid_edits_or_noop_fail_before_output(self):
        target = self.root / "target"
        with self.assertRaisesRegex(ReferenceError, "needs edits"):
            refine_rectified_masks(self.sparse, self.source_masks, target)
        self.assertFalse(target.exists())
        self.edits.write_text(json.dumps({"images": {"unknown.jpg": {
            "exclude_polygons": [[[0, 0], [1, 0], [1, 1]]]}}}))
        with self.assertRaisesRegex(ReferenceError, "unknown"):
            refine_rectified_masks(self.sparse, self.source_masks, target, edits=self.edits)
        self.assertFalse(target.exists())
        path = self.source_masks / "masks/view-00.jpg.mask.png"
        path.write_bytes(b"corrupted mask")
        with self.assertRaisesRegex(ReferenceError, "provenance changed"):
            refine_rectified_masks(self.sparse, self.source_masks, target,
                                   min_island_pixels=10)
        self.assertFalse(target.exists())

    def test_cli_refine_masks_and_reject_invalid_settings(self):
        target = self.root / "cli-r02"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["refine-dense-masks", "--reconstruction", str(self.sparse),
                                   "--mask-revision", str(self.source_masks),
                                   "--output", str(target),
                                   "--min-island-pixels", "10"]), 0)
        self.assertTrue((target / "mask-report.json").is_file())
        with self.assertRaisesRegex(ReferenceError, "min_island_pixels"):
            refine_rectified_masks(self.sparse, self.source_masks, self.root / "bad",
                                   min_island_pixels=-1)


if __name__ == "__main__":
    unittest.main()
