"""Multi-view mask consistency QA with genuine COLMAP binary/text layouts."""

import contextlib
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.mask_consistency import audit_multiview_masks
from kumiki_blender.reconstruction.references import ReferenceError


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaskConsistencyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sparse = self.root / "sparse"
        photos = self.sparse / "undistorted/images"
        photos.mkdir(parents=True)
        model = self.sparse / "undistorted/sparse"
        model.mkdir(parents=True)
        (model / "cameras.bin").write_bytes(b"first-party camera fixture")
        (model / "points3D.bin").write_bytes(b"first-party 3D fixture")
        self.masks = self.root / "masks"
        self.masks.mkdir()
        images = []
        for index in range(3):
            name = f"view-{index:02d}.jpg"
            path = photos / name
            Image.new("RGB", (100, 80), (118, 112, 105)).save(path)
            images.append({"filename": name, "sha256": _digest(path), "dimensions": [100, 80]})
            mask = Image.new("L", (100, 80), 0)
            mask.paste(255, (20, 10, 80, 75))
            mask.save(self.masks / (name + ".mask.png"))
        (self.sparse / "undistorted-reference-index.json").write_text(
            json.dumps({"images": images}))
        (self.sparse / "reconstruction-report.json").write_text(json.dumps({
            "status": "candidate / visual-review-required", "registered_images": 3,
            "staged_independent_images": 3, "undistorted_images": 3,
        }))
        self.points = [
            [(30, 30, 100), (4, 30, 101), (30, 40, 102), (4, 60, 103), (50, 20, -1)],
            [(30, 30, 100), (4, 30, 101), (30, 40, 102), (30, 50, 103)],
            [(30, 30, 100), (4, 30, 101), (4, 40, 102)],
        ]
        self._write_model()

    def _write_model(self):
        content = bytearray(struct.pack("<Q", 3))
        for index, points in enumerate(self.points):
            name = f"view-{index:02d}.jpg"
            content.extend(struct.pack("<i7di", index+1, 1, 0, 0, 0, 0, 0, 0, 1))
            content.extend(name.encode() + b"\0")
            content.extend(struct.pack("<Q", len(points)))
            for x, y, point_id in points:
                content.extend(struct.pack("<ddq", x, y, point_id))
        (self.sparse / "undistorted/sparse/images.bin").write_bytes(content)

    def test_flags_one_majority_outlier_and_ambiguous_two_view_track(self):
        target = self.root / "audit"
        result = audit_multiview_masks(self.sparse, self.masks, target)
        self.assertEqual(result["compared_3d_tracks"], 4)
        self.assertEqual(result["mixed_label_tracks"], 2)
        self.assertEqual(result["ambiguous_label_tracks"], 1)
        self.assertFalse(result["approved_for_reconstruction"])
        self.assertFalse(result["distribution_allowed"])
        self.assertEqual(len(result["images"]), 3)
        self.assertEqual(result["images"][2]["consensus_outliers"], 1)
        self.assertEqual(result["images"][2]["review_priority"], "high")
        self.assertEqual(result["images"][0]["review_priority"], "medium")
        self.assertTrue((target / result["contact_sheet"]).is_file())
        self.assertEqual(result["model_sha256"], _digest(self.sparse /
                         "undistorted/sparse/images.bin"))
        for entry in result["images"]:
            self.assertTrue((target / entry["review"]).is_file())
            self.assertEqual(entry["mask_sha256"],
                             _digest(self.masks / (entry["filename"] + ".mask.png")))

    def test_text_model_fallback_and_cli(self):
        (self.sparse / "undistorted/sparse/images.bin").unlink()
        lines = ["# Rectified COLMAP image track model"]
        for index, points in enumerate(self.points):
            lines.append(f"{index+1} 1 0 0 0 0 0 0 1 view-{index:02d}.jpg")
            lines.append(" ".join(f"{x} {y} {point_id}" for x, y, point_id in points))
        (self.sparse / "undistorted/sparse/images.txt").write_text("\n".join(lines) + "\n")
        target = self.root / "cli-audit"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["audit-dense-masks", "--reconstruction", str(self.sparse),
                                   "--rectified-masks", str(self.masks),
                                   "--output", str(target),
                                   "--max-overlay-points", "10"]), 0)
        report = json.loads((target / "mask-consistency-report.json").read_text())
        self.assertTrue(report["model_file"].endswith("images.txt"))
        self.assertEqual(report["mixed_label_tracks"], 2)

    def test_overlay_budget_keeps_red_majority_outliers_ahead_of_ambiguous_ties(self):
        # The two-view tie appears before the 3-view disagreement in the model.
        # A one-marker preview must still show the higher-priority red marker.
        self.points[0].insert(2, (4, 60, 104))
        self.points[2].insert(2, (30, 65, 104))
        self._write_model()
        target = self.root / "budget-audit"
        result = audit_multiview_masks(self.sparse, self.masks, target,
                                       max_overlay_points=1)
        image = result["images"][2]
        self.assertEqual(image["consensus_outliers"], 1)
        self.assertEqual(image["ambiguous_observations"], 1)
        with Image.open(target / image["review"]) as overlay:
            red = overlay.convert("RGB").getpixel((9, 40))
        self.assertGreater(red[0], red[1] * 1.5)
        self.assertGreater(red[0], red[2] * 1.5)

    def test_tampering_corrupt_tracks_or_no_overlap_rejects_without_output(self):
        target = self.root / "invalid"
        binary = self.sparse / "undistorted/sparse/images.bin"
        original = binary.read_bytes()
        binary.write_bytes(b"truncated")
        with self.assertRaisesRegex(ReferenceError, "registered image count|truncated"):
            audit_multiview_masks(self.sparse, self.masks, target)
        self.assertFalse(target.exists())
        binary.write_bytes(original)
        mask_path = self.masks / "view-00.jpg.mask.png"
        with Image.open(mask_path) as image:
            bad = image.copy()
        bad.putpixel((10, 10), 128)
        bad.save(mask_path)
        with self.assertRaisesRegex(ReferenceError, "binary pixels"):
            audit_multiview_masks(self.sparse, self.masks, target)
        self.assertFalse(target.exists())
        Image.new("L", (100, 80), 0).save(mask_path)
        # Use a valid binary mask again, then replace the tracks with unique IDs.
        with Image.open(self.masks / "view-01.jpg.mask.png") as other:
            other.save(mask_path)
        self.points = [[(30, 30, 100+idx)] for idx in range(3)]
        self._write_model()
        with self.assertRaisesRegex(ReferenceError, "No shared 3D feature tracks"):
            audit_multiview_masks(self.sparse, self.masks, target)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
