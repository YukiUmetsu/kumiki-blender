"""Mask-only SfM-camera geometry QA with no web-photo redistribution."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.silhouettes import compare_silhouettes


class SilhouetteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.render_root = self.root / "renders"
        self.render_root.mkdir()
        cameras = []
        for index in range(3):
            name = f"view-{index}"
            original = self.root / (name + "-source.png")
            mask = self.root / (name + "-mask.png")
            frame = self.render_root / (name + ".png")
            Image.new("RGB", (120, 90), "#333333").save(original)
            silhouette = Image.new("L", (120, 90), 0)
            ImageDraw.Draw(silhouette).rectangle((30, 10, 80, 75), fill=255)
            silhouette.save(mask)
            clay = Image.new("RGBA", (120, 90), (120, 120, 120, 0))
            clay.putalpha(silhouette)
            clay.save(frame)
            cameras.append({
                "image_id": name, "source_photo_path": str(original),
                "source_mask_path": str(mask),
                "intrinsics": {"width": 120, "height": 90, "fx_px": 100,
                               "fy_px": 100, "cx_px": 60, "cy_px": 45},
                "translation_m": [0, 0, 0],
                "rotation_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            })
        self.plan = self.root / "camera-plan.json"
        self.plan.write_text(json.dumps({"schema_version": 1, "asset_id": "statue",
                                         "status": "alignment-candidate / artistic-review-required",
                                         "cameras": cameras}))

    def test_scores_correct_alpha_masks_without_exporting_photos(self):
        output = self.root / "qa"
        report = compare_silhouettes(self.plan, self.render_root, output)
        self.assertEqual(report["status"], "mask-threshold-candidate / artistic-review-required")
        self.assertEqual(report["scored_cameras"], 3)
        self.assertTrue(all(view["iou"] == 1.0 for view in report["views"]))
        self.assertEqual(len(list(output.iterdir())), 4)
        self.assertEqual(json.loads((output / "silhouette-qa.json").read_text()), report)
        with Image.open(output / "overlay-view-0.png") as file:
            self.assertEqual(file.getpixel((45, 20)), (127, 127, 127))

    def test_bad_overlap_and_missing_views_never_claim_pass(self):
        Image.new("RGBA", (120, 90), (0, 0, 0, 0)).save(self.render_root / "view-0.png")
        (self.render_root / "view-2.png").unlink()
        report = compare_silhouettes(self.plan, self.render_root, self.root / "qa")
        self.assertEqual(report["status"], "review-required")
        self.assertEqual(report["views"][0]["iou"], 0)
        self.assertEqual(report["views"][2]["status"], "unscored")

    def test_refuses_reuse_of_output_and_rgb_render(self):
        dest = self.root / "qa"
        dest.mkdir()
        with self.assertRaisesRegex(ReferenceError, "already exists"):
            compare_silhouettes(self.plan, self.render_root, dest)
        Image.new("RGB", (120, 90), "red").save(self.render_root / "view-0.png")
        with self.assertRaisesRegex(ReferenceError, "RGBA"):
            compare_silhouettes(self.plan, self.render_root, self.root / "qa2")

    def test_all_white_staging_masks_do_not_create_fake_quality_score(self):
        plan = json.loads(self.plan.read_text())
        Image.new("L", (120, 90), 255).save(plan["cameras"][0]["source_mask_path"])
        report = compare_silhouettes(self.plan, self.render_root, self.root / "qa")
        self.assertEqual(report["views"][0]["status"], "unscored")
        self.assertEqual(report["scored_cameras"], 2)
        self.assertEqual(report["status"], "review-required")

    def test_regional_paw_failure_blocks_global_silhouette_pass(self):
        # Removing a small paw/bottom band leaves IoU > 0.8 overall, but
        # an annotated paw region reveals severe missing reference geometry.
        clay = Image.open(self.render_root / "view-0.png").convert("RGBA")
        alpha = clay.getchannel("A")
        ImageDraw.Draw(alpha).rectangle((0, 66, 119, 89), fill=0)
        clay.putalpha(alpha)
        clay.save(self.render_root / "view-0.png")
        global_only = compare_silhouettes(self.plan, self.render_root,
                                          self.root / "global")
        self.assertEqual(global_only["status"], "mask-threshold-candidate / artistic-review-required")
        config = self.root / "regions.json"
        config.write_text(json.dumps({"schema_version": 1, "views": {
            "view-0": [
                {"label": "front_paws", "box_norm": [0.23, 0.71, 0.69, 0.87]},
                {"label": "face", "box_norm": [0.23, 0.11, 0.69, 0.45]},
            ]}}))
        report = compare_silhouettes(self.plan, self.render_root,
                                     self.root / "regional", regions_file=config)
        self.assertEqual(report["status"], "review-required")
        self.assertEqual(report["review_regions_scored"], 2)
        paw, face = report["views"][0]["regions"]
        self.assertLess(paw["iou"], 0.4)
        self.assertGreater(paw["missing_reference_fraction"], 0.6)
        self.assertEqual(face["iou"], 1)
        self.assertTrue((self.root / "regional" / paw["overlay"]).is_file())
        self.assertNotIn("overlay_image", paw)

    def test_unscored_region_or_unscored_view_prevents_promotion(self):
        config = self.root / "regions.json"
        config.write_text(json.dumps({"schema_version": 1, "views": {
            "view-1": [{"label": "no_statue", "box_norm": [0, 0, 0.1, 0.1]}],
            "view-2": [{"label": "rear", "box_norm": [0.2, 0.2, 0.8, 0.8]}],
        }}))
        (self.render_root / "view-2.png").unlink()
        report = compare_silhouettes(self.plan, self.render_root,
                                     self.root / "regional", regions_file=config)
        self.assertEqual(report["status"], "review-required")
        self.assertEqual(report["review_regions_requested"], 2)
        self.assertEqual(report["review_regions_scored"], 0)
        self.assertEqual(report["views"][1]["regions"][0]["status"], "unscored")

    def test_rejects_malformed_or_unknown_region_annotations(self):
        config = self.root / "regions.json"
        for views in (
            {"unknown-view": [{"label": "paws", "box_norm": [0, 0, 1, 1]}]},
            {"view-0": [{"label": "paws", "box_norm": [0, 0, 1.2, 1]}]},
            {"view-0": [{"label": "paws", "box_norm": [0, 0, 1, 1]},
                         {"label": "paws", "box_norm": [0, 0, 1, 1]}]},
            {"view-0": [{"label": "paws", "box_norm": [0, False, 1, 1]}]},
        ):
            config.write_text(json.dumps({"schema_version": 1, "views": views}))
            with self.subTest(views=views), self.assertRaises(ReferenceError):
                compare_silhouettes(self.plan, self.render_root, self.root / "regional",
                                    regions_file=config)


if __name__ == "__main__":
    unittest.main()
