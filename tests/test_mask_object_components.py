"""Object-neutral component prompts and conservative automatic union QA."""

import unittest

from PIL import Image, ImageDraw

from kumiki_blender.reconstruction.mask_object_components import discover_visible_regions
from kumiki_blender.reconstruction.mask_object_recovery import evaluate_component_recovery


def rectangle(left, top, right, bottom, size=(180, 130)):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle((left, top, right, bottom), fill=255)
    return mask


class AutomaticVisibleRegionTests(unittest.TestCase):
    def setUp(self):
        self.photo = Image.new("RGB", (180, 130), (168, 143, 113))
        self.old = rectangle(35, 35, 105, 105)
        self.grounded = rectangle(35, 35, 125, 105)

    def test_discovers_multiple_local_regions_for_generic_object(self):
        guide, report = discover_visible_regions(self.photo, self.old, self.grounded)
        self.assertIsNotNone(guide)
        self.assertGreaterEqual(report["discovered_regions"], 3)
        self.assertLessEqual(report["discovered_regions"], 8)
        self.assertEqual(set(guide["component_boxes"]), set(guide["part_points"]))
        self.assertTrue(all(name.startswith("region_") for name in guide["component_boxes"]))
        for name, (left, top, right, bottom) in guide["component_boxes"].items():
            x, y = guide["part_points"][name][0]
            self.assertTrue(left <= x <= right and top <= y <= bottom)

    def test_ignores_unlocalized_detector_box_and_supports_local_detector_box(self):
        detector = {"component_boxes": {"subject": [0, 0, 1, 1],
                                        "nearly_all": [0, 0, .95, .95],
                                        "local": [.30, .30, .50, .60],
                                        "bad": ["a", 0, 1, 1]}}
        guide, report = discover_visible_regions(
            self.photo, self.old, self.grounded, detector_guide=detector)
        self.assertIsNotNone(guide)
        sources = [item["source"] for item in report["components"]]
        self.assertIn("localized-text-detection", sources)
        self.assertNotIn("nearly_all", guide["component_boxes"])
        self.assertTrue(all((right-left)*(bottom-top) <= .95
                            for left, top, right, bottom in guide["component_boxes"].values()))

    def test_abstains_on_invalid_empty_or_full_frame_proposals(self):
        empty = Image.new("L", self.photo.size, 0)
        self.assertIsNone(discover_visible_regions(self.photo, empty, empty)[0])
        filled = Image.new("L", self.photo.size, 255)
        self.assertIsNone(discover_visible_regions(self.photo, filled, filled)[0])
        self.assertIsNone(discover_visible_regions(self.photo, self.old, self.grounded,
                                                   max_regions=20)[0])

    def test_irregular_vertical_and_diagonal_subjects_need_no_anatomy_names(self):
        for shape in ("upright", "diagonal"):
            with self.subTest(shape=shape):
                proposal = Image.new("L", self.photo.size, 0)
                drawing = ImageDraw.Draw(proposal)
                if shape == "upright":
                    drawing.ellipse((65, 10, 130, 115), fill=255)
                    drawing.rectangle((50, 80, 145, 100), fill=255)
                else:
                    drawing.line((12, 105, 160, 18), fill=255, width=29)
                    drawing.ellipse((115, 5, 170, 64), fill=255)
                guide, report = discover_visible_regions(self.photo, proposal, proposal)
                self.assertIsNotNone(guide, report)
                self.assertGreaterEqual(report["discovered_regions"], 3)

    def test_accepts_well_supported_contiguous_expansion(self):
        guide, _ = discover_visible_regions(self.photo, self.old, self.grounded)
        result, report = evaluate_component_recovery(
            self.photo, self.old, self.grounded, self.grounded, guide=guide,
            foreground_holdouts=[], background_seeds=[], background_holdouts=[])
        self.assertIsNotNone(result, report)
        self.assertTrue(report["accepted"])
        self.assertGreater(report["area_growth"], 1.1)

    def test_rejects_confirmed_background_and_unsupported_growth(self):
        guide, _ = discover_visible_regions(self.photo, self.old, self.grounded)
        ground = [{"normalized_xy": [.65, .5]}]
        result, report = evaluate_component_recovery(
            self.photo, self.old, self.grounded, self.grounded, guide=guide,
            foreground_holdouts=[], background_seeds=ground, background_holdouts=[])
        self.assertIsNone(result)
        self.assertIn("verified-background-point-in-component-union", report["reasons"])
        wildly_expanded = rectangle(0, 0, 179, 129)
        result, report = evaluate_component_recovery(
            self.photo, self.old, wildly_expanded, self.grounded, guide=guide,
            foreground_holdouts=[], background_seeds=[], background_holdouts=[])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
