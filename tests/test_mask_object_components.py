"""Object-neutral component prompts and conservative automatic union QA."""

import unittest
from importlib.util import find_spec

from PIL import Image, ImageDraw

from kumiki_blender.reconstruction.mask_object_components import discover_visible_regions
from kumiki_blender.reconstruction.mask_object_recovery import evaluate_component_recovery


def rectangle(left, top, right, bottom, size=(180, 130)):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle((left, top, right, bottom), fill=255)
    return mask


@unittest.skipUnless(find_spec("scipy") is not None,
                     "Real pixel-level SAM2 prompt QA uses optional scipy")
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

    def test_cross_view_seeds_prompt_missing_nearby_parts_not_distant_background(self):
        tracks = [{"normalized_xy": [.76, .52]},
                  {"normalized_xy": [.49, .50]},
                  {"normalized_xy": [.98, .02]}]
        guide, report = discover_visible_regions(
            self.photo, self.old, self.grounded, foreground_seeds=tracks)
        self.assertIsNotNone(guide)
        sources = [component["source"] for component in report["components"]]
        self.assertIn("nearby-cross-view-foreground-track", sources)
        self.assertIn("cross-view-track-click", sources)
        guided_points = [point for region in guide["part_points"].values()
                         for point in region]
        self.assertIn([.76, .52], guided_points)
        self.assertNotIn([.98, .02], guided_points)

    def test_photo_color_distinct_exterior_provides_unverified_negative_clicks(self):
        scene = Image.new("RGB", self.photo.size, (40, 43, 47))
        ImageDraw.Draw(scene).rectangle((35, 35, 125, 105), fill=(170, 143, 113))
        guide, receipt = discover_visible_regions(scene, self.old, self.grounded)
        self.assertIsNotNone(guide)
        exterior = receipt["heuristic_exterior_negatives"]
        self.assertTrue(exterior)
        self.assertTrue(all(y >= .63 for _, y in exterior))
        self.assertTrue(all(point in guide["background_points"] for point in exterior))
        protected, _ = discover_visible_regions(
            scene, self.old, self.grounded,
            foreground_seeds=[{"normalized_xy": exterior[0]}])
        self.assertNotIn(exterior[0], protected["background_points"])
        _, uniform_report = discover_visible_regions(self.photo, self.old, self.grounded)
        self.assertEqual(uniform_report["heuristic_exterior_negatives"], [])

    def test_new_mask_cannot_cover_a_negative_prompt_even_without_track_evidence(self):
        guide, _ = discover_visible_regions(self.photo, self.old, self.grounded)
        guide["background_points"].append([.65, .5])
        candidate, receipt = evaluate_component_recovery(
            self.photo, self.old, self.grounded, self.grounded, guide=guide,
            foreground_holdouts=[], background_seeds=[], background_holdouts=[])
        self.assertIsNone(candidate)
        self.assertIn("sam2-negative-prompt-violated", receipt["reasons"])

    def test_accepts_well_supported_contiguous_expansion(self):
        guide, _ = discover_visible_regions(self.photo, self.old, self.grounded)
        result, report = evaluate_component_recovery(
            self.photo, self.old, self.grounded, self.grounded, guide=guide,
            foreground_holdouts=[], background_seeds=[], background_holdouts=[])
        self.assertIsNotNone(result, report)
        self.assertTrue(report["accepted"])
        self.assertGreater(report["area_growth"], 1.1)

    def test_keeps_verified_detached_object_region_behind_occlusion(self):
        old = rectangle(70, 25, 115, 95)
        proposed = rectangle(65, 22, 121, 99)
        ImageDraw.Draw(proposed).rectangle((22, 55, 43, 86), fill=255)
        guide = {"part_points": {"region_01": [[.52, .4]],
                                 "tracked_02": [[.17, .53]]},
                 "background_points": []}
        selected, receipt = evaluate_component_recovery(
            self.photo, old, proposed, proposed, guide=guide,
            foreground_holdouts=[{"normalized_xy": [.52, .4]}],
            background_seeds=[], background_holdouts=[])
        self.assertIsNotNone(selected, receipt)
        self.assertEqual(len(receipt["retained_tracked_islands"]), 1)
        self.assertGreater(receipt["area_growth"], 1.45)
        self.assertGreater(selected.getpixel((30, 69)), 128)

        # Without independent cross-view evidence the detached region must
        # not override the ordinary 45% growth limit.
        untracked = {"part_points": {"region_01": [[.52, .4]],
                                     "region_02": [[.17, .53]]},
                     "background_points": []}
        rejected, conservative = evaluate_component_recovery(
            self.photo, old, proposed, proposed, guide=untracked,
            foreground_holdouts=[], background_seeds=[], background_holdouts=[])
        self.assertIsNone(rejected)
        self.assertEqual(conservative["retained_tracked_islands"], [])

    def test_detached_region_needs_grounded_support_and_respects_background(self):
        old = rectangle(70, 25, 115, 95)
        proposed = rectangle(65, 22, 121, 99)
        ImageDraw.Draw(proposed).rectangle((22, 55, 43, 86), fill=255)
        guide = {"part_points": {"region_01": [[.52, .4]],
                                 "tracked_02": [[.17, .53]]},
                 "background_points": []}
        no_support = rectangle(65, 22, 121, 99)
        result, report = evaluate_component_recovery(
            self.photo, old, proposed, no_support, guide=guide,
            foreground_holdouts=[], background_seeds=[], background_holdouts=[])
        self.assertIsNone(result)
        self.assertEqual(report["retained_tracked_islands"], [])

        result, report = evaluate_component_recovery(
            self.photo, old, proposed, proposed, guide=guide,
            foreground_holdouts=[], background_seeds=[{"normalized_xy": [.17, .53]}],
            background_holdouts=[])
        self.assertIsNone(result)
        self.assertIn("verified-background-point-in-component-union", report["reasons"])

    def test_cropped_object_preserves_real_pixels_at_image_border(self):
        previous = rectangle(40, 35, 115, 129)
        candidate = rectangle(35, 31, 123, 129)
        guide = {"part_points": {"region_01": [[.4, .55]],
                                 "region_02": [[.4, .96]]},
                 "background_points": []}
        selected, report = evaluate_component_recovery(
            self.photo, previous, candidate, candidate, guide=guide,
            foreground_holdouts=[], background_seeds=[], background_holdouts=[])
        self.assertIsNotNone(selected, report)
        self.assertEqual(selected.getpixel((75, 129)), 255)

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

    def test_trim_requires_independent_ground_and_preserves_foreground(self):
        old = rectangle(25, 25, 140, 105)
        trim = rectangle(25, 25, 105, 105)
        guide, _ = discover_visible_regions(self.photo, old, old)
        ground = [{"normalized_xy": [x, .55]} for x in (.68, .73)]
        foreground = [{"normalized_xy": [x, .55]} for x in (.20, .35, .50)]
        chosen, receipt = evaluate_component_recovery(
            self.photo, old, trim, old, guide=guide,
            foreground_holdouts=foreground, background_seeds=ground,
            background_holdouts=[])
        self.assertIsNotNone(chosen, receipt)
        self.assertEqual(receipt["operation"], "ground-trim")
        self.assertEqual(receipt["verified_background_seed_hits_before"], 2)
        self.assertEqual(receipt["verified_background_seed_hits_after"], 0)

        without_evidence, receipt = evaluate_component_recovery(
            self.photo, old, trim, old, guide=guide,
            foreground_holdouts=foreground, background_seeds=[],
            background_holdouts=[])
        self.assertIsNone(without_evidence)
        without_foreground, receipt = evaluate_component_recovery(
            self.photo, old, trim, old, guide=guide,
            foreground_holdouts=[], background_seeds=ground,
            background_holdouts=[])
        self.assertIsNone(without_foreground)


if __name__ == "__main__":
    unittest.main()
