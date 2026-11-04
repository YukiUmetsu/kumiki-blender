"""Foreground proposals must never turn a competing guess into ground truth."""

import unittest

import numpy as np

from kumiki_blender.reconstruction.source_mask_refinement import (
    propose_refinement_guide, select_refined_candidate, mask_agreement,
)


class SourceMaskRefinementTests(unittest.TestCase):
    def setUp(self):
        self.initial = np.zeros((100, 200), dtype=np.uint8)
        self.initial[25:80, 18:185] = 255  # Subject plus leaked ground.
        self.aux = np.zeros_like(self.initial)
        self.aux[30:70, 40:160] = 255

    def test_rembg_guides_only_safe_peripheral_negatives(self):
        guide, report = propose_refinement_guide(self.initial, self.aux)
        self.assertEqual(report["status"], "candidate / guided-refinement")
        self.assertGreater(len(guide["background_points"]), 0)
        self.assertGreater(len(guide["foreground_points"]), 0)
        self.assertGreater(max(p[1] for p in guide["foreground_points"]) -
                           min(p[1] for p in guide["foreground_points"]), 0.30)
        for x, y in guide["background_points"]:
            # Do not use an IS-Net missing torso interior as a negative prompt.
            self.assertTrue(x < 0.16 or x > 0.84 or y < 0.12 or y > 0.88)

    def test_bad_rembg_foreground_is_not_used_for_negative_prompts(self):
        tiny = np.zeros_like(self.initial)
        tiny[30:32, 60:62] = 255
        guide, report = propose_refinement_guide(self.initial, tiny)
        self.assertIsNone(guide)
        self.assertEqual(report["status"], "suspect / unusable-auxiliary-mask")

    def test_refinement_rejects_loss_of_consensus_subject(self):
        refined = np.zeros_like(self.initial)
        refined[25:78, 30:175] = 255
        selected, result = select_refined_candidate(self.initial, refined, self.aux)
        self.assertTrue(result["attempted"])
        self.assertTrue(result["accepted"])
        self.assertEqual(int(selected[50, 80]), 255)
        damaged = np.zeros_like(self.initial)
        damaged[30:40, 40:60] = 255
        selected, result = select_refined_candidate(self.initial, damaged, self.aux)
        self.assertFalse(result["accepted"])
        np.testing.assert_array_equal(selected, self.initial)

    def test_disagreements_are_uncertain_not_background(self):
        agreement = mask_agreement(self.initial, self.aux)
        self.assertTrue(agreement["usable"])
        self.assertEqual(int(agreement["labels"][40, 80]), 255)
        self.assertEqual(int(agreement["labels"][26, 18]), 128)
        self.assertEqual(int(agreement["labels"][5, 5]), 0)


if __name__ == "__main__":
    unittest.main()
