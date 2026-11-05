"""Deterministic safeguards against head-only or contaminated grounded masks."""

import unittest

import numpy as np

from kumiki_blender.reconstruction.mask_candidate_ensemble import select_mask_candidate


class MaskCandidateEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.broad = np.zeros((100, 120), dtype=np.uint8)
        self.broad[12:85, 15:110] = 255
        self.detection = {"status": "candidate / text-grounded-box"}

    def test_head_only_detection_does_not_erase_statue_body(self):
        head_only = np.zeros_like(self.broad)
        head_only[12:37, 43:77] = 255
        choice, result = select_mask_candidate(self.broad, head_only, self.detection)
        self.assertEqual(choice, "broad")
        self.assertIn("grounded-candidate-loses-over-30-percent-of-broad-foreground",
                      result["reasons"])
        self.assertTrue(result["status"].startswith("uncertain"))

    def test_background_expansion_is_not_automatically_approved(self):
        leaked = np.full_like(self.broad, 255)
        choice, result = select_mask_candidate(self.broad, leaked, self.detection)
        self.assertEqual(choice, "broad")
        self.assertIn("grounded-candidate-grows-over-35-percent-beyond-broad-foreground",
                      result["reasons"])

    def test_agreeing_masks_can_select_grounded_for_later_track_audit(self):
        slight_trim = self.broad.copy()
        slight_trim[12:16, 15:110] = 0
        choice, report = select_mask_candidate(self.broad, slight_trim, self.detection)
        self.assertEqual(choice, "grounded")
        self.assertTrue(report["status"].startswith("candidate"))
        self.assertIn("not verified", report["limitations"])

    def test_detection_failure_keeps_broad_but_requires_review(self):
        choice, report = select_mask_candidate(self.broad, self.broad,
                                               {"status": "uncertain / no-confident-detection"})
        self.assertEqual(choice, "broad")
        self.assertIn("no-confident-subject-detection", report["reasons"])

    def test_mismatched_image_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            select_mask_candidate(self.broad, np.zeros((10, 10), dtype=np.uint8), self.detection)


if __name__ == "__main__":
    unittest.main()
