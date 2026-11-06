"""Held-out track evidence must actually be separate from prompt evidence."""
import unittest

import numpy as np

from kumiki_blender.reconstruction.mask_track_guidance import evaluate_track_resegmentation


class TrackProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.original = np.zeros((100, 100), dtype=np.uint8)
        self.original[20:80, 20:70] = 255
        self.proposed = self.original.copy()
        self.proposed[20:80, 70:80] = 255
        self.seeds = [{"point3d_id": 10 + i, "normalized_xy": [x, .4]}
                      for i, x in enumerate((.72, .74, .76))]
        self.holdouts = [{"point3d_id": 20 + i, "normalized_xy": [x, .5]}
                         for i, x in enumerate((.72, .76, .3, .4, .5))]

    def test_prompt_track_cannot_be_reused_as_holdout(self):
        self.holdouts[0]["point3d_id"] = self.seeds[0]["point3d_id"]
        result = evaluate_track_resegmentation(self.original, self.proposed,
                                              self.seeds, self.holdouts)
        self.assertFalse(result["accepted"])
        self.assertIn("overlapping-or-duplicate-track-evidence", result["reasons"])

    def test_missing_track_identity_cannot_prove_independence(self):
        del self.holdouts[0]["point3d_id"]
        result = evaluate_track_resegmentation(self.original, self.proposed,
                                              self.seeds, self.holdouts)
        self.assertFalse(result["accepted"])
        self.assertIn("missing-track-provenance", result["reasons"])

    def test_distinct_tracks_are_eligible(self):
        result = evaluate_track_resegmentation(self.original, self.proposed,
                                              self.seeds, self.holdouts)
        self.assertTrue(result["accepted"])

    def test_duplicate_track_in_one_group_is_not_extra_support(self):
        self.holdouts[1]["point3d_id"] = self.holdouts[0]["point3d_id"]
        result = evaluate_track_resegmentation(self.original, self.proposed,
                                              self.seeds, self.holdouts)
        self.assertFalse(result["accepted"])
        self.assertIn("overlapping-or-duplicate-track-evidence", result["reasons"])

    def test_negative_prompt_cannot_also_be_a_holdout(self):
        negative = [{"point3d_id": self.holdouts[0]["point3d_id"],
                     "normalized_xy": [.1, .1]}]
        result = evaluate_track_resegmentation(self.original, self.proposed,
                                              self.seeds, self.holdouts, negative)
        self.assertFalse(result["accepted"])
        self.assertIn("overlapping-or-duplicate-track-evidence", result["reasons"])

    def test_ground_repair_cannot_override_provenance_rejection(self):
        original = np.zeros((100, 100), dtype=np.uint8)
        original[20:80, 20:70] = 255
        proposed = original.copy()
        proposed[73:80, 20:70] = 0
        context = [{"point3d_id": 10+i, "normalized_xy": [x, .4]}
                   for i, x in enumerate((.3, .4, .5, .6))]
        heldout = [{"point3d_id": 20+i, "normalized_xy": [x, .5]}
                   for i, x in enumerate((.3, .35, .4, .45, .5, .55))]
        negatives = [{"point3d_id": 30+i, "normalized_xy": [x, .75]}
                     for i, x in enumerate((.3, .5))]
        ground_holdouts = [{"point3d_id": 30+i, "normalized_xy": [x, .75]}
                           for i, x in enumerate((.4, .6))]
        result = evaluate_track_resegmentation(
            original, proposed, context, heldout, negatives, ground_holdouts,
            ground_repair=True, foreground_context_count=4)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["ground_repair_verified"])
        self.assertIn("overlapping-or-duplicate-track-evidence", result["reasons"])
