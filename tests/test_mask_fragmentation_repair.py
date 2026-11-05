"""Stable image-only cleanup can recover masonry without claiming ground truth."""

import unittest

import numpy as np
from PIL import Image, ImageDraw

from kumiki_blender.reconstruction.mask_fragmentation_repair import (
    propose_fragmentation_repair,
)


class FragmentationRepairTests(unittest.TestCase):
    def setUp(self):
        width, height = 256, 144
        self.photo = Image.new("RGB", (width, height), (85, 145, 210))
        scene = ImageDraw.Draw(self.photo)
        scene.rectangle((0, 95, 255, 143), fill=(190, 172, 126))
        scene.polygon([(95, 17), (147, 17), (151, 48), (174, 64),
                       (197, 102), (223, 112), (217, 137), (14, 137),
                       (15, 122), (67, 108), (71, 60), (92, 49)],
                      fill=(158, 141, 107))
        subject = np.asarray(self.photo)[:, :, 0] == 158
        y, x = np.indices((height, width))
        # A periodic low-confidence SAM2 mask with tiny gaps across the full
        # subject, and a broad candidate that sees only part of its lower paw.
        holes = ((x % 5) <= 3) & ((y % 5) <= 3)
        grounded = subject & holes
        broad = grounded & (x >= 75) & (y >= 83)
        self.broad = Image.fromarray((broad * 255).astype("uint8"), "L")
        self.grounded = Image.fromarray((grounded * 255).astype("uint8"), "L")

    def test_nested_fragmented_candidates_recover_stable_connected_body(self):
        repaired, info = propose_fragmentation_repair(
            self.photo, self.broad, self.broad, self.grounded,
            background_seeds=[], background_holdouts=[], foreground_holdouts=[])
        self.assertIsNotNone(repaired, info)
        self.assertTrue(info["accepted"])
        self.assertGreater(info["stable_scale_iou"], .96)
        output = np.asarray(repaired) > 127
        self.assertTrue(output[70, 110])  # Chest missing from broad candidate
        self.assertTrue(output[127, 55])  # Front paw
        self.assertFalse(output[50, 15])  # Blue sky
        self.assertFalse(output[140, 245])  # Nearby ground

    def test_independent_ground_track_vetoes_a_proposal(self):
        proposed, info = propose_fragmentation_repair(
            self.photo, self.broad, self.broad, self.grounded,
            background_seeds=[{"normalized_xy": [110/255, 100/143],
                               "source": "triangulated-low-ground-plane"}],
            background_holdouts=[], foreground_holdouts=[])
        self.assertIsNone(proposed)
        self.assertIn("verified-background-track-inside-proposal", info["reasons"])

    def test_non_nested_or_dense_masks_do_not_grow_automatically(self):
        proposed, info = propose_fragmentation_repair(
            self.photo, self.grounded, self.grounded, self.grounded,
            background_seeds=[], background_holdouts=[], foreground_holdouts=[])
        self.assertIsNone(proposed)
        self.assertIn("not-a-fragmented-nested-candidate", info["reasons"])
