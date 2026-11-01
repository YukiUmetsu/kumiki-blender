"""Physical ratios aid reference fitting without altering the production head."""

import math
import unittest

from kumiki_blender.liberty_face_measurements import scaled_nps_head_targets


class LibertyFaceMeasurementTests(unittest.TestCase):
    def test_actual_nps_dimensions_and_scaled_review_cage(self):
        original = scaled_nps_head_targets(5.26)
        self.assertAlmostEqual(original.width, 3.05)
        self.assertAlmostEqual(original.eye_width, .76)
        self.assertAlmostEqual(original.nose_length, 1.37)
        self.assertAlmostEqual(original.mouth_width, .91)
        target = scaled_nps_head_targets(3.52)
        self.assertAlmostEqual(target.width, 2.041, delta=.002)
        self.assertAlmostEqual(target.eye_width, .509, delta=.002)
        self.assertAlmostEqual(target.nose_length, .917, delta=.002)
        self.assertAlmostEqual(target.mouth_width, .609, delta=.002)

    def test_scale_validation(self):
        for bad in (True, float("nan"), math.inf, 0, -1, 101, "five"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                scaled_nps_head_targets(bad)
