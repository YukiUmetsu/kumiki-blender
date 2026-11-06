import unittest

import numpy as np

from kumiki_blender.reconstruction.silhouette_qa import compare_silhouettes


class SilhouetteComparisonTests(unittest.TestCase):
    def test_missing_and_excess_with_uncertain_edges(self):
        reference = np.zeros((9, 9), dtype=np.uint8)
        reference[2:7, 2:7] = 255
        rendered = reference.copy()
        rendered[4, 4] = 0
        rendered[2, 1] = 255
        metrics, difference = compare_silhouettes(reference, rendered, boundary_tolerance=1)
        self.assertEqual((metrics['missing_pixels'], metrics['excess_pixels']), (1, 1))
        self.assertEqual(metrics['boundary_uncertain_pixels'], 1)
        self.assertAlmostEqual(metrics['iou'], 24 / 26)
        self.assertAlmostEqual(metrics['precision'], 24 / 25)
        self.assertAlmostEqual(metrics['recall'], 24 / 25)
        np.testing.assert_array_equal(difference[4, 4], [235, 50, 45])
        np.testing.assert_array_equal(difference[2, 1], [255, 210, 40])

    def test_empty_and_invalid_masks(self):
        empty = np.zeros((2, 2), dtype=np.uint8)
        metrics, _ = compare_silhouettes(empty, empty)
        self.assertIsNone(metrics['iou'])
        self.assertIsNone(metrics['precision'])
        self.assertIsNone(metrics['recall'])
        with self.assertRaises(ValueError):
            compare_silhouettes(empty, np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            compare_silhouettes(empty, np.full((2, 2), 128))
