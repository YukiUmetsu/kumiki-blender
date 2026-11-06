import unittest

import numpy as np

from kumiki_blender.reconstruction.discrepancy_regions import discrepancy_regions


class DiscrepancyRegionTests(unittest.TestCase):
    def test_separate_missing_excess_and_boundary_uncertainty(self):
        reference = np.zeros((24, 24), dtype=np.uint8)
        reference[3:21, 3:21] = 255
        rendered = reference.copy()
        rendered[8:12, 8:13] = 0
        rendered[3:5, 2] = 255
        regions = discrepancy_regions(reference, rendered, boundary_tolerance=1, min_pixels=1)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0]['kind'], 'reference_only')
        self.assertEqual(regions[0]['bbox_xyxy'], [8, 8, 13, 12])
        self.assertEqual(regions[0]['interior_pixels'], 20)
        self.assertEqual(regions[1]['kind'], 'render_only')
        self.assertEqual(regions[1]['evidence'], 'mask-boundary-only')

    def test_four_connected_components_and_empty_masks(self):
        reference = np.zeros((4, 4), dtype=np.uint8)
        reference[0, 0] = reference[1, 1] = 255
        regions = discrepancy_regions(reference, np.zeros_like(reference), boundary_tolerance=0, min_pixels=1)
        self.assertEqual(len(regions), 2)
        self.assertEqual(discrepancy_regions(reference, reference), [])
