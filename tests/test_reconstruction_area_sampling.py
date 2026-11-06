import unittest

import numpy as np

from kumiki_blender.reconstruction.photographic_qa import _area_stratified_samples
from kumiki_blender.reconstruction.references import ReferenceError


class AreaSamplingTests(unittest.TestCase):
    def test_samples_follow_surface_area_and_vary_within_face(self):
        xyz = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                        [4, 0, 0], [5, 0, 0], [4, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        samples, indices, area = _area_stratified_samples(xyz, faces, 100, np)
        self.assertEqual(area, 2.5)
        self.assertEqual(np.count_nonzero(indices == 0), 80)
        self.assertEqual(np.count_nonzero(indices == 1), 20)
        self.assertEqual(len(np.unique(samples, axis=0)), 100)
        self.assertTrue(np.allclose(samples[:, 2], 0))

    def test_degenerate_mesh_rejected(self):
        with self.assertRaises(ReferenceError):
            _area_stratified_samples(np.zeros((3, 3)), np.array([[0, 1, 2]]), 100, np)
