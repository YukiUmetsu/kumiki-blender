"""Spatial contracts for the scan-registered Liberty face experiment."""

import unittest

from kumiki_blender.liberty_face_region import (
    face_radius, face_weight, face_jawline_z, keep_face_vertex, conform_depth,
    sculpt_classical_face,
)


class LibertyFaceRegionTests(unittest.TestCase):
    def test_front_face_region_excludes_slab_neck_and_scalp(self):
        self.assertTrue(keep_face_vertex(-.30, -.65, 8.60))
        self.assertFalse(keep_face_vertex(-.30, -.65, 7.55))
        self.assertFalse(keep_face_vertex(-.30, -.65, 9.40))
        self.assertFalse(keep_face_vertex(-.30, .45, 8.60))
        self.assertFalse(keep_face_vertex(.42, -.48, 8.60))

    def test_lower_face_follows_chin_curve_instead_of_including_donor_neck(self):
        self.assertLess(face_jawline_z(-.30), face_jawline_z(.10))
        self.assertTrue(keep_face_vertex(-.30, -.60, 8.01))
        self.assertFalse(keep_face_vertex(.10, -.60, 8.05))
        self.assertFalse(keep_face_vertex(-.30, -.45, 7.83))

    def test_feather_is_symmetric_and_monotonic(self):
        self.assertAlmostEqual(face_radius(-.50, 8.56), face_radius(-.10, 8.56))
        weights = [face_weight(r) for r in (.0, .7, .8, .9, 1.0)]
        self.assertEqual(weights[:2], [1.0, 1.0])
        self.assertEqual(weights[-1], 0.0)
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_boundary_snaps_to_scanned_depth_without_changing_center(self):
        self.assertAlmostEqual(conform_depth(-.65, -.32, .5), -.65)
        self.assertAlmostEqual(conform_depth(-.65, -.32, 1.0), -.324, places=3)
        self.assertTrue(-.65 < conform_depth(-.65, -.32, .85) < -.324)
        self.assertAlmostEqual(conform_depth(-.65, None, .9), -.65)

    def test_classical_sculpt_narrows_eyes_and_projects_brow(self):
        upper = sculpt_classical_face(-.55, -.62, 8.82)
        lower = sculpt_classical_face(-.55, -.62, 8.70)
        brow = sculpt_classical_face(-.55, -.50, 8.92)
        self.assertLess(upper[2], 8.82)
        self.assertGreater(lower[2], 8.70)
        self.assertLess(brow[1], -.50)
        self.assertAlmostEqual(upper[2], sculpt_classical_face(-.05, -.62, 8.82)[2])
        self.assertEqual(sculpt_classical_face(-1.1, -.6, 8.72), (-1.1, -.6, 8.72))


if __name__ == "__main__":
    unittest.main()
