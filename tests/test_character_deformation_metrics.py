"""Pure-Python regressions for source-locked Blender rig pose evidence."""
from __future__ import annotations

import math
import unittest

from kumiki_blender.character_build.deformation_metrics import collision_delta, pose_motion


class DeformationMetricsTest(unittest.TestCase):
    def test_nonmoving_pose_is_observable(self):
        points = [(0, 0, 0), (0, 1, 0), (1, 0, 0)]
        report = pose_motion(points, points)
        self.assertTrue(report["finite"])
        self.assertEqual(report["max_vertex_displacement_m"], 0)
        self.assertEqual(report["moving_vertices"], 0)

    def test_motion_is_measured_in_world_space_without_registration(self):
        points = [(0, 0, 0), (0, 1, 0), (1, 0, 0)]
        posed = [(0, 0, 0), (0, 1.02, 0), (1, 0, 0)]
        report = pose_motion(points, posed)
        self.assertAlmostEqual(report["max_vertex_displacement_m"], .02)
        self.assertEqual(report["moving_vertices"], 1)
        self.assertAlmostEqual(report["p95_vertex_displacement_m"], .02)

    def test_nonfinite_geometry_is_not_misreported_as_zero_motion(self):
        report = pose_motion([(0, 0, 0)], [(math.nan, 0, 0)])
        self.assertFalse(report["finite"])
        self.assertIsNone(report["max_vertex_displacement_m"])

    def test_missing_vertices_and_malformed_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "count"):
            pose_motion([(0, 0, 0)], [(0, 0, 0), (1, 0, 0)])
        with self.assertRaisesRegex(ValueError, "empty"):
            pose_motion([], [])
        with self.assertRaisesRegex(ValueError, "epsilon"):
            pose_motion([(0, 0, 0)], [(0, 0, 0)], moving_epsilon_m=-1)

    def test_existing_contact_is_not_a_new_collision(self):
        result = collision_delta([(1, 2), (3, 4)], [(3, 4), (1, 2)])
        self.assertEqual(result["new_overlapping_triangle_pairs"], 0)
        self.assertEqual(result["status"], "existing-contact-review")

    def test_contact_delta_detects_new_and_resolved_overlaps(self):
        result = collision_delta([(1, 2), (3, 4)], [(1, 2), (5, 6)])
        self.assertEqual(result["neutral_overlapping_triangle_pairs"], 2)
        self.assertEqual(result["posed_overlapping_triangle_pairs"], 2)
        self.assertEqual(result["new_overlapping_triangle_pairs"], 1)
        self.assertEqual(result["resolved_overlapping_triangle_pairs"], 1)
        self.assertEqual(result["status"], "new-contact-review")


if __name__ == "__main__":
    unittest.main()
