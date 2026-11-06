"""Locating new photos: warp sampling, location checks and pose comparison."""
import unittest
from importlib.util import find_spec

import numpy as np

from kumiki_blender.reconstruction.match_photo_location import (
    LocationSettings, check_location, fitted_size, pose_difference, reprojection_errors, sample_warp, solve_pose,
    unique_track_observations)
from kumiki_blender.reconstruction.references import ReferenceError


def result(**overrides):
    return {"inliers": 500, "inlier_anchors": 6, "focal_ratio": 1.0, "holdout_error_px": 3.0, **overrides}


class WarpSamplingTests(unittest.TestCase):
    def test_reads_the_cell_under_each_anchor_pixel(self):
        warp = np.zeros((10, 10, 4))
        warp[2, 5, 2:] = (.5, -.5)
        certainty = np.zeros((10, 10))
        certainty[2, 5] = .8
        target, score = sample_warp(warp, certainty, np.array([[505., 210.]]), (1000, 1000), (200, 100))
        np.testing.assert_allclose(target, [[150., 25.]])
        self.assertAlmostEqual(float(score[0]), .8)

    def test_edge_pixels_stay_inside_the_grid(self):
        warp, certainty = np.zeros((4, 4, 4)), np.ones((4, 4))
        target, _ = sample_warp(warp, certainty, np.array([[1000., 1000.]]), (1000, 1000), (100, 100))
        np.testing.assert_allclose(target, [[50., 50.]])


class LocationCheckTests(unittest.TestCase):
    def test_a_supported_location_has_no_reasons(self):
        self.assertEqual(check_location(result(), LocationSettings()), [])

    def test_each_failed_check_is_named(self):
        settings = LocationSettings()
        for overrides, reason in (({"inliers": 39}, "too-few-inliers"),
                                  ({"inlier_anchors": 2}, "too-few-anchors-agree"),
                                  ({"focal_ratio": 5000.}, "implausible-focal-length"),
                                  ({"focal_ratio": .1}, "implausible-focal-length"),
                                  ({"holdout_error_px": 11.}, "held-out-anchors-disagree"),
                                  ({"holdout_error_px": None}, "held-out-anchors-disagree")):
            self.assertIn(reason, check_location(result(**overrides), settings))

    def test_invalid_settings_name_the_problem(self):
        with self.assertRaisesRegex(ReferenceError, "anchors_used"):
            LocationSettings(anchors_used=0)


class PhotoSizeTests(unittest.TestCase):
    def test_large_photos_shrink_and_small_photos_stay(self):
        self.assertEqual(fitted_size((8256, 5504), 4000), (4000, 2667))
        self.assertEqual(fitted_size((1600, 1200), 4000), (1600, 1200))


class TrackObservationTests(unittest.TestCase):
    def test_keeps_one_observation_for_each_3d_point(self):
        pixels = np.array([[1., 1.], [2., 2.], [3., 3.]])
        rows = unique_track_observations(pixels, np.array([7, 8, 7]))
        self.assertEqual([point_id for _, point_id in rows], [7, 8])
        np.testing.assert_allclose(rows[0][0], [1., 1.])


class PoseTests(unittest.TestCase):
    def test_pose_difference_measures_rotation_and_center(self):
        angle = np.radians(10)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        solved = {"R": rotation, "t": np.array([0, 0, -2.])}
        known = {"R": np.eye(3), "center": np.zeros(3)}
        degrees, distance = pose_difference(solved, known)
        self.assertAlmostEqual(degrees, 10, places=6)
        self.assertAlmostEqual(distance, 2, places=6)

    def test_points_behind_a_camera_have_infinite_error(self):
        solution = {"K": np.eye(3), "R": np.eye(3), "t": np.zeros(3)}
        errors = reprojection_errors(solution, np.zeros((2, 2)), np.array([[0, 0, 1.], [0, 0, -1.]]))
        self.assertEqual(errors[0], 0)
        self.assertTrue(np.isinf(errors[1]))


@unittest.skipUnless(find_spec("pycolmap") is not None, "Pose solving needs pycolmap")
class PoseSolveTests(unittest.TestCase):
    def test_recovers_pose_and_focal_length_with_outliers(self):
        rng = np.random.default_rng(0)
        points = rng.uniform([-2, -2, 6], [2, 2, 12], (300, 3))
        focal, size = 1700.0, (1000, 800)
        rotation, translation = np.eye(3), np.array([.3, -.2, .5])
        local = points + translation
        pixels = np.c_[focal * local[:, 0] / local[:, 2] + 500, focal * local[:, 1] / local[:, 2] + 400]
        pixels[:40] += rng.uniform(-200, 200, (40, 2))
        solution = solve_pose(pixels, points, size, 6.0)
        self.assertIsNotNone(solution)
        self.assertLess(abs(float(solution["camera"].params[0]) - focal) / focal, .02)
        self.assertGreater(int(solution["inlier_mask"].sum()), 240)
        np.testing.assert_allclose(solution["R"], rotation, atol=.02)


if __name__ == "__main__":
    unittest.main()
