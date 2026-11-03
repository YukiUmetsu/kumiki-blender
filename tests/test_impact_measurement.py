"""Impact observations must be derived from solver evidence, not gravity guesses."""

from __future__ import annotations

import unittest

from kumiki_blender.impact_measurement import (GroundSample, PairRelativeSample, ground_impact_frames,
                                               pair_impact_frames, surface_impact_speed)


def sample(frame: int, z: float, x: float = 0) -> GroundSample:
    return GroundSample(frame, z - .05, z, x, 0)


class ImpactMeasurementTests(unittest.TestCase):
    def test_surface_witness_detects_rotating_contact_with_stationary_body_origins(self):
        # A long, rotating bar's tip moves toward a wall and rebounds. Both
        # centers remain fixed, so pair_impact_frames must miss this event.
        centers = [PairRelativeSample(i, 0, 0, 1) for i in range(4)]
        self.assertEqual(pair_impact_frames(centers, lambda _: True), [])
        tip_relative = [(0, 0, gap) for gap in (.23, .09, .012, .10)]
        self.assertEqual(surface_impact_speed(tip_relative, (0, 0, 1)), 4.2)

    def test_surface_witness_does_not_invent_rubbing_or_pass_through_impacts(self):
        for points, normal in (
            ([(i*.1, 0, .005) for i in range(4)], (0, 0, 1)),
            ([(0, 0, z) for z in (.3, .2, .1, 0)], (0, 0, 1)),
            ([(0, 0, z) for z in (.3, .2, .1, .11)], (0, 0, -1)),
            ([(0, 0, z) for z in (.3, .2, .1, .11)], (0, 0, float("nan"))),
        ):
            with self.subTest(points=points, normal=normal):
                self.assertIsNone(surface_impact_speed(points, normal))

    def test_pair_impacts_require_convergence_surface_proximity_and_rebound(self):
        samples = [PairRelativeSample(i, 0, 0, z) for i, z in enumerate(
            [1, .9, .75, .55, .38, .31, .33, .5])]
        self.assertEqual(pair_impact_frames(samples, lambda i: i == 5), [(5, 5.1)])
        self.assertEqual(pair_impact_frames(samples, lambda i: False), [])
        self.assertEqual(pair_impact_frames([PairRelativeSample(i, 0, 0, .3) for i in range(8)],
                                            lambda _: True), [])
        passing = [PairRelativeSample(i, 0, 0, z) for i, z in enumerate(
            [1, .9, .75, .55, .38, .2, .02, -.2])]
        self.assertEqual(pair_impact_frames(passing, lambda _: True), [])

    def test_pair_impacts_reject_nonfinite_and_nonconsecutive_samples(self):
        samples = [PairRelativeSample(i, 0, 0, value) for i, value in enumerate(
            [1, .9, .75, .55, .38, float("nan"), .33, .5])]
        self.assertEqual(pair_impact_frames(samples, lambda _: True), [])
        skipped = [PairRelativeSample(2*i, 0, 0, .6-.04*i) for i in range(8)]
        self.assertEqual(pair_impact_frames(skipped, lambda _: True), [])

    def test_reports_fractional_ground_collision_from_trajectory_and_rebound(self):
        frames = [sample(i, z) for i, z in enumerate([.5, .45, .36, .24, .12, .075, .053, .08, .1])]
        cues = ground_impact_frames(frames, height_m=0, half_extent_m=8)
        self.assertEqual(len(cues), 1)
        self.assertGreater(cues[0][0], 4)
        self.assertLess(cues[0][0], 6)
        self.assertGreater(cues[0][1], 1)

    def test_refuses_falling_through_ground_moving_outside_or_too_slowly(self):
        positions = [.5, .45, .36, .24, .12, .075, -.08, -.4, -.7]
        for frames in ([sample(i, z) for i, z in enumerate(positions)],
                       [sample(i, z, x=10) for i, z in enumerate(positions)],
                       [sample(i, .055 - .0001*i) for i in range(9)]):
            with self.subTest(frames=frames):
                self.assertEqual(ground_impact_frames(frames, height_m=0, half_extent_m=8), [])

    def test_handles_nan_without_generating_an_event(self):
        samples = [sample(i, z) for i, z in enumerate([.5, .45, .36, .24, float("nan"), .075, .053, .08, .1])]
        self.assertEqual(ground_impact_frames(samples, height_m=0, half_extent_m=8), [])


if __name__ == "__main__":
    unittest.main()
