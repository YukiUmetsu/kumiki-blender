"""Pure actor geometry: inflation, bend chains, skin weights, Kumiki matrix parity."""

from __future__ import annotations

import math
import unittest

import numpy as np

from kumiki_blender.actor_geometry import (
    chain_weights, heightfield_mesh, inflate_mask, local_matrix, part_transform, trace_chain,
)


def disc(size: int = 80) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    return (x - size / 2) ** 2 + (y - size / 2) ** 2 < (size * .4) ** 2


def bent_arm() -> np.ndarray:
    """An L-shaped limb: 60 px across, then 60 px down."""
    mask = np.zeros((90, 90), bool)
    mask[10:20, 10:70] = True
    mask[10:80, 60:70] = True
    return mask


class Inflation(unittest.TestCase):
    def test_zero_outside_and_peak_in_the_middle(self):
        height = inflate_mask(disc())
        self.assertEqual(float(height[0, 0]), 0.0)
        self.assertAlmostEqual(float(height.max()), 1.0, places=3)
        peak_y, peak_x = np.unravel_index(height.argmax(), height.shape)
        self.assertLess(math.dist((peak_x, peak_y), (40, 40)), 8)

    def test_dome_rises_steeply_at_the_rim(self):
        """A round cross-section, not a cone: the first quarter gains over half the height."""
        height = inflate_mask(disc(), smooth=0)
        rim, centre = 8, 40
        quarter = rim + (centre - rim) // 4
        self.assertGreater(float(height[40, quarter]), .5)

    def test_empty_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            inflate_mask(np.zeros((8, 8), bool))


class Mesh(unittest.TestCase):
    def test_keeps_only_cells_touching_the_silhouette(self):
        height = inflate_mask(disc())
        mesh = heightfield_mesh(height, step=4)
        total = 2 * (height.shape[0] - 1) * (height.shape[1] - 1)
        self.assertLess(len(mesh.faces), total)
        self.assertGreater(len(mesh.faces), 100)

    def test_front_and_back_share_the_rim_plane(self):
        height = inflate_mask(disc())
        mesh = heightfield_mesh(height, step=4)
        rim = mesh.heights == 0
        self.assertTrue(rim.any())
        self.assertTrue((mesh.sides[mesh.heights > 0] != 0).any())


class Chain(unittest.TestCase):
    def test_follows_the_limb_around_the_corner(self):
        points = trace_chain(bent_arm(), (12, 15), joints=3)
        self.assertEqual(len(points), 4)
        start, *_, tip = points
        self.assertLess(math.dist(start, (12, 15)), 6)
        self.assertLess(math.dist(tip, (65, 79)), 8)
        # Every joint must stay on the art, or the bone leaves the limb.
        mask = bent_arm()
        for x, y in points:
            self.assertTrue(mask[int(y), int(x)], (x, y))

    def test_tip_hint_stops_the_chain_short(self):
        points = trace_chain(bent_arm(), (12, 15), joints=2, tip=(65, 40))
        self.assertLess(math.dist(points[-1], (65, 40)), 6)

    def test_start_off_the_art_snaps_to_it(self):
        points = trace_chain(bent_arm(), (0, 0), joints=2)
        self.assertTrue(bent_arm()[int(points[0][1]), int(points[0][0])])

    def test_joints_stay_distinct_on_a_wide_mass(self):
        """A feathered wing is wide. Joints must not collapse onto one medial point."""
        wing = np.zeros((600, 700), bool)
        wing[120:480, 40:660] = True
        points = trace_chain(wing, (60, 300), joints=3, tip=(640, 300))
        for a, b in zip(points, points[1:]):
            self.assertGreater(math.dist(a, b), 60)

    def test_empty_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            trace_chain(np.zeros((8, 8), bool), (1, 1), joints=2)


class Weights(unittest.TestCase):
    def test_rows_sum_to_one_and_ends_are_rigid(self):
        chain = [(0, 0), (50, 0), (100, 0)]
        vertices = np.array([[1., 3.], [50., 2.], [99., -3.]])
        weights = chain_weights(vertices, chain, blend=10)
        self.assertEqual(weights.shape, (3, 2))
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        self.assertGreater(weights[0, 0], .99)
        self.assertGreater(weights[2, 1], .99)
        self.assertAlmostEqual(float(weights[1, 0]), .5, delta=.1)


class PartTransform(unittest.TestCase):
    def transform(self, **turn):
        return part_transform(pivot=(100, 200), x=10, y=20, rotation=15, scale_x=1.2, scale_y=1.2, metres_per_px=.001, **turn)

    @staticmethod
    def apply(transform, u, v):
        location, rotation, scale = transform
        point = np.array((u * .001 * scale[0], 0, -v * .001 * scale[2]))
        return np.array(location) + rotation @ point

    def test_pivot_stays_fixed_under_a_3d_turn(self):
        still = self.apply(self.transform(), 100, 200)
        turned = self.apply(self.transform(yaw=20, pitch=-15), 100, 200)
        np.testing.assert_allclose(turned, still, atol=1e-12)

    def test_matches_the_flat_matrix_without_a_turn(self):
        matrix = local_matrix(pivot=(100, 200), x=10, y=20, rotation=15, scale_x=1.2, scale_y=1.2)
        flat = matrix @ np.array((140, 260, 1.))
        placed = self.apply(self.transform(), 140, 260)
        np.testing.assert_allclose(placed, (flat[0] * .001, 0, -flat[1] * .001), atol=1e-12)

    def test_positive_yaw_turns_the_face_toward_screen_right(self):
        _, rotation, _ = self.transform(yaw=30)
        self.assertGreater((rotation @ np.array((0, -1, 0)))[0], 0)

    def test_positive_pitch_looks_up(self):
        _, rotation, _ = part_transform(pivot=(0, 0), x=0, y=0, rotation=0, scale_x=1, scale_y=1, metres_per_px=.001, pitch=30)
        self.assertGreater((rotation @ np.array((0, -1, 0)))[2], 0)


class KumikiMatrixParity(unittest.TestCase):
    def test_matches_part_local_matrix_formula(self):
        """pivot + (x, y) + S·R(u − pivot), clockwise positive, y down."""
        matrix = local_matrix(pivot=(10, 20), x=5, y=-3, rotation=90, scale_x=2, scale_y=2)
        u = np.array([20, 20, 1.])
        out = matrix @ u
        # u − pivot = (10, 0); scaled (20, 0); 90° clockwise in y-down = (0, 20).
        np.testing.assert_allclose(out[:2], (15, 37), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
