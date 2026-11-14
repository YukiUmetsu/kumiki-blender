"""Surface-geodesic sculpt regression fixtures; no Blender installation needed."""
from __future__ import annotations

import math
import unittest

from kumiki_blender.character_build.surface_distance import surface_distances


class SurfaceDistanceTest(unittest.TestCase):
    def test_nearby_but_disconnected_sheet_does_not_move(self):
        # Two sheet strips 0.001m apart in space but not topologically connected.
        positions = [
            (0, 0, 0), (.01, 0, 0), (.02, 0, 0),
            (0, 0, .001), (.01, 0, .001), (.02, 0, .001),
        ]
        edges = [(0, 1), (1, 2), (3, 4), (4, 5)]
        result = surface_distances(positions, edges, 0, .015)
        self.assertEqual(set(result), {0, 1})
        self.assertAlmostEqual(result[1], .01)
        self.assertNotIn(3, result)

    def test_protected_boundary_blocks_traversal(self):
        result = surface_distances([(0, 0, 0), (.005, 0, 0), (.01, 0, 0)],
                                   [(0, 1), (1, 2)], 0, .02,
                                   protected=frozenset({1}))
        self.assertEqual(result, {0: 0.0})

    def test_shortest_surface_path_not_euclidean_distance(self):
        positions = [(0, 0, 0), (0, .01, 0), (.01, .01, 0), (.01, 0, 0)]
        edges = [(0, 1), (1, 2), (2, 3)]
        result = surface_distances(positions, edges, 0, .025)
        self.assertEqual(set(result), {0, 1, 2})
        self.assertAlmostEqual(result[2], .02)
        self.assertNotIn(3, result)

    def test_reject_bad_sources(self):
        pts = [(0, 0, 0), (.01, 0, 0)]
        for seed, radius, protected in [
            (2, .02, frozenset()), (0, 0, frozenset()),
            (0, float("nan"), frozenset()), (0, .02, frozenset({0})),
            (0, .02, frozenset({2})), (True, .02, frozenset()),
        ]:
            with self.subTest(seed=seed, radius=radius, protected=protected):
                with self.assertRaises(ValueError):
                    surface_distances(pts, [(0, 1)], seed, radius, protected=protected)
        with self.assertRaisesRegex(ValueError, "edge"):
            surface_distances(pts, [(0, 5)], 0, .02)
        with self.assertRaisesRegex(ValueError, "finite"):
            surface_distances([(math.nan, 0, 0)], [], 0, .02)


if __name__ == "__main__":
    unittest.main()
