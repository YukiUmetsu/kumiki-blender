import math
import unittest

from kumiki_blender.blender_textures import cylindrical_uv_ring


class CylindricalUVTests(unittest.TestCase):
    def test_crossing_back_seam_uses_short_continuous_u_interval(self):
        points = tuple((0.35*math.cos(a), 0.35*math.sin(a), z)
                       for a, z in ((6.20, 0.0), (0.08, 0.0), (0.08, 1.0), (6.20, 1.0)))
        coords = cylindrical_uv_ring(points, radius=0.35, tile_size_m=1.0)
        self.assertLess(abs(coords[0][0]-coords[1][0]), 0.1)
        self.assertAlmostEqual(coords[0][1], 0)
        self.assertAlmostEqual(coords[2][1], 1)

    def test_wrap_count_is_integral_for_seamless_repeat(self):
        uv = cylindrical_uv_ring(((1.0, 0.0, 0.0),), radius=1.0, tile_size_m=1.0)
        repeat = round(math.tau)
        self.assertEqual(uv, ((0.0, 0.0),))
        near_end = cylindrical_uv_ring(((math.cos(6.282), math.sin(6.282), 0),),
                                        radius=1.0, tile_size_m=1.0)
        self.assertAlmostEqual(near_end[0][0], repeat, delta=0.002)

    def test_rejects_invalid_scale(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cylindrical_uv_ring(((1, 0, 0),), radius=1, tile_size_m=value)
