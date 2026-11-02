"""Carved stone grooves must be smooth, bounded and physically plausible."""
import math
import unittest

from kumiki_blender.sculpt_hair_flow import catmull_rom_3d, stone_hair_groove


class SculptHairFlowTests(unittest.TestCase):
    def test_center_is_recessed_and_bank_is_subtle(self):
        self.assertLess(stone_hair_groove(0, .5), -.0017)
        self.assertGreater(stone_hair_groove(.009, .5), 0)
        self.assertLess(stone_hair_groove(.009, .5), .0007)
        self.assertAlmostEqual(stone_hair_groove(.1, .5), 0, places=8)

    def test_ends_fade_to_unchanged_stone(self):
        self.assertEqual(stone_hair_groove(0, 0), 0)
        self.assertEqual(stone_hair_groove(0, 1), 0)
        self.assertLess(abs(stone_hair_groove(0, .05)),
                        abs(stone_hair_groove(0, .3)))

    def test_invalid_inputs_are_rejected(self):
        for d,t in ((-1,.5),(.01,-.1),(.01,1.1),(math.nan,.5)):
            with self.subTest(d=d,t=t), self.assertRaises(ValueError):
                stone_hair_groove(d,t)

    def test_three_dimensional_guides_interpolate_exact_endpoints(self):
        points=[(0,0,0),(.1,-.05,.1),(.2,-.02,.3),(.3,.05,.4)]
        curve=catmull_rom_3d(points,steps=8)
        self.assertEqual(len(curve),25)
        self.assertEqual(curve[0],points[0])
        self.assertEqual(curve[-1],points[-1])
        self.assertTrue(all(math.isfinite(x) for point in curve for x in point))


if __name__ == '__main__':
    unittest.main()
