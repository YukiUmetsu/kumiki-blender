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


class ContinuousGrooveProjectionTests(unittest.TestCase):
    def test_segment_projection_is_continuous_between_samples(self):
        from kumiki_blender.sculpt_hair_flow import polyline_arc_lengths, nearest_polyline_segment
        path = [(float(x),0.,0.) for x in range(5)]
        lengths = polyline_arc_lengths(path)
        for x in (1.49,1.50,1.51):
            dist,arc = nearest_polyline_segment((x,.3,0),path,lengths,2)
            self.assertAlmostEqual(dist,.3)
            self.assertAlmostEqual(arc,x/4)

    def test_curved_three_dimensional_path_and_end_taper(self):
        from kumiki_blender.sculpt_hair_flow import polyline_arc_lengths, nearest_polyline_segment
        path = [(0.,0.,0.),(0.,0.,1.),(1.,0.,1.)]
        lengths = polyline_arc_lengths(path)
        dist,arc = nearest_polyline_segment((.4,.2,1.),path,lengths,1)
        self.assertAlmostEqual(dist,.2)
        self.assertAlmostEqual(arc,.7)

    def test_rejects_bad_guide_geometry(self):
        from kumiki_blender.sculpt_hair_flow import polyline_arc_lengths, nearest_polyline_segment
        with self.assertRaises(ValueError):
            polyline_arc_lengths([(0.,0.,0.)]*2)
        with self.assertRaises(ValueError):
            polyline_arc_lengths([(0.,0.,0.),(float('nan'),0.,0.)])
        with self.assertRaises(ValueError):
            nearest_polyline_segment((0.,0.,0.),[(0.,0.,0.),(1.,0.,0.)],[0.,1.],5)

if __name__ == '__main__':
    unittest.main()
