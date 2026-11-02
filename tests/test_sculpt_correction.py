"""Synthetic, physically bounded sculpt offsets; no real monument claim."""
import unittest
from kumiki_blender.sculpt_correction import bounded_sculpt_offsets, _wendland_c2


class SculptCorrectionTests(unittest.TestCase):
    def test_anchor_moves_by_exact_offset_and_far_vertices_stay_fixed(self):
        vertices=[(0,0,0),(.1,0,0),(.25,0,0),(.5,0,0)]
        candidates=[{"source_world_xyz":[0,0,0],"target_world_xyz":[0,-.08,.04]}]
        offsets=bounded_sculpt_offsets(vertices,candidates,influence_radius_m=.3)
        self.assertEqual(offsets[0],(0,-.08,.04))
        self.assertEqual(offsets[-1],(0,0,0))
        self.assertGreater(abs(offsets[1][1]),abs(offsets[2][1]))

    def test_overlap_does_not_add_two_full_displacements(self):
        candidates=[{"source_world_xyz":[0,0,0],"target_world_xyz":[0,-.12,0]},
                    {"source_world_xyz":[.02,0,0],"target_world_xyz":[.02,-.12,0]}]
        offsets=bounded_sculpt_offsets([(0,0,0)],candidates,influence_radius_m=.2)
        self.assertGreater(offsets[0][1],-.13)
        self.assertLess(offsets[0][1],-.11)

    def test_smooth_compact_kernel(self):
        self.assertEqual(_wendland_c2(0),1)
        self.assertEqual(_wendland_c2(1),0)
        self.assertEqual(_wendland_c2(2),0)
        self.assertGreater(_wendland_c2(.1),_wendland_c2(.4))

    def test_excessive_correction_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"cap"):
            bounded_sculpt_offsets([(0,0,0)],
                [{"source_world_xyz":[0,0,0],"target_world_xyz":[0,-1,0]}])
        with self.assertRaises(ValueError):
            bounded_sculpt_offsets([(0,0,0)],[],influence_radius_m=.2)


if __name__=='__main__':
    unittest.main()
