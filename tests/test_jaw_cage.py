"""Synthetic jaw-cage geometry constraints; no monument likeness assertion."""
import unittest
import numpy as np
from kumiki_blender.jaw_cage import experimental_jaw_cage


class JawCageTests(unittest.TestCase):
    def test_preserves_crown_and_neck_foot_and_moves_chin_down(self):
        p=np.array([[0,37.2,2],[0,35.,1.8],[0,34.,1.8],[.5,35.,1.8]])
        q,audit=experimental_jaw_cage(p)
        np.testing.assert_array_equal(q[0],p[0])
        np.testing.assert_array_equal(q[2],p[2])
        self.assertLess(q[1,1],p[1,1])
        self.assertGreater(q[1,2],p[1,2])
        self.assertGreater(q[3,0],p[3,0])
        self.assertIn("HOLD",audit["jaw_status"])

    def test_symmetry_and_continuity(self):
        yy=np.linspace(33.8,36.5,200)
        p=np.vstack([np.column_stack([np.full_like(yy,x),yy,np.full_like(yy,1.8)])
                     for x in (-.5,.5)])
        q,_=experimental_jaw_cage(p)
        np.testing.assert_allclose(q[:200,0],-q[200:,0])
        self.assertLess(np.max(np.abs(np.diff(q[:200,1]-p[:200,1]))),.02)

    def test_rejects_invalid_and_extreme(self):
        for p in ([[float("nan"),35,1]],[[1,2]],[[1,2,3,4]]):
            with self.assertRaises(ValueError):
                experimental_jaw_cage(p)
        with self.assertRaises(ValueError):
            experimental_jaw_cage([[0,35,1]],drop_m=.6)


if __name__=="__main__":
    unittest.main()
