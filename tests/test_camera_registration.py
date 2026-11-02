"""Synthetic camera fit verifies projection math, not monument likeness."""

import math
import unittest

try:
    from kumiki_blender.camera_registration import fit_landmarks, project
    import numpy as np
    import scipy
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Install optional reconstruction dependencies")
class CameraRegistrationTests(unittest.TestCase):
    def test_recovers_angled_camera_with_interpretable_pixel_residuals(self):
        xyz = np.array([(-1.0,-.6,36.8),(.9,-.5,36.8),
                        (-.7,-.8,36.2),(.8,-.9,36.2),
                        (0.,-1.3,35.7),(-.5,-.9,35.3),
                        (.4,-1.,35.3),(0.,-.65,34.9),
                        (-.8,.1,35.0),(.8,.2,35.2)])
        truth = dict(target=[.02,-.13,35.8], azimuth=.08,
                     elevation=math.radians(-31), distance=13.,
                     focal_px=1640., image_size=[1536,1024], roll=.015)
        screen,depth=project(xyz,**truth)
        self.assertGreater(min(depth),0)
        records=[{"name":str(i),"world_xyz":v.tolist(),
                  "photo_uv":uv.tolist()} for i,(v,uv) in
                 enumerate(zip(xyz,screen))]
        initial={k:truth[k] for k in ("target","azimuth","elevation",
                                      "distance","focal_px","roll")}
        fitted=fit_landmarks(records,[1536,1024],initial=initial)
        self.assertEqual(fitted["status"],"provisional_fit")
        self.assertLess(fitted["rms_reprojection_px"],.01)
        self.assertEqual(len(fitted["landmarks"]),10)

    def test_rejects_insufficient_anchors_or_invalid_camera(self):
        with self.assertRaisesRegex(ValueError,"At least eight"):
            fit_landmarks([], [700,700])
        with self.assertRaisesRegex(ValueError,"positive"):
            project([[0,0,0]],target=[0,0,0],azimuth=0,elevation=0,
                    distance=0,focal_px=900,image_size=[700,700])

    def test_outlier_landmark_blocks_registration_acceptance(self):
        xyz=np.array([(x,y,z) for x,y,z in [
            (-.8,-.8,36.4),(.8,-.8,36.4),(-.9,-.7,35.9),
            (.9,-.7,35.9),(-.5,-1.2,35.2),(.5,-1.2,35.2),
            (0.,-1.5,35.6),(0.,-.7,34.8),(.4,.1,35.1)]])
        camera=dict(target=[0,0,36],azimuth=.02,elevation=-.4,
                    distance=14.,focal_px=1650.,image_size=[1536,1024])
        pixel,_=project(xyz,**camera)
        records=[{"name":str(i),"world_xyz":p.tolist(),
                  "photo_uv":v.tolist()} for i,(p,v) in enumerate(zip(xyz,pixel))]
        records[7]["photo_uv"][1]+=130
        report=fit_landmarks(records,[1536,1024],max_error_px=12)
        self.assertTrue(report["status"].startswith("HOLD"))
        self.assertGreater(report["rms_reprojection_px"],12)


if __name__=="__main__":
    unittest.main()
