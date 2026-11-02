"""Weak-perspective camera fitting: synthetic truth and mismatch gates."""
import unittest
import numpy as np
from kumiki_blender.orthographic_registration import project_ortho,fit_ortho_landmarks


class OrthoRegistrationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(1234)
        self.xyz=rng.normal(size=(18,3))*[.6,.3,.8]+[0,-1.5,36]
        self.image_size=(800,900)
        self.camera=dict(target=self.xyz.mean(axis=0),azimuth=.2,elevation=-.3,
                         roll=.04,scale_px_per_m=210.,offset_uv=(410.,450.))
        self.uv=project_ortho(self.xyz,**self.camera)
        self.landmarks=[{"name":f"point-{i}","world_xyz":xyz.tolist(),
                         "photo_uv":uv.tolist()}
                        for i,(xyz,uv) in enumerate(zip(self.xyz,self.uv))]

    def test_recovers_camera_without_distance_focal_degeneracy(self):
        r=fit_ortho_landmarks(self.landmarks,self.image_size)
        self.assertEqual(r["status"],"provisional_ortho_fit")
        self.assertLess(r["rms_reprojection_px"],.1)
        self.assertAlmostEqual(r["camera"]["scale_px_per_m"],210,delta=.2)

    def test_held_when_target_shape_is_wrong(self):
        pts=[dict(p) for p in self.landmarks]
        pts[0]["photo_uv"]=[pts[0]["photo_uv"][0],pts[0]["photo_uv"][1]+75]
        r=fit_ortho_landmarks(pts,self.image_size,max_error_px=3)
        self.assertEqual(r["status"],"HOLD_high_reprojection_error")

    def test_rejects_invalid_correspondence(self):
        with self.assertRaises(ValueError):
            fit_ortho_landmarks(self.landmarks[:4],self.image_size)
        with self.assertRaises(ValueError):
            project_ortho([[0,0,0]],target=[0,0,0],azimuth=0,elevation=0,
                          roll=0,scale_px_per_m=-1,offset_uv=(0,0))


if __name__=="__main__":
    unittest.main()
