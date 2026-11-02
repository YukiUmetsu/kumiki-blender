"""Synthetic scan registration: rigid similarity, outliers and failure gates."""
import math
import unittest
import numpy as np
from kumiki_blender.scan_registration import fit_scan_landmarks,ScanRegistrationError


class ScanRegistrationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(43)
        self.src=rng.uniform(-1,1,(18,3))*[1.2,.7,1.6]
        angle=.41
        self.rot=np.array([[math.cos(angle),-math.sin(angle),0],
                           [math.sin(angle),math.cos(angle),0],[0,0,1.]])
        self.scale=1.8
        self.trans=np.array([.5,35.5,-.2])
        self.dst=self.scale*self.src@self.rot.T+self.trans

    def anchors(self,dst=None):
        dst=self.dst if dst is None else dst
        return [{"name":f"feature-{i}","source_xyz":a.tolist(),
                 "target_xyz":b.tolist()} for i,(a,b) in enumerate(zip(self.src,dst))]

    def test_recovers_rotation_scale_and_translation(self):
        report=fit_scan_landmarks(self.anchors())
        self.assertEqual(report['inlier_count'],18)
        self.assertLess(report['rms_inlier_m'],1e-5)
        self.assertAlmostEqual(report['uniform_scale'],self.scale,places=6)
        m=np.array(report['matrix_source_to_target'])
        np.testing.assert_allclose(m[:3,:3],self.scale*self.rot,atol=1e-7)
        np.testing.assert_allclose(m[:3,3],self.trans,atol=1e-7)
        self.assertIn('HOLD',report['status'])

    def test_outliers_do_not_distort_fitted_scan(self):
        noisy=self.dst.copy()
        noisy[2]+=[.8,.4,-.6]
        noisy[11]+=[-.9,.6,.2]
        report=fit_scan_landmarks(self.anchors(noisy))
        self.assertEqual(report['inlier_count'],16)
        self.assertEqual(set(report['outlier_names']),{'feature-2','feature-11'})
        self.assertLess(report['rms_inlier_m'],1e-5)

    def test_refuses_reflection(self):
        mirrored=self.dst.copy();mirrored[:,0]*=-1
        with self.assertRaisesRegex(ScanRegistrationError,'mirrored'):
            fit_scan_landmarks(self.anchors(mirrored))

    def test_refuses_anisotropic_scale(self):
        warped=self.dst.copy();warped[:,2]*=1.5
        with self.assertRaises(ScanRegistrationError):
            fit_scan_landmarks(self.anchors(warped))

    def test_refuses_coplanar_and_duplicate_landmarks(self):
        flat=self.anchors()
        for p in flat:
            p['source_xyz'][2]=0
        with self.assertRaisesRegex(ScanRegistrationError,'non-coplanar'):
            fit_scan_landmarks(flat)
        duplicates=self.anchors()
        duplicates[0]['name']=duplicates[1]['name']
        with self.assertRaisesRegex(ScanRegistrationError,'unique'):
            fit_scan_landmarks(duplicates)

    def test_refuses_bad_coordinates_and_few_points(self):
        with self.assertRaisesRegex(ScanRegistrationError,'6..200'):
            fit_scan_landmarks(self.anchors()[:4])
        points=self.anchors()
        points[1]['target_xyz'][2]=float('nan')
        with self.assertRaisesRegex(ScanRegistrationError,'finite'):
            fit_scan_landmarks(points)


if __name__=='__main__':unittest.main()
