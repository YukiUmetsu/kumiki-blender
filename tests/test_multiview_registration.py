"""Synthetic camera validation does not imply actual-monument registration."""

import math
import unittest

try:
    import numpy as np
    from kumiki_blender.camera_registration import project
    from kumiki_blender.multiview_registration import assess_multiview
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,"Install optional reconstruction dependencies")
class MultiViewRegistrationTests(unittest.TestCase):
    def make_views(self):
        points=np.array([(-1.,-.6,36.8),(.9,-.5,36.8),
                         (-.7,-.8,36.2),(.8,-.9,36.2),
                         (0.,-1.3,35.7),(-.5,-.9,35.3),
                         (.4,-1.,35.3),(0.,-.65,34.9),
                         (-.8,.1,35.0),(.8,.2,35.2)])
        result=[]
        for view_id,azimuth,elevation in (("front",.08,-.42),("oblique",.55,-.32)):
            camera=dict(target=[0.,-.2,35.8],azimuth=azimuth,
                        elevation=elevation,distance=13.,focal_px=1600.,
                        image_size=[1536,1024],roll=.015)
            pixels,depth=project(points,**camera)
            self.assertTrue(np.all(depth>0))
            result.append({"view_id":view_id,"image_size":[1536,1024],
                           "model_revision":"anatomy-test",
                           "subject_photo_url":"https://example.org/"+view_id+".jpg",
                           "initial":{k:v for k,v in camera.items() if k!="image_size"},
                           "landmarks":[{"name":f"p{i}","world_xyz":p.tolist(),
                                         "photo_uv":uv.tolist()}
                                        for i,(p,uv) in enumerate(zip(points,pixels))]})
        return result

    def test_two_synthetic_views_fit_same_head(self):
        report=assess_multiview(self.make_views())
        self.assertEqual(report["status"],"provisional_multiview")
        self.assertEqual(report["common_landmark_count"],10)
        self.assertFalse(report["persistent_mismatch_candidates"])
        for view in report["views"]:
            self.assertLess(view["rms_reprojection_px"],.01)

    def test_rejects_inconsistent_3d_anchor(self):
        views=self.make_views()
        views[1]["landmarks"][4]["world_xyz"][2]+=.1
        with self.assertRaisesRegex(ValueError,"Inconsistent world_xyz"):
            assess_multiview(views)

    def test_tolerates_small_independent_annotation_offsets(self):
        views=self.make_views()
        views[1]["landmarks"][4]["world_xyz"][2]+=.012
        report=assess_multiview(views)
        self.assertEqual(report["status"],"provisional_multiview")
        self.assertAlmostEqual(report["max_shared_anchor_spread_m"],.012)

    def test_requires_independent_overlapping_views(self):
        views=self.make_views()
        with self.assertRaisesRegex(ValueError,"At least two"):
            assess_multiview(views[:1])
        views[1]["landmarks"][:6]=[
            {**p,"name":"other_"+p["name"]} for p in views[1]["landmarks"][:6]]
        with self.assertRaisesRegex(ValueError,"common facial landmarks"):
            assess_multiview(views)

    def test_rejects_same_photo_or_different_mesh(self):
        views=self.make_views()
        views[1]["subject_photo_url"]=views[0]["subject_photo_url"]
        with self.assertRaisesRegex(ValueError,"same source photograph"):
            assess_multiview(views)
        views=self.make_views()
        views[1]["model_revision"]="another-mesh"
        with self.assertRaisesRegex(ValueError,"same model_revision"):
            assess_multiview(views)

    def test_flags_missing_source_provenance(self):
        views=self.make_views()
        views[1].pop("subject_photo_url")
        report=assess_multiview(views)
        self.assertEqual(report["status"],"HOLD_unverified_source_independence")

    def test_flags_persistent_mismatch_but_never_auto_sculpts(self):
        views=self.make_views()
        for view in views:
            view["landmarks"][7]["photo_uv"][1]+=180.
        report=assess_multiview(views,max_error_px=10)
        self.assertTrue(report["status"].startswith("HOLD"))
        self.assertGreater(max(v["rms_reprojection_px"] for v in report["views"]),10)


if __name__=="__main__":
    unittest.main()
