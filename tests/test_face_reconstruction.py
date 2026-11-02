"""Camera-held-out facial fitting: synthetic examples, not real monument measurements."""
import math
import unittest
import numpy as np

from kumiki_blender.camera_registration import project
from kumiki_blender.face_reconstruction import (
    _camera_ray, _triangulate_rays, propose_face_corrections)


class FaceReconstructionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(421)
        self.donor = rng.normal(size=(16, 3)) * [.52, .33, .6] + [0., -.1, 36.]
        self.true = self.donor.copy()
        self.true[7] += [0.04, -0.07, -0.015]
        self.cameras = [
            dict(target=[0., -.15, 36.], azimuth=.08, elevation=-.32,
                 distance=13., focal_px=1900., roll=.012, image_size=[1400, 1000]),
            dict(target=[0., -.15, 36.], azimuth=.65, elevation=-.25,
                 distance=14., focal_px=1850., roll=-.015, image_size=[1400, 1000]),
        ]

    def _views(self):
        views = []
        for i, camera in enumerate(self.cameras):
            pixels, depths = project(self.true, **camera)
            self.assertTrue(np.all(depths > 0))
            views.append({"view_id": f"v{i}", "subject_photo_url": f"https://example.com/v{i}.jpg",
                          "model_revision": "synthetic-head-r1", "image_size": camera["image_size"],
                          "initial": {k: v for k, v in camera.items() if k != "image_size"},
                          "landmarks": [{"name": f"point_{j}",
                                         "world_xyz": xyz.tolist(), "photo_uv": uv.tolist()}
                                        for j, (xyz, uv) in enumerate(zip(self.donor, pixels))]})
        return views

    def test_exact_ray_round_trip(self):
        point = self.true[7]
        rays = []
        for camera in self.cameras:
            uv, _ = project(np.array([point]), **camera)
            meta = {"target_world_m": camera["target"],
                    "azimuth_deg": math.degrees(camera["azimuth"]),
                    "elevation_deg": math.degrees(camera["elevation"]),
                    "roll_deg": math.degrees(camera["roll"]),
                    "distance_m": camera["distance"], "focal_px": camera["focal_px"]}
            rays.append(_camera_ray(meta, uv[0], camera["image_size"]))
        recovered, diagnostic = _triangulate_rays(rays)
        np.testing.assert_allclose(recovered, point, atol=1e-8)
        self.assertLess(diagnostic["max_ray_gap_m"], 1e-7)

    def test_parallel_views_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "parallel"):
            _triangulate_rays([((0,0,0),(0,0,1)),((1,0,0),(0,0,1))])

    def test_wrong_correspondence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            _triangulate_rays([((0,0,0),(0,0,1)),((1,0,0),(-.8,.2,1))],
                              max_ray_gap_m=.001)

    def test_leave_one_out_recovers_moved_chin(self):
        result = propose_face_corrections(self._views(), max_camera_rms_px=10,
                                          max_target_error_px=5)
        self.assertEqual(result["status"], "HOLD_needs_visual_review")
        candidates = {c["name"]: c for c in result["candidates"]}
        self.assertIn("point_7", candidates, result["held_features"])
        np.testing.assert_allclose(candidates["point_7"]["target_world_xyz"],
                                   self.true[7], atol=.03)

    def test_provenance_and_same_mesh_required(self):
        views = self._views()
        views[1]["subject_photo_url"] = views[0]["subject_photo_url"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            propose_face_corrections(views)
        views = self._views()
        views[1]["model_revision"] = "different-mesh"
        with self.assertRaisesRegex(ValueError, "model_revision"):
            propose_face_corrections(views)


if __name__ == '__main__':
    unittest.main()
