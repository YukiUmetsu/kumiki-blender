"""Gap filling with triangulated matches: cloud format, geometry and vetoes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kumiki_blender.reconstruction.match_densification import (
    DensificationSettings, encode_points, mask_veto, read_dense_cloud, scene_image_order,
    thin_to_voxels, triangulate_matches, views_agree_with_cameras, views_of_record, write_dense_cloud)
from kumiki_blender.reconstruction.references import ReferenceError


def camera(center, focal=1000.0, size=(1000, 800)):
    center = np.asarray(center, dtype=float)
    return {"K": np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1.]]),
            "R": np.eye(3), "t": -center, "center": center, "width": size[0], "height": size[1]}


def project(cam, points):
    pixels = (cam["K"] @ (points + cam["t"]).T).T
    return pixels[:, :2] / pixels[:, 2:]


HEADER = (b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
          b"property float32 x\nproperty float32 y\nproperty float32 z\n"
          b"property uint8 red\nproperty uint8 green\nproperty uint8 blue\n"
          b"property float32 nx\nproperty float32 ny\nproperty float32 nz\n"
          b"property list uint8 uint32 view_indices\nproperty list uint8 float32 view_weights\n"
          b"end_header\n")


class DenseCloudFormatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "cloud.ply"
        self.xyz = np.array([[1., 2., 3.], [4., 5., 6.]])
        body = encode_points(self.xyz, np.array([[1, 2, 3], [4, 5, 6]]), np.zeros((2, 3)),
                             np.array([[7, 8], [9, 3]]))
        self.path.write_bytes(HEADER + body)

    def test_records_round_trip_with_their_views(self):
        _, body, xyz, offsets = read_dense_cloud(self.path)
        np.testing.assert_allclose(xyz, self.xyz)
        self.assertEqual(views_of_record(body, offsets[1]), [9, 3])

    def test_records_of_different_length_are_read(self):
        header, body, _, _ = read_dense_cloud(self.path)
        three_views = (np.array([1., 1., 1.], "<f4").tobytes() + bytes(3) + bytes(12) + b"\x03" +
                       np.array([1, 2, 5], "<u4").tobytes() + b"\x01" +
                       np.array([1.], "<f4").tobytes())
        self.path.write_bytes(header.replace(b"vertex 2", b"vertex 1") + three_views)
        _, body, xyz, offsets = read_dense_cloud(self.path)
        self.assertEqual(views_of_record(body, offsets[0]), [1, 2, 5])

    def test_appended_points_keep_old_records_and_update_the_count(self):
        header, body, _, offsets = read_dense_cloud(self.path)
        extra = encode_points(np.array([[9., 9., 9.]]), np.array([[1, 1, 1]]), np.zeros((1, 3)),
                              np.array([[0, 1]]))
        write_dense_cloud(self.path, header, body, 2, extra, 1)
        _, new_body, xyz, new_offsets = read_dense_cloud(self.path)
        self.assertEqual(len(xyz), 3)
        np.testing.assert_allclose(xyz[2], [9, 9, 9])
        self.assertEqual(views_of_record(new_body, new_offsets[0]), [7, 8])

    def test_other_layouts_are_refused(self):
        self.path.write_bytes(HEADER.replace(b"view_weights", b"other") + bytes(10))
        with self.assertRaisesRegex(ReferenceError, "unsupported layout"):
            read_dense_cloud(self.path)


class SceneOrderTests(unittest.TestCase):
    def test_order_comes_from_the_scene_file(self):
        scene = b"xx/images/b.png\x00yy/images/a.png\x00zz/images/c.png"
        self.assertEqual(scene_image_order(scene, ["a.png", "b.png", "c.png"]),
                         {"b.png": 0, "a.png": 1, "c.png": 2})

    def test_missing_photo_is_refused(self):
        with self.assertRaisesRegex(ReferenceError, "does not contain"):
            scene_image_order(b"/a.png", ["a.png", "b.png"])

    def test_wrong_order_lowers_view_agreement(self):
        good, far = camera((0, 0, 0)), camera((0, 0, 100))
        points = np.array([[.1, .1, 5.]] * 4)
        self.assertEqual(views_agree_with_cameras(points, [[0]] * 4, {0: good, 1: far}), 1.0)
        self.assertEqual(views_agree_with_cameras(points, [[1]] * 4, {0: good, 1: far}), 0.0)


class TriangulationTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = camera((0, 0, 0)), camera((1.5, 0, 0))
        self.world = np.random.default_rng(0).uniform([-1, -1, 4], [1, 1, 7], (200, 3))

    def test_exact_matches_give_their_points(self):
        points, keep = triangulate_matches(self.a, self.b, project(self.a, self.world),
                                           project(self.b, self.world), 3.0, 3.0)
        self.assertTrue(keep.all())
        np.testing.assert_allclose(points, self.world, atol=1e-6)

    def test_a_small_ray_angle_is_rejected(self):
        near = camera((.01, 0, 0))
        _, keep = triangulate_matches(self.a, near, project(self.a, self.world),
                                      project(near, self.world), 3.0, 3.0)
        self.assertFalse(keep.any())

    def test_a_large_epipolar_error_is_rejected(self):
        wrong = project(self.b, self.world) + np.array([0, 25.])
        _, keep = triangulate_matches(self.a, self.b, project(self.a, self.world), wrong, 3.0, 3.0)
        self.assertFalse(keep.any())


class MaskVetoTests(unittest.TestCase):
    def setUp(self):
        self.cameras = {name: camera((index * .3, 0, 0)) for index, name in enumerate("abcde")}
        self.point = np.array([[.5, 0., 6.]])
        self.settings = DensificationSettings()

    def masks(self, background):
        return {name: np.full((800, 1000), name not in background) for name in self.cameras}

    def test_cameras_that_see_background_veto_a_point(self):
        self.assertTrue(mask_veto(self.point, self.cameras, self.masks("abcd"), self.settings)[0])

    def test_one_disagreeing_mask_does_not_veto(self):
        self.assertFalse(mask_veto(self.point, self.cameras, self.masks("a"), self.settings)[0])

    def test_too_few_cameras_cannot_veto(self):
        cameras = {name: self.cameras[name] for name in "ab"}
        self.assertFalse(mask_veto(self.point, cameras, self.masks("ab"), self.settings)[0])


class ThinningTests(unittest.TestCase):
    def test_keeps_one_point_per_voxel(self):
        points = np.array([[.1, .1, .1], [.2, .2, .2], [1.5, .1, .1]])
        self.assertEqual(len(thin_to_voxels(points, 1.0)), 2)


if __name__ == "__main__":
    unittest.main()
