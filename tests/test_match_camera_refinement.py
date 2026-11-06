"""Dense-match camera refinement: pair choice, tracks, and a camera that truly improves."""
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from PIL import Image

from kumiki_blender.reconstruction.match_camera_refinement import (
    RefinementSettings, epipolar_distance, fundamental_from_cameras, holdout_error, lattice_cells,
    merge_tracks, select_pairs, similarity_between, triangulate_rays)
from kumiki_blender.reconstruction.references import ReferenceError


def camera(center, focal=1000.0, size=(1000, 800)):
    """Build a camera at `center` that looks along +z, with R = identity."""
    rotation = np.eye(3)
    center = np.asarray(center, dtype=float)
    return {"K": np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1.]]),
            "R": rotation, "t": -rotation @ center, "center": center,
            "width": size[0], "height": size[1], "image_id": 1}


def project(cam, points):
    local = (cam["R"] @ points.T).T + cam["t"]
    pixels = (cam["K"] @ local.T).T
    return pixels[:, :2] / pixels[:, 2:]


class PairSelectionTests(unittest.TestCase):
    def test_a_weakly_connected_camera_keeps_its_best_partner(self):
        shared = {("a", "b"): 100, ("a", "c"): 90, ("b", "c"): 80, ("a", "aerial"): 3}
        self.assertEqual(select_pairs(shared, 1), [("a", "aerial"), ("a", "b"), ("a", "c")])

    def test_pairs_are_unique_and_ordered(self):
        pairs = select_pairs({("a", "b"): 5, ("b", "c"): 5}, 6)
        self.assertEqual(pairs, [("a", "b"), ("b", "c")])


class TrackMergeTests(unittest.TestCase):
    def test_matches_sharing_a_cell_join_into_one_track(self):
        tracks = merge_tracks([
            ("a", "b", np.array([[10., 10.]]), np.array([[20., 20.]])),
            ("b", "c", np.array([[21., 21.]]), np.array([[30., 30.]]))], 4.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(sorted(name for name, _, _ in tracks[0]), ["a", "b", "c"])

    def test_a_track_with_two_cells_in_one_image_is_dropped(self):
        tracks = merge_tracks([
            ("a", "b", np.array([[10., 10.]]), np.array([[20., 20.]])),
            ("b", "c", np.array([[21., 21.]]), np.array([[30., 30.]])),
            ("c", "a", np.array([[31., 31.]]), np.array([[90., 90.]]))], 4.0)
        self.assertEqual(tracks, [])


class GeometryTests(unittest.TestCase):
    def test_correct_cameras_give_zero_epipolar_distance(self):
        first, second = camera((0, 0, 0)), camera((1, .2, 0))
        points = np.random.default_rng(0).uniform([-1, -1, 4], [1, 1, 8], (50, 3))
        distance = epipolar_distance(fundamental_from_cameras(first, second),
                                     project(first, points), project(second, points))
        self.assertLess(distance.max(), 1e-6)

    def test_triangulation_recovers_a_point_and_refuses_parallel_rays(self):
        point = np.array([.3, -.2, 5.])
        centers = [np.zeros(3), np.array([1., 0, 0])]
        directions = [(point - c) / np.linalg.norm(point - c) for c in centers]
        np.testing.assert_allclose(triangulate_rays(centers, directions), point, atol=1e-9)
        self.assertIsNone(triangulate_rays(centers, [directions[0], directions[0]]))

    def test_holdout_error_rises_when_a_camera_is_wrong(self):
        good = {"a": camera((0, 0, 0)), "b": camera((1, .2, 0))}
        points = np.random.default_rng(0).uniform([-1, -1, 4], [1, 1, 8], (200, 3))
        pair = {"first": "a", "second": "b", "holdout_a": project(good["a"], points),
                "holdout_b": project(good["b"], points)}
        wrong = {**good, "b": camera((1, .2, 0), focal=1100.0)}
        self.assertLess(holdout_error(good, [pair])["median_px"], 1e-3)
        self.assertGreater(holdout_error(wrong, [pair])["median_px"], 1.0)

    def test_pair_mean_shows_one_bad_pair_that_the_overall_median_hides(self):
        good = {name: camera((index * .6, 0, 0)) for index, name in enumerate("abcd")}
        points = np.random.default_rng(0).uniform([-1, -1, 4], [1, 1, 8], (200, 3))
        cameras = {**good, "d": camera((1.8, 0, 0), focal=1100.0)}
        pairs = [{"first": a, "second": b, "holdout_a": project(good[a], points),
                  "holdout_b": project(good[b], points)}
                 for a, b in (("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"))]
        result = holdout_error(cameras, pairs)
        self.assertLess(result["median_px"], 1e-3)
        self.assertGreater(result["pair_mean_px"], 1.0)

    def test_similarity_recovers_a_known_transform(self):
        points = np.random.default_rng(1).normal(size=(20, 3))
        angle = .7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        moved = 2.5 * (rotation @ points.T).T + np.array([1., -2., 3.])
        scale, fitted, translation = similarity_between(points, moved)
        self.assertAlmostEqual(scale, 2.5, places=9)
        np.testing.assert_allclose(fitted, rotation, atol=1e-9)
        np.testing.assert_allclose(translation, [1., -2., 3.], atol=1e-9)

    def test_held_out_cells_are_fixed_and_exclude_nothing_twice(self):
        rows, columns = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
        first = lattice_cells(rows.ravel(), columns.ravel(), 5)
        self.assertEqual(int(first.sum()), 20)
        np.testing.assert_array_equal(first, lattice_cells(rows.ravel(), columns.ravel(), 5))


class SettingsTests(unittest.TestCase):
    def test_invalid_settings_name_the_problem(self):
        with self.assertRaisesRegex(ReferenceError, "held-out"):
            RefinementSettings(holdout_every=1)
        with self.assertRaisesRegex(ReferenceError, "lattice_step"):
            RefinementSettings(lattice_step=0)
        with self.assertRaisesRegex(ReferenceError, "reprojection_schedule_px"):
            RefinementSettings(reprojection_schedule_px=())


@unittest.skipUnless(find_spec("pycolmap") is not None and find_spec("cv2") is not None,
                     "End-to-end refinement needs pycolmap and OpenCV")
class EndToEndRefinementTests(unittest.TestCase):
    """A camera with a wrong focal length improves, and held-out matches show it."""

    def setUp(self):
        import pycolmap

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        rng = np.random.default_rng(3)
        self.world = rng.uniform([-1.5, -1.2, 4], [1.5, 1.2, 8], (5000, 3))
        self.size = (1000, 800)
        centers = [(0, 0, 0), (.6, 0, 0), (0, .5, 0), (.6, .5, 0), (-.5, .2, 0)]
        self.truth = {f"view-{i}.png": camera(c) for i, c in enumerate(centers)}
        revision = self.root / "cameras"
        rectified = revision / "undistorted"
        (rectified / "sparse").mkdir(parents=True)
        (rectified / "images").mkdir()
        self.masks = self.root / "masks"
        self.masks.mkdir()
        model = pycolmap.Reconstruction()
        sparse = self.world[:300]
        element_ids = {}
        for index, (name, truth) in enumerate(self.truth.items(), 1):
            focal = 1100.0 if name == "view-4.png" else 1000.0
            model.add_camera_with_trivial_rig(pycolmap.Camera(
                model="PINHOLE", width=1000, height=800, params=[focal, focal, 500, 400], camera_id=index))
            xy = project(truth, sparse)
            image = pycolmap.Image(name=name, camera_id=index, image_id=index,
                                   points2D=pycolmap.Point2DList([pycolmap.Point2D(p) for p in xy]))
            model.add_image_with_trivial_frame(image, pycolmap.Rigid3d(
                pycolmap.Rotation3d(), truth["t"]))
            element_ids[index] = len(xy)
            Image.new("RGB", self.size, "gray").save(rectified / "images" / name)
            Image.new("L", self.size, 255).save(self.masks / (name + ".mask.png"))
        for point_index, xyz in enumerate(sparse):
            track = pycolmap.Track()
            track.add_elements([pycolmap.TrackElement(i, point_index) for i in element_ids])
            model.add_point3D(xyz, track)
        model.write(str(rectified / "sparse"))
        (revision / "reconstruction-report.json").write_text(json.dumps(
            {"status": "candidate / visual-review-required"}))
        (revision / "staged-reference-index.json").write_text(json.dumps({"images": []}))
        (revision / "undistorted-reference-index.json").write_text(json.dumps({"images": [
            {"filename": name, "sha256": "0" * 64} for name in self.truth]}))
        self.revision = revision

    def predictor(self):
        truth, world = self.truth, self.world

        class SyntheticMatcher:
            """Return a warp grid that holds exact matches at each lattice sample."""

            def dense_warp(self, first, second):
                rows, columns = 96, 96
                warp, certainty = np.zeros((rows, columns, 4)), np.zeros((rows, columns))
                a, b = truth[first.name], truth[second.name]
                cells = np.stack(np.mgrid[6:rows:12, 6:columns:12], -1).reshape(-1, 2)
                for index, (row, column) in enumerate(cells):
                    point = world[index * 7 % len(world)]
                    pa, pb = project(a, point[None])[0], project(b, point[None])[0]
                    if (0 <= pa[0] < 1000 and 0 <= pa[1] < 800 and 0 <= pb[0] < 1000 and 0 <= pb[1] < 800):
                        warp[row, column] = (pa[0] / 500 - 1, pa[1] / 400 - 1,
                                             pb[0] / 500 - 1, pb[1] / 400 - 1)
                        certainty[row, column] = .9
                return warp, certainty

        return SyntheticMatcher()

    def test_wrong_focal_length_is_corrected_and_pixels_are_kept(self):
        import pycolmap

        from kumiki_blender.reconstruction.match_camera_refinement import (
            STAGED_STATUS, refine_cameras_with_matches)

        output = self.root / "refined"
        report = refine_cameras_with_matches(
            self.revision, self.masks, output, predictor=self.predictor(),
            settings=RefinementSettings(pairs_per_camera=4, min_verified_matches=8))
        self.assertEqual(report["status"], STAGED_STATUS, report.get("reason"))
        self.assertLess(report["holdout_after"]["pair_mean_px"], .1)
        self.assertGreater(report["holdout_before"]["pair_mean_px"], 3.0)
        self.assertFalse(report["approved_for_reconstruction"])
        refined = pycolmap.Reconstruction(str(output / "undistorted/sparse"))
        focal = next(refined.cameras[image.camera_id].params[0] for image in refined.images.values()
                     if image.name == "view-4.png")
        self.assertLess(abs(focal - 1000.0), 40.0)
        self.assertEqual((output / "undistorted/images/view-0.png").read_bytes(),
                         (self.revision / "undistorted/images/view-0.png").read_bytes())
        self.assertTrue((output / "undistorted/text-model/cameras.txt").is_file())

    def test_rejects_a_camera_to_exclude_that_is_not_registered(self):
        from kumiki_blender.reconstruction.match_camera_refinement import refine_cameras_with_matches

        with self.assertRaisesRegex(ReferenceError, "not registered"):
            refine_cameras_with_matches(self.revision, self.masks, self.root / "x",
                                        predictor=self.predictor(), exclude_cameras=["absent.png"])


if __name__ == "__main__":
    unittest.main()
