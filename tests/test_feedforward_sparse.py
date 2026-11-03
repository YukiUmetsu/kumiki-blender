"""MapAnything feed-forward sparse engine: geometry gates and revision contract.

No torch or model weights are needed. The worker boundary is replaced by a fake
runner that writes exact synthetic predictions (a plane seen by several cameras,
plus one view whose depth is wrong).
"""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from kumiki_blender.reconstruction import cameras
from kumiki_blender.reconstruction.feedforward_sparse import (
    CANVAS, build_feedforward_sparse_reconstruction, canvas_placement,
    colmap_pose_from_cam2world, covisibility_matrices, largest_component,
    pose_is_valid, quaternion_from_rotation,
)
from kumiki_blender.reconstruction.mapanything_worker import choose_device
from kumiki_blender.reconstruction.photo_model import PhotoModelSettings, build_photo_model
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.splat_input import load_photo_folder

FOCAL = 400.0
PLANE_DEPTH = 5.0
PHOTO_COUNT = 8
# Small baselines keep every consistent pair inside the photo; the last view is bad.
CENTERS = [(0.02 * index, 0.01 * index, 0.0) for index in range(PHOTO_COUNT)]


def _matrix_from_quaternion(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _cam2world(center):
    pose = np.eye(4)
    pose[:3, 3] = center
    return pose


class GeometryTests(unittest.TestCase):
    def test_canvas_placement_keeps_whole_photo_and_centres_it(self):
        self.assertEqual(canvas_placement(96, 96), (CANVAS, CANVAS, 0, 0))
        w, h, x0, y0 = canvas_placement(852, 480)
        self.assertEqual((w, h, x0), (CANVAS, 292, 0))
        self.assertEqual(y0, (CANVAS - h) // 2)
        self.assertLessEqual(max(w, h), CANVAS)

    def test_quaternion_round_trip_for_axis_and_random_rotations(self):
        rng = np.random.default_rng(3)
        rotations = [np.eye(3)]
        for angle in (np.pi / 2, np.pi, 3.0):
            for axis in np.eye(3):
                c, s = np.cos(angle), np.sin(angle)
                k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                rotations.append(np.eye(3) + s * k + (1 - c) * k @ k)
        for _ in range(20):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            rotations.append(q)
        for rotation in rotations:
            quaternion = quaternion_from_rotation(rotation)
            self.assertGreaterEqual(quaternion[0], 0)
            self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=9)
            np.testing.assert_allclose(_matrix_from_quaternion(quaternion), rotation, atol=1e-9)

    def test_colmap_pose_is_world_to_camera(self):
        rng = np.random.default_rng(5)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q if np.linalg.det(q) > 0 else -q
        cam2world = np.eye(4)
        cam2world[:3, :3], cam2world[:3, 3] = q, [1.5, -2.0, 7.0]
        quaternion, translation = colmap_pose_from_cam2world(cam2world)
        point = np.array([0.3, 0.4, 9.0])
        in_camera = _matrix_from_quaternion(quaternion) @ point + translation
        expected = q.T @ (point - cam2world[:3, 3])
        np.testing.assert_allclose(in_camera, expected, atol=1e-9)

    def test_pose_validity_rejects_scale_and_non_finite(self):
        self.assertTrue(pose_is_valid(_cam2world([0, 0, 1])))
        scaled = _cam2world([0, 0, 1])
        scaled[:3, :3] *= 2.0
        self.assertFalse(pose_is_valid(scaled))
        broken = _cam2world([0, 0, 1])
        broken[0, 3] = np.nan
        self.assertFalse(pose_is_valid(broken))

    def test_largest_component_prefers_size_then_lowest_index(self):
        adjacency = np.zeros((5, 5), dtype=bool)
        adjacency[0, 1] = adjacency[1, 0] = True
        adjacency[2, 3] = adjacency[3, 2] = adjacency[3, 4] = adjacency[4, 3] = True
        self.assertEqual(largest_component(adjacency), [2, 3, 4])
        tied = np.zeros((4, 4), dtype=bool)
        tied[0, 1] = tied[1, 0] = tied[2, 3] = tied[3, 2] = True
        self.assertEqual(largest_component(tied), [0, 1])

    def test_device_policy_is_explicit_and_never_silent_for_cuda_or_mps(self):
        self.assertEqual(choose_device("auto", cuda_available=True, mps_available=True), "cuda")
        self.assertEqual(choose_device("auto", cuda_available=False, mps_available=True), "mps")
        self.assertEqual(choose_device("auto", cuda_available=False, mps_available=False), "cpu")
        with self.assertRaises(ValueError):
            choose_device("cuda", cuda_available=False, mps_available=True)
        with self.assertRaises(ValueError):
            choose_device("mps", cuda_available=True, mps_available=False)
        with self.assertRaises(ValueError):
            choose_device("tpu", cuda_available=True, mps_available=True)


def _plane_view(center, depth_value=PLANE_DEPTH):
    """Plane at z=PLANE_DEPTH seen by a camera at ``center`` (identity orientation)."""
    grid = np.arange(CANVAS, dtype=np.float64)
    cols, rows = np.meshgrid(grid, grid)
    c = CANVAS / 2
    points = np.stack([center[0] + (cols - c) * PLANE_DEPTH / FOCAL,
                       center[1] + (rows - c) * PLANE_DEPTH / FOCAL,
                       np.full_like(cols, PLANE_DEPTH)], axis=-1)
    intrinsics = np.array([[FOCAL, 0, c], [0, FOCAL, c], [0, 0, 1.0]])
    return points, np.full((CANVAS, CANVAS), depth_value), intrinsics


class CovisibilityTests(unittest.TestCase):
    def _views(self, centers, depth_values):
        views = []
        for center, depth_value in zip(centers, depth_values):
            points, depth, intrinsics = _plane_view(center, depth_value)
            sample = points[::40, ::40].reshape(-1, 3)
            views.append({"K": intrinsics, "depth": depth, "rect": (0, 0, CANVAS, CANVAS),
                          "points": sample, "world_to_cam": np.linalg.inv(_cam2world(center))})
        return views

    def test_consistent_neighbours_agree_and_distant_views_do_not_overlap(self):
        views = self._views([(0, 0, 0), (0.2, 0, 0), (10, 0, 0)], [PLANE_DEPTH] * 3)
        fraction, count = covisibility_matrices(views)
        self.assertGreater(fraction[0, 1], 0.99)
        self.assertGreater(fraction[1, 0], 0.99)
        self.assertEqual(count[0, 2], 0)
        self.assertEqual(fraction[0, 2], 0.0)

    def test_tolerance_sets_how_strict_depth_agreement_is(self):
        views = self._views([(0, 0, 0), (0.1, 0, 0)], [PLANE_DEPTH, 4.0])  # 25% depth error
        loose, _ = covisibility_matrices(views, tolerance=0.30)
        tight, _ = covisibility_matrices(views, tolerance=0.05)
        self.assertGreater(loose[0, 1], 0.99)
        self.assertLess(tight[0, 1], 0.01)

    def test_view_with_wrong_depth_is_not_re_observed(self):
        views = self._views([(0, 0, 0), (0.1, 0, 0)], [PLANE_DEPTH, 3.0])
        fraction, _ = covisibility_matrices(views)
        # View 0's samples are checked against view 1's wrong depth: not re-observed.
        self.assertLess(fraction[0, 1], 0.01)
        # View 1's samples land on view 0's correct depth, so the pair is still
        # rejected by the symmetric gate, which needs both directions.
        self.assertGreater(fraction[1, 0], 0.99)
        self.assertLess(min(fraction[0, 1], fraction[1, 0]), 0.30)


def _pycolmap_or_skip(test):
    try:
        import pycolmap
    except ImportError:
        test.skipTest("pycolmap is not installed")
    return pycolmap


def _write_photos(root):
    photos = root / "photos"
    photos.mkdir()
    rng = np.random.default_rng(11)
    for index in range(PHOTO_COUNT):  # load_photo_folder requires at least 8 photos
        pixels = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(photos / f"photo-{index:02}.png")
    return photos


def _fake_worker(bogus_index=None, failing=False):
    """Return a runner that writes exact predictions for the request it is given."""

    def runner(command, log, timeout):
        input_path = Path(command[command.index("--input") + 1])
        output_path = Path(command[command.index("--output") + 1])
        log.write_text("fake worker\n", encoding="utf-8")
        if failing:
            return 2
        request = json.loads(input_path.read_text(encoding="utf-8"))
        arrays = {}
        for index, entry in enumerate(request["images"]):
            center = CENTERS[index]
            bogus = index == (len(request["images"]) - 1 if bogus_index is None else bogus_index)
            points, depth, intrinsics = _plane_view(center, 3.0 if bogus else PLANE_DEPTH)
            w, h, x0, y0 = entry["width"], entry["height"], entry["x0"], entry["y0"]
            mask = np.zeros((CANVAS, CANVAS), dtype=bool)
            mask[y0:y0 + h, x0:x0 + w] = True
            arrays[f"pose_{index}"] = _cam2world(center).astype(np.float32)
            arrays[f"K_{index}"] = intrinsics.astype(np.float32)
            arrays[f"pts_{index}"] = points.astype(np.float32)
            arrays[f"depth_{index}"] = depth.astype(np.float32)
            arrays[f"mask_{index}"] = mask
        np.savez(output_path, **arrays)
        return 0

    return runner


class FeedForwardBuilderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.references = load_photo_folder(_write_photos(self.root), "plane-test")

    def test_consistent_views_register_and_bad_view_is_excluded(self):
        target = self.root / "sparse-r01"
        report = build_feedforward_sparse_reconstruction(
            self.references, target, runner=_fake_worker())
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertIsNone(report["failed_stage"])
        self.assertEqual(report["registered_images"], PHOTO_COUNT - 1)
        self.assertEqual(report["undistorted_images"], PHOTO_COUNT - 1)
        self.assertEqual(len(report["unregistered_filenames"]), 1)
        self.assertEqual(report["model_license"], "Apache-2.0")
        self.assertFalse(report["distribution_allowed"])

        index = json.loads((target / "undistorted-reference-index.json").read_text())["images"]
        self.assertEqual(sorted(entry["filename"] for entry in index), sorted(report["registered_filenames"]))
        for entry in index:
            self.assertTrue((target / "undistorted/images" / entry["filename"]).is_file())
            self.assertEqual(entry["dimensions"], [CANVAS, CANVAS])

    def test_cameras_round_trip_through_existing_colmap_readers(self):
        target = self.root / "sparse-r02"
        build_feedforward_sparse_reconstruction(self.references, target, runner=_fake_worker())
        intrinsics = cameras._read_intrinsics(target / "undistorted/text-model/cameras.txt")
        poses = cameras._read_poses(target / "undistorted/text-model/images.txt")
        self.assertEqual(len(poses), PHOTO_COUNT - 1)
        for camera_id, data in intrinsics.items():
            self.assertAlmostEqual(data["fx_px"], FOCAL, places=3)
            self.assertEqual((data["width"], data["height"]), (CANVAS, CANVAS))

    def test_pycolmap_can_load_the_text_revision_when_installed(self):
        pycolmap = _pycolmap_or_skip(self)
        target = self.root / "sparse-r03"
        build_feedforward_sparse_reconstruction(self.references, target, runner=_fake_worker())
        reconstruction = pycolmap.Reconstruction(str(target / "undistorted/sparse"))
        self.assertEqual(len(reconstruction.images), PHOTO_COUNT - 1)

    def test_worker_failure_checkpoints_without_undistorted_revision(self):
        target = self.root / "sparse-r04"
        report = build_feedforward_sparse_reconstruction(
            self.references, target, runner=_fake_worker(failing=True))
        self.assertEqual(report["failed_stage"], "mapanything-inference")
        self.assertEqual(report["status"], "review-required")
        self.assertFalse((target / "undistorted").exists())
        saved = json.loads((target / "reconstruction-report.json").read_text())
        self.assertEqual(saved["failed_stage"], "mapanything-inference")

    def test_too_few_consistent_views_is_not_a_candidate(self):
        target = self.root / "sparse-r05"

        def scattered(command, log, timeout):
            _fake_worker()(command, log, timeout)
            output = Path(command[command.index("--output") + 1])
            with np.load(output) as data:
                arrays = {key: data[key] for key in data.files}
            for index in range(PHOTO_COUNT):
                arrays[f"pose_{index}"] = _cam2world((index * 50.0, 0, 0)).astype(np.float32)
            np.savez(output, **arrays)
            return 0

        report = build_feedforward_sparse_reconstruction(self.references, target, runner=scattered)
        self.assertEqual(report["status"], "review-required")
        self.assertEqual(report["failed_stage"], "insufficient_registered_views")
        self.assertFalse((target / "undistorted").exists())

    def test_rejects_out_of_range_depth_tolerance(self):
        with self.assertRaisesRegex(ReferenceError, "depth_tolerance"):
            build_feedforward_sparse_reconstruction(
                self.references, self.root / "tol", depth_tolerance=0.0, runner=_fake_worker())

    def test_rejects_masks_bad_model_ids_and_existing_output(self):
        mask_path = self.root / "mask.png"
        Image.new("L", (64, 64), 255).save(mask_path)
        masked = dataclasses.replace(self.references, images=tuple(
            dataclasses.replace(image, mask=mask_path) for image in self.references.images))
        with self.assertRaisesRegex(ReferenceError, "object masks"):
            build_feedforward_sparse_reconstruction(masked, self.root / "x", runner=_fake_worker())
        with self.assertRaisesRegex(ReferenceError, "Hugging Face repo id"):
            build_feedforward_sparse_reconstruction(
                self.references, self.root / "y", model_id="../../etc", runner=_fake_worker())
        existing = self.root / "exists"
        existing.mkdir()
        with self.assertRaisesRegex(ReferenceError, "already exists"):
            build_feedforward_sparse_reconstruction(self.references, existing, runner=_fake_worker())


class PhotoModelEngineTests(unittest.TestCase):
    def test_mapanything_engine_settings_validate(self):
        PhotoModelSettings(engine="mapanything").validate()
        with self.assertRaisesRegex(ReferenceError, "mapanything_device"):
            PhotoModelSettings(engine="mapanything", mapanything_device="tpu").validate()
        with self.assertRaisesRegex(ReferenceError, "mapanything_model"):
            PhotoModelSettings(engine="mapanything", mapanything_model=" ").validate()

    def test_build_dispatches_to_mapanything_builder(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        references = load_photo_folder(_write_photos(root), "plane-test")
        settings = PhotoModelSettings(engine="mapanything", mapanything_model="facebook/map-anything-apache",
                                      mapanything_device="cpu", render_previews=False)
        dependencies = ({"InterfaceCOLMAP": "/bin/true", "DensifyPointCloud": "/bin/true",
                         "ReconstructMesh": "/bin/true", "RefineMesh": "/bin/true",
                         "TextureMesh": "/bin/true", "blender": "/bin/true"}, [])
        fake_report = {"status": "review-required", "failed_stage": "mapanything-inference"}
        with patch("kumiki_blender.reconstruction.photo_model.build_feedforward_sparse_reconstruction",
                   return_value=fake_report) as builder:
            report = build_photo_model(references, root / "out", settings, dependencies=dependencies)
        self.assertEqual(builder.call_args.kwargs["model_id"], "facebook/map-anything-apache")
        self.assertEqual(builder.call_args.kwargs["device"], "cpu")
        self.assertEqual(report["failed_stage"], "sparse-registration")


if __name__ == "__main__":
    unittest.main()
