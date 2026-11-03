"""MapAnything feed-forward sparse engine: geometry gates and revision contract.

No torch or model weights are needed. The worker boundary is replaced by a fake
runner that writes exact synthetic predictions (a plane seen by several cameras,
plus one view whose depth is wrong).
"""

import dataclasses
import json
import subprocess
import sys
import tempfile
import types
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
from kumiki_blender.reconstruction.mapanything_worker import _run, choose_device
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
        # One landscape photo exercises the canvas offset (x0 = 0, y0 = 86).
        size = (96, 64) if index == 0 else (64, 64)
        pixels = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(pixels).save(photos / f"photo-{index:02}.png")
    return photos


def _fake_worker(bogus_index=None, failing=False, asymmetric=None):
    """Return a runner that writes exact predictions for the request it is given.

    ``asymmetric`` maps a view index to a fy multiplier, to test per-axis focals.
    """

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
            intrinsics = intrinsics.copy()
            intrinsics[1, 1] *= (asymmetric or {}).get(index, 1.0)
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
            self.assertEqual(max(entry["dimensions"]), CANVAS)

    def test_cameras_round_trip_through_existing_colmap_readers(self):
        target = self.root / "sparse-r02"
        build_feedforward_sparse_reconstruction(self.references, target, runner=_fake_worker())
        intrinsics = cameras._read_intrinsics(target / "undistorted/text-model/cameras.txt")
        poses = cameras._read_poses(target / "undistorted/text-model/images.txt")
        self.assertEqual(len(poses), PHOTO_COUNT - 1)
        for camera_id, data in intrinsics.items():
            self.assertAlmostEqual(data["fx_px"], FOCAL, places=3)
            self.assertEqual(max(data["width"], data["height"]), CANVAS)

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


class FocalAndCanvasTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.references = load_photo_folder(_write_photos(self.root), "plane-test")

    def _cameras(self, target):
        lines = [line.split() for line in (target / "undistorted/sparse/cameras.txt").read_text().splitlines()
                 if line and not line.startswith("#")]
        return {int(line[0]): line for line in lines}

    def test_per_axis_focals_are_written_and_only_mapanything_plan_readers_accept_them(self):
        target = self.root / "asym"
        report = build_feedforward_sparse_reconstruction(
            self.references, target, runner=_fake_worker(asymmetric={1: 1.02}))
        self.assertEqual(report["status"], "candidate / visual-review-required")
        camera_file = target / "undistorted/text-model/cameras.txt"
        # The camera for view 1 keeps fx and fy separately.
        fx_fy = [(float(line[4]), float(line[5])) for line in self._cameras(target).values()]
        self.assertTrue(any(abs(fx - fy) / max(fx, fy) > 0.015 for fx, fy in fx_fy))
        with self.assertRaisesRegex(ReferenceError, "fx/fy"):
            cameras._read_intrinsics(camera_file)  # default square-pixel COLMAP rule
        loaded = cameras._read_intrinsics(camera_file, require_square_pixels=False,
                                          max_focal_asymmetry=0.05)
        self.assertEqual(len(loaded), PHOTO_COUNT - 1)

    def test_views_over_the_asymmetry_cap_are_not_registered(self):
        target = self.root / "cap"
        report = build_feedforward_sparse_reconstruction(
            self.references, target, runner=_fake_worker(asymmetric={2: 1.08}))
        # The bogus last view is excluded by depth; view 2 is excluded by its focals.
        self.assertEqual(report["registered_images"], PHOTO_COUNT - 2)
        self.assertEqual(len(report["unregistered_filenames"]), 2)
        loaded = cameras._read_intrinsics(target / "undistorted/text-model/cameras.txt",
                                          require_square_pixels=False, max_focal_asymmetry=0.05)
        self.assertTrue(all(abs(c["fx_px"] - c["fy_px"]) / max(c["fx_px"], c["fy_px"]) < 0.05
                            for c in loaded.values()))

    def test_offset_camera_has_principal_point_shifted_into_photo_coordinates(self):
        target = self.root / "offset"
        build_feedforward_sparse_reconstruction(self.references, target, runner=_fake_worker())
        # Landscape 96x64 -> 518x345 on the canvas, so y0 = (518 - 345) // 2 = 86.
        landscape = [line for line in self._cameras(target).values()
                     if (int(line[2]), int(line[3])) == (CANVAS, 345)]
        self.assertEqual(len(landscape), 1)
        self.assertAlmostEqual(float(landscape[0][7]), CANVAS / 2 - 86, places=6)  # cy
        self.assertAlmostEqual(float(landscape[0][6]), CANVAS / 2, places=6)  # cx, no x offset

    def test_points_take_colour_from_the_photo_they_were_sampled_from(self):
        target = self.root / "colour"
        build_feedforward_sparse_reconstruction(self.references, target, runner=_fake_worker())
        text_model = target / "undistorted/text-model"
        poses = cameras._read_poses(text_model / "images.txt")
        intrinsics = cameras._read_intrinsics(text_model / "cameras.txt")
        images = {name: np.asarray(Image.open(target / "undistorted/images" / name).convert("RGB"))
                  for _, _, name, _, _ in poses}
        points = [line.split() for line in (text_model / "points3D.txt").read_text().splitlines()
                  if line and not line.startswith("#")]
        for line in points[:40]:
            xyz = np.array([float(v) for v in line[1:4]])
            rgb = np.array([int(v) for v in line[4:7]])
            matched = False
            for _, camera_id, name, quaternion, tvec in poses:
                k = intrinsics[camera_id]
                cam = _matrix_from_quaternion(quaternion) @ xyz + np.array(tvec)
                if cam[2] <= 0:
                    continue
                u = round(k["fx_px"] * cam[0] / cam[2] + k["cx_px"])
                v = round(k["fy_px"] * cam[1] / cam[2] + k["cy_px"])
                image = images[name]
                if 0 <= v < image.shape[0] and 0 <= u < image.shape[1] and (image[v, u] == rgb).all():
                    matched = True
                    break
            self.assertTrue(matched, f"point {line[0]} colour matches no source photo pixel")


class MapAnythingWorkerCanvasTests(unittest.TestCase):
    """Runs the real worker _run() with stub torch/MapAnything modules (no weights)."""

    class _Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def __getitem__(self, key):
            return MapAnythingWorkerCanvasTests._Tensor(self.array[key])

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    def _stub_modules(self, seen):
        torch = types.ModuleType("torch")
        torch.__version__ = "stub"
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        torch.device = lambda name: types.SimpleNamespace(type=name)
        torch.from_numpy = lambda array: MapAnythingWorkerCanvasTests._Tensor(array)

        class _Scope:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False
        torch.inference_mode = lambda: _Scope()

        class _Model:
            def to(self, device):
                return self

            def eval(self):
                return self

            def infer(self, processed, **kwargs):
                seen.extend(view["img"].array.copy() for view in processed)
                size = CANVAS
                return [{
                    "camera_poses": self._t(np.eye(4)[None]),
                    "intrinsics": self._t(np.array([[[FOCAL, 0, size / 2], [0, FOCAL, size / 2], [0, 0, 1.0]]])),
                    "pts3d": self._t(np.zeros((1, size, size, 3))),
                    "depth_z": self._t(np.ones((1, size, size, 1))),
                    "mask": self._t(np.ones((1, size, size, 1))),
                } for _ in processed]

            @staticmethod
            def _t(array):
                return MapAnythingWorkerCanvasTests._Tensor(array)

        class MapAnything:
            @staticmethod
            def from_pretrained(model_id):
                return _Model()

        mapanything_models = types.ModuleType("mapanything.models")
        mapanything_models.MapAnything = MapAnything
        mapanything_image = types.ModuleType("mapanything.utils.image")
        mapanything_image.preprocess_inputs = lambda views: views
        mapanything = types.ModuleType("mapanything")
        return {"torch": torch, "mapanything": mapanything,
                "mapanything.models": mapanything_models,
                "mapanything.utils": types.ModuleType("mapanything.utils"),
                "mapanything.utils.image": mapanything_image}

    def test_worker_pastes_each_photo_at_its_canvas_offset_and_writes_contract(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        photo = root / "landscape.png"
        pixels = np.random.default_rng(3).integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(photo)
        w, h, x0, y0 = canvas_placement(96, 64)
        request = {"schema_version": 1, "model_id": "facebook/map-anything-apache", "device": "cpu",
                   "canvas": CANVAS,
                   "images": [{"path": str(photo), "width": w, "height": h, "x0": x0, "y0": y0}]}
        (root / "in.json").write_text(json.dumps(request))
        seen = []
        with patch.dict(sys.modules, self._stub_modules(seen)):
            code = _run(root / "in.json", root / "pred.npz", root / "meta.json")
        self.assertEqual(code, 0)
        canvas = seen[0]
        self.assertEqual(canvas.shape, (CANVAS, CANVAS, 3))
        resized = np.asarray(Image.open(photo).convert("RGB").resize((w, h), Image.Resampling.LANCZOS))
        np.testing.assert_array_equal(canvas[y0:y0 + h, x0:x0 + w], resized)
        self.assertEqual(int(canvas[:y0].sum()) + int(canvas[y0 + h:].sum()), 0)  # padding is black
        meta = json.loads((root / "meta.json").read_text())
        self.assertEqual((meta["device"], meta["canvas"], meta["views"]), ("cpu", CANVAS, 1))
        with np.load(root / "pred.npz") as data:
            self.assertEqual(sorted(data.files), ["K_0", "depth_0", "mask_0", "pose_0", "pts_0"])
            self.assertEqual(data["pts_0"].shape, (CANVAS, CANVAS, 3))


class WorkerFailureTests(unittest.TestCase):
    def test_missing_mapanything_is_a_clean_worker_failure(self):
        code = (
            "import sys, runpy; sys.modules['mapanything'] = None; "
            "runpy.run_module('kumiki_blender.reconstruction.mapanything_worker', run_name='__main__')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, "--input", "/nonexistent.json",
             "--output", "/tmp/unused.npz", "--meta", "/tmp/unused-meta.json"],
            capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("MapAnything worker failed", completed.stderr)


class PhotoModelEngineTests(unittest.TestCase):
    def test_mapanything_engine_settings_validate(self):
        PhotoModelSettings(engine="mapanything").validate()
        with self.assertRaisesRegex(ReferenceError, "mapanything_device"):
            PhotoModelSettings(engine="mapanything", mapanything_device="tpu").validate()
        for bad in (" ", "../../etc/passwd", "/abs/path", "no-slash"):
            with self.assertRaisesRegex(ReferenceError, "Hugging Face repo id"):
                PhotoModelSettings(engine="mapanything", mapanything_model=bad).validate()

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
