"""Real-camera recovery must not silently promote learned or invalid poses."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.reconstruction.registered_camera_recovery import (
    find_implausible_cameras, prioritize_missing_camera_views, recover_registered_cameras,
    remove_implausible_cameras, validate_camera_recovery, copy_subject_qa_provenance,
)
from kumiki_blender.reconstruction.learned_camera_hypotheses import (
    align_learned_camera_hypotheses, select_learned_views,
)


class FakeCamera:
    def __init__(self, focal=1000.0, width=1000, height=800):
        self.params = [focal, width / 2, height / 2]
        self.width, self.height = width, height


class FakeImage:
    def __init__(self, name, center, direction=(0.0, 0.0, 1.0), tracks=50, focal=1000.0):
        self.name = name
        self.center = center
        self.direction = direction
        self.num_points3D = tracks
        self.camera = FakeCamera(focal)
        self.camera_id = name

    def projection_center(self):
        return self.center

    def viewing_direction(self):
        return self.direction


class FakeModel:
    def __init__(self, positions):
        self.images = {index: FakeImage(name, center, tracks=tracks)
                       for index, (name, center, tracks) in enumerate(positions, 1)}

    def reg_image_ids(self):
        return list(self.images)

    @property
    def cameras(self):
        return {image.camera_id: image.camera for image in self.images.values()}


class RecoveryTests(unittest.TestCase):
    def test_recovered_revision_keeps_original_subject_confidence_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root / "source", root / "target"
            source.mkdir()
            target.mkdir()
            raster = root / "subject.png"
            raster.write_bytes(b"bytes")
            record = {"schema": "subject-feature-index-v1", "images": {
                "photo.png": {"confidence_path": str(raster),
                              "confidence_sha256": hashlib.sha256(b"bytes").hexdigest()}}}
            (source / "subject-feature-index.json").write_text(json.dumps(record))
            self.assertIsNotNone(copy_subject_qa_provenance(source, target))
            self.assertEqual((source / "subject-feature-index.json").read_bytes(),
                             (target / "subject-feature-index.json").read_bytes())
            raster.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                copy_subject_qa_provenance(source, target)
    def setUp(self):
        self.original = [("a", (0, 0, 0), 50), ("b", (1, 0, 0), 50),
                         ("c", (0, 1, 0), 50), ("d", (1, 1, 0), 50)]

    def test_accepts_real_new_pose_with_tracks_and_stable_baseline(self):
        result = validate_camera_recovery(
            FakeModel(self.original),
            FakeModel(self.original + [("new", (0.5, 0.5, 1), 35)]))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["new_cameras"], 1)

    def test_rejects_generated_pose_without_colmap_tracks(self):
        result = validate_camera_recovery(
            FakeModel(self.original),
            FakeModel(self.original + [("estimated", (0.5, 0.5, 1), 0)]))
        self.assertFalse(result["accepted"])
        self.assertIn("new-camera-lacks-verified-3d-tracks", result["reasons"])

    def test_rejects_nonfinite_new_camera_even_with_tracks(self):
        result = validate_camera_recovery(
            FakeModel(self.original),
            FakeModel(self.original + [("estimated", (float("nan"), 0, 1), 35)]))
        self.assertFalse(result["accepted"])
        self.assertIn("new-camera-has-invalid-pose", result["reasons"])

    def test_rejects_original_camera_drift(self):
        displaced = [("a", (0.4, 0, 0), 50)] + self.original[1:]
        result = validate_camera_recovery(
            FakeModel(self.original),
            FakeModel(displaced + [("new", (0.5, 0.5, 1), 35)]))
        self.assertFalse(result["accepted"])
        self.assertIn("baseline-camera-center-drift", result["reasons"])

    def test_rejects_missing_original_camera(self):
        result = validate_camera_recovery(
            FakeModel(self.original),
            FakeModel(self.original[:-1] + [("new", (0.5, 0.5, 1), 35)]))
        self.assertFalse(result["accepted"])
        self.assertIn("lost-original-registered-camera", result["reasons"])

    def test_no_new_cameras_cannot_pass(self):
        result = validate_camera_recovery(FakeModel(self.original), FakeModel(self.original))
        self.assertFalse(result["accepted"])
        self.assertIn("no-new-registered-cameras", result["reasons"])

    def test_native_success_without_new_real_tracks_stays_unverified(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "source", Path(temporary) / "recovered"
            model = source / "sparse" / "0"
            model.mkdir(parents=True)
            (source / "images").mkdir()
            (source / "database.db").write_bytes(b"database")
            (source / "staged-reference-index.json").write_text(json.dumps({"images": []}))
            (source / "reconstruction-report.json").write_text(json.dumps({
                "status": "candidate / visual-review-required", "engine": "colmap",
                "source_sparse_model": "0"}))
            with patch("kumiki_blender.reconstruction.registered_camera_recovery.shutil.which",
                       return_value="/usr/bin/colmap"), \
                 patch("kumiki_blender.reconstruction.sfm_cohort_recovery.recover_cohort",
                       side_effect=ValueError("no real matching tracks")) as cohort_attempt, \
                 patch("kumiki_blender.reconstruction.learned_camera_hypotheses.build_learned_camera_hypotheses",
                       return_value={"status": "unavailable / install-mapanything-extra"}):
                result = recover_registered_cameras(
                    source, target, runner=lambda _command, _log, _timeout: 0,
                    model_loader=lambda _path: FakeModel(self.original))
            self.assertEqual(result["status"], "no-verified-camera-recovery")
            self.assertFalse(result["candidate_accepted"])
            self.assertIn("image-registrator-produced-no-model",
                          result["initial_validation"]["reasons"])
            self.assertFalse((target / "reconstruction-report.json").exists())
            self.assertEqual([call.kwargs["settings"].min_pair_inliers
                              for call in cohort_attempt.call_args_list], [25, 18, 12, 10])
            self.assertTrue(all(call.kwargs["settings"].seed_strategy == "anchor-first"
                                for call in cohort_attempt.call_args_list))
            self.assertEqual(len(result["cohort_attempts"]), 4)
            self.assertEqual(result["remaining_elevated_filename_hints"], [])

    def test_existing_pnp_success_does_not_skip_missing_elevated_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()  # macOS temporary paths are symlinks.
            source, target = root / "source", root / "recovered"
            model = source / "sparse" / "0"
            model.mkdir(parents=True)
            (source / "images").mkdir()
            (source / "database.db").write_bytes(b"database")
            index = []
            for i in range(6):
                name = f"photo-{i:04d}.png"
                photo = source / "images" / name
                photo.write_bytes(f"photo{i}".encode())
                index.append({"filename": name, "original_filename":
                              "sphinx-top-elevated.jpg" if i == 5 else f"photo{i}.jpg",
                              "sha256": hashlib.sha256(photo.read_bytes()).hexdigest()})
            (source / "staged-reference-index.json").write_text(json.dumps({"images": index}))
            (source / "reconstruction-report.json").write_text(json.dumps({
                "status": "candidate / visual-review-required", "engine": "colmap",
                "source_sparse_model": "0"}))
            # Source names and selected elevated candidate are distinct; the
            # fake model intentionally leaves that elevated photo unregistered.
            source_names = {"photo-0000.png", "photo-0001.png", "photo-0002.png", "photo-0003.png"}
            originals = FakeModel([(name, center, tracks) for name, (_, center, tracks)
                                   in zip(sorted(source_names), self.original)])
            recovered = FakeModel([(name, center, tracks) for name, (_, center, tracks)
                                   in zip(sorted(source_names), self.original)] +
                                  [("photo-0004.png", (0.5, 0.5, 1), 38)])
            calls = []
            commands = []

            def runner(argv, _log, _timeout):
                calls.append(argv[1])
                commands.append(argv)
                if argv[1] == "image_registrator":
                    for suffix in ("cameras.bin", "images.bin", "points3D.bin"):
                        (target / "registered" / suffix).write_bytes(b"fake-model")
                    return 0
                return 1

            def model_loader(path):
                return recovered if path == target / "registered" else originals

            with patch("kumiki_blender.reconstruction.registered_camera_recovery.shutil.which",
                       return_value="/usr/bin/colmap"), \
                 patch("kumiki_blender.reconstruction.sfm_cohort_recovery.recover_cohort",
                       side_effect=ValueError("no valid independent cohort")), \
                 patch("kumiki_blender.reconstruction.learned_camera_hypotheses.build_learned_camera_hypotheses",
                       return_value={"status": "unavailable / model-not-installed"}):
                result = recover_registered_cameras(
                    source, target, runner=runner, model_loader=model_loader)
            self.assertTrue(result["initial_validation"]["accepted"])
            self.assertIn("point_triangulator", calls)
            triangulator = next(command for command in commands if command[1] == "point_triangulator")
            self.assertEqual(triangulator[triangulator.index("--clear_points") + 1], "0")
            self.assertIn("cohort_error", result)
            self.assertEqual(result["remaining_elevated_filename_hints"], ["photo-0005.png"])
            self.assertIn("photo-0005.png", result["elevated_filename_hints"])

    def test_recovered_revision_keeps_the_source_rectified_pixel_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "source", Path(temporary) / "recovered"
            (source / "sparse" / "0").mkdir(parents=True)
            (source / "images").mkdir()
            (source / "database.db").write_bytes(b"database")
            (source / "staged-reference-index.json").write_text(json.dumps({"images": []}))
            report = {"status": "candidate / visual-review-required", "engine": "colmap",
                      "source_sparse_model": "0", "max_image_size": 4000}
            (source / "reconstruction-report.json").write_text(json.dumps(report))
            originals = FakeModel(self.original)
            recovered = FakeModel(self.original + [("photo-new.png", (0.5, 0.5, 1), 38)])
            undistort = []

            def runner(argv, _log, _timeout):
                if argv[1] == "image_registrator":
                    for suffix in ("cameras.bin", "images.bin", "points3D.bin"):
                        (target / "registered" / suffix).write_bytes(b"fake-model")
                if argv[1] == "image_undistorter":
                    undistort.append(argv)
                    return 1
                return 0

            with patch("kumiki_blender.reconstruction.registered_camera_recovery.shutil.which",
                       return_value="/usr/bin/colmap"), \
                 patch("kumiki_blender.reconstruction.sfm_cohort_recovery.recover_cohort",
                       side_effect=ValueError("no valid independent cohort")), \
                 patch("kumiki_blender.reconstruction.learned_camera_hypotheses.build_learned_camera_hypotheses",
                       return_value={"status": "unavailable / model-not-installed"}):
                recover_registered_cameras(
                    source, target, runner=runner,
                    model_loader=lambda path: originals if path.name == "0" else recovered)
                self.assertEqual(len(undistort), 1)
                self.assertEqual(undistort[0][undistort[0].index("--max_image_size") + 1], "4000")
                report["max_image_size"] = "huge"
                (source / "reconstruction-report.json").write_text(json.dumps(report))
                with self.assertRaisesRegex(Exception, "max_image_size"):
                    recover_registered_cameras(source, Path(temporary) / "second",
                                               runner=runner, model_loader=lambda _path: originals)

    def test_degenerate_new_camera_is_named_and_a_good_one_is_not(self):
        baseline = FakeModel(self.original)
        candidate = FakeModel(self.original + [("telephoto.png", (20, 20, 20), 40),
                                               ("degenerate.png", (900, 900, 900), 40)])
        candidate.images[5].camera = FakeCamera(focal=10000.0)
        candidate.images[6].camera = FakeCamera(focal=5_000_000.0)
        self.assertEqual(find_implausible_cameras(baseline, candidate), {
            "degenerate.png": ["focal-length-too-long", "camera-too-far-from-other-cameras"]})

    def test_removing_implausible_cameras_keeps_a_clean_model_untouched(self):
        baseline = FakeModel(self.original)
        clean = FakeModel(self.original + [("good.png", (2, 2, 2), 40)])
        bad = FakeModel(self.original + [("good.png", (2, 2, 2), 40),
                                         ("degenerate.png", (900, 900, 900), 40)])
        bad.images[6].camera = FakeCamera(focal=5_000_000.0)
        calls, report = [], {}

        def dropper(model, names, output):
            calls.append((model, names))
            return output

        self.assertEqual(remove_implausible_cameras(baseline, Path("clean"), Path("out"), report,
                                                    lambda _path: clean, dropper), Path("clean"))
        self.assertEqual(calls, [])
        self.assertEqual(remove_implausible_cameras(baseline, Path("bad"), Path("out"), report,
                                                    lambda _path: bad, dropper), Path("out"))
        self.assertEqual(calls, [(Path("bad"), ["degenerate.png"])])
        self.assertIn("degenerate.png", report["implausible_new_cameras"])

    def test_elevated_photos_rank_before_generic_missing_views(self):
        staged = [{"filename": "generic.png", "original_filename": "sphinx-side.jpg"},
                  {"filename": "top.png", "original_filename": "sphinx-top-right.jpg"}]
        self.assertEqual(prioritize_missing_camera_views(staged, set())[0], "top.png")

    def test_learned_view_selection_prioritizes_existing_elevated_photos(self):
        index = [{"filename": f"photo-{i:04}.png", "original_filename":
                  (f"sphinx-top-{i}.jpg" if i in (8, 9) else f"photo-{i}.jpg")}
                 for i in range(1, 11)]
        selected = select_learned_views(index, {f"photo-{i:04}.png" for i in range(1, 8)})
        self.assertEqual(len(selected), 8)
        self.assertIn("photo-0008.png", selected)
        self.assertIn("photo-0009.png", selected)

    def test_learned_camera_poses_remain_unverified_even_when_anchors_align(self):
        baseline = FakeModel(self.original)
        predicted = FakeModel(self.original + [("top", (0, 0, 2), 0)])
        result = align_learned_camera_hypotheses(baseline, predicted)
        self.assertEqual(result["status"],
                         "learned-camera-hypotheses / requires-pnp-verification")
        self.assertEqual(result["pose_hypotheses"][0]["filename"], "top")
        self.assertIn("unverified", result["pose_hypotheses"][0]["provenance"])


if __name__ == "__main__":
    unittest.main()
