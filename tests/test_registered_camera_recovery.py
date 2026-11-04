"""Real-camera recovery must not silently promote learned or invalid poses."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.reconstruction.registered_camera_recovery import (
    recover_registered_cameras, validate_camera_recovery,
)
from kumiki_blender.reconstruction.learned_camera_hypotheses import (
    align_learned_camera_hypotheses, select_learned_views,
)


class FakeImage:
    def __init__(self, name, center, direction=(0.0, 0.0, 1.0), tracks=50):
        self.name = name
        self.center = center
        self.direction = direction
        self.num_points3D = tracks

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


class RecoveryTests(unittest.TestCase):
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
                       side_effect=ValueError("no real matching tracks")), \
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
