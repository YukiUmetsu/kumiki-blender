"""Predicted-match cameras are staged only with an unchanged baseline, and are always labelled."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.reconstruction.predicted_camera_staging import (
    STAGED_STATUS, STAGING_REPORT, photos_with_predicted_keypoints, stage_predicted_cameras)
from kumiki_blender.reconstruction.references import ReferenceError


class FakeImage:
    def __init__(self, name, center, tracks):
        self.name, self.center, self.num_points3D = name, center, tracks

    def projection_center(self):
        return self.center

    def viewing_direction(self):
        return (0.0, 0.0, 1.0)


class FakeModel:
    def __init__(self, cameras):
        self.images = {index: FakeImage(*camera) for index, camera in enumerate(cameras, 1)}

    def reg_image_ids(self):
        return list(self.images)


BASELINE = [("a.png", (0, 0, 0), 50), ("b.png", (1, 0, 0), 50), ("c.png", (0, 1, 0), 50)]


def write_database(path, keypoints):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER)")
        for index, (name, rows) in enumerate(keypoints.items(), 1):
            db.execute("INSERT INTO images VALUES (?, ?)", (index, name))
            db.execute("INSERT INTO keypoints VALUES (?, ?)", (index, rows))


class PredictedKeypointTests(unittest.TestCase):
    def test_names_only_photos_with_added_keypoints(self):
        self.assertEqual(photos_with_predicted_keypoints(
            {"a.png": 10, "top.png": 5, "face.png": 7},
            {"a.png": 10, "top.png": 45, "face.png": 7}), {"top.png"})

    def test_rejects_a_database_of_other_photos_or_with_removed_keypoints(self):
        for candidate in ({"a.png": 10}, {"a.png": 10, "other.png": 5}, {"a.png": 9, "top.png": 5}):
            with self.assertRaisesRegex(ReferenceError, "prepare-roma-candidate-database"):
                photos_with_predicted_keypoints({"a.png": 10, "top.png": 5}, candidate)


class StagingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.source = self.root / "cameras"
        (self.source / "sparse" / "0").mkdir(parents=True)
        (self.source / "images").mkdir()
        names = [name for name, _, _ in BASELINE] + ["top.png", "face.png"]
        index = []
        for name in names:
            photo = self.source / "images" / name
            photo.write_bytes(name.encode())
            index.append({"filename": name, "sha256": hashlib.sha256(name.encode()).hexdigest()})
        (self.source / "staged-reference-index.json").write_text(json.dumps({"images": index}))
        (self.source / "reconstruction-report.json").write_text(json.dumps({
            "status": "candidate / visual-review-required", "source_sparse_model": "0",
            "max_image_size": 4000}))
        write_database(self.source / "database.db", {name: 10 for name in names})
        self.candidate = self.root / "candidate.db"
        write_database(self.candidate, {**{name: 10 for name in names}, "top.png": 60})
        self.target = self.root / "staged"
        self.commands = []

    def stage(self, registered):
        def runner(argv, _log, _timeout):
            self.commands.append(argv)
            if argv[1] == "image_registrator":
                for name in ("cameras.bin", "images.bin", "points3D.bin"):
                    (self.target / "registered" / name).write_bytes(b"model")
            return 1 if argv[1] == "image_undistorter" else 0

        return stage_predicted_cameras(
            self.source, self.candidate, self.target, runner=runner,
            model_loader=lambda path: FakeModel(BASELINE) if path.name == "0"
            else FakeModel(registered))

    def test_labels_predicted_and_measured_cameras_then_exports_at_the_source_size(self):
        report = self.stage(BASELINE + [("top.png", (2, 2, 2), 40), ("face.png", (3, 0, 1), 30)])
        self.assertEqual(report["predicted_match_cameras"], ["top.png"])
        self.assertEqual(report["measured_feature_cameras"], ["face.png"])
        self.assertTrue(report["validation"]["accepted"])
        self.assertFalse(report["approved_for_reconstruction"])
        stages = [argv[1] for argv in self.commands]
        self.assertEqual(stages[:2], ["geometric_verifier", "image_registrator"])
        # COLMAP verifies in a private copy; the candidate database stays as it was.
        self.assertIn(str(self.target / "database.db"), self.commands[0])
        undistort = next(argv for argv in self.commands if argv[1] == "image_undistorter")
        self.assertEqual(undistort[undistort.index("--max_image_size") + 1], "4000")
        # The fake undistorter fails, so the staged status is not reached.
        self.assertEqual(report["reason"], "image-undistorter-failed")
        self.assertNotEqual(report["status"], STAGED_STATUS)
        self.assertEqual(json.loads((self.target / STAGING_REPORT).read_text()), report)

    def test_a_moved_baseline_camera_blocks_the_export(self):
        moved = [("a.png", (5, 5, 5), 50)] + BASELINE[1:] + [("top.png", (2, 2, 2), 40)]
        report = self.stage(moved)
        self.assertEqual(report["reason"], "baseline-or-track-validation-failed")
        self.assertIn("baseline-camera-center-drift", report["validation"]["reasons"])
        self.assertFalse((self.target / "undistorted").exists())

    def test_a_predicted_camera_with_few_tracks_is_not_staged(self):
        report = self.stage(BASELINE + [("top.png", (2, 2, 2), 4)])
        self.assertEqual(report["reason"], "baseline-or-track-validation-failed")
        self.assertIn("new-camera-lacks-verified-3d-tracks", report["validation"]["reasons"])

    def test_only_measured_cameras_registered_is_not_a_predicted_staging(self):
        report = self.stage(BASELINE + [("face.png", (3, 0, 1), 30)])
        self.assertEqual(report["reason"], "no-predicted-photo-was-registered")
        self.assertEqual(report["unregistered_predicted_photos"], ["top.png"])

    def test_database_without_predictions_points_to_measured_recovery(self):
        with self.assertRaisesRegex(ReferenceError, "recover-registered-cameras"):
            stage_predicted_cameras(self.source, self.source / "database.db", self.target,
                                    runner=lambda *_: 0, model_loader=lambda _path: FakeModel(BASELINE))
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
