"""Object-agnostic SfM cohort recovery, geometry QA and immutable inputs."""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kumiki_blender.reconstruction import sfm_cohort_recovery as sfm
from kumiki_blender.cli import main as cli_main


class Camera:
    def __init__(self, name, center, direction=(0, 0, 1)):
        self.name = name
        self.center = np.asarray(center, dtype=float)
        self.direction = np.asarray(direction, dtype=float)

    def projection_center(self):
        return self.center

    def viewing_direction(self):
        return self.direction


class Reconstruction:
    def __init__(self, cameras, *, points=100):
        self.images = {image_id: Camera(name, pos, direction)
                       for image_id, (name, pos, direction) in cameras.items()}
        self.points = points

    def reg_image_ids(self):
        return list(self.images)

    def num_points3D(self):
        return self.points

    def write(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model-marker.txt").write_text("ok", encoding="utf-8")


BASE_CAMERAS = {
    1: ("a.png", (0, 0, 0), (0, 0, 1)),
    2: ("b.png", (10, 0, 0), (0, 0, 1)),
    3: ("c.png", (0, 10, 0), (0, 0, 1)),
    4: ("d.png", (8, 8, 3), (0, 0, 1)),
}


def cohort_model(*, corrupted=False, id_collision=False, gain=True):
    models = {}
    for image_id, (name, pos, direction) in BASE_CAMERAS.items():
        loc = np.asarray(pos) * 2 + np.array([20, -10, 6])
        if corrupted and name == "d.png":
            loc += (15, 5, 0)
        models[image_id + (100 if id_collision and name == "a.png" else 0)] = (
            name, loc, direction)
    if gain:
        models[5] = ("u.png", (12, 4, 7), (0, 0, 1))
        models[6] = ("v.png", (15, 4, 7), (0, 0, 1))
    return Reconstruction(models, points=300)


def fixture(root: Path) -> Path:
    sparse = root / "revision" / "sparse-reconstruction"
    (sparse / "sparse" / "0").mkdir(parents=True)
    (sparse / "text-models" / "0").mkdir(parents=True)
    (sparse / "images").mkdir()
    names = ["a.png", "b.png", "c.png", "d.png", "u.png", "v.png", "w.png"]
    manifest = []
    for name in names:
        data = f"data-for-{name}".encode()
        (sparse / "images" / name).write_bytes(data)
        manifest.append({"filename": name, "sha256": hashlib.sha256(data).hexdigest()})
    (sparse / "staged-reference-index.json").write_text(json.dumps({"images": manifest}))
    (sparse / "text-models" / "0" / "images.txt").write_text("".join(
        f"{index} 1 0 0 0 0 0 0 1 {name}\n0 0 -1\n" for index, name in enumerate(names[:4], 1)))
    with sqlite3.connect(sparse / "database.db") as db:
        db.executescript("CREATE TABLE images (image_id INTEGER, name TEXT);"
                         "CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, config INTEGER);")
        db.executemany("INSERT INTO images VALUES (?, ?)", enumerate(names, 1))
        db.executemany("INSERT INTO two_view_geometries VALUES (?,?,?)", [
            (min(a, b) * sfm.MAX_IMAGE_ID + max(a, b), count, 2)
            for a, b, count in [(1, 5, 250), (2, 5, 120), (3, 6, 90),
                                (4, 6, 85), (5, 6, 220), (5, 7, 40), (1, 2, 500)]] + [
            (1 * sfm.MAX_IMAGE_ID + 7, 900, 1)])
    return sparse.parent


class FakePyCOLMAP:
    def __init__(self, candidate=None, *, map_failure=False):
        self.baseline = Reconstruction(BASE_CAMERAS)
        self.candidate = candidate or cohort_model()
        self.merged = Reconstruction({**BASE_CAMERAS,
                                      5: ("u.png", (0, 6, 4), (0, 0, 1)),
                                      6: ("v.png", (2, 6, 4), (0, 0, 1))}, points=400)
        self.calls = []
        self.map_failure = map_failure

    def Reconstruction(self, path):
        if Path(path).name == "merge-under-review":
            return self.merged
        return self.baseline

    @staticmethod
    def IncrementalPipelineOptions():
        return SimpleNamespace(image_names=[], init_image_id1=-1, init_image_id2=-1,
                               max_runtime_seconds=-1, multiple_models=True, ba_use_gpu=True,
                               min_model_size=10)

    def incremental_mapping(self, database, images, output, *, options):
        self.calls.append((Path(database), options.image_names,
                           options.init_image_id1, options.init_image_id2,
                           options.ba_use_gpu, options.max_runtime_seconds))
        if self.map_failure:
            raise RuntimeError("fake native mapper failure")
        return {0: self.candidate}


class CohortRecoveryTests(unittest.TestCase):
    def test_default_cohort_is_missing_photos_and_verified_bridge_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = fixture(Path(tmp))
            proposal = sfm.plan_cohort_recovery(revision)
            self.assertEqual(proposal["status"], "ready-to-attempt")
            self.assertEqual(proposal["missing_target_filenames"], ["u.png", "v.png", "w.png"])
            self.assertEqual(proposal["anchor_filenames"], sorted(BASE_CAMERAS[i][0] for i in BASE_CAMERAS))
            self.assertEqual({proposal["seed_pairs"][0]["a"], proposal["seed_pairs"][0]["b"]},
                             {"a.png", "u.png"})
            self.assertNotIn("w.png", proposal["registered_seed_filenames"])
            self.assertEqual(proposal["source_image_ids"]["u.png"], 5)

    def test_arbitrary_target_manifest_names_and_preferred_anchor_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = fixture(Path(tmp))
            proposal = sfm.plan_cohort_recovery(revision, targets=["u.png", "a.png"],
                                                preferred_anchors=["c.png"])
            self.assertEqual(proposal["missing_target_filenames"], ["u.png"])
            self.assertIn("a.png", proposal["anchor_filenames"])
            self.assertIn("c.png", proposal["anchor_filenames"])
            with self.assertRaisesRegex(ValueError, "already be registered"):
                sfm.plan_cohort_recovery(revision, preferred_anchors=["w.png"])
            with self.assertRaisesRegex(ValueError, "source database"):
                sfm.plan_cohort_recovery(revision, targets=["other-object.jpg"])

    def test_public_cli_plan_supports_arbitrary_filenames_without_pycolmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            names = root / "selected.json"
            names.write_text(json.dumps(["u.png", "v.png", "b.png"]))
            captured = io.StringIO()
            with redirect_stdout(captured):
                result = cli_main(["plan-sfm-cohort-recovery", "--revision", str(revision),
                                   "--targets", str(names), "--anchor", "a.png"])
            self.assertEqual(result, 0)
            plan = json.loads(captured.getvalue())
            self.assertEqual(plan["missing_target_filenames"], ["u.png", "v.png"])
            self.assertIn("a.png", plan["anchor_filenames"])
            self.assertEqual(plan["registered_seed_image_ids"]["d.png"], 4)

    def test_readonly_alignment_inspection_uses_specific_camera_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            fake = FakePyCOLMAP()
            model_dir = root / "independent"
            model_dir.mkdir()
            landmarks_path = root / "landmarks.json"
            landmarks = [{"id": str(i), "cohort_xyz": list(coords),
                          "baseline_xyz": list(np.asarray(coords) * 0.5 - [10, -5, 3])}
                         for i, coords in enumerate(((2, 2, 2), (4, 7, 2), (8, 2, 5)))]
            landmarks_path.write_text(json.dumps({"landmarks": landmarks}))
            fake.Reconstruction = lambda path: (
                fake.baseline if Path(path).name == "0" else fake.candidate)
            report = sfm.inspect_cohort_alignment(
                revision, model_dir, landmarks_path=landmarks_path, pycolmap_module=fake)
            self.assertEqual(report["status"], "alignment-evidence-passes / visual-review-required")
            self.assertEqual(report["object_landmarks"]["count"], 3)
            landmarks[0]["baseline_xyz"] = [float("nan"), 2.0, 3.0]
            landmarks_path.write_text(json.dumps({"landmarks": landmarks}))
            with self.assertRaisesRegex(ValueError, "finite baseline_xyz"):
                sfm.inspect_cohort_alignment(
                    revision, model_dir, landmarks_path=landmarks_path, pycolmap_module=fake)

    def test_cannot_mutate_or_use_unattributed_staged_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = fixture(Path(tmp))
            (revision / "sparse-reconstruction" / "images" / "u.png").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "modified staged image"):
                sfm.plan_cohort_recovery(revision)

    def test_refuses_to_write_outputs_under_source_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = fixture(Path(tmp))
            target = revision / "sparse-reconstruction" / "unsafe-child"
            with self.assertRaisesRegex(ValueError, "outside the source sparse revision"):
                sfm.recover_cohort(revision, target, pycolmap_module=FakePyCOLMAP(),
                                   merge_runner=lambda *a, **kw: None)
            self.assertFalse(target.exists())

    def test_distinct_camera_frames_align_via_proper_similarity(self):
        result = sfm.assess_alignment(Reconstruction(BASE_CAMERAS), cohort_model(),
                                      sfm.CohortRecoverySettings())
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["shared_cameras"], 4)
        self.assertEqual(result["new_cameras"], 2)
        self.assertAlmostEqual(result["scale"], 0.5)
        self.assertLess(result["camera_max_residual_ratio"], 1e-9)

    def test_rejects_wrong_image_ids_and_wrong_camera_poses(self):
        baseline = Reconstruction(BASE_CAMERAS)
        settings = sfm.CohortRecoverySettings()
        collision = sfm.assess_alignment(baseline, cohort_model(id_collision=True), settings)
        self.assertFalse(collision["accepted"])
        self.assertIn("image-id-collision-or-different-database", collision["reasons"])
        wrong = sfm.assess_alignment(baseline, cohort_model(corrupted=True), settings)
        self.assertFalse(wrong["accepted"])
        self.assertIn("common-camera-center-mismatch", wrong["reasons"])

    def test_rejects_collinear_anchor_centers_and_fewer_than_three(self):
        baseline = Reconstruction({1: BASE_CAMERAS[1], 2: BASE_CAMERAS[2], 3: BASE_CAMERAS[3]})
        almost = cohort_model()
        almost.images.pop(3)
        result = sfm.assess_alignment(baseline, almost, sfm.CohortRecoverySettings())
        self.assertIn("insufficient-shared-registered-cameras", result["reasons"])
        cols = {i: (name, (i * 10, 0, 0), direction)
                for i, (name, _, direction) in BASE_CAMERAS.items()}
        result = sfm.assess_alignment(Reconstruction(cols), cohort_model(),
                                      sfm.CohortRecoverySettings())
        self.assertIn("invalid-geometric-evidence", result["reasons"][0])

    def test_fails_closed_on_inconsistent_object_landmarks(self):
        baseline = Reconstruction(BASE_CAMERAS)
        landmarks = [{"id": str(i), "cohort_xyz": pos,
                      "baseline_xyz": list(np.asarray(pos) * 0.5 - [10, -5, 3])}
                     for i, pos in enumerate(((2, 2, 2), (4, 7, 2), (8, 2, 5)))]
        good = sfm.assess_alignment(baseline, cohort_model(),
                                    sfm.CohortRecoverySettings(), landmarks)
        self.assertTrue(good["accepted"], good)
        landmarks[2]["baseline_xyz"] = [100, 100, 100]
        bad = sfm.assess_alignment(baseline, cohort_model(),
                                   sfm.CohortRecoverySettings(), landmarks)
        self.assertFalse(bad["accepted"])
        self.assertTrue(any("landmark" in cause or "geometric-evidence" in cause
                            for cause in bad["reasons"]))

    def test_independent_models_are_kept_when_alignment_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            original_db = sfm._sha256(revision / "sparse-reconstruction" / "database.db")
            fake = FakePyCOLMAP(cohort_model(corrupted=True))
            out = root / "fail-alignment"
            report = sfm.recover_cohort(revision, out, pycolmap_module=fake,
                                        merge_runner=lambda *a, **kw: self.fail("must not merge"))
            self.assertEqual(report["status"], "independent-models-only / alignment-not-validated")
            self.assertEqual(sfm._sha256(out / "database.db"), original_db)
            self.assertEqual(sfm._sha256(revision / "sparse-reconstruction" / "database.db"), original_db)
            self.assertTrue((out / "trial-01" / "sparse" / "0" / "model-marker.txt").is_file())
            self.assertEqual(len(fake.calls), 5)
            for _, names, id1, id2, gpu, seconds in fake.calls:
                self.assertEqual(set(names),
                                 {"a.png", "b.png", "c.png", "d.png", "u.png", "v.png", "w.png"})
                self.assertIn(id1, range(1, 8))
                self.assertIn(id2, range(1, 8))
                self.assertFalse(gpu)
                self.assertEqual(seconds, 1200)
            with self.assertRaises(FileExistsError):
                sfm.recover_cohort(revision, out, pycolmap_module=fake)

    def test_accepted_models_use_colmap_merger_and_report_review_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            fake = FakePyCOLMAP()
            commands = []
            def merge(command, **kwargs):
                commands.append((command, kwargs))
                return SimpleNamespace(returncode=0, stdout="Merged", stderr="")
            result = sfm.recover_cohort(revision, root / "merged", pycolmap_module=fake,
                                        merge_runner=merge)
            self.assertEqual(result["status"], "candidate / needs-object-and-camera-visual-review")
            self.assertEqual(result["merge"]["registered_cameras"], 6)
            self.assertTrue((root / "merged" / "merged-candidate").is_dir())
            self.assertEqual(len(commands), 1)
            self.assertIn("model_merger", commands[0][0])
            self.assertEqual(commands[0][1]["timeout"], 1200)
            saved = json.loads((root / "merged" / "cohort-recovery-report.json").read_text())
            self.assertEqual(saved["status"], result["status"])

    def test_merger_rejects_native_failure_or_lost_baseline_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            fake = FakePyCOLMAP()
            denied = sfm.recover_cohort(revision, root / "denied", pycolmap_module=fake,
                merge_runner=lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="bad"))
            self.assertEqual(denied["status"], "merge-rejected / colmap-geometric-check")
            fake.merged = Reconstruction({2: BASE_CAMERAS[2], 5: ("u.png", (0, 0, 0), (0, 0, 1))})
            rejected = sfm.recover_cohort(revision, root / "rejected", pycolmap_module=fake,
                merge_runner=lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
            self.assertEqual(rejected["status"], "merge-rejected / lost-original-cameras")
            self.assertFalse((root / "rejected" / "merged-candidate").exists())

    def test_failed_mapper_keeps_report_and_does_not_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            result = sfm.recover_cohort(revision, root / "failure",
                pycolmap_module=FakePyCOLMAP(map_failure=True),
                merge_runner=lambda *a, **kw: self.fail("should not merge"))
            self.assertEqual(result["status"], "no-independent-model / all-trials-failed")
            self.assertTrue(all(entry["status"] == "failed" for entry in result["trials"]))
            self.assertIn("fake native mapper", result["trials"][0]["error"])


if __name__ == "__main__":
    unittest.main()
