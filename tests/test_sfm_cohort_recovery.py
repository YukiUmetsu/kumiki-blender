"""Object-agnostic SfM cohort recovery, geometry QA and immutable inputs."""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr, redirect_stdout
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
    def test_disconnected_group_without_subject_qa_is_kept_unmerged(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = fixture(Path(tmp))
            independent = Reconstruction({5: ("u.png", (0, 0, 0), (0, 0, 1)),
                                          6: ("v.png", (2, 0, 0), (0, 0, 1)),
                                          7: ("w.png", (1, 2, 0), (0, 0, 1))})
            qa = sfm.assess_disconnected_alignment(
                Reconstruction(BASE_CAMERAS), independent,
                revision / "sparse-reconstruction/database.db", None,
                sfm.CohortRecoverySettings())
            self.assertFalse(qa["accepted"])
            self.assertIn("subject-confidence-unavailable", qa["reasons"])

    def test_disconnected_verified_path_does_not_call_common_camera_merger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            sparse = revision / "sparse-reconstruction"
            (sparse / "subject-feature-index.json").write_text('{"schema":"subject-feature-index-v1","images":{}}')
            candidate = Reconstruction({5: ("u.png", (0, 0, 0), (0, 0, 1)),
                                        6: ("v.png", (2, 0, 0), (0, 0, 1)),
                                        7: ("w.png", (1, 2, 0), (0, 0, 1))})
            fake = FakePyCOLMAP(candidate)
            fake.merged = Reconstruction({**BASE_CAMERAS,
                                          5: ("u.png", (0, 4, 2), (0, 0, 1)),
                                          6: ("v.png", (1, 4, 2), (0, 0, 1)),
                                          7: ("w.png", (2, 4, 2), (0, 0, 1))}, points=400)
            accepted = {"accepted": True, "reasons": [], "method": "verified-3d-points",
                        "scale": 1., "rotation": np.eye(3).tolist(),
                        "translation": [0, 0, 0], "inliers": 22,
                        "shared_cameras": 0, "new_cameras": 3,
                        "new_filenames": ["u.png", "v.png", "w.png"], "shared_filenames": []}
            with patch.object(sfm, "assess_disconnected_alignment", return_value=accepted), \
                 patch("kumiki_blender.reconstruction.sfm_point_alignment.fuse_disjoint_models",
                       return_value=fake.merged):
                report = sfm.recover_cohort(revision, root / "disconnected", pycolmap_module=fake,
                    merge_runner=lambda *a, **kw: self.fail("common camera merger must not run"))
            self.assertEqual(report["status"], "candidate / needs-object-and-camera-visual-review")
            self.assertEqual(report["merge"]["alignment"]["method"], "verified-3d-points")
            self.assertEqual(report["merge"]["registered_cameras"], 7)
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

    def test_weak_verified_cross_group_pair_is_seeded_after_confident_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = fixture(Path(tmp))
            database = revision / "sparse-reconstruction/database.db"
            with sqlite3.connect(database) as db:
                # Only 12 geometrically verified correspondences connect the
                # main registered group to w; strong u/v support remains.
                db.execute("INSERT INTO two_view_geometries VALUES (?,?,?)",
                           (1 * sfm.MAX_IMAGE_ID + 7, 12, 2))
                db.execute("UPDATE two_view_geometries SET rows=400 WHERE pair_id=?",
                           (5 * sfm.MAX_IMAGE_ID + 6,))
            strict = sfm.plan_cohort_recovery(revision)
            self.assertFalse(any({edge["a"], edge["b"]} == {"a.png", "w.png"}
                                 for edge in strict["seed_pairs"]))
            settings = sfm.CohortRecoverySettings(min_pair_inliers=12,
                                                   seed_strategy="anchor-first")
            relaxed = sfm.plan_cohort_recovery(revision, settings=settings)
            self.assertTrue(any({edge["a"], edge["b"]} == {"a.png", "w.png"}
                                for edge in relaxed["seed_pairs"]))
            self.assertTrue(all({edge["a"], edge["b"]} & set(relaxed["anchor_filenames"])
                                for edge in relaxed["seed_pairs"][:3]))
            # Low seed count cannot override the downstream geometry QA.
            self.assertEqual(settings.min_shared_cameras, sfm.CohortRecoverySettings().min_shared_cameras)
            self.assertEqual(settings.max_camera_residual_ratio,
                             sfm.CohortRecoverySettings().max_camera_residual_ratio)

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

    def test_geometry_aware_seeds_deprioritize_duplicate_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = fixture(Path(tmp))
            database = revision / "sparse-reconstruction" / "database.db"
            points = 250
            with sqlite3.connect(database) as db:
                db.executescript("""
                    ALTER TABLE images ADD camera_id INTEGER;
                    CREATE TABLE cameras (camera_id INTEGER, model INTEGER, width INTEGER,
                                          height INTEGER, params BLOB);
                    CREATE TABLE keypoints (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
                    ALTER TABLE two_view_geometries ADD cols INTEGER;
                    ALTER TABLE two_view_geometries ADD data BLOB;
                    ALTER TABLE two_view_geometries ADD qvec BLOB;
                    ALTER TABLE two_view_geometries ADD tvec BLOB;
                """)
                db.execute("INSERT INTO cameras VALUES (1,0,1000,800,?)",
                           (struct.pack("<3d", 1000, 500, 400),))
                db.execute("UPDATE images SET camera_id=1")
                for image_id in range(1, 8):
                    x_offset = 80 if image_id == 6 else 0
                    keypoints = b"".join(struct.pack("<ff", 200 + i * 2.0 + x_offset,
                                                        100 + (i * 7) % 600)
                                        for i in range(points))
                    db.execute("INSERT INTO keypoints VALUES (?,?,?,?)",
                               (image_id, points, 2, keypoints))
                # Both pairs have hundreds of verified matches, but a/u uses
                # the same bearings (zero triangulation-angle evidence).
                for first, second, count in ((1, 5, 250), (5, 6, 220)):
                    pair_id = first * sfm.MAX_IMAGE_ID + second
                    matches = b"".join(struct.pack("<II", i, i) for i in range(count))
                    db.execute("UPDATE two_view_geometries SET cols=2, data=?, "
                               "qvec=?, tvec=? WHERE pair_id=?",
                               (matches, struct.pack("<4d", 1, 0, 0, 0),
                                struct.pack("<3d", 1, 0, 0), pair_id))
            plan = sfm.plan_cohort_recovery(revision)
            self.assertTrue(any({edge["a"], edge["b"]} == {"a.png", "u.png"}
                                for edge in plan["rejected_low_parallax_seed_pairs"]))
            useful = next(edge for edge in plan["seed_pairs"]
                          if {edge["a"], edge["b"]} == {"u.png", "v.png"})
            self.assertGreater(useful["quality"]["median_ray_angle_degrees"], 1.5)
            self.assertGreater(useful["seed_score"], 0)
            self.assertTrue(plan["coverage_gaps"])

    def test_coverage_gaps_recommend_photographic_bridges_without_inventing_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = sfm.plan_cohort_recovery(fixture(Path(tmp)))
            groups = report["coverage_gaps"]
            self.assertGreaterEqual(len(groups), 1)
            self.assertEqual(sorted(name for item in groups for name in
                                    item["unregistered_filenames"]),
                             ["u.png", "v.png", "w.png"])
            self.assertTrue(all("intermediate" in item["acquisition_guidance"]
                                for item in groups))
            self.assertTrue(any(item["verified_pairs_to_registered"] > 0
                                for item in groups))

    def test_photo_request_names_original_sources_and_marks_angle_clues_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = fixture(Path(tmp))
            index = revision / "sparse-reconstruction" / "staged-reference-index.json"
            manifest = json.loads(index.read_text())
            originals = {"a.png": "registered-oblique.jpg", "u.png": "target-left-side2.jpeg",
                         "v.png": "target-screenshot5.jpg", "w.png": "target-top-view.jpg"}
            for item in manifest["images"]:
                item["original_filename"] = originals.get(item["filename"], item["filename"])
            index.write_text(json.dumps(manifest))
            plan = sfm.plan_cohort_recovery(revision)
            self.assertEqual(len(plan["first_photos_to_find"]), len(plan["coverage_gaps"]))
            for gap in plan["coverage_gaps"]:
                request = gap["photo_request"]
                self.assertIsNone(request["measured_missing_camera_angle"])
                self.assertIn("original_filename", request["unregistered"])
                self.assertIn("intermediate photos", request["next_photos_to_find"])
            views = {p["original_filename"]: p["view_hint"] for group in
                     plan["coverage_gaps"] for p in group["photo_request"]["other_missing_views"]}
            self.assertEqual(views["target-left-side2.jpeg"]["label"], "left side/profile")
            self.assertEqual(views["target-top-view.jpg"]["label"], "elevated or overhead view")
            self.assertEqual(views["target-screenshot5.jpg"]["label"],
                             "angle unknown; inspect photograph")

    def test_missing_angle_guide_markdown_and_visual_artifact(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow optional")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            sparse = revision / "sparse-reconstruction"
            index = sparse / "staged-reference-index.json"
            manifest = json.loads(index.read_text())
            for item in manifest["images"]:
                photo = sparse / "images" / item["filename"]
                Image.new("RGB", (40, 30), "#d7b889").save(photo)
                item["sha256"] = hashlib.sha256(photo.read_bytes()).hexdigest()
                item["original_filename"] = (
                    "unregistered-left-side.jpeg" if item["filename"] == "u.png"
                    else "my-original-" + item["filename"])
            index.write_text(json.dumps(manifest))
            guide = root / "angle-guide"
            output = io.StringIO()
            with redirect_stdout(output):
                rc = cli_main(["diagnose-sfm-coverage", "--revision", str(revision),
                               "--guide-dir", str(guide), "--format", "markdown"])
            self.assertEqual(rc, 0)
            self.assertIn("unregistered-left-side.jpeg", output.getvalue())
            self.assertIn("Registered reference", output.getvalue())
            self.assertIn("no measured COLMAP camera angles", output.getvalue())
            self.assertTrue((guide / "missing-angle-guide.md").is_file())
            self.assertTrue((guide / "missing-camera-angles.jpg").is_file())
            saved = json.loads((guide / "missing-angle-guide.json").read_text())
            self.assertTrue(saved["coverage_gaps"])
            self.assertTrue(all("photo_request" in group for group in saved["coverage_gaps"]))
            self.assertIn("![Missing-view contact sheet]", (guide / "missing-angle-guide.md").read_text())
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertNotEqual(cli_main(["diagnose-sfm-coverage", "--revision",
                                              str(revision), "--guide-dir", str(guide)]), 0)
            self.assertIn("new directory", stderr.getvalue())
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertNotEqual(cli_main(["diagnose-sfm-coverage", "--revision",
                                              str(revision), "--guide-dir",
                                              str(sparse / "unsafe")]), 0)
            self.assertIn("outside the source revision", stderr.getvalue())

    def test_coverage_report_counts_conservative_existing_3d_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = fixture(Path(tmp))
            sparse = revision / "sparse-reconstruction"
            (sparse / "text-models" / "0" / "points3D.txt").write_text(
                "42 1 2 3 255 255 255 1.0 1 0 3 999\n")
            with sqlite3.connect(sparse / "database.db") as db:
                db.execute("ALTER TABLE two_view_geometries ADD cols INTEGER")
                db.execute("ALTER TABLE two_view_geometries ADD data BLOB")
                for pair_id, count in db.execute(
                        "SELECT pair_id, rows FROM two_view_geometries").fetchall():
                    db.execute("UPDATE two_view_geometries SET cols=2, data=? "
                               "WHERE pair_id=?", (b"".join(struct.pack("<II", i, i)
                               for i in range(count)), pair_id))
            plan = sfm.plan_cohort_recovery(revision)
            group = next(group for group in plan["coverage_gaps"]
                         if "u.png" in group["unregistered_filenames"])
            self.assertEqual(group["existing_3d_track_support"]["u.png"][
                "conservative_one_to_one_2d3d"], 1)
            self.assertEqual(group["pnp_support_warning"],
                             "existing-3d-tracks-insufficient-for-default-pnp")

    def test_read_only_failed_run_correlation_tracks_camera_and_point_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            plan = sfm.plan_cohort_recovery(revision)
            past = root / "previous.json"
            past.write_text(json.dumps({
                "source_database_sha256": plan["source_database_sha256"],
                "source_model_index": 0,
                "status": "independent-models-only / alignment-not-validated",
                "trials": [{"trial": 1, "seed_pair": {"a": "u.png", "b": "v.png"},
                            "models": [{"registered_cameras": 2, "points3D": 41,
                                        "alignment": {"shared_cameras": 1}}]}],
            }))
            result = sfm.annotate_failed_recovery(plan, past)
            group = next(group for group in result["coverage_gaps"]
                         if "u.png" in group["unregistered_filenames"])
            self.assertEqual(group["previous_trial_results"][0]["model_sparse_points"], [41])
            self.assertEqual(group["previous_failure_evidence"],
                             "no-prior-trial-formed-3-shared-camera-anchors")
            self.assertEqual(result["previous_recovery"]["trials"], 1)
            with self.assertRaisesRegex(ValueError, "does not match"):
                past.write_text(json.dumps({"source_database_sha256": "wrong",
                                           "source_model_index": 0}))
                sfm.annotate_failed_recovery(plan, past)

    def test_public_coverage_diagnostic_works_without_pycolmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision = fixture(Path(tmp))
            out = io.StringIO()
            with redirect_stdout(out):
                rc = cli_main(["diagnose-sfm-coverage", "--revision", str(revision)])
            self.assertEqual(rc, 0)
            report = json.loads(out.getvalue())
            self.assertIn("coverage_gaps", report)
            self.assertIn("seed_pairs", report)
            self.assertTrue(report["coverage_gaps"])

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
