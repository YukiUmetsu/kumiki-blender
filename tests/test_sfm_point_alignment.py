"""Independent camera groups need real cross-track points, not common cameras."""
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kumiki_blender.reconstruction.sfm_point_alignment import (
    estimate_point_alignment, extract_verified_3d_matches, fuse_disjoint_models,
    validate_cross_reprojection, _subject_lookup,
)
from kumiki_blender.reconstruction.sfm_cohort_recovery import (
    CohortRecoverySettings, assess_partial_alignment,
)


class Point:
    def __init__(self, pid, xy):
        self.point3D_id = pid
        self.xy = np.asarray(xy, dtype=float)

    def has_point3D(self):
        return True


class Image:
    def __init__(self, image_id, name, point_ids):
        self.image_id, self.name = image_id, name
        self.points2D = [Point(pid, (i * 3 + 10, i * 2 + 5))
                         for i, pid in enumerate(point_ids)]

    def project_point(self, xyz):
        return np.asarray(xyz[:2])


class Model:
    def __init__(self, images, points):
        self.images = {image.image_id: image for image in images}
        self.points3D = {i: SimpleNamespace(xyz=np.asarray(xyz, dtype=float))
                         for i, xyz in points.items()}

    def reg_image_ids(self):
        return list(self.images)


def synthetic_pairs(*, outliers=0):
    matches = []
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for i in range(40):
        a = np.array([float(i % 5) * 3, float((i // 5) % 4) * 5,
                      float(i // 20) * 7 + (i % 3) * 0.4])
        b = (rotation @ a) * 1.8 + np.array([20, 3, -9])
        if i < outliers:
            b += [i * 7 + 100, -i * 5 - 20, 80]
        matches.append({"baseline_xyz": a, "cohort_xyz": b,
                        "baseline_point_id": i + 1, "cohort_point_id": i + 100,
                        "image_pairs": [(i % 4 + 1, 10 + i % 3)],
                        "subject_verified": True, "reprojection_px": 0.4})
    return matches


class DisconnectedAlignmentTests(unittest.TestCase):
    def test_one_shared_camera_hybrid_aligns_real_triangulated_surface_tracks(self):
        from PIL import Image as PILImage
        import hashlib
        import json

        class HybridImage(Image):
            def __init__(self, image_id, point_ids, xyz):
                super().__init__(image_id, f"img-{image_id}.png", point_ids)
                for index, point in enumerate(self.points2D):
                    point.xy = np.asarray(xyz[index][:2], dtype=float) + 50

            def projection_center(self):
                return np.asarray([self.image_id * 3., 0, 0], dtype=float)

            def viewing_direction(self):
                return np.asarray([0, 0, 1], dtype=float)

            def project_point(self, xyz):
                return np.asarray(xyz[:2]) + 50

        def model(image_ids, offset, xyz):
            images = [HybridImage(image_id, [i + offset for i in range(len(xyz))], xyz)
                      for image_id in image_ids]
            reconstructed = Model(images, {i + offset: value for i, value in enumerate(xyz)})
            for index in range(len(xyz)):
                reconstructed.points3D[index + offset].track = SimpleNamespace(elements=[
                    SimpleNamespace(image_id=image_id, point2D_idx=index)
                    for image_id in image_ids])
            return reconstructed

        xyz = [(i % 5 * 3., (i // 5) % 3 * 5., i // 15 * 7. + i % 3 * 0.4)
               for i in range(30)]
        baseline = model([1, 2, 3], 0, xyz)
        cohort = model([1, 10], 100, xyz)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "database.db"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, config INTEGER, data BLOB)")
                for left, right in ((1, 2), (1, 3), (2, 10), (3, 10)):
                    blob = b"".join(struct.pack("<II", i, i) for i in range(len(xyz)))
                    conn.execute("INSERT INTO two_view_geometries VALUES (?,?,?,?,?)",
                                 (left * 2147483647 + right, len(xyz), 2, 2, blob))
            confidence = root / "subject.png"
            PILImage.new("L", (160, 160), 255).save(confidence)
            sha = hashlib.sha256(confidence.read_bytes()).hexdigest()
            qa = root / "subject-feature-index.json"
            qa.write_text(json.dumps({"schema": "subject-feature-index-v1", "images": {
                f"img-{image_id}.png": {"confidence_path": str(confidence),
                                       "confidence_sha256": sha, "dimensions": [160, 160]}
                for image_id in (1, 2, 3, 10)}}))
            aligned = assess_partial_alignment(
                baseline, cohort, db, qa, CohortRecoverySettings(min_new_cameras=1))
            self.assertTrue(aligned["accepted"], aligned["reasons"])
            self.assertEqual(aligned["shared_cameras"], 1)
            self.assertEqual(aligned["evidence_audit"]["shared_anchor_2d3d_pairs"], 30)
            self.assertGreaterEqual(aligned["new_camera_3d_support"]["10"]["count"], 25)
            self.assertEqual(aligned["method"], "verified-hybrid-3d-points")

            class NativeStylePointMap:
                """PyCOLMAP exposes indexed Point3DMap access, not dict.get."""
                def __init__(self, items):
                    self.items_by_id = items

                def __getitem__(self, key):
                    return self.items_by_id[key]

                def __contains__(self, key):
                    return key in self.items_by_id

                def items(self):
                    return self.items_by_id.items()

                def values(self):
                    return self.items_by_id.values()

                def __len__(self):
                    return len(self.items_by_id)

            original_base_points, original_cohort_points = baseline.points3D, cohort.points3D
            baseline.points3D = NativeStylePointMap(original_base_points)
            cohort.points3D = NativeStylePointMap(original_cohort_points)
            try:
                native_map_result = assess_partial_alignment(
                    baseline, cohort, db, qa, CohortRecoverySettings(min_new_cameras=1))
                self.assertTrue(native_map_result["accepted"], native_map_result["reasons"])
            finally:
                baseline.points3D, cohort.points3D = original_base_points, original_cohort_points

            two_shared = model([1, 2, 10], 100, xyz)
            aligned_two = assess_partial_alignment(
                baseline, two_shared, db, qa, CohortRecoverySettings(min_new_cameras=1))
            self.assertTrue(aligned_two["accepted"], aligned_two["reasons"])
            self.assertEqual(aligned_two["shared_cameras"], 2)
            # A scale/pose fit based on plausible stone features cannot hide
            # an incompatible second anchor camera center.
            two_shared.images[2].projection_center = lambda: np.asarray([100., 0., 0.])
            bad_anchor = assess_partial_alignment(
                baseline, two_shared, db, qa, CohortRecoverySettings(min_new_cameras=1))
            self.assertFalse(bad_anchor["accepted"])
            self.assertIn("shared-camera-center-mismatch", bad_anchor["reasons"])

            blank = root / "unverified-subject.png"
            PILImage.new("L", (160, 160), 128).save(blank)
            index = json.loads(qa.read_text())
            index["images"]["img-10.png"] = {
                "confidence_path": str(blank),
                "confidence_sha256": hashlib.sha256(blank.read_bytes()).hexdigest(),
                "dimensions": [160, 160],
            }
            qa.write_text(json.dumps(index))
            unverified = assess_partial_alignment(
                baseline, cohort, db, qa, CohortRecoverySettings(min_new_cameras=1))
            self.assertFalse(unverified["accepted"])
            self.assertIn("insufficient-verified-new-camera-3d-support", unverified["reasons"])

            # A model that sees a completely different object can have one
            # common camera ID and many 2D matches, but no valid 3D Sim(3).
            misleading = model([1, 10], 100, xyz)
            generator = np.random.default_rng(48)
            for point in misleading.points3D.values():
                point.xyz = generator.normal(0, 60, size=3)
            rejected = assess_partial_alignment(
                baseline, misleading, db, qa, CohortRecoverySettings(min_new_cameras=1))
            self.assertFalse(rejected["accepted"])
            self.assertTrue(rejected["reasons"])

    def test_subject_lookup_lazily_loads_only_participating_images(self):
        from PIL import Image as PILImage
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mask = root / "small.png"
            PILImage.new("L", (20, 20), 255).save(mask)
            good = {"confidence_path": str(mask),
                    "confidence_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
                    "dimensions": [20, 20]}
            irrelevant = {f"irrelevant-{n}.png": {
                "confidence_path": str(root / f"missing-{n}.png"),
                "confidence_sha256": "bad"} for n in range(256)}
            index = root / "index.json"
            index.write_text(json.dumps({"schema": "subject-feature-index-v1",
                                         "images": {"used.png": good, **irrelevant}}))
            lookup = _subject_lookup(index)
            self.assertTrue(lookup("used.png", (5., 5.)))
            self.assertFalse(lookup("not-staged.png", (5., 5.)))
            with self.assertRaisesRegex(ValueError, "subject|confidence"):
                lookup("irrelevant-0.png", (5., 5.))
    def test_native_pycolmap_merger_preserves_disjoint_tracks_and_camera_ids(self):
        try:
            import pycolmap
        except ImportError:
            self.skipTest("Optional photogrammetry dependency is absent")
        def make(image_ids):
            model = pycolmap.Reconstruction()
            camera = pycolmap.Camera(model="SIMPLE_PINHOLE", width=800, height=600,
                                    params=[900, 400, 300], camera_id=1)
            model.add_camera_with_trivial_rig(camera)
            for image_id in image_ids:
                frame = pycolmap.Image(name=f"{image_id}.png", camera_id=1, image_id=image_id,
                                       keypoints=np.array([[400., 300.], [410., 310.]]))
                model.add_image_with_trivial_frame(frame, pycolmap.Rigid3d())
            track = pycolmap.Track()
            for image_id in image_ids:
                track.add_element(image_id, 0)
            model.add_point3D(np.array([0., 0., 5.]), track)
            return model
        baseline, cohort = make([1, 2]), make([3, 4])
        candidate = fuse_disjoint_models(baseline, cohort, {
            "accepted": True, "scale": 2.0, "rotation": np.eye(3).tolist(),
            "translation": [10, 0, 0]})
        self.assertTrue(candidate.is_valid())
        self.assertEqual(set(candidate.reg_image_ids()), {1, 2, 3, 4})
        self.assertEqual(candidate.num_points3D(), 2)
        self.assertEqual(baseline.num_points3D(), 1)
        self.assertEqual(cohort.num_points3D(), 1)
        self.assertEqual(sorted(round(float(p.xyz[0])) for p in candidate.points3D.values()), [0, 10])

    def test_native_pycolmap_hybrid_merger_links_only_verified_new_camera_observations(self):
        try:
            import pycolmap
        except ImportError:
            self.skipTest("Optional photogrammetry dependency is absent")

        def make(image_ids):
            model = pycolmap.Reconstruction()
            camera = pycolmap.Camera(model="SIMPLE_PINHOLE", width=800, height=600,
                                    params=[900, 400, 300], camera_id=1)
            model.add_camera_with_trivial_rig(camera)
            for image_id in image_ids:
                frame = pycolmap.Image(
                    name=f"{image_id}.png", camera_id=1, image_id=image_id,
                    keypoints=np.asarray([[400 + i, 300 + i] for i in range(12)], dtype=float))
                model.add_image_with_trivial_frame(frame, pycolmap.Rigid3d())
            for index in range(12):
                track = pycolmap.Track()
                for image_id in image_ids:
                    track.add_element(image_id, index)
                model.add_point3D(np.asarray([float(index), float(index % 3), 5.]), track)
            return model

        baseline, cohort = make([1, 2]), make([2, 3])
        matches = [
            [base_pid, co_pid, 3, index]
            for index, (base_pid, co_pid) in enumerate(zip(
                sorted(baseline.points3D), sorted(cohort.points3D)))]
        merged = fuse_disjoint_models(baseline, cohort, {
            "accepted": True, "method": "verified-hybrid-3d-points",
            "scale": 1., "rotation": np.eye(3).tolist(), "translation": [0., 0., 0.],
            "inlier_point_ids": [row[:2] for row in matches],
            "verified_new_camera_observations": matches})
        self.assertTrue(merged.is_valid())
        self.assertEqual(set(merged.reg_image_ids()), {1, 2, 3})
        self.assertEqual(merged.num_points3D(), baseline.num_points3D())
        self.assertEqual(sum(bool(p.has_point3D()) for p in merged.images[3].points2D), 12)
        self.assertEqual(sum(bool(p.has_point3D()) for p in baseline.images[1].points2D), 12)
        for image_id in (1, 2):
            self.assertTrue(np.allclose(merged.images[image_id].projection_center(),
                                        baseline.images[image_id].projection_center()))
    def test_zero_common_cameras_robust_sim3_rejects_outliers(self):
        evidence = synthetic_pairs(outliers=9)
        result = estimate_point_alignment(evidence, baseline_cloud=[x["baseline_xyz"] for x in evidence],
                                          cohort_cloud=[x["cohort_xyz"] for x in evidence])
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertGreaterEqual(result["inliers"], 30)
        self.assertAlmostEqual(result["scale"], 1 / 1.8, places=5)
        self.assertLess(result["median_residual_ratio"], 0.001)

    def test_rejects_unrelated_and_spatially_degenerate_correspondences(self):
        random = np.random.default_rng(45)
        false = synthetic_pairs()
        for row in false:
            row["cohort_xyz"] = random.normal(size=3) * 50
        self.assertFalse(estimate_point_alignment(false)["accepted"])
        line = synthetic_pairs()
        for i, row in enumerate(line):
            row["baseline_xyz"] = [i, 0, 0]
            row["cohort_xyz"] = [i * 2, 0, 0]
        rejected = estimate_point_alignment(line)
        self.assertFalse(rejected["accepted"])
        self.assertTrue(any("spread" in x or "degenerate" in x for x in rejected["reasons"]))

    def test_rejects_background_only_evidence_and_local_patch(self):
        pairs = synthetic_pairs()
        for row in pairs:
            row["subject_verified"] = False
        self.assertFalse(estimate_point_alignment(pairs)["accepted"])
        pairs = synthetic_pairs()
        # A 10 cm patch cannot anchor a monument-scale global Sim(3).
        large = list(np.array([i * 100, 0, 0]) for i in range(100))
        self.assertFalse(estimate_point_alignment(pairs, baseline_cloud=large)["accepted"])

    def test_colmap_verified_feature_indices_bridge_models_without_shared_images(self):
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "database.db"
            with sqlite3.connect(database) as db:
                db.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, config INTEGER, data BLOB)")
                for left, right in ((1, 10), (2, 11), (3, 12)):
                    blob = b"".join(struct.pack("<II", i, i) for i in range(15))
                    db.execute("INSERT INTO two_view_geometries VALUES (?,?,?,?,?)",
                               (left * 2147483647 + right, 15, 2, 2, blob))
                # A misleading watermark/background pair must never count as
                # statue-surface evidence even with many verified keypoints.
                blob = b"".join(struct.pack("<II", i, i) for i in range(15))
                for left, right, geometry in ((1, 11, 4), (2, 10, 5), (2, 12, 6),
                                              (3, 10, 7), (3, 11, 8)):
                    db.execute("INSERT INTO two_view_geometries VALUES (?,?,?,?,?)",
                               (left * 2147483647 + right, 15, 2, geometry, blob))
            positions = {i: (i * 2., i % 4 * 3., i % 3 * 5.) for i in range(15)}
            baseline = Model([Image(i, f"base-{i}.png", list(range(15))) for i in (1, 2, 3)], positions)
            alternate = Model([Image(i, f"new-{i}.png", list(range(15))) for i in (10, 11, 12)], positions)
            audit = {}
            pairs = extract_verified_3d_matches(database, baseline, alternate, audit=audit)
            self.assertEqual(len(pairs), 15)
            self.assertEqual(audit["cross_registered_geometry_pairs"], 8)
            self.assertEqual(audit["nonplanar_geometry_pairs"], 3)
            self.assertEqual(audit["cross_keypoint_observations"], 45)
            self.assertEqual(audit["both_points_triangulated"], 45)
            self.assertEqual(audit["unique_3d_correspondences"], 15)
            self.assertEqual(len(pairs[0]["image_pairs"]), 3)
            self.assertEqual(set(baseline.images) & set(alternate.images), set())

    def test_cross_camera_reprojection_rejects_false_3d_fit(self):
        matches = synthetic_pairs()
        baseline_images, alternate_images = {}, {}
        for index, row in enumerate(matches):
            base_id, cohort_id = index + 1, index + 101
            row["sample"] = {"baseline_image_id": base_id, "cohort_image_id": cohort_id,
                             "baseline_keypoint": 0, "cohort_keypoint": 0}
            baseline_images[base_id] = Image(base_id, f"base-{base_id}", [index])
            alternate_images[cohort_id] = Image(cohort_id, f"new-{cohort_id}", [index])
            baseline_images[base_id].points2D[0].xy = np.asarray(row["baseline_xyz"][:2])
            alternate_images[cohort_id].points2D[0].xy = np.asarray(row["cohort_xyz"][:2])
        model_a, model_b = Model(list(baseline_images.values()), {}), Model(list(alternate_images.values()), {})
        alignment = estimate_point_alignment(matches)
        self.assertTrue(alignment["accepted"])
        self.assertTrue(validate_cross_reprojection(matches, alignment, model_a, model_b)["accepted"])
        for image_id in range(1, 7):
            model_a.images[image_id].points2D[0].xy += [1000, 500]
        self.assertFalse(validate_cross_reprojection(matches, alignment, model_a, model_b)["accepted"])
