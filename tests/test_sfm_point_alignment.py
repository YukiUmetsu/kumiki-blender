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
            pairs = extract_verified_3d_matches(database, baseline, alternate)
            self.assertEqual(len(pairs), 15)
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
