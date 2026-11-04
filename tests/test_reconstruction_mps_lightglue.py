"""Check feature-index integrity before optional Apple Silicon ML execution."""

import sqlite3
import struct
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.reconstruction.colmap import _cap_colmap_sift_features
from kumiki_blender.reconstruction.mps_lightglue_worker import (
    MAX_IMAGE_ID, _load_features, main, pair_id, read_pairs, select_spatial_keypoints,
    write_matches,
)
from kumiki_blender.reconstruction.references import ReferenceError


class MPSLightGlueFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / "colmap.db"
        with sqlite3.connect(self.database) as db:
            db.executescript("""
                CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, width INTEGER, height INTEGER);
                CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
                CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
                CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
                CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
                CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            """)
            db.execute("INSERT INTO cameras VALUES (1, 4000, 2400)")
            for image_id, name in ((1, "photo-a.png"), (2, "photo-b.png")):
                count = 300
                points = b"".join(struct.pack("<ffff", (i * 53) % 4000 + .5,
                                               (i * 37) % 2400 + .5,
                                               1.0 + i / 1000, i / 100) for i in range(count))
                descriptor = b"".join(bytes([i % 255]) * 128 for i in range(count))
                db.execute("INSERT INTO images VALUES (?, ?, 1)", (image_id, name))
                db.execute("INSERT INTO keypoints VALUES (?, ?, 4, ?)", (image_id, count, points))
                db.execute("INSERT INTO descriptors VALUES (?, ?, 128, ?)", (image_id, count, descriptor))

    def _set_affine_keypoints(self, image_id=1):
        """Encode COLMAP's six-column affine shape, including rotation and shear."""
        rotation, shear = 0.5, 0.2
        with sqlite3.connect(self.database) as db:
            points = db.execute("SELECT rows, data FROM keypoints WHERE image_id=?",
                                (image_id,)).fetchone()
            affine = b"".join(
                struct.pack("<ffffff", x, y, 2 * math.cos(rotation),
                            -3 * math.sin(rotation + shear), 2 * math.sin(rotation),
                            3 * math.cos(rotation + shear))
                for x, y, _, _ in struct.iter_unpack("<ffff", points[1]))
            db.execute("UPDATE keypoints SET cols=6, data=? WHERE image_id=?",
                       (affine, image_id))

    def test_spatial_selection_is_exact_deterministic_and_spatially_diverse(self):
        points = [(i % 100, i // 100, 1.0 + i / 100) for i in range(3000)]
        chosen = select_spatial_keypoints(points, 100, 30, 256)
        self.assertEqual(len(chosen), 256)
        self.assertEqual(len(set(chosen)), 256)
        self.assertEqual(chosen, select_spatial_keypoints(points, 100, 30, 256))
        self.assertEqual(chosen, sorted(chosen))
        self.assertEqual(len(select_spatial_keypoints(points[:100], 100, 30, 256)), 100)
        self.assertGreater(len({(points[i][0] // 13, points[i][1] // 4) for i in chosen}), 24)

    def test_cpu_cap_rewrites_consistent_features_without_resizing_camera(self):
        result = _cap_colmap_sift_features(self.database, 256)
        self.assertEqual(result["original_keypoints"], 600)
        self.assertEqual(result["removed_keypoints"], 88)
        self.assertEqual(result["max_selected"], 256)
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute("SELECT width, height FROM cameras").fetchone(), (4000, 2400))
            for image_id in (1, 2):
                kp = db.execute("SELECT rows, data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
                desc = db.execute("SELECT rows, data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
                self.assertEqual((kp[0], desc[0]), (256, 256))
                for index in range(256):
                    source_index = round((struct.unpack_from("<f", kp[1], index * 16 + 8)[0] - 1.0) * 1000)
                    self.assertEqual(desc[1][index * 128], source_index % 255)

    def test_cpu_cap_preserves_six_column_affine_keypoint_rows_and_descriptors(self):
        self._set_affine_keypoints(image_id=1)
        result = _cap_colmap_sift_features(self.database, 256)
        self.assertEqual(result["max_selected"], 256)
        with sqlite3.connect(self.database) as db:
            rows, cols, blob = db.execute(
                "SELECT rows, cols, data FROM keypoints WHERE image_id=1").fetchone()
            descriptors = db.execute("SELECT rows, data FROM descriptors WHERE image_id=1").fetchone()
            self.assertEqual((rows, cols, len(blob)), (256, 6, 256 * 24))
            self.assertEqual(descriptors[0], 256)
            self.assertEqual(len(descriptors[1]), 256 * 128)

    def test_mps_accepts_affine_geometry_with_colmap_scale_and_orientation(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is an optional MPS worker dependency")
        self._set_affine_keypoints(image_id=1)
        with sqlite3.connect(self.database) as db:
            features = _load_features(db, "photo-a.png", 256)
            np.testing.assert_allclose(features["scales"], 2.5, atol=1e-5)
            np.testing.assert_allclose(features["oris"], 0.5, atol=1e-5)
            self.assertEqual(len(features["indices"]), 256)
            self.assertEqual(features["original_count"], 300)
            self.assertEqual(db.execute("SELECT cols FROM keypoints WHERE image_id=1").fetchone()[0], 6)
            # No reindexing in MPS: neural correspondence indices map back to
            # original database rows for COLMAP geometric verification.
            right = _load_features(db, "photo-b.png", 256)
            write_matches(db, features, right, np.array([[2, 3]], dtype=np.int64))
            match = db.execute("SELECT data FROM matches").fetchone()[0]
            self.assertEqual(struct.unpack("<II", match),
                             (int(features["indices"][2]), int(right["indices"][3])))

    def test_refuses_reindexing_after_any_match_has_been_saved(self):
        with sqlite3.connect(self.database) as db:
            db.execute("INSERT INTO matches VALUES (?, 1, 2, ?)", (pair_id(1, 2), struct.pack("<II", 0, 0)))
        with self.assertRaisesRegex(ReferenceError, "after any matching"):
            _cap_colmap_sift_features(self.database, 256)

    def test_mps_feature_cap_keeps_original_row_indices_for_match_storage(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is an optional MPS worker dependency")
        with sqlite3.connect(self.database) as db:
            left = _load_features(db, "photo-b.png", 256)
            right = _load_features(db, "photo-a.png", 256)
            self.assertEqual(len(left["indices"]), 256)
            self.assertEqual(left["original_count"], 300)
            self.assertEqual(left["image_size"], (4000, 2400))
            result = write_matches(db, left, right, np.array([[3, 5], [4, 6]], dtype=np.int64))
            self.assertEqual(result, 2)
            row = db.execute("SELECT rows, cols, data FROM matches WHERE pair_id=?", (pair_id(1, 2),)).fetchone()
            expected = sorted([(int(right["indices"][5]), int(left["indices"][3])),
                               (int(right["indices"][6]), int(left["indices"][4]))])
            self.assertEqual((row[0], row[1]), (2, 2))
            self.assertEqual(list(struct.iter_unpack("<II", row[2])), expected)
            self.assertEqual(db.execute("SELECT rows FROM keypoints WHERE image_id=1").fetchone()[0], 300)

    def test_pair_parser_and_pair_identifier_validate_input(self):
        pairs_file = self.root / "pairs.txt"
        pairs_file.write_text("photo-b.png photo-a.png\n", encoding="utf-8")
        self.assertEqual(read_pairs(pairs_file), [("photo-b.png", "photo-a.png")])
        self.assertEqual(pair_id(2, 1), MAX_IMAGE_ID + 2)
        pairs_file.write_text("photo-a.png photo-a.png\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid pair"):
            read_pairs(pairs_file)

    def test_worker_check_validates_environment_without_requiring_match_inputs(self):
        with patch("kumiki_blender.reconstruction.mps_lightglue_worker._check_dependencies") as check:
            self.assertEqual(main(["check", "--device", "mps"]), 0)
        check.assert_called_once_with("mps")

    def test_worker_check_reports_missing_mps_support_without_matching(self):
        with patch("kumiki_blender.reconstruction.mps_lightglue_worker._check_dependencies",
                   side_effect=RuntimeError("PyTorch MPS is unavailable")):
            self.assertEqual(main(["check", "--device", "mps"]), 1)


if __name__ == "__main__":
    unittest.main()
