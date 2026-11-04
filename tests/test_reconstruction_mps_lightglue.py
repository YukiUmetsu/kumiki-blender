"""Check feature-index integrity before optional Apple Silicon ML execution."""

import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.reconstruction.colmap import _cap_colmap_sift_features
from kumiki_blender.reconstruction.mps_lightglue_worker import (
    MAX_IMAGE_ID, _load_features, pair_id, read_pairs, select_spatial_keypoints,
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


if __name__ == "__main__":
    unittest.main()
