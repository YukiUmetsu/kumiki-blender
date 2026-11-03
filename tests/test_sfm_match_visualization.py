"""COLMAP verified-correspondence QA: image-space evidence, not fake geometry."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "visualize_sfm_matches.py"
SPEC = importlib.util.spec_from_file_location("visualize_sfm_matches", SCRIPT)
assert SPEC and SPEC.loader
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


def _fixture(root: Path) -> tuple[Path, Path]:
    revision = root / "revision"
    sparse = revision / "sparse-reconstruction"
    photos = sparse / "images"
    photos.mkdir(parents=True)
    originals = []
    for index in range(1, 4):
        filename = f"photo-{index:04d}.png"
        photo = photos / filename
        Image.new("RGB", (160, 100), (index * 40, 90, 100)).save(photo)
        originals.append({"filename": filename,
                          "original_filename": f"source-{index}.jpg",
                          "original_sha256": hashlib.sha256(photo.read_bytes()).hexdigest()})
    (sparse / "staged-reference-index.json").write_text(json.dumps({"images": originals}))
    model = sparse / "text-models" / "0"
    model.mkdir(parents=True)
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 photo-0001.png\n0 0 -1\n"
        "2 1 0 0 0 0 0 0 1 photo-0002.png\n0 0 -1\n"
    )
    coords = [(12.0, 10.0), (52.0, 25.0), (98.0, 60.0), (145.0, 85.0)]
    points = b"".join(struct.pack("<ff", *xy) for xy in coords)
    matches = b"".join(struct.pack("<II", i, i) for i in range(4))
    with sqlite3.connect(sparse / "database.db") as db:
        db.executescript(
            "CREATE TABLE images (image_id INTEGER, name TEXT);"
            "CREATE TABLE keypoints (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);"
            "CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB, config INTEGER);"
        )
        db.executemany("INSERT INTO images VALUES (?,?)",
                       [(i, f"photo-{i:04d}.png") for i in range(1, 4)])
        db.executemany("INSERT INTO keypoints VALUES (?,?,?,?)",
                       [(i, 4, 2, points) for i in range(1, 4)])
        db.executemany("INSERT INTO two_view_geometries VALUES (?,?,?,?,?)",
                       [(1 * qa.MAX_IMAGE_ID + 2, 4, 2, matches, 2),
                        (1 * qa.MAX_IMAGE_ID + 3, 4, 2, matches, 2),
                        (2 * qa.MAX_IMAGE_ID + 3, 4, 2, matches, 2)])
    index = root / "cohort-source-index.json"
    index.write_text(json.dumps({"images": [
        {"filename": "original-0020.png", "staged_sha256": originals[2]["original_sha256"]}
    ]}))
    return revision, index


class SfmMatchQaTests(unittest.TestCase):
    def test_visualizes_actual_correspondences_with_registration_categories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            revision, index = _fixture(root)
            output = root / "match-qa"
            report = qa.audit(revision, output, min_inliers=2, max_pairs=3,
                              max_draw=4, source_index=index)
            self.assertEqual(report["registered_in_largest_model"], 2)
            self.assertEqual(report["verified_pairs"], 3)
            self.assertEqual(report["categories"]["bridge"], 2)
            self.assertEqual(report["categories"]["registered"], 1)
            self.assertTrue(all(pair["status"] == "rendered" for pair in report["pairs"]))
            self.assertTrue(any(pair.get("original_b") == "original-0020.png"
                                for pair in report["pairs"]))
            self.assertTrue(all(pair["coverage_a"] > 0 for pair in report["pairs"]))
            self.assertTrue((output / "match-contact-sheet.jpg").is_file())
            self.assertTrue((output / "match-qa.json").is_file())
            self.assertIn("NOT establish parallax", (output / "match-qa.json").read_text())
            with self.assertRaises(FileExistsError):
                qa.audit(revision, output, min_inliers=2)

    def test_rejects_malformed_keypoint_or_match_blobs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            revision, _ = _fixture(root)
            db_path = revision / "sparse-reconstruction" / "database.db"
            with sqlite3.connect(db_path) as db:
                db.execute("UPDATE keypoints SET data=? WHERE image_id=1", (b"short",))
            with sqlite3.connect(db_path) as db:
                with self.assertRaisesRegex(ValueError, "keypoint BLOB"):
                    qa._read_keypoints(db, 1)
                with self.assertRaisesRegex(ValueError, "inlier BLOB"):
                    qa._decode_inliers(5, 2, b"short")

    def test_grid_coverage_and_input_validation(self):
        self.assertEqual(qa._grid_coverage([(10, 10), (90, 90)], (100, 100)), 0.125)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            revision, _ = _fixture(root)
            with self.assertRaisesRegex(ValueError, "max-pairs"):
                qa.audit(revision, root / "output", max_pairs=0)


if __name__ == "__main__":
    unittest.main()
