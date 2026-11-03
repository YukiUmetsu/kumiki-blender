"""Read-only missing-camera 2D-3D candidate support from actual COLMAP BLOBs."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_sfm_pnp_support.py"
SPEC = importlib.util.spec_from_file_location("diagnose_sfm_pnp_support", SCRIPT)
assert SPEC and SPEC.loader
pnp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pnp)


def _fixture(root: Path) -> Path:
    sparse = root / "sparse-reconstruction"
    model = sparse / "text-models" / "0"
    model.mkdir(parents=True)
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 photo-0001.png\n0 0 -1\n"
        "2 1 0 0 0 -1 0 0 2 photo-0002.png\n0 0 -1\n"
    )
    (model / "points3D.txt").write_text(
        "# point ID X Y Z R G B ERROR TRACKS\n" +
        "".join(f"{100+i} 0 0 10 255 255 255 0.1 1 {i} 2 {i}\n"
                for i in range(6))
    )
    (sparse / "staged-reference-index.json").write_text(json.dumps({
        "images": [
            {"filename": f"photo-{i:04d}.png",
             "original_filename": f"orig-{i}.jpg",
             "original_sha256": f"sha{i}"}
            for i in range(1, 5)
        ]
    }))
    coords = [(10.0 + 20 * i, 10.0 + 9 * i) for i in range(8)]
    keypoints = b"".join(struct.pack("<ff", *xy) for xy in coords)
    def blob(matches):
        return b"".join(struct.pack("<II", *match) for match in matches)
    with sqlite3.connect(sparse / "database.db") as db:
        db.executescript(
            "CREATE TABLE images (image_id INTEGER, name TEXT, camera_id INTEGER);"
            "CREATE TABLE cameras (camera_id INTEGER, width INTEGER, height INTEGER);"
            "CREATE TABLE keypoints (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);"
            "CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB, config INTEGER);"
        )
        db.executemany("INSERT INTO images VALUES (?,?,?)",
                       [(i, f"photo-{i:04d}.png", i) for i in range(1, 5)])
        db.executemany("INSERT INTO cameras VALUES (?,?,?)",
                       [(i, 160, 100) for i in range(1, 5)])
        db.executemany("INSERT INTO keypoints VALUES (?,?,?,?)",
                       [(i, 8, 2, keypoints) for i in range(1, 5)])
        a = [(i, i) for i in range(6)]
        b = a + [(1, 0), (0, 6), (2, 7)]
        pairs = [
            (1, 3, a, 2),
            (2, 3, b, 2),
            (1, 4, [(6, 0), (7, 1)], 2),
            (3, 4, [(0, 0)], 1),  # degenerate, must not count
        ]
        db.executemany("INSERT INTO two_view_geometries VALUES (?,?,?,?,?)", [
            (i * pnp.MAX_IMAGE_ID + j, len(matches), 2, blob(matches), config)
            for i, j, matches, config in pairs
        ])
    return sparse


class PnpSupportTests(unittest.TestCase):
    def test_2d3d_support_conflicts_and_duplicate_point_associations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture(root)
            report = pnp.analyze(root)
            self.assertEqual(report["status"], "candidate-2d3d-support-only")
            model = report["models"][0]
            self.assertEqual(model["registered"], 2)
            self.assertEqual(model["existing_3d_track_observations"], 12)
            missing = {item["filename"]: item for item in model["unregistered_images"]}
            image = missing["photo-0003.png"]
            self.assertEqual(image["registered_neighbors"], 2)
            self.assertEqual(image["verified_bridge_pairs"], 2)
            self.assertEqual(image["verified_bridge_inliers"], 15)
            self.assertEqual(image["inliers_touching_existing_3d_tracks"], 15)
            self.assertEqual(image["candidate_query_keypoints"], 8)
            self.assertEqual(image["ambiguous_query_keypoints"], 1)
            self.assertEqual(image["unambiguous_query_keypoints"], 7)
            self.assertEqual(image["distinct_existing_3d_points"], 6)
            self.assertEqual(image["conservative_one_to_one_2d3d"], 5)
            self.assertGreater(image["one_to_one_4x4_image_coverage"], 0)
            self.assertEqual(missing["photo-0004.png"]["verified_bridge_inliers"], 2)
            self.assertEqual(missing["photo-0004.png"]["conservative_one_to_one_2d3d"], 0)
            self.assertIn("NOT", report["interpretation"])

    def test_cohort_source_id_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture(root)
            source = root / "source-index.json"
            source.write_text(json.dumps({"images": [
                {"filename": "photo-0020.png", "staged_sha256": "sha3"}
            ]}))
            report = pnp.analyze(root, source)
            self.assertEqual(report["models"][0]["unregistered_images"][0]["original_filename"],
                             "photo-0020.png")

    def test_invalid_verified_match_blob_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sparse = _fixture(root)
            with sqlite3.connect(sparse / "database.db") as db:
                db.execute("UPDATE two_view_geometries SET data=? WHERE pair_id=?",
                           (b"bad", 1 * pnp.MAX_IMAGE_ID + 3))
            with self.assertRaisesRegex(ValueError, "verified-inlier BLOB"):
                pnp.analyze(root)

    def test_model_database_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sparse = _fixture(root)
            path = sparse / "text-models" / "0" / "images.txt"
            path.write_text(path.read_text().replace("photo-0001.png", "incorrect.png"))
            with self.assertRaisesRegex(ValueError, "model/database image mismatch"):
                pnp.analyze(root)


if __name__ == "__main__":
    unittest.main()
