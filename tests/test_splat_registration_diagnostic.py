"""Read-only COLMAP match-graph diagnostics for failed splat revisions."""
from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_splat_registration.py"
SPEC = importlib.util.spec_from_file_location("diagnose_splat_registration", SCRIPT)
assert SPEC and SPEC.loader
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class SplatRegistrationDiagnosticTests(unittest.TestCase):
    def test_pair_graph_and_registered_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            sparse = Path(tmp) / "sparse-reconstruction"
            model = sparse / "text-models" / "0"
            model.mkdir(parents=True)
            (model / "images.txt").write_text(
                "# registered images\n"
                "1 1 0 0 0 0 0 0 1 photo-0001.jpg\n0 0 -1\n"
                "2 1 0 0 0 0 0 0 1 photo-0002.jpg\n0 0 -1\n"
            )
            with sqlite3.connect(sparse / "database.db") as db:
                db.executescript(
                    "CREATE TABLE images (image_id INTEGER, name TEXT);"
                    "CREATE TABLE keypoints (image_id INTEGER, rows INTEGER);"
                    "CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, config INTEGER);"
                )
                db.executemany("INSERT INTO images VALUES (?,?)", [
                    (1, "photo-0001.jpg"), (2, "photo-0002.jpg"), (3, "photo-0003.jpg")])
                db.executemany("INSERT INTO keypoints VALUES (?,?)", [(1, 200), (2, 180), (3, 90)])
                db.execute("INSERT INTO two_view_geometries VALUES (?,?,?)",
                           (diagnostic.MAX_IMAGE_ID + 2, 42, 2))
                db.execute("INSERT INTO two_view_geometries VALUES (?,?,?)",
                           (2 * diagnostic.MAX_IMAGE_ID + 3, 100, 1))
            result = diagnostic.analyze(Path(tmp), 15)
            self.assertEqual(result["photos"], 3)
            self.assertEqual(result["registered"], 2)
            self.assertEqual(result["connected_component_sizes"], [2, 1])
            self.assertEqual(result["strong_pair_count"], 1)
            self.assertEqual([image["filename"] for image in result["weak_or_isolated"]],
                             ["photo-0003.jpg"])
            self.assertEqual(result["strongest_pairs"][0]["inliers"], 42)

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                diagnostic.analyze(root, 15)
            self.assertFalse((root / "database.db").exists())


if __name__ == "__main__":
    unittest.main()
