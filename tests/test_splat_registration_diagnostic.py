"""Read-only COLMAP match-graph diagnostics for failed splat revisions."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
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
            (sparse / "staged-reference-index.json").write_text(json.dumps({
                "images": [{"filename": f"photo-{index:04d}.jpg"} for index in range(1, 5)]
            }))
            (sparse / "images").mkdir()
            (sparse / "images" / "photo-0004.jpg").write_bytes(b"staged")
            (sparse / "logs").mkdir()
            (sparse / "logs" / "01-feature_extractor.log").write_text(
                "WARNING: failed to read photo-0004.jpg\n"
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
            self.assertEqual(result["staged_photos"], 4)
            self.assertEqual(len(result["missing_from_database"]), 1)
            self.assertEqual(result["missing_from_database"][0]["filename"], "photo-0004.jpg")
            self.assertTrue(result["missing_from_database"][0]["staged_file_exists"])
            self.assertEqual(result["missing_from_database"][0]["staged_bytes"], 6)
            self.assertIn("photo-0004.jpg", result["feature_extraction_warnings"][0])
            self.assertEqual(result["connected_components"][0], ["photo-0001.jpg", "photo-0002.jpg"])
            self.assertEqual(result["connected_component_sizes"], [2, 1])
            self.assertEqual(result["strong_pair_count"], 1)
            self.assertEqual([image["filename"] for image in result["weak_or_isolated"]],
                             ["photo-0003.jpg"])
            self.assertEqual(result["strongest_pairs"][0]["inliers"], 42)
            other = sparse / "text-models" / "1"
            other.mkdir()
            (other / "images.txt").write_text(
                "3 1 0 0 0 0 0 0 1 photo-0003.jpg\n0 0 -1\n"
            )
            with_second = diagnostic.analyze(Path(tmp), 15)
            self.assertEqual(with_second["registered"], 2)
            self.assertEqual([item["registered"] for item in with_second["sparse_models"]], [2, 1])
            self.assertEqual(with_second["registered_filenames"],
                             ["photo-0001.jpg", "photo-0002.jpg"])
            json_file = Path(tmp) / "registration-report.json"
            with patch.object(sys, "argv", ["diagnose", str(Path(tmp)),
                                             "--json-output", str(json_file)]):
                with redirect_stdout(StringIO()):
                    self.assertEqual(diagnostic.main(), 0)
            saved = json.loads(json_file.read_text())
            self.assertEqual(saved["sparse_models"], with_second["sparse_models"])
            with patch.object(sys, "argv", ["diagnose", str(Path(tmp)),
                                             "--json-output", str(json_file)]):
                with redirect_stdout(StringIO()), self.assertRaises(SystemExit):
                    diagnostic.main()

            try:
                from PIL import Image
            except ImportError:
                return
            for index in range(1, 5):
                Image.new("RGB", (24, 16), (80, 100, 120)).save(
                    sparse / "images" / f"photo-{index:04d}.jpg"
                )
            sheet = Path(tmp) / "registration-sheet.jpg"
            diagnostic.make_contact_sheet(Path(tmp), result, sheet)
            self.assertTrue(sheet.is_file())
            with Image.open(sheet) as image:
                self.assertEqual(image.size, (1200, 248))
            with self.assertRaises(ValueError):
                diagnostic.make_contact_sheet(Path(tmp), result, sheet)

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                diagnostic.analyze(root, 15)
            self.assertFalse((root / "database.db").exists())


if __name__ == "__main__":
    unittest.main()
