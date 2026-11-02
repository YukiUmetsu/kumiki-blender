"""Regression checks for a non-procedural, fail-closed SfM experiment."""

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts" / "liberty_haer_photogrammetry_probe.py"
spec = importlib.util.spec_from_file_location("liberty_haer_sfm_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class LibertyPhotogrammetryProbeTests(unittest.TestCase):
    def test_no_registered_reconstruction_is_an_explicit_hold(self):
        report = probe.summarize_reconstructions({})
        self.assertEqual(report["candidates"], [])
        self.assertFalse(report["candidate_for_manual_registration_review"])
        self.assertIn("HOLD", report["model_release"])

    def test_sparse_registration_does_not_claim_sculpt_geometry(self):
        points = {i: SimpleNamespace(error=.8) for i in range(330)}
        rec = SimpleNamespace(
            points3D=points,
            images={i: SimpleNamespace(name=f"photo-{i}.jpg") for i in range(5)},
            num_reg_images=lambda: 5,
            num_points3D=lambda: 330,
            reg_image_ids=lambda: range(5),
        )
        report = probe.summarize_reconstructions({1: rec})
        self.assertTrue(report["candidate_for_manual_registration_review"])
        self.assertIn("HOLD", report["model_release"])
        self.assertEqual(report["candidates"][0]["registered_images"], 5)

    def test_strong_panorama_matches_are_not_mistaken_for_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "colmap.db"
            with sqlite3.connect(db_path) as db:
                db.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
                db.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, config INTEGER)")
                db.executemany("INSERT INTO images VALUES (?,?)",
                               [(1, "side-A.jpg"), (2, "side-B.jpg")])
                db.execute("INSERT INTO two_view_geometries VALUES (?,?,?)",
                           (1 * 2147483647 + 2, 920, 6))
            report = probe.summarize_image_matches(db_path)
            self.assertEqual(report["strongest_pairs"][0]["inlier_matches"], 920)
            self.assertEqual(report["strongest_pairs"][0]["geometric_configuration"],
                             "planar_or_panoramic")


if __name__ == "__main__":
    unittest.main()
