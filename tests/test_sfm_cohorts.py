"""Photo cohort extraction from a failed COLMAP revision (no synthetic geometry)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_sfm_cohorts.py"
SPEC = importlib.util.spec_from_file_location("prepare_sfm_cohorts", SCRIPT)
assert SPEC and SPEC.loader
cohorts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cohorts)


def fixture(root: Path) -> Path:
    sparse = root / "revision" / "sparse-reconstruction"
    images = sparse / "images"
    images.mkdir(parents=True)
    entries = []
    for i in range(1, 9):
        filename = f"photo-{i:04d}.png"
        source = images / filename
        Image.new("RGB", (64, 64), (i * 20, 50, 60)).save(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        entries.append({"filename": filename, "original_filename": f"source-{i}.jpg",
                        "original_sha256": "a" * 64, "sha256": digest,
                        "source_url": "local-unattributed"})
    (sparse / "staged-reference-index.json").write_text(json.dumps({"images": entries}))
    with sqlite3.connect(sparse / "database.db") as db:
        db.executescript(
            "CREATE TABLE images (image_id INTEGER, name TEXT);"
            "CREATE TABLE keypoints (image_id INTEGER, rows INTEGER);"
            "CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, config INTEGER);"
        )
        db.executemany("INSERT INTO images VALUES (?,?)",
                       [(i, f"photo-{i:04d}.png") for i in range(1, 9)])
        db.executemany("INSERT INTO keypoints VALUES (?,?)", [(i, 500) for i in range(1, 9)])
        db.executemany("INSERT INTO two_view_geometries VALUES (?,?,?)",
                       [(a * 2147483647 + b, 100, 2)
                        for a, b in [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7)]])
    for model, indices in [(0, [1, 2, 3]), (1, [5, 6, 7])]:
        model_dir = sparse / "text-models" / str(model)
        model_dir.mkdir(parents=True)
        (model_dir / "images.txt").write_text("".join(
            f"{i} 1 0 0 0 0 0 0 1 photo-{i:04d}.png\n0 0 -1\n" for i in indices
        ))
    return root / "revision"


class CohortTests(unittest.TestCase):
    def test_creates_separate_verified_components_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            target = root / "cohorts"
            report = cohorts.prepare(revision, target)
            self.assertEqual([c["photos"] for c in report["cohorts"]], [4, 3])
            self.assertEqual(report["registered_largest_model"], 3)
            self.assertEqual(report["excluded_filenames"], ["photo-0008.png"])
            self.assertEqual(report["cohorts"][0]["required_cameras_for_splat"], 8)
            self.assertFalse(report["cohorts"][0]["can_attempt_production_splat"])
            self.assertEqual([m["model"] for m in report["cohorts"][0]["registered_in_existing_models"]],
                             ["0"])
            self.assertEqual([m["model"] for m in report["cohorts"][1]["registered_in_existing_models"]],
                             ["1"])
            for c in report["cohorts"]:
                photos = target / c["id"] / "photos"
                self.assertEqual(len(list(photos.iterdir())), c["photos"])
                index = json.loads((photos.parent / "source-index.json").read_text())
                self.assertFalse(index["distribution_allowed"])
                for entry in index["images"]:
                    self.assertEqual(hashlib.sha256((photos / entry["filename"]).read_bytes()).hexdigest(),
                                     entry["staged_sha256"])
            self.assertTrue((target / "cohort-plan.json").exists())
            with self.assertRaises(FileExistsError):
                cohorts.prepare(revision, target)

    def test_rejects_modified_source_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            source = revision / "sparse-reconstruction" / "images" / "photo-0001.png"
            source.write_bytes(b"modified")
            target = root / "cohorts"
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                cohorts.prepare(revision, target)
            self.assertFalse(target.exists())

    def test_does_not_include_isolated_images_or_mix_independent_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            report = cohorts.plan(revision, min_photos=4)
            self.assertEqual(len(report["cohorts"]), 1)
            self.assertEqual(report["cohorts"][0]["filenames"],
                             [f"photo-{i:04d}.png" for i in range(1, 5)])
            self.assertEqual(len(report["excluded_filenames"]), 4)
            with self.assertRaisesRegex(ValueError, "min-photos"):
                cohorts.plan(revision, min_photos=2)


if __name__ == "__main__":
    unittest.main()
