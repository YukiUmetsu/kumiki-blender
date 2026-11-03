"""Seeded SfM recovery must be immutable and never promote weak models."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recover_sfm_tracks.py"
SPEC = importlib.util.spec_from_file_location("recover_sfm_tracks", SCRIPT)
assert SPEC and SPEC.loader
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class FakeReconstruction:
    def __init__(self, names, points=100):
        self._names = list(names)
        self._points = points
        self.images = {index: SimpleNamespace(name=name)
                       for index, name in enumerate(self._names, 1)}

    def reg_image_ids(self):
        return list(self.images)

    def num_points3D(self):
        return self._points

    def write(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model-marker.txt").write_text(",".join(self._names))


class FakePyCOLMAP:
    def __init__(self, recovered=None, *, fail=False):
        self.source = FakeReconstruction(["a.png", "b.png", "c.png", "d.png"])
        self.recovered = recovered or ["a.png", "b.png", "c.png", "d.png", "e.png"]
        self.fail = fail
        self.calls = []
        self.Reconstruction = lambda path: self.source

    def IncrementalPipelineOptions(self):
        return SimpleNamespace(min_model_size=0, fix_existing_frames=False)

    def triangulate_points(self, model, database, images, output,
                           *, clear_points, options, refine_intrinsics):
        self.calls.append(("triangulate", clear_points, options.fix_existing_frames,
                           refine_intrinsics))
        model._points += 60

    def incremental_mapping(self, database, images, output,
                            *, options, input_path):
        self.calls.append(("resume", options.fix_existing_frames, Path(input_path).is_dir()))
        if self.fail:
            raise RuntimeError("synthetic mapping failure")
        return {0: FakeReconstruction(self.recovered, 170)}


def _fixture(root: Path) -> Path:
    sparse = root / "revision" / "sparse-reconstruction"
    (sparse / "sparse" / "0").mkdir(parents=True)
    (sparse / "images").mkdir()
    with sqlite3.connect(sparse / "database.db") as db:
        db.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
        db.executemany("INSERT INTO images VALUES (?,?)",
                       [(i, f"{chr(96+i)}.png") for i in range(1, 11)])
    return sparse.parent


class SfmTrackRecoveryTests(unittest.TestCase):
    def test_preserves_source_and_retriangulates_before_seeded_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = _fixture(root)
            source_db = revision / "sparse-reconstruction" / "database.db"
            original_hash = recovery._sha256(source_db)
            engine = FakePyCOLMAP()
            output = root / "recovered"
            report = recovery.recover(revision, output, pycolmap_module=engine)
            self.assertEqual(report["seed_registered_images"], 4)
            self.assertEqual(report["seed_after_triangulation_points3D"], 160)
            self.assertEqual(report["models"][0]["registered_images"], 5)
            self.assertEqual(report["models"][0]["new_registered_filenames"], ["e.png"])
            self.assertFalse(report["camera_gate_met"])
            self.assertEqual(report["status"], "candidate / needs-geometric-review")
            self.assertEqual(engine.calls, [
                ("triangulate", False, True, False),
                ("resume", True, True),
            ])
            self.assertEqual(recovery._sha256(source_db), original_hash)
            self.assertEqual(recovery._sha256(output / "database.db"), original_hash)
            self.assertTrue((output / "seed" / "model-marker.txt").is_file())
            self.assertTrue((output / "sparse" / "0" / "model-marker.txt").is_file())
            self.assertEqual(json.loads((output / "recovery-report.json").read_text()),
                             report)
            with self.assertRaises(FileExistsError):
                recovery.recover(revision, output, pycolmap_module=engine)

    def test_failed_mapper_retains_diagnostic_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = _fixture(root)
            output = root / "failure"
            with self.assertRaisesRegex(RuntimeError, "synthetic mapping failure"):
                recovery.recover(revision, output, pycolmap_module=FakePyCOLMAP(fail=True))
            report = json.loads((output / "recovery-report.json").read_text())
            self.assertEqual(report["status"], "failed / review-required")
            self.assertIn("synthetic mapping failure", report["error"])

    def test_no_retriangulation_and_seed_pose_refinement_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = _fixture(root)
            engine = FakePyCOLMAP(recovered=["a.png", "b.png", "c.png", "d.png"])
            report = recovery.recover(
                revision, root / "output", retriangulate=False,
                fix_seed_poses=False, pycolmap_module=engine)
            self.assertEqual(report["status"], "no-registration-improvement")
            self.assertEqual(report["seed_after_triangulation_points3D"], 100)
            self.assertEqual(engine.calls, [("resume", False, True)])

    def test_unsafe_source_and_model_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = _fixture(root)
            with self.assertRaises(FileNotFoundError):
                recovery.recover(revision, root / "out", model=99,
                                 pycolmap_module=FakePyCOLMAP())
            bad = FakePyCOLMAP()
            bad.source = FakeReconstruction(["a.png", "b.png", "c.png", "unknown.png"])
            with self.assertRaisesRegex(ValueError, "do not match"):
                recovery.recover(revision, root / "out", pycolmap_module=bad)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
