"""Retry training on a verified staged dataset, preserving the failed revision."""
from __future__ import annotations

import importlib.util
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.reconstruction.splat_input import stage_splat_dataset
from tests.test_reconstruction_splats import SplatDatasetTests

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retry_opensplat_training.py"
SPEC = importlib.util.spec_from_file_location("retry_opensplat_training", SCRIPT)
assert SPEC and SPEC.loader
retry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry)


class OpenSplatRetryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SplatDatasetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.root = self.fixture.root
        self.source = self.root / "failed-revision"
        self.source.mkdir()
        stage_splat_dataset(self.fixture.source, self.source / "dataset")

    def test_retry_uses_copied_dataset_and_validates_real_splat(self):
        trainer = self.fixture._fake_opensplat()
        original = self.source / "dataset/images/photo-0000.jpg"
        before = retry._sha256(original)
        output = self.root / "retry-success"
        report = retry.train_existing(
            self.source, output, opensplat=str(trainer),
            iterations=300, downscale=4.0)
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertEqual(report["registered_images"], 8)
        self.assertEqual(report["artifact"]["gaussians"], 2)
        self.assertEqual(report["downscale_factor"], 4.0)
        self.assertEqual(report["command"][1], str(output / "dataset"))
        self.assertEqual(retry._sha256(original), before)
        self.assertEqual(retry._sha256(output / "dataset/images/photo-0000.jpg"), before)
        self.assertTrue((output / "logs/opensplat.log").is_file())
        self.assertEqual(json.loads((output / "retry-report.json").read_text()), report)
        with self.assertRaises(FileExistsError):
            retry.train_existing(self.source, output, opensplat=str(trainer))

    def test_native_sigsegv_is_reported_without_erasing_camera_reconstruction(self):
        trainer = self.root / "opensplat-segv"
        trainer.write_text(f"#!{sys.executable}\nimport os, signal\n"
                           "os.kill(os.getpid(), signal.SIGSEGV)\n")
        trainer.chmod(0o755)
        output = self.root / "retry-segv"
        report = retry.train_existing(self.source, output, opensplat=str(trainer),
                                      iterations=100, downscale=8, cpu=True)
        self.assertEqual(report["status"], "failed / native-trainer")
        self.assertEqual(report["exit_code"], -signal.SIGSEGV)
        self.assertEqual(report["signal_name"], "SIGSEGV")
        self.assertIn("--cpu", report["command"])
        self.assertIn("lldb", report["debug_command"])
        self.assertTrue((self.source / "dataset/sparse/0/points3D.bin").is_file())
        self.assertTrue((output / "retry-report.json").is_file())

    def test_tampered_source_refused_before_creating_output(self):
        photo = self.source / "dataset/images/photo-0000.jpg"
        photo.write_bytes(photo.read_bytes() + b"tamper")
        output = self.root / "should-not-exist"
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            retry.train_existing(self.source, output,
                                 opensplat=str(self.fixture._fake_opensplat()))
        self.assertFalse(output.exists())

    def test_invalid_downscale_and_nested_output_refused(self):
        trainer = str(self.fixture._fake_opensplat())
        with self.assertRaisesRegex(ValueError, "Downscale"):
            retry.train_existing(self.source, self.root / "bad", opensplat=trainer,
                                 downscale=float("nan"))
        with self.assertRaisesRegex(ValueError, "inside the source"):
            retry.train_existing(self.source, self.source / "child", opensplat=trainer)


if __name__ == "__main__":
    unittest.main()
