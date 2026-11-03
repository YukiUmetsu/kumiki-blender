"""Retry training on a verified staged dataset, preserving the failed revision."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
        # macOS resolves /var/... to /private/var/... for the same directory.
        self.assertEqual(Path(report["command"][1]).resolve(),
                         (output / "dataset").resolve())
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
        self.assertIn(" -k ", report["debug_command"])
        self.assertIn("crash-qa", report["next_action"])
        self.assertTrue((self.source / "dataset/sparse/0/points3D.bin").is_file())
        self.assertTrue((output / "retry-report.json").is_file())

    def test_mac_shell_wrapper_runs_without_empty_array_on_bash_nounset(self):
        # Regression: macOS Bash 3.2 rejects "${empty_array[@]}" with set -u.
        # Exercise the actual wrapper, not just its syntax.
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts/mac_splat_test.sh"
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        cli = bin_dir / "kumiki-blender"
        cli.write_text("#!/bin/sh\nexit 0\n")
        cli.chmod(0o755)
        trainer = self.fixture._fake_opensplat()
        for cpu in ("0", "1"):
            output_root = self.root / ("retries-cpu-" + cpu)
            env = os.environ.copy()
            env.update({
                "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
                "OPENSPLAT": str(trainer),
                "OUTPUT_ROOT": str(output_root),
                "ITERATIONS": "300",
                "DOWNSCALE_FACTOR": "4",
                "TRAIN_CPU": cpu,
                "PYTHONPATH": str(repo_root / "src") +
                    os.pathsep + env.get("PYTHONPATH", ""),
            })
            completed = subprocess.run(
                ["bash", str(script), "train-existing", str(self.source)],
                cwd=repo_root, env=env, text=True, capture_output=True,
                timeout=30, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout +
                             completed.stderr)
            revisions = list(output_root.glob("splat-retry-*"))
            self.assertEqual(len(revisions), 1)
            report = json.loads((revisions[0] / "retry-report.json").read_text())
            self.assertEqual("--cpu" in report["command"], cpu == "1")
            self.assertEqual(report["status"], "candidate / visual-review-required")

    def test_tampered_source_refused_before_creating_output(self):
        photo = self.source / "dataset/images/photo-0000.jpg"
        photo.write_bytes(photo.read_bytes() + b"tamper")
        output = self.root / "should-not-exist"
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            retry.train_existing(self.source, output,
                                 opensplat=str(self.fixture._fake_opensplat()))
        self.assertFalse(output.exists())

    def test_foreground_train_refuses_unreviewed_masks_before_training(self):
        repo_root = Path(__file__).resolve().parents[1]
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        cli = bin_dir / "kumiki-blender"
        cli.write_text("#!/bin/sh\nexit 0\n")
        cli.chmod(0o755)
        (self.source / "splat-report.json").write_text('{"status":"failed"}')
        (self.source / "sparse-reconstruction").mkdir()
        env = os.environ.copy()
        env.update({"PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
                    "OUTPUT_ROOT": str(self.root / "foreground-output"),
                    "MASKS_REVIEWED": "0"})
        result = subprocess.run(
            ["bash", str(repo_root / "scripts/mac_splat_test.sh"),
             "foreground-train", str(self.source), "unused-masks", "unused-audit"],
            cwd=repo_root, env=env, text=True, capture_output=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MASKS_REVIEWED=1", result.stderr)
        self.assertFalse((self.root / "foreground-output").exists())

    def test_invalid_downscale_and_nested_output_refused(self):
        trainer = str(self.fixture._fake_opensplat())
        with self.assertRaisesRegex(ValueError, "Downscale"):
            retry.train_existing(self.source, self.root / "bad", opensplat=trainer,
                                 downscale=float("nan"))
        with self.assertRaisesRegex(ValueError, "inside the source"):
            retry.train_existing(self.source, self.source / "child", opensplat=trainer)


if __name__ == "__main__":
    unittest.main()
