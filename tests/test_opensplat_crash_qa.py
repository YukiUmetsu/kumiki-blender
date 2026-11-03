"""LLDB crash collection is reproducible, non-destructive and honest."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from kumiki_blender.reconstruction.splat_input import stage_splat_dataset
from tests.test_reconstruction_splats import SplatDatasetTests

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_opensplat_crash.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("diagnose_opensplat_crash", SCRIPT)
assert SPEC and SPEC.loader
crash = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(crash)


class OpenSplatCrashQATests(unittest.TestCase):
    def setUp(self):
        self.fixture = SplatDatasetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.root = self.fixture.root
        self.retry = self.root / "failed-retry"
        self.retry.mkdir()
        stage_splat_dataset(self.fixture.source, self.retry / "dataset")
        self.binary = self.root / "opensplat"
        self.binary.write_text("#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o755)
        self.original = {
            "status": "failed / native-trainer",
            "exit_code": -11,
            "mode": "cpu",
            "registered_images": 8,
            "command": [
                str(self.binary), str(self.retry / "dataset"), "-n", "100",
                "-d", "8", "-o", str(self.retry / "splat.ply"), "--cpu"
            ],
        }
        (self.retry / "retry-report.json").write_text(
            json.dumps(self.original), encoding="utf-8")
        self.debugger = self.root / "fake-lldb"
        self.debugger.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "assert '--batch' in sys.argv and '-k' in sys.argv\n"
            "assert 'thread backtrace all' in sys.argv\n"
            "print('stop reason = EXC_BAD_ACCESS')\n"
            "print('frame #0: 0x0000 libtorch_cpu.dylib native_function')\n"
            "print('frame #1: 0x0001 OpenSplat main')\n"
        )
        self.debugger.chmod(0o755)

    def test_crash_qa_copies_dataset_and_captures_stack(self):
        original_image = self.retry / "dataset/images/photo-0000.jpg"
        digest = crash.validate_dataset(self.retry)["photo_hashes"][0]["sha256"]
        output = self.root / "debug"
        report = crash.diagnose(self.retry, output, lldb=str(self.debugger))
        self.assertEqual(report["status"], "trace-captured / inspect-log")
        self.assertEqual(report["original_exit_code"], -11)
        self.assertEqual(report["original_mode"], "cpu")
        self.assertTrue(report["saw_native_crash"])
        self.assertTrue(any("frame #0" in frame for frame in report["frame_excerpt"]))
        self.assertIn("--cpu", report["lldb_command"])
        self.assertEqual(Path(report["lldb_command"][-2]).resolve(),
                         (output / "splat.ply").resolve())
        self.assertTrue((output / "dataset/sparse/0/points3D.bin").is_file())
        self.assertEqual(hashlib.sha256(original_image.read_bytes()).hexdigest(), digest)
        self.assertTrue((output / "lldb.log").is_file())
        self.assertEqual(json.loads((output / "crash-qa.json").read_text()), report)
        with self.assertRaises(FileExistsError):
            crash.diagnose(self.retry, output, lldb=str(self.debugger))

    def test_mismatched_retry_input_rejected_before_debugging(self):
        self.original["command"][1] = str(self.root / "different-dataset")
        (self.retry / "retry-report.json").write_text(json.dumps(self.original))
        output = self.root / "refuse"
        with self.assertRaisesRegex(ValueError, "different input dataset"):
            crash.diagnose(self.retry, output, lldb=str(self.debugger))
        self.assertFalse(output.exists())

    def test_debugger_launch_failure_is_not_reported_as_captured_trace(self):
        self.debugger.write_text(
            "#!/usr/bin/env python3\n"
            "print('error: process launch failed: Operation not permitted')\n"
        )
        self.debugger.chmod(0o755)
        output = self.root / "blocked-debugger"
        report = crash.diagnose(self.retry, output, lldb=str(self.debugger))
        self.assertEqual(report["status"], "debugger-launch-failed")
        self.assertFalse(report["saw_native_crash"])
        self.assertFalse(any("frame #" in line for line in report["frame_excerpt"]))
        self.assertIn("process launch failed", report["frame_excerpt"][0])
        self.assertIn("process launch failed", report["log_tail"])

    def test_debugger_without_crash_or_backtrace_is_inconclusive(self):
        self.debugger.write_text(
            "#!/usr/bin/env python3\nprint('Process exited with status = 0')\n"
        )
        self.debugger.chmod(0o755)
        report = crash.diagnose(self.retry, self.root / "no-crash",
                                lldb=str(self.debugger))
        self.assertEqual(report["status"], "trace-inconclusive")
        self.assertFalse(report["saw_native_crash"])

    def test_debugger_breakpoint_is_not_a_native_crash(self):
        self.debugger.write_text(
            "#!/usr/bin/env python3\n"
            "print('stop reason = breakpoint 1.1')\n"
            "print('frame #0: opensplat main')\n"
        )
        self.debugger.chmod(0o755)
        report = crash.diagnose(self.retry, self.root / "breakpoint",
                                lldb=str(self.debugger))
        self.assertEqual(report["status"], "trace-inconclusive")
        self.assertFalse(report["saw_native_crash"])

    def test_crash_marker_without_backtrace_is_not_a_complete_trace(self):
        self.debugger.write_text(
            "#!/usr/bin/env python3\nprint('stop reason = EXC_BAD_ACCESS')\n"
        )
        self.debugger.chmod(0o755)
        report = crash.diagnose(self.retry, self.root / "no-frames",
                                lldb=str(self.debugger))
        self.assertEqual(report["status"], "crash-seen / no-backtrace")
        self.assertTrue(report["saw_native_crash"])

    def test_long_debugger_log_preserves_first_crashing_frame(self):
        self.debugger.write_text(
            "#!/usr/bin/env python3\n"
            "print('stop reason = EXC_BAD_ACCESS')\n"
            "print('frame #0: libtorch_cpu.dylib crashing_function')\n"
            "print('verbose diagnostic' * 100000)\n"
        )
        self.debugger.chmod(0o755)
        report = crash.diagnose(self.retry, self.root / "long-log",
                                lldb=str(self.debugger))
        self.assertEqual(report["status"], "trace-captured / inspect-log")
        self.assertTrue(report["saw_native_crash"])
        self.assertTrue(any("frame #0" in frame for frame in report["frame_excerpt"]))
        self.assertLessEqual(len(report["log_tail"]), 12000)

    def test_wrapper_crash_qa_mode(self):
        bindir = self.root / "bin"
        bindir.mkdir()
        cli = bindir / "kumiki-blender"
        cli.write_text("#!/bin/sh\nexit 0\n")
        cli.chmod(0o755)
        lldb = bindir / "lldb"
        shutil.copyfile(self.debugger, lldb)
        lldb.chmod(0o755)
        outroot = self.root / "wrapper"
        env = os.environ.copy()
        env.update({
            "PATH": str(bindir) + os.pathsep + env.get("PATH", ""),
            "OUTPUT_ROOT": str(outroot),
            "PYTHONPATH": str(ROOT / "src") + os.pathsep +
                          env.get("PYTHONPATH", ""),
        })
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/mac_splat_test.sh"),
             "crash-qa", str(self.retry)],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        outputs = list(outroot.glob("opensplat-crash-qa-*"))
        self.assertEqual(len(outputs), 1)
        self.assertTrue((outputs[0] / "crash-qa.json").is_file())


if __name__ == "__main__":
    unittest.main()
