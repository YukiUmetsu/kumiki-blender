"""TTY/non-TTY progress UX and monitored worker regression tests."""

import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from kumiki_blender.reconstruction.live_progress import (
    LiveProgress, _safe, _stage_detail, run_monitored_worker,
)


class MutableClock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class LiveProgressTests(unittest.TestCase):
    def test_plain_progress_shows_phases_real_completion_and_heartbeat(self):
        stream = io.StringIO()
        clock = MutableClock()
        progress = LiveProgress("sphinx", stream=stream, interactive=False, clock=clock)
        progress.emit({"kind": "start", "phase": "preflight"})
        clock.now += 16
        progress.emit({"kind": "tick", "phase": "preflight", "detail": "Reading photos"})
        clock.now += 16
        progress.emit({"kind": "tick", "phase": "preflight"})
        progress.emit({"kind": "done", "phase": "preflight"})
        progress.emit({"kind": "start", "phase": "cameras"})
        progress.emit({"kind": "blocked", "phase": "cameras", "detail": "Insufficient views"})
        output = stream.getvalue()
        self.assertIn("Checking dependencies", output)
        self.assertIn("Reading photos", output)
        self.assertGreaterEqual(output.count("Reading photos"), 2)
        self.assertIn("9% phases", output)  # one of eleven phases has completed
        self.assertIn("Insufficient views", output)
        self.assertNotIn("\x1b", output)

    def test_tty_redraw_and_skip(self):
        stream = io.StringIO()
        clock = MutableClock()
        progress = LiveProgress("sphinx", stream=stream, interactive=True, clock=clock,
                                ai_bridges=True)
        progress.emit({"kind": "start", "phase": "preflight"})
        progress.emit({"kind": "skip", "phase": "preflight"})
        progress.emit({"kind": "start", "phase": "cameras"})
        progress.emit({"kind": "tick", "phase": "cameras", "detail": "COLMAP: 50/100 pairs"})
        self.assertIn("Skipped", stream.getvalue())
        self.assertIn("1/12", stream.getvalue())
        self.assertIn("\x1b[2K", stream.getvalue())
        progress.emit({"kind": "finished"})

    def test_untrusted_worker_text_sanitized(self):
        self.assertEqual(_safe("abc\x1b[2J\n\rdef"), "abc def")

    def test_heartbeat_refreshes_a_silent_phase_without_worker_events(self):
        stream = io.StringIO()
        clock = MutableClock()
        progress = LiveProgress("sphinx", stream=stream, interactive=True, clock=clock)
        progress.emit({"kind": "start", "phase": "cameras"})
        clock.now += 5
        time.sleep(1.15)
        progress.emit({"kind": "finished"})
        self.assertIn("00:05", stream.getvalue())

    def test_real_receipt_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cameras").mkdir()
            (root / "masks").mkdir()
            (root / "dense").mkdir()
            (root / "cameras/reconstruction-report.json").write_text(json.dumps({
                "matching_progress": {"completed_pairs": 9, "total_pairs": 30},
                "undistorted_images": 4,
            }))
            (root / "masks/mask-report.json").write_text(json.dumps({
                "images": [{"filename": "a"}, {"filename": "b"}], "cache": {"hits": 1},
            }))
            (root / "dense/dense-report.json").write_text(json.dumps({
                "commands": [{"stage": "interface"}, {"stage": "densify"}],
            }))
            self.assertIn("9/30 pairs (30%)", _stage_detail(root, "cameras"))
            self.assertIn("2/4 photos (50%)", _stage_detail(root, "masks"))
            self.assertIn("2/3 steps", _stage_detail(root, "dense"))
            (root / "dense/dense-report.json").write_text("invalid-json")
            self.assertIsNone(_stage_detail(root, "dense"))

    def test_worker_preserves_stdout_and_stderr_in_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / "worker.log"
            events = []
            code = run_monitored_worker(
                [sys.executable, "-c", "import sys;print('hello');"
                 "print('warning',file=sys.stderr)"],
                log, os.environ.copy(), 10, stage="coverage", root=root, emit=events.append)
            self.assertEqual(code, 0)
            self.assertIn("hello", log.read_text())
            self.assertIn("warning", log.read_text())
            self.assertTrue(any(event["kind"] == "tick" for event in events))

    def test_worker_timeout_is_logged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / "worker.log"
            code = run_monitored_worker(
                [sys.executable, "-c", "import time;time.sleep(10)"],
                log, os.environ.copy(), 0, stage="dense", root=root)
            self.assertEqual(code, 124)
            self.assertIn("timed out", log.read_text())


if __name__ == "__main__":
    unittest.main()
