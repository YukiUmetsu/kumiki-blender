"""Source-file locks, frame sampling and safe CLI for roundtrip GLB geometry."""
from __future__ import annotations

import hashlib
import io
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.character_build.__main__ import main
from kumiki_blender.character_build.roundtrip_plan import (
    file_sha256, frame_samples, make_roundtrip_plan, validate_roundtrip_plan,
)


class RoundtripPlanTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.blend = self.root / "baked.blend"
        self.glb = self.root / "animated.glb"
        self.blend.write_bytes(b"synthetic Blender asset for plan contract")
        self.glb.write_bytes(b"synthetic GLB asset for plan contract")

    def test_locks_exact_source_files(self):
        data = make_roundtrip_plan(self.blend, self.glb,
                                   first_frame=1, last_frame=12, samples=5)
        self.assertEqual(data["source_sha256"], hashlib.sha256(self.blend.read_bytes()).hexdigest())
        self.assertEqual(data["glb_sha256"], hashlib.sha256(self.glb.read_bytes()).hexdigest())
        self.assertEqual(data["first_frame"], 1)
        self.assertEqual(frame_samples(1, 12, 5), [1, 3, 6, 9, 12])
        validate_roundtrip_plan(data)
        self.assertEqual(file_sha256(self.blend), data["source_sha256"])

    def test_strict_budgets_and_unsafe_values(self):
        def attempt(**kwargs):
            return make_roundtrip_plan(self.blend, self.glb,
                                       first_frame=1, last_frame=12, **kwargs)
        for opts in [
            {"samples": 12}, {"samples": 2}, {"samples": True},
            {"fps": 0}, {"fps": 121}, {"fps": True},
            {"max_vertices_per_mesh": 9},
            {"max_vertices_per_mesh": 500_001},
            {"max_p95_error_m": -1}, {"max_p95_error_m": math.nan},
            {"max_max_error_m": math.inf},
            {"max_p95_error_m": .03, "max_max_error_m": .02},
            {"armature": "../something"},
        ]:
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                attempt(**opts)
        with self.assertRaises(ValueError):
            frame_samples(1, 3, 5)
        with self.assertRaises(ValueError):
            frame_samples(3, 3, 3)

    def test_incorrect_file_extensions_and_fields_rejected(self):
        wrong = self.root / "source.txt"
        wrong.write_bytes(b"not blend")
        with self.assertRaisesRegex(ValueError, "Blender"):
            make_roundtrip_plan(wrong, self.glb)
        with self.assertRaisesRegex(ValueError, "GLB|glb"):
            make_roundtrip_plan(self.blend, wrong)
        data = make_roundtrip_plan(self.blend, self.glb)
        data["source_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "SHA"):
            validate_roundtrip_plan(data)
        data["source_sha256"] = "0" * 64
        data["extra"] = True
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_roundtrip_plan(data)

    def test_cli_invokes_locked_blender_worker_and_propagates_failed_status(self):
        with patch("kumiki_blender.character_build.blender.run_worker") as worker:
            worker.return_value = {"status": "evaluated-glb-roundtrip-review-required"}
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = main([
                    "audit-animated-glb",
                    "--blend", str(self.blend),
                    "--glb", str(self.glb),
                    "--start", "1", "--end", "12",
                    "--samples", "5",
                    "--output", str(self.root / "check"),
                ])
            self.assertEqual(result, 0)
            self.assertEqual(worker.call_args.args[0], "glb-surface-qa")
            plan = worker.call_args.args[1]["plan"]
            self.assertEqual(plan["source_sha256"], file_sha256(self.blend))
            self.assertEqual(plan["glb_sha256"], file_sha256(self.glb))
            self.assertEqual(worker.call_args.kwargs["blend_file"], self.blend.resolve())
            worker.return_value = {"status": "structural-fail"}
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = main([
                    "audit-animated-glb",
                    "--blend", str(self.blend),
                    "--glb", str(self.glb),
                    "--start", "1", "--end", "12",
                    "--output", str(self.root / "check-fail"),
                ])
            self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
