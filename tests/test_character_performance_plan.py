"""Source-locked, bounded body+face animation plan contracts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.performance_plan import (
    load_performance_plan, make_performance_plan, save_performance_plan,
    validate_performance_plan,
)


class PerformancePlanTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.blend = self.root / "combined.blend"
        self.blend.write_bytes(b"BLENDER-anim-source")

    def test_save_and_stale_input_detection(self):
        plan = save_performance_plan(self.blend, "talk_wave", 1, 24, self.root / "plan")
        self.assertEqual(plan["clip"], "talk_wave")
        self.assertEqual(load_performance_plan(self.root / "plan" / "performance-plan.json"), plan)
        with self.assertRaises(FileExistsError):
            save_performance_plan(self.blend, "talk_wave", 1, 24, self.root / "plan")
        self.blend.write_bytes(b"BLENDER-source-was-updated")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_performance_plan(self.root / "plan" / "performance-plan.json")

    def test_reject_invalid_names_and_ranges(self):
        for name in ("", "../bad", "two words"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "clip"):
                make_performance_plan(self.blend, name, 1, 20)
        for end in (-1, 0, 10000):
            with self.subTest(end=end), self.assertRaisesRegex(ValueError, "frames"):
                make_performance_plan(self.blend, "talk", 1, end)
        with self.assertRaisesRegex(ValueError, "fps"):
            make_performance_plan(self.blend, "talk", 1, 20, fps=0)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            make_performance_plan(self.blend, "talk", 1, 20, time_alignment_tolerance_s=float("nan"))

    def test_reject_extra_fields_and_mutations(self):
        plan = make_performance_plan(self.blend, "talk", 1, 24)
        plan["script"] = "import os"
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_performance_plan(plan)
        plan.pop("script")
        plan["fps"] = True
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_performance_plan(plan)


if __name__ == "__main__":
    unittest.main()
