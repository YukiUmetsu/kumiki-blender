"""Animation bake plan tests without Blender, cloud models or private rigs."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.animation_bake_plan import (
    load_bake_plan, make_bake_plan, save_bake_plan, validate_bake_plan,
)


class AnimationBakePlanTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.blend = self.root / "animated.blend"
        self.blend.write_bytes(b"BLENDER-test")
 
    def test_source_locked_plan_and_immutable_revision(self):
        saved = save_bake_plan(self.blend, "wing_fold", 1, 24, self.root / "plan")
        self.assertEqual(saved["fps"], 24)
        self.assertEqual(saved["end_frame"], 24)
        self.assertEqual(load_bake_plan(self.root / "plan" / "bake-plan.json"), saved)
        with self.assertRaises(FileExistsError):
            save_bake_plan(self.blend, "wing_fold", 1, 24, self.root / "plan")
        self.blend.write_bytes(b"BLENDER-changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_bake_plan(self.root / "plan" / "bake-plan.json")

    def test_reject_unsafe_and_unbounded_clips(self):
        for name in ("../unsafe", "evil name", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                make_bake_plan(self.blend, name, 1, 10)
        for frame in (0, -1, 14000):
            with self.subTest(end=frame), self.assertRaises(ValueError):
                make_bake_plan(self.blend, "clip", 1, frame)
        with self.assertRaises(ValueError):
            make_bake_plan(self.blend, "clip", 1, 10000)
        with self.assertRaisesRegex(ValueError, "fps"):
            make_bake_plan(self.blend, "clip", 1, 10, fps=0)
        with self.assertRaisesRegex(ValueError, "rotation"):
            make_bake_plan(self.blend, "clip", 1, 10,
                           max_rotation_error_deg=float("nan"))

    def test_reject_mutated_shape_and_types(self):
        plan = make_bake_plan(self.blend, "walk", 1, 12)
        plan["untrusted-script"] = "print('bad')"
        with self.assertRaises(ValueError):
            validate_bake_plan(plan)
        plan.pop("untrusted-script")
        plan["fps"] = True
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_bake_plan(plan)
        plan["fps"] = 24
        plan["source_sha256"] = "mismatched"
        with self.assertRaisesRegex(ValueError, "SHA"):
            validate_bake_plan(plan)


if __name__ == "__main__":
    unittest.main()
