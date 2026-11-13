"""Automated body Action + local cue plan tests without Blender or Rhubarb."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.talking_plan import (
    load_talking_plan, make_talking_plan, save_talking_plan,
)


class TalkingPlanTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "baked.blend"
        self.source.write_bytes(b"BLENDER-baked-skeleton")
        self.cues = self.root / "mouth.json"
        self.cues.write_text(json.dumps({"mouthCues": [
            {"start": 0, "end": .1, "value": "X"},
            {"start": .1, "end": .35, "value": "A"},
            {"start": .35, "end": .45, "value": "X"},
        ]}), encoding="utf-8")

    def test_saved_plan_reloads_and_locks_sources(self):
        plan = save_talking_plan(self.source, self.cues, "talk", 1, 12,
                                 self.root / "plan")
        self.assertEqual(plan["performance"]["clip"], "talk")
        self.assertEqual(plan["visemes"]["first_frame"], 1)
        self.assertEqual(load_talking_plan(self.root / "plan" / "talking-plan.json"), plan)
        with self.assertRaises(FileExistsError):
            save_talking_plan(self.source, self.cues, "talk", 1, 12,
                              self.root / "plan")
        self.cues.write_text(self.cues.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_talking_plan(self.root / "plan" / "talking-plan.json")

    def test_reject_clips_shorter_than_cue_timeline(self):
        with self.assertRaisesRegex(ValueError, "beyond"):
            make_talking_plan(self.source, self.cues, "short", 1, 5)

    def test_reject_invalid_source_and_rig_names(self):
        with self.assertRaisesRegex(ValueError, "armature"):
            make_talking_plan(self.source, self.cues, "talk", 1, 12,
                              armature="../unsafe")
        with self.assertRaisesRegex(ValueError, "fps"):
            make_talking_plan(self.source, self.cues, "talk", 1, 12, fps=0)

    def test_detects_plan_option_mutation(self):
        saved = save_talking_plan(self.source, self.cues, "talk", 1, 12,
                                 self.root / "plan")
        path = self.root / "plan" / "talking-plan.json"
        saved["visemes"]["ramp_frames"] = 4.0
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "changed"):
            load_talking_plan(path)


if __name__ == "__main__":
    unittest.main()
