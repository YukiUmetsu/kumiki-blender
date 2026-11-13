"""Character facial cue contract tests (no Blender or audio model required)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.viseme_plan import (
    load_cues, load_viseme_plan, make_viseme_plan, save_viseme_plan,
)


class VisemePlanTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.blend = self.root / "authored-face.blend"
        self.blend.write_bytes(b"BLENDER-fixture")
        self.cues = self.root / "mouth-cues.json"
        self.data = {"metadata": {"soundFile": "local.wav", "duration": 0.9},
                     "mouthCues": [
                         {"start": 0.0, "end": 0.18, "value": "X"},
                         {"start": 0.18, "end": 0.48, "value": "A"},
                         {"start": 0.48, "end": 0.9, "value": "C"},
                     ]}
        self.save()

    def save(self):
        self.cues.write_text(json.dumps(self.data), encoding="utf-8")

    def test_rhubarb_json_contract_preserves_cue_sequence(self):
        cues = load_cues(self.cues)
        self.assertEqual([cue["value"] for cue in cues], ["X", "A", "C"])
        self.assertEqual(len(cues), 3)

    def test_plan_and_stale_source_guard(self):
        saved = save_viseme_plan(self.blend, self.cues, "Mouth", "talk_test",
                                 self.root / "plan", fps=24)
        self.assertEqual(saved["shape_keys"], {"A": "viseme_A", "C": "viseme_C"})
        self.assertEqual(load_viseme_plan(self.root / "plan" / "viseme-plan.json"), saved)
        with self.assertRaises(FileExistsError):
            save_viseme_plan(self.blend, self.cues, "Mouth", "talk_test",
                             self.root / "plan")
        self.blend.write_bytes(b"BLENDER-mutated")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_viseme_plan(self.root / "plan" / "viseme-plan.json")

    def test_cue_source_changes_invalidate_plan(self):
        save_viseme_plan(self.blend, self.cues, "Mouth", "speech", self.root / "plan")
        self.data["mouthCues"][1]["value"] = "B"
        self.save()
        with self.assertRaisesRegex(ValueError, "changed"):
            load_viseme_plan(self.root / "plan" / "viseme-plan.json")

    def test_invalid_and_overlapping_cues(self):
        self.data["mouthCues"][1]["start"] = 0.17
        self.save()
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_cues(self.cues)
        self.data["mouthCues"][1]["start"] = 0.18
        self.data["mouthCues"][1]["value"] = "Z"
        self.save()
        with self.assertRaisesRegex(ValueError, "mouth code"):
            load_cues(self.cues)
        self.data["mouthCues"][1]["value"] = "A"
        self.data["mouthCues"][1]["end"] = float("nan")
        self.save()
        with self.assertRaisesRegex(ValueError, "timestamps"):
            load_cues(self.cues)

    def test_reject_missing_mappings_unsafe_names_and_timelines(self):
        with self.assertRaisesRegex(ValueError, "mapping"):
            make_viseme_plan(self.blend, self.cues, "Mouth", "speech",
                             shape_keys={"A": "A"})
        with self.assertRaisesRegex(ValueError, "shape-key mapping"):
            make_viseme_plan(self.blend, self.cues, "Mouth", "speech",
                             shape_keys={"A": "bad name", "C": "viseme_C"})
        with self.assertRaisesRegex(ValueError, "mouth_object"):
            make_viseme_plan(self.blend, self.cues, "../../bad", "speech")
        with self.assertRaisesRegex(ValueError, "fps"):
            make_viseme_plan(self.blend, self.cues, "Mouth", "speech", fps=0)
        with self.assertRaisesRegex(ValueError, "ramp_frames"):
            make_viseme_plan(self.blend, self.cues, "Mouth", "speech", ramp_frames=0)

    def test_only_silence_is_not_a_meaningful_clip(self):
        self.data["mouthCues"] = [{"start": 0, "end": 0.9, "value": "X"}]
        self.save()
        with self.assertRaisesRegex(ValueError, "non-silence"):
            make_viseme_plan(self.blend, self.cues, "Mouth", "silent")


if __name__ == "__main__":
    unittest.main()
