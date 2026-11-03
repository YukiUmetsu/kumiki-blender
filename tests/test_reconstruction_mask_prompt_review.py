"""A reviewer can accept individual COLMAP prompt suggestions without auto-painting."""

import json
import contextlib
import io
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.masking import _load_guides
from kumiki_blender.reconstruction.mask_prompt_review import accept_mask_point_suggestions
from kumiki_blender.reconstruction.references import ReferenceError
from tests import test_reconstruction_mask_consistency as fixtures


class MaskPromptReviewTests(unittest.TestCase):
    setUp = fixtures.MaskConsistencyTests.setUp
    _write_model = fixtures.MaskConsistencyTests._write_model
    _foreground_pipeline_candidates = fixtures.MaskConsistencyTests._foreground_pipeline_candidates

    def test_accepts_only_selected_cyan_point_into_new_reviewable_guide_revision(self):
        masks, audit = self._foreground_pipeline_candidates()
        original = (masks / "mask-report.json").read_bytes()
        output = self.root / "accepted-guide-revision"
        result = accept_mask_point_suggestions(
            self.sparse, masks, audit, output, accept=["view-02.jpg:102"])
        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["status"], "candidate / reviewed-mask-prompts")
        guides_file = output / "mask-guides.json"
        doc = json.loads(guides_file.read_text())
        guides, _ = _load_guides(guides_file,
                                [f"view-{i:02d}.jpg" for i in range(3)],
                                (0.12, 0.04, 0.88, 0.96))
        self.assertEqual(len(doc["images"]), 3)
        self.assertEqual(guides["view-02.jpg"]["foreground_points"], [[0.040404, 0.506329]])
        self.assertEqual(guides["view-00.jpg"]["foreground_points"], [])
        self.assertTrue((output / "mask-guide-decision-report.json").is_file())
        self.assertEqual((masks / "mask-report.json").read_bytes(), original)
        with Image.open(masks / "masks/view-02.jpg.mask.png") as source:
            self.assertEqual(source.getpixel((4, 40)), 0)  # never edits masks
        with self.assertRaisesRegex(ReferenceError, "already exists"):
            accept_mask_point_suggestions(
                self.sparse, masks, audit, output, accept=["view-02.jpg:102"])

    def test_rejects_unknown_ambiguous_or_duplicate_points_before_creating_output(self):
        masks, audit = self._foreground_pipeline_candidates()
        for index, decisions in enumerate((
            ["view-00.jpg:99999"],  # missing review target
            ["view-00.jpg:101"],    # two-view ambiguity, no majority
            ["view-02.jpg:102", "view-02.jpg:102"],
            ["view-02.jpg:102", "view-02.jpg:00102"],
            ["../other.jpg:102"],
            [{"not": "a decision"}],
        )):
            output = self.root / f"refused-{index}"
            with self.subTest(decisions=decisions), self.assertRaises(ReferenceError):
                accept_mask_point_suggestions(self.sparse, masks, audit, output, accept=decisions)
            self.assertFalse(output.exists())

    def test_rejects_stale_camera_or_mask_evidence_and_outside_component_point(self):
        masks, audit = self._foreground_pipeline_candidates()
        model = self.sparse / "undistorted/sparse/images.bin"
        original = model.read_bytes()
        model.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ReferenceError, "COLMAP model changed"):
            accept_mask_point_suggestions(self.sparse, masks, audit,
                                          self.root / "stale-model", accept=["view-02.jpg:102"])
        self.assertFalse((self.root / "stale-model").exists())
        model.write_bytes(original)
        report_file = masks / "mask-report.json"
        report = json.loads(report_file.read_text())
        report["images"][2]["guide"] = {
            "component_boxes": {"head": [0.2, 0.1, 0.8, 0.8]},
            "part_points": {"head": [[0.3, 0.4]]},
        }
        report_file.write_text(json.dumps(report))
        with self.assertRaisesRegex(ReferenceError, "outside every component box"):
            accept_mask_point_suggestions(self.sparse, masks, audit,
                                          self.root / "wrong-part", accept=["view-02.jpg:102"])
        self.assertFalse((self.root / "wrong-part").exists())

    def test_accepts_confirmed_red_background_leak_as_negative_prompt(self):
        for index, point in enumerate(((4, 70, 105), (4, 70, 105), (30, 70, 105))):
            self.points[index].append(point)
        self._write_model()
        masks, audit = self._foreground_pipeline_candidates()
        output = self.root / "negative-guides"
        accept_mask_point_suggestions(self.sparse, masks, audit, output,
                                      accept=["view-02.jpg:105"])
        doc = json.loads((output / "mask-guides.json").read_text())
        self.assertEqual(doc["images"]["view-02.jpg"]["background_points"],
                         [[0.30303, 0.886076]])

    def test_cli_writes_revised_guide_from_explicit_point_decision(self):
        masks, audit = self._foreground_pipeline_candidates()
        output = self.root / "cli-guide-revision"
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["review-mask-suggestions", "--reconstruction", str(self.sparse),
                         "--mask-revision", str(masks), "--audit-revision", str(audit),
                         "--output", str(output), "--accept", "view-02.jpg:102"])
        self.assertEqual(code, 0)
        self.assertTrue((output / "mask-guides.json").is_file())


if __name__ == "__main__":
    unittest.main()
