"""Host-side contact-sheet and regression checks, without a Blender runtime."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.visual_qa import VisualQAError, inspect_visual_revision


class VisualQATests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="kumiki-visual-qa-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.views = ["preview.png", "previews/front.png", "previews/detail.png"]

    def revision(self, name: str, colors: list[tuple[int, int, int]],
                 *, asset_id: str = "sample-lamp", views: list[str] | None = None) -> Path:
        path = self.root / name
        path.mkdir()
        names = views if views is not None else self.views
        for view, color in zip(names, colors):
            frame_path = path / view
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6), color).save(frame_path)
        (path / "manifest.json").write_text(json.dumps({
            "asset_id": asset_id, "files": {"quality_views": names},
        }), encoding="utf-8")
        return path

    @staticmethod
    def verified(root: Path) -> dict[str, str]:
        return {"asset_id": json.loads((root / "manifest.json").read_text())['asset_id']}

    def test_contact_sheet_without_baseline_is_explicitly_not_artistic_acceptance(self) -> None:
        revision = self.revision("r01", [(20, 30, 40), (100, 120, 140), (250, 250, 250)])
        output = self.root / "inspection"
        with patch("kumiki_blender.visual_qa.audit_revision", side_effect=self.verified):
            report = inspect_visual_revision(revision, output)

        self.assertEqual(report["status"], "inspection-only / artistic-review-required")
        self.assertEqual(report["mode"], "inspection")
        self.assertEqual(len(report["views"]), 3)
        self.assertAlmostEqual(report["views"][0]["exposure"]["mean_luma"], 0.114, delta=0.012)
        self.assertTrue((output / "contact-sheet.png").is_file())
        with Image.open(output / "contact-sheet.png") as sheet:
            self.assertEqual(sheet.size, (1328, 836))
        self.assertEqual(json.loads((output / "visual-qa.json").read_text()), report)

    def test_matching_baseline_passes_without_diff_artifacts(self) -> None:
        colors = [(10, 30, 50), (100, 100, 100), (220, 230, 240)]
        before = self.revision("r01", colors)
        current = self.revision("r02", colors)
        output = self.root / "comparison"
        with patch("kumiki_blender.visual_qa.audit_revision", side_effect=self.verified):
            report = inspect_visual_revision(current, output, baseline=before)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["mode"], "regression")
        for view in report["views"]:
            self.assertEqual(view["regression"]["rms_normalized"], 0)
            self.assertEqual(view["regression"]["changed_fraction"], 0)
            self.assertIsNone(view["regression"]["diff_image"])
        self.assertFalse(list(output.glob("diff-*")))

    def test_changed_image_requires_review_and_creates_diff(self) -> None:
        before = self.revision("r01", [(0, 0, 0)] * 3)
        current = self.revision("r02", [(0, 0, 0), (255, 0, 0), (0, 0, 0)])
        output = self.root / "comparison"
        with patch("kumiki_blender.visual_qa.audit_revision", side_effect=self.verified):
            report = inspect_visual_revision(current, output, baseline=before)

        self.assertEqual(report["status"], "review-required")
        regression = report["views"][1]["regression"]
        self.assertEqual(regression["status"], "review-required")
        self.assertEqual(regression["changed_fraction"], 1.0)
        self.assertGreater(regression["rms_normalized"], 0.5)
        self.assertTrue((output / regression["diff_image"]).is_file())

    def test_baseline_must_match_asset_and_ordered_views(self) -> None:
        current = self.revision("r01", [(0, 0, 0)] * 3)
        wrong_asset = self.revision("r02", [(0, 0, 0)] * 3, asset_id="another-lamp")
        wrong_views = self.revision("r03", [(0, 0, 0)] * 3, views=list(reversed(self.views)))
        output = self.root / "comparison"
        with patch("kumiki_blender.visual_qa.audit_revision", side_effect=self.verified):
            for baseline in (wrong_asset, wrong_views):
                with self.subTest(baseline=baseline.name):
                    with self.assertRaisesRegex(VisualQAError, "same asset ID and ordered camera"):
                        inspect_visual_revision(current, output, baseline=baseline)
        self.assertFalse(output.exists())

    def test_invalid_thresholds_and_idiff_without_baseline_are_rejected(self) -> None:
        for kwargs in ({"max_rms": math.nan}, {"max_changed": -0.1},
                       {"pixel_threshold": 256}, {"pixel_threshold": 12.2},
                       {"use_idiff": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(VisualQAError):
                inspect_visual_revision("unused", self.root / "unused-output", **kwargs)

    def test_output_must_not_change_revision_and_cannot_be_reused(self) -> None:
        revision = self.revision("r01", [(20, 30, 40)] * 3)
        with patch("kumiki_blender.visual_qa.audit_revision", side_effect=self.verified):
            with self.assertRaisesRegex(VisualQAError, "outside the immutable"):
                inspect_visual_revision(revision, revision / "new-inspection")
            output = self.root / "inspection"
            inspect_visual_revision(revision, output)
            with self.assertRaisesRegex(VisualQAError, "already exists"):
                inspect_visual_revision(revision, output)

    def test_optional_openimageio_idiff_report(self) -> None:
        before = self.revision("r01", [(20, 30, 40)] * 3)
        current = self.revision("r02", [(20, 30, 40)] * 3)
        with (patch("kumiki_blender.visual_qa.audit_revision", side_effect=self.verified),
              patch("kumiki_blender.visual_qa.shutil.which", return_value="/usr/bin/idiff"),
              patch("kumiki_blender.visual_qa.subprocess.run", return_value=subprocess.CompletedProcess(
                  args=[], returncode=0, stdout="PASS", stderr="")) as run):
            report = inspect_visual_revision(current, self.root / "inspection", baseline=before,
                                             use_idiff=True)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(run.call_count, 3)
        self.assertIn("-failpercent", run.call_args.args[0])
        self.assertEqual(report["views"][0]["openimageio"]["status"], "pass")

    def test_cli_exit_code_three_on_visual_regression(self) -> None:
        before = self.revision("r01", [(0, 0, 0)] * 3)
        current = self.revision("r02", [(255, 255, 255)] * 3)
        captured = StringIO()
        with (patch("kumiki_blender.visual_qa.audit_revision", side_effect=self.verified),
              redirect_stdout(captured)):
            exit_code = main(["visual-qa", "--revision", str(current), "--baseline", str(before),
                              "--output", str(self.root / "inspection")])
        self.assertEqual(exit_code, 3)
        self.assertEqual(json.loads(captured.getvalue())["status"], "review-required")


if __name__ == "__main__":
    unittest.main()
