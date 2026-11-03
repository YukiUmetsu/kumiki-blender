"""Explain reconstruction evidence and missing photo angles without inventing poses."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tests.test_sfm_pnp_support import _fixture

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_photo_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_photo_coverage", SCRIPT)
assert SPEC and SPEC.loader
advice = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(advice)


def _add_camera_geometry(sparse: Path):
    model = sparse / "text-models" / "0"
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 photo-0001.png\n0 0 -1\n"
        "2 1 0 0 0 -5 0 0 2 photo-0002.png\n0 0 -1\n"
    )
    (model / "points3D.txt").write_text(
        "".join(f"{100+i} {i*0.01} 0 10 255 255 255 0.1 1 {i} 2 {i}\n"
                for i in range(12))
    )


class PhotoCoverageAdviceTests(unittest.TestCase):
    def test_per_photo_support_and_unknown_angles_without_sufficient_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture(root)
            report = advice.analyze(root)
            self.assertEqual(report["registered_in_best_model"], 2)
            self.assertEqual(report["photos"], 4)
            self.assertFalse(report["camera_gate_met"])
            self.assertEqual(report["required_registered_at_current_photo_count"], 8)
            self.assertEqual(report["additional_registrations_needed_at_current_photo_count"], 6)
            images = {p["filename"]: p for p in report["images"]}
            self.assertEqual(images["photo-0001.png"]["assessment"], "keep")
            self.assertEqual(images["photo-0003.png"]["pnp_support"]["conservative_one_to_one_2d3d"], 5)
            self.assertEqual(images["photo-0003.png"]["assessment"], "low-current-value")
            self.assertEqual(images["photo-0003.png"]["original_filename"], "orig-3.jpg")
            self.assertIsNone(images["photo-0003.png"]["estimated_view"])
            self.assertEqual(report["angle_reference"]["kind"], "unavailable")
            self.assertIsNone(report["recommended_photos"][0]["target_azimuth_deg"])
            self.assertFalse(advice._angle_map(report, root / "unavailable.svg"))
            self.assertFalse((root / "unavailable.svg").exists())
            self.assertIn("not that the image is bad", advice._markdown(report))

    def test_front_anchored_relative_angles_never_assigned_to_missing_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sparse = _fixture(root)
            _add_camera_geometry(sparse)
            report = advice.analyze(root, front_photo="orig-1.jpg")
            self.assertEqual(report["angle_reference"]["kind"], "annotated-front")
            images = {p["filename"]: p for p in report["images"]}
            self.assertEqual(images["photo-0001.png"]["estimated_view"]["relative_azimuth_deg"], 0)
            self.assertEqual(images["photo-0002.png"]["estimated_view"]["label"], "right three-quarter")
            self.assertGreater(images["photo-0002.png"]["estimated_view"]["relative_azimuth_deg"], 20)
            self.assertIsNone(images["photo-0003.png"]["estimated_view"])
            self.assertTrue(report["recommended_photos"])
            self.assertTrue(any(p["target_azimuth_deg"] is not None
                                for p in report["recommended_photos"]))
            self.assertTrue(all(p["evidence"].startswith("estimated-colmap")
                                for p in report["recommended_photos"]))
            orbit = root / "camera-orbit.svg"
            self.assertTrue(advice._angle_map(report, orbit))
            svg = orbit.read_text()
            self.assertIn("Front-relative camera orbit", svg)
            self.assertIn("orig-2.jpg", svg)
            self.assertIn("Suggested overlapping shot", svg)

    def test_arbitrary_anchor_is_not_falsely_named_front(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sparse = _fixture(root)
            _add_camera_geometry(sparse)
            report = advice.analyze(root, front_photo="orig-3.jpg")
            self.assertEqual(report["angle_reference"]["kind"], "arbitrary-reference-camera")
            self.assertEqual(report["angle_reference"]["requested_front_status"],
                             "not-registered-in-best-model")
            self.assertIn("Front anchor unavailable", advice._markdown(report))
            self.assertIn("not a verified front/side",
                          report["images"][0]["estimated_view"]["label"])
            self.assertIsNone(next(p for p in report["images"]
                                   if p["filename"] == "photo-0003.png")["estimated_view"])

    def test_source_index_annotations_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sparse = _fixture(root)
            source = root / "source-index.json"
            source.write_text(json.dumps({"images": [
                {"filename": "photo-0020.png", "staged_sha256": "sha3"}
            ]}))
            tags = root / "tags.json"
            tags.write_text(json.dumps({"photo-0020.png": {
                "view": "left profile", "framing": "face close-up"
            }}))
            report = advice.analyze(root, source_index=source, annotations=tags)
            third = next(p for p in report["images"] if p["filename"] == "photo-0003.png")
            self.assertEqual(third["original_filename"], "photo-0020.png")
            self.assertEqual(third["annotations"]["view"], "left profile")
            output = root / "advice"
            with patch.object(sys, "argv", ["audit", str(root), "--output", str(output),
                                             "--source-index", str(source),
                                             "--annotations", str(tags)]):
                with redirect_stdout(StringIO()):
                    self.assertEqual(advice.main(), 0)
            saved = json.loads((output / "photo-advice.json").read_text())
            self.assertEqual(saved["images"], report["images"])
            self.assertIn("photo-0020.png", (output / "photo-advice.md").read_text())
            with patch.object(sys, "argv", ["audit", str(root), "--output", str(output)]):
                with redirect_stdout(StringIO()), self.assertRaises(SystemExit):
                    advice.main()

    def test_missing_import_and_invalid_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sparse = _fixture(root)
            with sqlite3.connect(sparse / "database.db") as db:
                db.execute("DELETE FROM images WHERE image_id=4")
            report = advice.analyze(root)
            last = next(p for p in report["images"] if p["filename"] == "photo-0004.png")
            self.assertEqual(last["assessment"], "cannot-use")
            self.assertIn("not imported", last["reasons"][0].lower())
            tags = root / "bad-tags.json"
            tags.write_text(json.dumps({"photo-0001.png": ["not a dict"]}))
            with self.assertRaisesRegex(ValueError, "Invalid annotation"):
                advice.analyze(root, annotations=tags)


if __name__ == "__main__":
    unittest.main()
