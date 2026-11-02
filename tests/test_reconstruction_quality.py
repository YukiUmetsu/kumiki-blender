"""Numerical reconstruction gates must reject fabricated or missing evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.reconstruction_quality import (
    ReconstructionEvidenceError, compare_registered, review_reconstruction,
)


class ReconstructionQualityTests(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is optional; image measurements need Pillow")
        self.Image = Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _png(self, name: str, color: int, *, mask: bool = False):
        image = self.Image.new("L" if mask else "RGB", (64, 64), 0 if mask else (color, color, color))
        if mask:
            from PIL import ImageDraw
            ImageDraw.Draw(image).rectangle((14, 9, 48, 55), fill=255)
        path = self.root / name
        image.save(path)
        return path

    def _points(self, name: str):
        path = self.root / name
        path.write_text(json.dumps({"eye_left": [22, 21], "nose_tip": [31, 33],
                                    "chin": [31, 53]}), encoding="utf-8")
        return path

    def _view(self, name: str, index: int):
        ref = self._png(f"{name}-reference.png", 30+index)
        render = self._png(f"{name}-render.png", 80+index)
        ref_mask = self._png(f"{name}-ref-mask.png", 0, mask=True)
        render_mask = self._png(f"{name}-render-mask.png", 0, mask=True)
        ref_points = self._points(f"{name}-ref-points.json")
        render_points = self._points(f"{name}-render-points.json")
        return {"id": name, "reference": ref.name, "render": render.name,
                "registration": "calibrated", "camera_evidence": "Shared 3D fiducial markers",
                "reference_mask": ref_mask.name, "render_mask": render_mask.name,
                "reference_landmarks": ref_points.name,
                "render_landmarks": render_points.name,
                "thresholds": {"min_iou": .95, "max_landmark_error_ratio": .02,
                               "rationale": "Synthetic same-camera geometry verification"},
                "status": "pass"}

    def _manifest(self):
        data = {"goal": "photo-faithful", "subject": "synthetic regression fixture",
                "foundation": {"method": "sculpt", "target_verified": True,
                               "sources": ["Test fixture"], "scale_basis": "Known grid",
                               "constraints": ["Shared camera control points"]},
                "views": [self._view(name, index) for index, name in enumerate(
                    ("front", "profile", "three-quarter"))],
                "critical_features": [{"name": "face", "status": "pass"}],
                "unresolved_blockers": []}
        path = self.root / "review.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path, data

    def test_three_registered_views_pass_when_actual_masks_and_landmarks_match(self):
        path, _ = self._manifest()
        report = review_reconstruction(path)
        self.assertEqual(report["status"], "PASS", report["reasons"])
        self.assertEqual(set(report["view_metrics"]), {"front", "profile", "three-quarter"})
        self.assertTrue(all(v["iou"] == 1 for v in report["view_metrics"].values()))

    def test_string_truthiness_cannot_fake_authentic_foundation(self):
        path, data = self._manifest()
        data["foundation"]["target_verified"] = "true"
        path.write_text(json.dumps(data), encoding="utf-8")
        report = review_reconstruction(path)
        self.assertEqual(report["status"], "HOLD")
        self.assertIn("Foundation is not authenticated to the real target",
                      report["reasons"])

    def test_measured_failure_overrides_manifest_claim_of_pass(self):
        path, data = self._manifest()
        profile = data["views"][1]
        self.Image.new("L", (64, 64), 255).save(self.root / profile["render_mask"])
        report = review_reconstruction(path)
        self.assertEqual(report["status"], "HOLD")
        self.assertTrue(any("profile: silhouette IoU" in r for r in report["reasons"]))

    def test_unregistered_view_duplicate_reference_or_path_escape_holds(self):
        path, data = self._manifest()
        data["views"][0]["registration"] = "unregistered"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(review_reconstruction(path)["status"], "HOLD")

        data["views"][0]["registration"] = "calibrated"
        data["views"][1]["reference"] = data["views"][0]["reference"]
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertTrue(any("Duplicate reference" in r
                            for r in review_reconstruction(path)["reasons"]))

        data["views"][1]["reference"] = "../../missing.png"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(review_reconstruction(path)["status"], "HOLD")

    def test_baseline_nonregression_and_strict_binary_mask(self):
        view = self._view("front", 1)
        def evaluate(**extras):
            return compare_registered(
                reference_mask=self.root / view["reference_mask"],
                render_mask=self.root / view["render_mask"],
                reference_landmarks=self.root / view["reference_landmarks"],
                render_landmarks=self.root / view["render_landmarks"],
                registration="calibrated", camera_evidence="Shared 3D markers",
                min_iou=.80, max_landmark_error=.03, **extras)
        baseline_mask = self.root / "baseline.png"
        self.Image.open(self.root / view["render_mask"]).save(baseline_mask)
        baseline_points = self._points("baseline.json")
        self.assertEqual(evaluate(baseline_render_mask=baseline_mask,
                                  baseline_render_landmarks=baseline_points,
                                  require_nonregression=True)["status"], "PASS")
        from PIL import ImageDraw
        changed = self.Image.open(self.root / view["render_mask"])
        ImageDraw.Draw(changed).rectangle((45, 14, 58, 40), fill=0)
        changed.save(self.root / view["render_mask"])
        result = evaluate(baseline_render_mask=baseline_mask,
                          baseline_render_landmarks=baseline_points,
                          require_nonregression=True)
        self.assertEqual(result["status"], "HOLD")
        self.assertIn("silhouette regressed against baseline", result["reasons"])

        self.Image.new("L", (64, 64), 128).save(self.root / view["render_mask"])
        with self.assertRaises(ReconstructionEvidenceError):
            evaluate()


class LibertyReleaseHoldTests(unittest.TestCase):
    def test_current_procedural_face_is_explicitly_blocked(self):
        repo = Path(__file__).resolve().parents[1]
        report = review_reconstruction(repo / "docs" / "STATUE_OF_LIBERTY_FACE_HOLD.json")
        self.assertEqual(report["status"], "HOLD")
        self.assertEqual(report["view_metrics"], {})
        self.assertTrue(any("Foundation is not authenticated" in value
                            for value in report["reasons"]))
        self.assertTrue(any("profile: anatomical reviewer" in value
                            for value in report["reasons"]))
