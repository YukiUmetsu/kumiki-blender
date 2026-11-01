"""No images, external models, Pillow or Blender needed for landmark QA."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.cli import main
from kumiki_blender.sphinx_landmarks import (
    LandmarkError, LandmarkView, compare_landmark_views, load_landmark_view,
)


class SphinxLandmarkTests(unittest.TestCase):
    def setUp(self):
        self.points = {
            "crown": (120., 20.), "left-wing": (30., 120.),
            "right-wing": (210., 120.), "left-eye": (85., 70.),
            "right-eye": (155., 70.), "mouth": (120., 99.),
            "chin": (120., 130.),
        }
        self.anchors = ("crown", "left-wing", "right-wing")

    def view(self, points):
        return LandmarkView("frontal", (400, 250), self.anchors, points)

    def test_translation_and_uniform_scale_do_not_create_mismatch(self):
        ref = self.view(self.points)
        candidate = self.view({name: (x*.8+10, y*.8+14)
                               for name, (x, y) in self.points.items()})
        review = compare_landmark_views(ref, candidate)
        self.assertEqual(review["mean_distance_fraction"], 0.)
        self.assertAlmostEqual(review["alignment"]["uniform_scale"], 1.25)
        self.assertFalse(review["survey_accuracy_claimed"])

    def test_compressed_face_cannot_hide_behind_anisotropic_fit(self):
        candidate = dict(self.points)
        candidate["mouth"] = (120., 78.)
        candidate["chin"] = (120., 87.)
        report = compare_landmark_views(self.view(self.points), self.view(candidate))
        self.assertEqual(report["largest_mismatches"][0]["feature"], "chin")
        self.assertLess(report["largest_mismatches"][0]["vertical_error_fraction"], -.1)
        self.assertGreater(report["mean_distance_fraction"], .05)

    def test_mismatched_views_features_and_degenerate_anchors_fail(self):
        ref = self.view(self.points)
        for candidate in (
            LandmarkView("profile-left", ref.size, self.anchors, dict(self.points)),
            LandmarkView("frontal", ref.size, ("crown", "left-eye"), dict(self.points)),
            self.view({**self.points, "mouth": (120., 99.), "chin": (120., 130.),
                       "new-feature": (40., 45.)}),
            self.view({**self.points, "crown": (10., 10.),
                       "left-wing": (10., 10.), "right-wing": (10., 10.)}),
        ):
            with self.subTest(candidate=candidate), self.assertRaises(LandmarkError):
                compare_landmark_views(ref, candidate)

    def test_json_rejects_bad_coordinates_and_preserves_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"landmarks.json"
            data = {"view": "frontal", "image_size": [400, 250],
                    "anchors": list(self.anchors),
                    "landmarks": {name: list(point) for name, point in self.points.items()}}
            file.write_text(json.dumps(data), encoding="utf-8")
            loaded = load_landmark_view(file)
            self.assertEqual(loaded.landmarks, self.points)
            unchanged = file.read_bytes()
            self.assertEqual(unchanged, file.read_bytes())
            for bad in ([float("nan"), 99], [-1, 3], [401, 3], [2]):
                data["landmarks"]["mouth"] = bad
                file.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(LandmarkError):
                    load_landmark_view(file)

    def test_cli_reports_mismatches_from_two_annotation_files(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as directory:
            files = []
            for suffix, points in (("reference", self.points),
                                   ("candidate", {**self.points, "mouth": (120., 82.)})):
                path = Path(directory)/(suffix+".json")
                path.write_text(json.dumps({
                    "view": "frontal", "image_size": [400, 250],
                    "anchors": list(self.anchors),
                    "landmarks": {name: list(point) for name, point in points.items()},
                }), encoding="utf-8")
                files.append(path)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["compare-sphinx-landmarks", "--reference", str(files[0]),
                             "--candidate", str(files[1])])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["largest_mismatches"][0]["feature"], "mouth")


if __name__ == "__main__":
    unittest.main()
