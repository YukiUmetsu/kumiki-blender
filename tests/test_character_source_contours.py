"""Regression for explicit Blender Z-up source OBJ and thin feature contour fidelity."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kumiki_blender.character_build.mesh import read_mesh
from kumiki_blender.character_build.region_compare import _metrics
from kumiki_blender.character_build.region_contours import (
    boundary_pixels, silhouette_boundary_metrics,
)


class SourceFrameObjTest(unittest.TestCase):
    def test_native_and_legacy_obj_paths_have_separate_coordinate_contracts(self):
        world = np.array([
            [.022, -.074594, .253],
            [.05, -.052, .253],
            [.036, -.063, .287],
        ])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for filename in ("eye.L.source.obj", "landmark-fitted-source.obj"):
                path = directory / filename
                path.write_text(
                    "\n".join(
                        "v " + " ".join(f"{v:.7f}" for v in vertex)
                        for vertex in world
                    ) + "\nf 1 2 3\n", encoding="utf-8")
                result = read_mesh(path)
                np.testing.assert_allclose(result.vertices, world, atol=1e-9)
            # Existing OBJ interchange format stores Y-up, -Z-forward.
            # Its implicit conversion must stay unchanged for rig handoffs.
            legacy = directory / "eye.L.obj"
            values = [(x, z, -y) for x, y, z in world]
            legacy.write_text(
                "\n".join("v " + " ".join(f"{v:.7f}" for v in point)
                          for point in values) + "\nf 1 2 3\n", encoding="utf-8")
            result = read_mesh(legacy)
            np.testing.assert_allclose(result.vertices, world, atol=1e-9)

    def test_near_identical_name_without_source_suffix_is_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "source-geometry.obj"
            file.write_text("v 0.01 -0.05 0.27\nv 0.03 -0.05 0.27\n"
                            "v 0.02 -0.06 0.28\nf 1 2 3\n")
            result = read_mesh(file)
            self.assertAlmostEqual(result.vertices[0, 1], -.27, places=6)
            self.assertAlmostEqual(result.vertices[0, 2], -.05, places=6)


class RegisteredContoursTest(unittest.TestCase):
    def setUp(self):
        self.reference = np.zeros((300, 400), dtype=bool)
        self.reference[50:100, 50:110] = True
        # A narrow stylized feather/eyelid lip: 100 pixels out of >3000.
        self.reference[70:72, 110:160] = True

    def test_self_comparison_and_explicit_qa_pixel_resolution(self):
        result = silhouette_boundary_metrics(self.reference, self.reference)
        self.assertEqual(result["boundary_f1_1px"], 1.)
        self.assertEqual(result["boundary_f1_2px"], 1.)
        self.assertEqual(result["boundary_mean_distance_px"], 0.)
        self.assertEqual(result["boundary_p95_distance_px"], 0.)
        self.assertGreater(result["boundary_reference_pixels"], 0)
        self.assertIn("400x300", result["boundary_metric_grid"])

    def test_thin_feature_deleted_iou_still_high_but_contours_detect(self):
        candidate = self.reference.copy()
        candidate[70:72, 110:160] = False
        score = _metrics(self.reference, candidate)
        self.assertGreater(score["iou"], .96)
        self.assertLess(score["boundary_f1_2px"], .85)
        self.assertGreater(score["boundary_mean_distance_px"], 1)
        self.assertGreater(score["boundary_p95_distance_px"], 10)

    def test_displaced_contour_and_internal_hole_are_detected(self):
        shifted = np.zeros_like(self.reference)
        shifted[50:100, 56:116] = True
        shifted[70:72, 116:166] = True
        score = silhouette_boundary_metrics(self.reference, shifted)
        self.assertLess(score["boundary_f1_2px"], .8)
        hollow = self.reference.copy()
        hollow[63:82, 65:98] = False
        score = silhouette_boundary_metrics(self.reference, hollow)
        self.assertLess(score["boundary_f1_2px"], 1.)
        self.assertGreater(int(boundary_pixels(hollow).sum()),
                           int(boundary_pixels(self.reference).sum()))

    def test_empty_contour_fails_without_forged_success_score(self):
        blank = np.zeros_like(self.reference)
        score = silhouette_boundary_metrics(self.reference, blank)
        self.assertIsNone(score["boundary_f1_2px"])
        self.assertIsNone(score["boundary_p95_distance_px"])
        self.assertEqual(score["boundary_candidate_pixels"], 0)
        with self.assertRaisesRegex(ValueError, "equal"):
            silhouette_boundary_metrics(self.reference, blank[:, :-1])


if __name__ == "__main__":
    unittest.main()
