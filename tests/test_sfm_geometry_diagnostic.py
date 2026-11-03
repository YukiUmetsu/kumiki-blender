"""Test geometric SfM diagnostics against known COLMAP world-to-camera poses."""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_sfm_geometry.py"
SPEC = importlib.util.spec_from_file_location("analyze_sfm_geometry", SCRIPT)
assert SPEC and SPEC.loader
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


def _model(folder: Path, *, translation: float = -1.0) -> None:
    folder.mkdir(parents=True)
    (folder / "cameras.txt").write_text(
        "1 SIMPLE_RADIAL 1000 800 700 500 400 0\n"
        "2 PINHOLE 1000 800 650 660 500 400\n"
    )
    (folder / "images.txt").write_text(
        "# image ID quaternion translation camera ID filename\n"
        "1 1 0 0 0 0 0 0 1 a.png\n"
        "1 2 3\n"
        f"2 1 0 0 0 {translation} 0 0 2 b.png\n"
        "4 5 6\n"
    )
    (folder / "points3D.txt").write_text(
        "# point ID X Y Z R G B ERROR TRACKS\n"
        "1 0 0 10 255 255 255 0.5 1 0 2 0\n"
        "2 0 0 20 255 255 255 0.5 1 1 2 1\n"
    )


class SfmGeometryTests(unittest.TestCase):
    def test_known_baseline_yields_expected_parallax(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "text-models" / "0"
            _model(root)
            result = geometry.analyze(Path(tmp))
            self.assertEqual(result["status"], "estimated-camera-geometry")
            self.assertEqual(len(result["models"]), 1)
            model = result["models"][0]
            self.assertEqual(model["registered_images"], 2)
            self.assertEqual(model["intrinsics"]["camera_count"], 2)
            self.assertEqual(model["intrinsics"]["camera_models"],
                             {"PINHOLE": 1, "SIMPLE_RADIAL": 1})
            self.assertEqual(model["intrinsics"]["cameras"][0]["focal_to_max_dimension"], 0.7)
            self.assertEqual(model["triangulation"]["triangulated_points_with_two_cameras"], 2)
            self.assertAlmostEqual(model["triangulation"]["median_max_triangulation_angle_deg"],
                                   (math.degrees(math.atan(0.1)) +
                                    math.degrees(math.atan(0.05))) / 2, delta=0.002)
            self.assertEqual(model["triangulation"]["percent_points_below_5_deg"], 50.0)
            self.assertEqual(model["triangulation"]["percent_points_below_2_deg"], 0.0)

    def test_same_camera_center_has_zero_parallax_despite_feature_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "text-models" / "0"
            _model(root, translation=0.0)
            result = geometry.analyze(Path(tmp))
            tri = result["models"][0]["triangulation"]
            self.assertEqual(tri["median_max_triangulation_angle_deg"], 0.0)
            self.assertEqual(tri["percent_points_below_1_deg"], 100.0)

    def test_rotation_quaternion_is_normalized_and_transpose_used(self):
        # 90 degrees about Z; world-to-camera t=(1,0,0) => C=(0,1,0).
        center = geometry._camera_center((math.sqrt(2), 0, 0, math.sqrt(2)), (1, 0, 0))
        self.assertAlmostEqual(center[0], 0, delta=1e-10)
        self.assertAlmostEqual(center[1], 1, delta=1e-10)
        self.assertAlmostEqual(center[2], 0, delta=1e-10)

    def test_missing_sparse_data_and_bad_quaternion(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(geometry.analyze(Path(tmp))["status"], "no-text-models")
            root = Path(tmp) / "text-models" / "0"
            _model(root)
            (root / "points3D.txt").unlink()
            self.assertEqual(geometry.analyze(Path(tmp))["models"][0]["triangulation"]["status"],
                             "missing-points3D")
        with self.assertRaisesRegex(ValueError, "Zero camera quaternion"):
            geometry._camera_center((0, 0, 0, 0), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
