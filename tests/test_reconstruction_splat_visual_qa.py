"""Calibrated multi-photo splat comparison tests; no FFmpeg or actual GPU needed."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.splat_visual_qa import (
    SplatViewQASettings, _camera_pose, _select_spread, plan_splat_view_qa,
    render_splat_view_qa)
from kumiki_blender.reconstruction.splats import SplatSettings, build_splat_revision
from tests import test_reconstruction_splats as splat_fixtures


class SplatVisualQATests(unittest.TestCase):
    def setUp(self):
        self.fixture = splat_fixtures.SplatDatasetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.root = self.fixture.root
        self.splat = self.root / "splats-r01"
        build_splat_revision(
            self.fixture.source, self.splat,
            SplatSettings(iterations=300, opensplat=str(self.fixture._fake_opensplat())))
        self.renderer = self.root / "fake-splat-transform"
        self.renderer.write_text(f'''#!{sys.executable}
from pathlib import Path
import sys
from PIL import Image
args = sys.argv
assert Path(args[1]).is_file()
assert '--camera-pos' in args and '--camera-target' in args and '--camera-up' in args
assert '--camera-fov' in args and '--camera-near' in args
size = tuple(map(int, args[args.index('--resolution') + 1].split('x')))
if False:
    sys.exit(9)
Image.new('RGB', size, (12, 77, 90)).save(args[2], lossless=True)
''')
        self.renderer.chmod(0o755)

    def settings(self, **overrides):
        values = {"max_views": 3, "splat_transform": str(self.renderer)}
        values.update(overrides)
        return SplatViewQASettings(**values)

    def test_camera_pose_matches_colmap_viewer_180_degree_z_transform(self):
        position, target, up = _camera_pose((1., 0, 0, 0), (1., 2, 3))
        self.assertEqual(position, [1., 2., -3.])
        self.assertEqual(target, [1., 2., -2.])
        self.assertEqual(up, [-0., 1., -0.])

    def test_preflight_pairs_registered_photos_and_pinned_cameras(self):
        output = self.root / "qa-r01"
        report = plan_splat_view_qa(self.splat, output, self.settings())
        self.assertFalse(output.exists())
        self.assertEqual(report["registered_views"], 3)
        self.assertEqual(len({v["photo_name"] for v in report["selected_views"]}), 3)
        self.assertEqual({v["source_resolution"][0] for v in report["selected_views"]}, {128})
        self.assertTrue(all(v["camera_up"] == [0, 1, 0] for v in report["selected_views"]))
        self.assertFalse(report["distribution_allowed"])
        self.assertTrue((self.splat / "dataset/cameras/images.txt").exists())
        self.assertIn("unselected", " ".join(report["warnings"]))

    def test_renders_actual_mock_gpu_pixels_and_multi_angle_contact_sheet(self):
        output = self.root / "qa-r01"
        report = render_splat_view_qa(self.splat, output, self.settings())
        self.assertTrue(report["status"].startswith("candidate"), report)
        self.assertEqual(report["comparison_count"], 3)
        self.assertEqual(len(report["worst_views"]), 3)
        self.assertTrue((output / "multi-angle-contact-sheet.jpg").is_file())
        for view in report["completed_views"]:
            self.assertGreater(view["mean_absolute_error_rgb"], 0)
            self.assertEqual(view["valid_pixel_fraction"], 1)
            self.assertTrue((output / "views" / f"{view['index']:03d}" /
                             "comparison/registered-render.png").is_file())
        report_saved = json.loads((output / "visual-qa-report.json").read_text())
        self.assertEqual(report_saved["comparison_count"], 3)

    def test_handles_off_center_and_non_square_intrinsics_without_false_alignment(self):
        camera = self.fixture.source / "undistorted/text-model/cameras.txt"
        camera.write_text("1 PINHOLE 128 96 140 110 74 39\n")
        splat = self.root / "anisotropic-splats"
        build_splat_revision(self.fixture.source, splat,
                             SplatSettings(iterations=300, opensplat=str(self.fixture._fake_opensplat())))
        plan = plan_splat_view_qa(splat, self.root / "qa-r01", self.settings())
        affine = plan["selected_views"][0]["pixel_warp"]
        self.assertAlmostEqual(affine[0], 110/140)
        self.assertAlmostEqual(affine[2], 64-(110/140)*74)
        report = render_splat_view_qa(splat, self.root / "qa-r01", self.settings())
        self.assertTrue(report["status"].startswith("candidate"), report)
        self.assertLess(report["completed_views"][0]["valid_pixel_fraction"], 1)

    def test_foreground_masks_allow_small_statue_region_and_ignore_background(self):
        masks = self.root / "source-masks"
        masks.mkdir()
        for index in range(8):
            mask = Image.new("L", (128, 96), 0)
            mask.paste(255, (40, 28, 80, 72))
            mask.save(masks / f"photo-{index:04d}.jpg.mask.png")
        splat = self.root / "masked-splats"
        build_splat_revision(self.fixture.source, splat,
                             SplatSettings(iterations=300, opensplat=str(self.fixture._fake_opensplat())),
                             rectified_masks=masks)
        report = render_splat_view_qa(splat, self.root / "masked-qa", self.settings())
        self.assertTrue(report["status"].startswith("candidate"), report)
        self.assertTrue(all(view["mask_applied"] for view in report["completed_views"]))
        self.assertTrue(all(0.1 < view["valid_pixel_fraction"] < 0.2
                            for view in report["completed_views"]))

    def test_rejects_empty_masks_without_promoting_visual_qa(self):
        masks = self.root / "empty-masks"
        masks.mkdir()
        for index in range(8):
            mask = Image.new("L", (128, 96), 0)
            mask.putpixel((64, 48), 255)
            mask.save(masks / f"photo-{index:04d}.jpg.mask.png")
        splat = self.root / "empty-mask-splats"
        build_splat_revision(self.fixture.source, splat,
                             SplatSettings(iterations=300, opensplat=str(self.fixture._fake_opensplat())),
                             rectified_masks=masks)
        report = render_splat_view_qa(splat, self.root / "empty-mask-qa", self.settings())
        self.assertTrue(report["status"].startswith("failed"), report)
        self.assertIn("Too few valid foreground pixels", report["failure"])

    def test_rejects_tampered_reference_and_camera_before_output_creation(self):
        photo = self.splat / "dataset/images/photo-0000.jpg"
        photo.write_bytes(photo.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ReferenceError, "photo changed"):
            plan_splat_view_qa(self.splat, self.root / "qa-r01", self.settings())
        self.assertFalse((self.root / "qa-r01").exists())
        photo.write_bytes((self.fixture.source / "undistorted/images/photo-0000.jpg").read_bytes())
        camera = self.splat / "dataset/cameras/images.txt"
        camera.write_text(camera.read_text() + "# tampered\n")
        with self.assertRaisesRegex(ReferenceError, "camera text changed"):
            plan_splat_view_qa(self.splat, self.root / "qa-r01", self.settings())

    def test_gpu_failure_is_held_with_recoverable_report(self):
        output = self.root / "qa-fail"
        report = render_splat_view_qa(self.splat, output,
                                      self.settings(splat_transform=str(self.renderer)))
        self.assertTrue(report["status"].startswith("candidate"), report)
        output2 = self.root / "qa-fail-again"
        original = self.renderer.read_text()
        self.renderer.write_text(original.replace("if False:", "if True:"))
        report = render_splat_view_qa(self.splat, output2, self.settings())
        self.assertTrue(report["status"].startswith("failed"))
        self.assertEqual(report["completed_views"], [])
        self.assertTrue((output2 / "visual-qa-report.json").is_file())

    def test_cli_plan_and_compare(self):
        def call(*args):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(args)
            return code, json.loads(stdout.getvalue())
        path = self.root / "cli-qa"
        base = ("--splat", str(self.splat), "--output", str(path),
                "--max-views", "3", "--splat-transform", str(self.renderer))
        code, result = call("plan-splat-view-qa", *base)
        self.assertEqual(code, 0)
        self.assertFalse(path.exists())
        code, result = call("compare-splat-views", *base)
        self.assertEqual(code, 0)
        self.assertEqual(result["comparison_count"], 3)

    def test_spread_sampling_does_not_only_pick_first_filenames(self):
        views = [{"camera_position": [float(i), 0., 0.],
                  "camera_target": [float(i), 0., 1.]} for i in range(8)]
        selected = _select_spread(views, 3)
        self.assertEqual(selected[0]["camera_position"][0], 0)
        self.assertEqual(selected[1]["camera_position"][0], 7)
        self.assertTrue(2 <= selected[2]["camera_position"][0] <= 5)


if __name__ == "__main__":
    unittest.main()
