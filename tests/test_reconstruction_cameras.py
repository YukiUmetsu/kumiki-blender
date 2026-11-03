"""Known-camera geometry, sparse alignment and provenance safeguards."""

import json
import math
import tempfile
import unittest
import contextlib
import io
import shutil
from pathlib import Path
from unittest.mock import patch, Mock

from PIL import Image

from kumiki_blender.reconstruction.cameras import create_camera_plan, fit_similarity
from kumiki_blender.reconstruction.blender_reference import load_camera_plan
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.cli import main


class CameraPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.recon = self.root / "reconstruction"
        (self.recon / "text-models/0").mkdir(parents=True)
        (self.recon / "images").mkdir()
        self.names = ("front.png", "left.png", "right.png")
        self.index = []
        for name in self.names:
            path = self.recon / "images" / name
            Image.new("RGB", (120, 90), "gray").save(path)
            import hashlib
            self.index.append({"image_id": name[:-4], "filename": name,
                               "dimensions": [120, 90],
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "source_url": "https://example.org/" + name,
                               "view": "front", "regions": ["face"]})
        (self.recon / "staged-reference-index.json").write_text(json.dumps({"images": self.index}))
        (self.recon / "reconstruction-report.json").write_text(json.dumps({
            "asset_id": "statue", "status": "candidate / visual-review-required",
            "models": [{"model": "0", "registered_images": 3}],
        }))
        model = self.recon / "text-models/0"
        (model / "cameras.txt").write_text("# cameras\n1 PINHOLE 120 90 100 100 60 45\n")
        (model / "images.txt").write_text(
            "# image records followed by 2D observations\n"
            "1 1 0 0 0 0 0 0 1 front.png\n5 5 -1\n"
            "2 1 0 0 0 -1 0 0 1 left.png\n\n"
            "3 1 0 0 0 0 -1 0 1 right.png\n\n")
        self.anchors = self.root / "anchors.json"
        self.write("anchors.json", {"schema_version": 1, "asset_id": "statue", "anchors": [
            {"colmap_xyz": [0, 0, 0], "blender_xyz_m": [10, 20, 30]},
            {"colmap_xyz": [1, 0, 0], "blender_xyz_m": [10, 22, 30]},
            {"colmap_xyz": [0, 1, 0], "blender_xyz_m": [8, 20, 30]},
            {"colmap_xyz": [0, 0, 1], "blender_xyz_m": [10, 20, 32]},
        ]})

    def write(self, name, document):
        (self.root / name).write_text(json.dumps(document))

    def test_known_scale_rotation_translation_and_camera_axes(self):
        report = create_camera_plan(self.recon, self.anchors, self.root / "camera-plan")
        self.assertEqual(report["status"], "alignment-candidate / artistic-review-required")
        self.assertEqual(report["registered_cameras"], 3)
        self.assertAlmostEqual(report["anchor_rms_m"], 0, places=7)
        front, left, right = report["cameras"]
        self.assertEqual(front["image_id"], "front")
        self.assertEqual(front["intrinsics"]["fx_px"], 100)
        for point, expected in ((front["translation_m"], (10, 20, 30)),
                                (left["translation_m"], (10, 22, 30)),
                                (right["translation_m"], (8, 20, 30))):
            for x, y in zip(point, expected):
                self.assertAlmostEqual(x, y, places=7)
        # COLMAP camera forward +Z becomes Blender camera forward -Z.
        rotation = front["rotation_3x3"]
        self.assertAlmostEqual(rotation[0][0], 0, places=7)
        self.assertAlmostEqual(rotation[1][0], 1, places=7)
        self.assertAlmostEqual(rotation[2][2], -1, places=7)
        self.assertEqual(json.loads((self.root / "camera-plan/camera-plan.json").read_text()), report)

    def test_similarity_preserves_proper_rotation_for_half_turn_but_not_reflection(self):
        source = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
        target = [(2*x+3, -2*y+5, -2*z+7) for x, y, z in source]
        scale, rotation, translation, rms = fit_similarity(list(zip(source, target)))
        self.assertAlmostEqual(scale, 2, places=7)
        self.assertAlmostEqual(rms, 0, places=7)
        self.assertAlmostEqual(rotation[0][0], 1, places=7)
        self.assertAlmostEqual(rotation[1][1], -1, places=7)
        self.assertAlmostEqual(rotation[2][2], -1, places=7)
        self.assertEqual(tuple(round(v) for v in translation), (3, 5, 7))
        reflected = [(-x+3, y+5, z+7) for x, y, z in source]
        _, _, _, reflection_rms = fit_similarity(list(zip(source, reflected)))
        self.assertGreater(reflection_rms, 0.1)

    def test_output_is_new_and_does_not_copy_source_photos(self):
        target = self.root / "camera-plan"
        create_camera_plan(self.recon, self.anchors, target)
        self.assertEqual(sorted(path.name for path in target.iterdir()), ["camera-plan.json"])
        with self.assertRaisesRegex(ReferenceError, "already exists"):
            create_camera_plan(self.recon, self.anchors, target)

    def test_photo_changes_after_plan_are_detected_before_blender_reads_it(self):
        report = create_camera_plan(self.recon, self.anchors, self.root / "plan")
        photo = Path(report["cameras"][0]["source_photo_path"])
        photo.write_bytes(b"replaced after planning")
        with self.assertRaisesRegex(ReferenceError, "photograph changed"):
            load_camera_plan(self.root / "plan/camera-plan.json")

    def test_rejects_changed_images_and_unregistered_names(self):
        (self.recon / "images/front.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ReferenceError, "changed"):
            create_camera_plan(self.recon, self.anchors, self.root / "plan")
        self.assertFalse((self.root / "plan").exists())

    def test_rejects_unregistered_names_and_distorted_cameras(self):
        model = self.recon / "text-models/0"
        images = model / "images.txt"
        images.write_text(images.read_text().replace("front.png", "other.png"))
        with self.assertRaisesRegex(ReferenceError, "not in staged"):
            create_camera_plan(self.recon, self.anchors, self.root / "plan")
        images.write_text(images.read_text().replace("other.png", "front.png"))
        (model / "cameras.txt").write_text("1 SIMPLE_RADIAL 120 90 100 60 45 0.25\n")
        with self.assertRaisesRegex(ReferenceError, "undistort"):
            create_camera_plan(self.recon, self.anchors, self.root / "plan")

    def test_rejects_collinear_anchors_and_inconsistent_fit(self):
        manifest = json.loads(self.anchors.read_text())
        manifest["anchors"][2]["colmap_xyz"] = [2, 0, 0]
        manifest["anchors"][3]["colmap_xyz"] = [3, 0, 0]
        self.anchors.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ReferenceError, "non-collinear"):
            create_camera_plan(self.recon, self.anchors, self.root / "plan")
        manifest["anchors"][2]["colmap_xyz"] = [0, 1, 0]
        manifest["anchors"][3]["colmap_xyz"] = [0, 0, 1]
        manifest["anchors"][2]["blender_xyz_m"] = [80, 80, 30]
        self.anchors.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ReferenceError, "anchor RMS"):
            create_camera_plan(self.recon, self.anchors, self.root / "plan", max_anchor_rms_m=0.05)

    def test_rejects_malformed_pose_and_focal_ratio(self):
        model = self.recon / "text-models/0"
        cameras = model / "cameras.txt"
        cameras.write_text("1 PINHOLE 120 90 100 90 60 45\n")
        with self.assertRaisesRegex(ReferenceError, "focal"):
            create_camera_plan(self.recon, self.anchors, self.root / "plan")
        cameras.write_text("1 PINHOLE 120 90 100 100 60 45\n")
        images = model / "images.txt"
        images.write_text(images.read_text().replace("1 1 0 0 0 0 0 0 1", "1 0 0 0 0 0 0 0 1"))
        with self.assertRaisesRegex(ReferenceError, "quaternion"):
            create_camera_plan(self.recon, self.anchors, self.root / "plan")

    def test_cli_camera_plan_and_blender_handoff(self):
        plan_dir = self.root / "camera-plan"
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = main(["prepare-camera-plan", "--reconstruction", str(self.recon),
                           "--alignment", str(self.anchors), "--output", str(plan_dir)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["registered_cameras"], 3)
        import sys
        with patch.dict(sys.modules, {"bpy": None}), \
             patch("kumiki_blender.cli.shutil.which", return_value="/bin/fake-blender"), \
             patch("kumiki_blender.cli.subprocess.run", return_value=Mock(returncode=0)) as run:
            result = main(["render-reference-scene", "--plan", str(plan_dir / "camera-plan.json"),
                           "--model", str(self.root / "sculpt.glb"), "--output", str(self.root / "review"),
                           "--render-clay", "--max-cameras", "2"])
        self.assertEqual(result, 0)
        self.assertIn("--render-clay", run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][-3:], ["--max-cameras", "2", "--render-clay"])

    def test_undistorted_cameras_require_new_rectified_silhouette_masks(self):
        rectify = self.recon / "undistorted"
        (rectify / "images").mkdir(parents=True)
        (rectify / "text-model").mkdir()
        for name in self.names:
            shutil.copyfile(self.recon / "images" / name, rectify / "images" / name)
        for name in ("cameras.txt", "images.txt"):
            shutil.copyfile(self.recon / "text-models/0" / name, rectify / "text-model" / name)
        (self.recon / "undistorted-reference-index.json").write_text(json.dumps({"images": self.index}))
        manifest = json.loads((self.recon / "reconstruction-report.json").read_text())
        manifest.update({"undistorted_model": "undistorted/text-model", "source_sparse_model": "0"})
        (self.recon / "reconstruction-report.json").write_text(json.dumps(manifest))
        report = create_camera_plan(self.recon, self.anchors, self.root / "plan")
        self.assertTrue(report["rectified_cameras"])
        self.assertTrue(all(item["source_mask_path"] is None for item in report["cameras"]))
        self.assertIn("undistorted/images", report["cameras"][0]["source_photo_path"])
        masks = self.root / "rectified-masks"
        masks.mkdir()
        for name in self.names:
            Image.new("L", (120, 90), 255).save(masks / (name + ".png"))
        with self.assertRaisesRegex(ReferenceError, "dimensions"):
            Image.new("L", (80, 80), 255).save(masks / "front.png.png")
            create_camera_plan(self.recon, self.anchors, self.root / "plan2", undistorted_masks=masks)
        Image.new("L", (120, 90), 255).save(masks / "front.png.png")
        masked = create_camera_plan(self.recon, self.anchors, self.root / "plan3", undistorted_masks=masks)
        self.assertTrue(all(item["source_mask_path"] for item in masked["cameras"]))


class MapAnythingFocalPlanTests(unittest.TestCase):
    """Per-axis focals: accepted only for mapanything revisions, capped, and carried by the plan."""

    setUp = CameraPlanTests.setUp
    write = CameraPlanTests.write

    def _set_engine(self, engine):
        report_path = self.recon / "reconstruction-report.json"
        report = json.loads(report_path.read_text())
        if engine is None:
            report.pop("engine", None)
        else:
            report["engine"] = engine
        report_path.write_text(json.dumps(report))

    def _set_focals(self, fx, fy):
        (self.recon / "text-models/0/cameras.txt").write_text(
            f"# cameras\n1 PINHOLE 120 90 {fx} {fy} 60 45\n")

    def test_asymmetry_rejected_for_colmap_and_accepted_for_mapanything(self):
        self._set_focals(100, 104)  # 3.8% asymmetry
        self._set_engine(None)
        with self.assertRaisesRegex(ReferenceError, "fx/fy"):
            create_camera_plan(self.recon, self.anchors, self.root / "plan-colmap")
        self._set_engine("mapanything")
        report = create_camera_plan(self.recon, self.anchors, self.root / "plan-mapanything")
        self.assertEqual(report["focal_asymmetry_limit"], 0.05)
        self.assertEqual(report["cameras"][0]["intrinsics"]["fx_px"], 100)
        self.assertEqual(report["cameras"][0]["intrinsics"]["fy_px"], 104)
        load_camera_plan(self.root / "plan-mapanything/camera-plan.json")

    def test_mapanything_asymmetry_above_cap_is_rejected_at_planning(self):
        self._set_focals(100, 110)  # 9.1% asymmetry
        self._set_engine("mapanything")
        with self.assertRaisesRegex(ReferenceError, "limit"):
            create_camera_plan(self.recon, self.anchors, self.root / "plan-over-cap")

    def test_plan_limit_binds_when_blender_reads_the_plan(self):
        self._set_focals(100, 104)
        self._set_engine("mapanything")
        create_camera_plan(self.recon, self.anchors, self.root / "plan")
        plan_path = self.root / "plan/camera-plan.json"
        plan = json.loads(plan_path.read_text())
        plan["focal_asymmetry_limit"] = 0.005  # a stricter limit than the 3.8% in the file
        plan_path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ReferenceError, "asymmetry"):
            load_camera_plan(plan_path)


if __name__ == "__main__":
    unittest.main()
