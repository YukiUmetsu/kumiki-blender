"""Source-photo to sculpt landmark constraints in registered camera frames."""

import json
import tempfile
import unittest
import contextlib
import io
from pathlib import Path

from PIL import Image

from kumiki_blender.reconstruction.landmarks import check_landmark_reprojection
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.cli import main


class LandmarkQATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        photo = self.root / "first-party.png"
        Image.new("RGB", (120, 90), "gray").save(photo)
        cameras = []
        for index in range(3):
            cameras.append({"image_id": f"cam-{index}",
                            "source_photo_path": str(photo), "source_mask_path": None,
                            "intrinsics": {"width": 120, "height": 90,
                                           "fx_px": 100, "fy_px": 100, "cx_px": 60, "cy_px": 45},
                            "translation_m": [0, 0, 5],
                            "rotation_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
        self.plan = self.root / "plan.json"
        self.plan.write_text(json.dumps({"schema_version": 1, "asset_id": "monument",
                                         "status": "alignment-candidate / artistic-review-required",
                                         "cameras": cameras}))
        self.annotations = self.root / "landmarks.json"
        records = []
        for cam in range(3):
            for landmark, xyz, pixel in (("nose", [0, 0, 0], [60, 45]),
                                         ("left-eye", [1, 0, 0], [80, 45]),
                                         ("right-eye", [0, 1, 0], [60, 25])):
                records.append({"image_id": f"cam-{cam}", "region": "face",
                                "landmark_id": landmark, "sculpt_xyz_m": xyz,
                                "image_xy_px": pixel})
        self.data = {"schema_version": 1, "asset_id": "monument", "required_regions": ["face"],
                     "measurements": records}
        self.save()

    def save(self):
        self.annotations.write_text(json.dumps(self.data))

    def test_exact_known_camera_projection_and_region_coverage(self):
        report = check_landmark_reprojection(self.plan, self.annotations, self.root / "qa")
        self.assertEqual(report["status"], "reprojection-threshold-candidate / artistic-review-required")
        self.assertEqual(report["annotated_cameras"], 3)
        self.assertEqual(report["regions"]["face"]["median_error_px"], 0)
        self.assertEqual(report["regions"]["face"]["mismatches"], 0)
        self.assertEqual(report["measurements"][1]["projected_px"], [80, 45])
        self.assertEqual(len(list((self.root / "qa").iterdir())), 1)
        self.assertEqual(json.loads((self.root / "qa/landmark-qa.json").read_text()), report)

    def test_face_mismatch_and_missing_region_are_review_required(self):
        self.data["required_regions"].append("drapery")
        self.data["measurements"][0]["image_xy_px"] = [22, 17]
        self.save()
        report = check_landmark_reprojection(self.plan, self.annotations, self.root / "qa", max_error_px=5)
        self.assertEqual(report["status"], "review-required")
        self.assertEqual(report["regions"]["drapery"]["coverage_ready"], False)
        self.assertEqual(report["measurements"][0]["status"], "mismatch")
        self.assertGreater(report["measurements"][0]["error_px"], 40)

    def test_geometry_behind_camera_never_counts_as_match(self):
        for record in self.data["measurements"]:
            if record["landmark_id"] == "nose":
                record["sculpt_xyz_m"] = [0, 0, 10]
        self.save()
        report = check_landmark_reprojection(self.plan, self.annotations, self.root / "qa")
        self.assertEqual(report["measurements"][0]["status"], "behind-camera")
        self.assertIsNone(report["measurements"][0]["error_px"])
        self.assertEqual(report["status"], "review-required")

    def test_rejects_same_landmark_with_inconsistent_sculpt_coordinates(self):
        self.data["measurements"][3]["sculpt_xyz_m"] = [7, 7, 7]
        self.save()
        with self.assertRaisesRegex(ReferenceError, "differs between camera"):
            check_landmark_reprojection(self.plan, self.annotations, self.root / "qa")
        self.assertFalse((self.root / "qa").exists())

    def test_rejects_unknown_camera_and_reused_output(self):
        self.data["measurements"][0]["image_id"] = "unknown"
        self.save()
        with self.assertRaisesRegex(ReferenceError, "unknown registered camera"):
            check_landmark_reprojection(self.plan, self.annotations, self.root / "qa")
        self.data["measurements"][0]["image_id"] = "cam-0"
        self.save()
        (self.root / "qa").mkdir()
        with self.assertRaisesRegex(ReferenceError, "already exists"):
            check_landmark_reprojection(self.plan, self.annotations, self.root / "qa")

    def test_cli_landmark_qa_reports_score_without_claiming_historical_fidelity(self):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            status = main(["landmark-qa", "--plan", str(self.plan),
                           "--annotations", str(self.annotations),
                           "--output", str(self.root / "qa"), "--max-error-px", "5"])
        report = json.loads(captured.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "reprojection-threshold-candidate / artistic-review-required")
        self.assertFalse(report["historical_fidelity_verified"])


if __name__ == "__main__":
    unittest.main()
