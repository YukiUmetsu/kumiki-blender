"""Real-photo annotations are independent, provisional, not metric scans."""
import json
import unittest
from pathlib import Path
from kumiki_blender.orthographic_registration import fit_ortho_landmarks

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"docs/CHRIST_REDEEMER_PHOTO_ANCHORS_R59_HOLD.json"


@unittest.skipUnless(DATA.is_file(),"Real-photo annotation fixture is not checked out")
class ChristPhotoAnchorTests(unittest.TestCase):
    def test_distinct_photos_shared_anatomical_landmarks_and_hold(self):
        data=json.loads(DATA.read_text())
        self.assertIn("HOLD",data["status"])
        self.assertEqual(len(data["views"]),2)
        self.assertEqual(len({v["subject_photo_url"] for v in data["views"]}),2)
        sets=[{p["name"]:p for p in v["landmarks"]} for v in data["views"]]
        self.assertEqual(set(sets[0]),set(sets[1]))
        self.assertGreaterEqual(len(sets[0]),12)
        for name in sets[0]:
            self.assertEqual(sets[0][name]["world_xyz"],sets[1][name]["world_xyz"])

    def test_generic_donor_is_not_falsely_marked_photo_faithful(self):
        data=json.loads(DATA.read_text())
        for view in data["views"]:
            with self.subTest(view=view["view_id"]):
                report=fit_ortho_landmarks(view["landmarks"],view["image_size"])
                self.assertEqual(report["status"],"HOLD_high_reprojection_error")
                self.assertGreater(report["rms_reprojection_px"],15)


if __name__=="__main__":
    unittest.main()
