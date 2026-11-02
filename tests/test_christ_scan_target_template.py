"""The scan-target template is an unfilled generic-donor scaffold, not a scan."""
import json
import unittest
from pathlib import Path

from kumiki_blender.scan_registration import fit_scan_landmarks,ScanRegistrationError

ROOT=Path(__file__).resolve().parents[1]


class ChristScanTemplateTests(unittest.TestCase):
    def test_unfilled_source_points_never_register(self):
        template=json.loads((ROOT/"docs/CHRIST_REDEEMER_SCAN_TARGET_ANCHORS_HOLD.json").read_text())
        self.assertIn("HOLD",template["target_geometry_status"])
        self.assertEqual(template["source_kind"],"REPLACE_WITH_VERIFIED_SOURCE_KIND")
        self.assertGreaterEqual(len(template["landmarks"]),12)
        self.assertTrue(all(p["source_xyz"] is None for p in template["landmarks"]))
        with self.assertRaises(ScanRegistrationError):
            fit_scan_landmarks(template["landmarks"])

    def test_target_frame_is_glb_y_up_and_matches_donor_anchor_positions(self):
        template=json.loads((ROOT/"docs/CHRIST_REDEEMER_SCAN_TARGET_ANCHORS_HOLD.json").read_text())
        photos=json.loads((ROOT/"docs/CHRIST_REDEEMER_PHOTO_ANCHORS_R59_HOLD.json").read_text())
        original={p["name"]:p["world_xyz"] for p in photos["views"][0]["landmarks"]}
        for p in template["landmarks"]:
            x,y,z=original[p["name"]]
            self.assertEqual(p["target_xyz"],[x,z,-y])
