"""Head-base fitting must preserve real anatomy and disclose landmark error."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.reconstruction.head_base import (
    HeadFitError, fit_head_landmarks, head_deformation_field,
    load_head_fit_plan, transform_point,
)
from tests.test_asset_import import fixture_glb, provenance_json


SOURCE = {
    "nose": (0.0, -0.12, 0.05),
    "left-eye": (-0.06, -0.10, 0.04),
    "right-eye": (0.06, -0.10, 0.04),
    "chin": (0.0, -0.11, -0.12),
    "right-ear": (0.10, 0.015, 0.0),
    "forehead": (0.0, -0.085, 0.13),
}


def target(point):
    # Independent hand-derived reference transform: 90-degree Z rotation,
    # uniform scale 2, then translation (3, -1, 5).
    return (3 - 2 * point[1], -1 + 2 * point[0], 5 + 2 * point[2])


class HeadBaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture_glb(self.root / "human-head.glb", materials=["Skin"])
        (self.root / "provenance.json").write_text(json.dumps(
            provenance_json(asset_id="reviewed-human-head", license="CC0-1.0")))
        self.data = {
            "schema_version": 1,
            "asset_id": "redeemer-head-study",
            "source_glb": "human-head.glb",
            "source_provenance": "provenance.json",
            "mesh_name": "Head",
            "fit_mode": "rigid",
            "landmarks": [
                {"id": name, "region": "face", "source_xyz_m": list(source),
                 "target_xyz_m": list(target(source)), "radius_m": 0.12}
                for name, source in SOURCE.items()
            ],
        }
        self.path = self.root / "head-fit.json"

    def load(self):
        self.path.write_text(json.dumps(self.data))
        return load_head_fit_plan(self.path)

    def test_recovers_rotated_scaled_translated_anatomical_head(self):
        fit = fit_head_landmarks(self.load())
        self.assertAlmostEqual(fit.scale, 2.0, places=5)
        self.assertAlmostEqual(fit.rms_error_m, 0.0, places=5)
        result = transform_point(SOURCE["nose"], fit)
        for got, expected in zip(result, target(SOURCE["nose"])):
            self.assertAlmostEqual(got, expected, places=5)
        self.assertEqual(len(fit.landmark_errors), 6)
        self.assertFalse(fit.to_report()["historical_fidelity_verified"])

    def test_incorrect_target_landmark_remains_visible_in_report(self):
        self.data["landmarks"][-1]["target_xyz_m"][0] += 0.04
        fit = fit_head_landmarks(self.load())
        self.assertGreater(fit.rms_error_m, 0.005)
        self.assertGreater(fit.max_error_m, fit.rms_error_m)
        self.assertEqual(fit.to_report()["status"], "head-fit-candidate / artistic-review-required")

    def test_cannot_escape_manifest_root_or_use_unlicensed_head(self):
        self.data["source_glb"] = "../human-head.glb"
        with self.assertRaises(HeadFitError):
            self.load()
        self.data["source_glb"] = "human-head.glb"
        (self.root / "provenance.json").write_text(json.dumps(
            provenance_json(redistribution_allowed=False)))
        with self.assertRaises(HeadFitError):
            self.load()

    def test_rejects_degenerate_landmarks_and_duplicate_ids(self):
        self.data["landmarks"][1]["id"] = "nose"
        with self.assertRaisesRegex(HeadFitError, "duplicate"):
            self.load()
        self.data["landmarks"][1]["id"] = "left-eye"
        for index, entry in enumerate(self.data["landmarks"]):
            entry["source_xyz_m"] = [index * 0.1, 0, 0]
        with self.assertRaisesRegex(HeadFitError, "degenerate"):
            fit_head_landmarks(self.load())

    def test_nonfinite_and_unexpected_keys_are_rejected(self):
        self.data["landmarks"][0]["target_xyz_m"][1] = math.inf
        with self.assertRaises(HeadFitError):
            self.load()

    def test_front_axis_is_explicit_and_validated(self):
        self.data["source_front_axis"] = "+X"
        self.assertEqual(self.load().source_front_axis, "+X")
        self.data["source_front_axis"] = "-Z"
        with self.assertRaisesRegex(HeadFitError, "source_front_axis"):
            self.load()

    def test_local_sculpt_correction_matches_anchor_and_fades_to_zero(self):
        self.data["fit_mode"] = "local-sculpt"
        self.data["landmarks"][0]["target_xyz_m"][0] += 0.018
        self.data["landmarks"][0]["radius_m"] = 0.09
        plan = self.load()
        fit = fit_head_landmarks(plan)
        field = head_deformation_field(plan, fit)
        for anchor in plan.landmarks:
            aligned = transform_point(anchor.source_xyz_m, fit)
            deformed = field.displace(aligned)
            self.assertLess(math.dist(deformed, anchor.target_xyz_m), 1e-4)
        remote = (20, 20, 20)
        self.assertEqual(field.displace(remote), remote)

    def test_rigid_mode_never_changes_fitted_anatomy(self):
        plan = self.load()
        fit = fit_head_landmarks(plan)
        field = head_deformation_field(plan, fit)
        point = (3.1, -1, 5)
        self.assertEqual(field.displace(point), point)

    def test_large_sculpt_deformation_is_rejected(self):
        self.data["fit_mode"] = "local-sculpt"
        self.data["landmarks"][0]["target_xyz_m"][0] += 1.5
        plan = self.load()
        fit = fit_head_landmarks(plan)
        with self.assertRaisesRegex(HeadFitError, "excessive"):
            head_deformation_field(plan, fit)
        self.data["landmarks"][0]["target_xyz_m"][1] = 1.0
        self.data["unknown"] = "not allowed"
        with self.assertRaises(HeadFitError):
            self.load()


if __name__ == "__main__":
    unittest.main()
