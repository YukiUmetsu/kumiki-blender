"""Registered source 3D landmark substitutions into existing anatomical templates."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from PIL import Image

from tests.test_character_anatomy import anatomy_fixture
from kumiki_blender.character_build.anatomy_plan import validate_anatomy_spec
from kumiki_blender.character_build.anatomy_landmark_resolver import resolve_anatomy_landmarks
from kumiki_blender.character_build.landmark_triangulation import (
    project, triangulate_landmarks,
)


class AnatomyLandmarkResolutionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        refs = self.root/"refs"
        refs.mkdir()
        for view in ("front", "right"):
            Image.new("RGB", (1600,1200)).save(refs/f"{view}.png")
        base = anatomy_fixture()
        coords = {
            "eye.L.center": base["parts"][0]["center_m"],
            "upper_lid.L.corner": base["parts"][1]["control_points_m"][0],
            "beak.root": base["parts"][2]["sections_m"][0][0],
        }
        source = {
            "schema_version": 1, "character_id": "fixture", "height_m": .35,
            "registered_views": {
                view: {"file": f"refs/{view}.png", "registration": "original-aligned"}
                for view in ("front", "right")
            },
            "landmarks": [{
                "id": key, "max_reprojection_error_px": .25,
                "observations": [
                    {"view": view, "xy_px": project(position, view, .35), "confidence": 1.0}
                    for view in ("front", "right")]
            } for key, position in coords.items()],
        }
        self.sources = self.root/"registered.json"
        self.sources.write_text(json.dumps(source), encoding="utf-8")
        self.report = self.root/"registered-output"
        triangulate_landmarks(self.sources, self.report)
        self.landmark_path = self.report/"landmarks.json"
        base["parts"][0]["center_m"] = {"landmark": "eye.L.center"}
        base["parts"][1]["control_points_m"][0] = {"landmark": "upper_lid.L.corner"}
        base["parts"][2]["sections_m"][0][0] = {"landmark": "beak.root"}
        self.template = self.root/"anatomy-template.json"
        self.template.write_text(json.dumps(base), encoding="utf-8")

    def test_exact_selected_landmarks_and_original_template_unchanged(self):
        destination = self.root/"resolved"
        receipt = resolve_anatomy_landmarks(self.template, self.landmark_path, destination)
        spec = json.loads((destination/"resolved-anatomy.json").read_text())
        validate_anatomy_spec(spec)
        self.assertEqual(len(receipt["used_landmarks"]), 3)
        self.assertEqual(spec["parts"][0]["center_m"], [.036, -.063, .27])
        self.assertEqual(spec["parts"][1]["control_points_m"][0], [.023, -.071, .275])
        self.assertEqual(spec["parts"][2]["sections_m"][0][0], [.014, -.068, .251])
        self.assertTrue(isinstance(json.loads(self.template.read_text())["parts"][0]["center_m"], dict))
        with self.assertRaises(FileExistsError):
            resolve_anatomy_landmarks(self.template, self.landmark_path, destination)

    def test_missing_landmark_does_not_emit_partial_files(self):
        data = json.loads(self.template.read_text())
        data["parts"][0]["center_m"] = {"landmark": "nonexistent"}
        self.template.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Missing independently"):
            resolve_anatomy_landmarks(self.template, self.landmark_path, self.root/"bad")
        self.assertFalse((self.root/"bad").exists())

    def test_verified_status_cannot_cite_unsolved_camera(self):
        data = json.loads(self.template.read_text())
        data["parts"][0]["evidence"]["status"] = "registered-verified"
        # Source triangulation only has front and right, but the template
        # claims a front-right registered view.
        self.template.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "do not support|do not support landmark"):
            resolve_anatomy_landmarks(self.template, self.landmark_path, self.root/"bad")

    def test_changed_source_image_invalidates_resolver(self):
        Image.new("RGB", (1600,1200), (10,20,30)).save(self.root/"refs/front.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            resolve_anatomy_landmarks(self.template, self.landmark_path, self.root/"bad")

    def test_no_placeholder_is_not_misrepresented_as_source_grounded(self):
        self.template.write_text(json.dumps(anatomy_fixture()))
        with self.assertRaisesRegex(ValueError, "no landmark"):
            resolve_anatomy_landmarks(self.template, self.landmark_path, self.root/"bad")


if __name__ == "__main__":
    unittest.main()
