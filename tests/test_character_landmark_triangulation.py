"""Multi-view, source-pinned 2D-to-3D anatomical landmark triangulation tests."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.character_build.landmark_triangulation import (
    load_triangulated_landmarks, project, triangulate_landmarks,
)
from kumiki_blender.character_build.views import camera


class TriangulationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root/"refs").mkdir()
        for name in ("front", "right", "back", "front_left", "az45_el30"):
            Image.new("RGB", (1600, 1200)).save(self.root/"refs"/f"{name}.png")
        self.known = [.033, -.046, .22]
        self.spec = self.root/"landmarks-spec.json"
        self.data = self._spec(["front", "right"])
        self.save()

    def _spec(self, cameras):
        return {
            "schema_version": 1, "character_id": "synthetic",
            "height_m": .35,
            "registered_views": {
                view: {"file": f"refs/{view}.png", "registration": "original-aligned"}
                for view in cameras
            },
            "landmarks": [{
                "id": "eye.L.center",
                "max_reprojection_error_px": .25,
                "observations": [
                    {"view": view, "xy_px": project(self.known, view, .35), "confidence": 1.0}
                    for view in cameras
                ],
            }],
        }

    def save(self):
        self.spec.write_text(json.dumps(self.data), encoding="utf-8")

    def test_projection_matches_existing_camera_convention(self):
        import numpy as np
        for name in ("front", "right", "front_left", "az45_el30"):
            with self.subTest(view=name):
                expected = camera(name).project(np.array([self.known]), .35/1000)
                result = project(self.known, name, .35)
                self.assertAlmostEqual(result[0], float(expected[0][0]), places=8)
                self.assertAlmostEqual(result[1], float(expected[1][0]), places=8)

    def test_reconstructs_world_point_from_verified_front_right(self):
        record = triangulate_landmarks(self.spec, self.root/"triangulated")
        point = record["points_m"]["eye.L.center"]
        self.assertTrue(all(abs(a-b) <= 1e-8 for a,b in zip(point, self.known)))
        self.assertLess(record["landmarks"][0]["max_reprojection_error_px"], 1e-4)
        self.assertEqual(load_triangulated_landmarks(
            self.root/"triangulated"/"landmarks.json"), record)
        with self.assertRaises(FileExistsError):
            triangulate_landmarks(self.spec, self.root/"triangulated")

    def test_elevated_diagonal_view_recovers_depth(self):
        self.data = self._spec(["front", "az45_el30"])
        self.save()
        record = triangulate_landmarks(self.spec, self.root/"oblique")
        self.assertTrue(all(abs(a-b) <= 1e-8 for a,b in
                            zip(record["points_m"]["eye.L.center"], self.known)))

    def test_rejects_front_back_ambiguity(self):
        self.data = self._spec(["front", "back"])
        self.save()
        with self.assertRaisesRegex(ValueError, "rank-deficient|baseline"):
            triangulate_landmarks(self.spec, self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_refuses_inconsistent_third_view(self):
        self.data = self._spec(["front", "right", "front_left"])
        self.data["landmarks"][0]["observations"][2]["xy_px"][0] += 24
        self.save()
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            triangulate_landmarks(self.spec, self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_hash_changes_and_invalid_registration(self):
        triangulate_landmarks(self.spec, self.root/"good")
        img = Image.new("RGB", (1600,1200), (3,4,5))
        img.save(self.root/"refs"/"right.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_triangulated_landmarks(self.root/"good/landmarks.json")
        self.data["registered_views"]["right"]["registration"] = "unverified"
        self.save()
        with self.assertRaisesRegex(ValueError, "Unverified"):
            triangulate_landmarks(self.spec, self.root/"bad")

    def test_reject_unsafe_source_and_misaligned_reference(self):
        self.data["registered_views"]["front"]["file"] = "../secret.png"
        self.save()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            triangulate_landmarks(self.spec, self.root/"bad")
        self.data["registered_views"]["front"]["file"] = "refs/front.png"
        Image.new("RGB", (200,100)).save(self.root/"refs/front.png")
        self.save()
        with self.assertRaisesRegex(ValueError, "not aligned"):
            triangulate_landmarks(self.spec, self.root/"bad")

    def test_observation_confidence_and_duplicate_views(self):
        self.data["landmarks"][0]["observations"][0]["confidence"] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "confidence"):
            triangulate_landmarks(self.spec, self.root/"bad")
        self.data = self._spec(["front", "right"])
        self.data["landmarks"][0]["observations"][1]["view"] = "front"
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            triangulate_landmarks(self.spec, self.root/"bad")


if __name__ == "__main__":
    unittest.main()
