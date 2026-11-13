"""Geodesic anatomical fitting and source-pinned landmark recipe regression tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from kumiki_blender.character_build.geodesic_fit import LandmarkPin, fit_landmark_surface
from kumiki_blender.character_build.landmark_fit_plan import (
    load_landmark_fit_plan, save_landmark_fit_plan, validate_fit_spec)
from kumiki_blender.character_build.landmark_triangulation import project, triangulate_landmarks


def grid(side=9, y=0.0):
    vertices = np.array([
        ((x-(side-1)/2)*.01, (z-(side-1)/2)*.01+y, 0.0)
        for z in range(side) for x in range(side)
    ], dtype=float)
    faces = []
    for row in range(side-1):
        for col in range(side-1):
            i = row*side+col
            faces.append([i, i+1, i+1+side, i+side])
    return vertices, faces


class GeodesicSurfaceTest(unittest.TestCase):
    def test_anchored_fit_preserves_source_mesh_and_distant_vertices(self):
        vertices, faces = grid()
        before = vertices.copy()
        anchor = 40
        pin = LandmarkPin("eye.L.center", anchor, (0.0, 0.0, .003), .027)
        after, report = fit_landmark_surface(vertices, faces, [pin],
             protected_vertices=(0, 8, 72, 80),
             max_displacement_m=.005, max_relative_edge_change=.30)
        self.assertAlmostEqual(after[anchor, 2], .003, places=9)
        np.testing.assert_array_equal(after[[0,8,72,80]], before[[0,8,72,80]])
        self.assertEqual(report["landmark_count"], 1)
        self.assertGreater(report["changed_vertices"], 1)
        self.assertLess(report["changed_vertices"], len(vertices))
        self.assertEqual(report["flipped_or_degenerate_faces"], 0)
        np.testing.assert_array_equal(vertices, before)

    def test_geodesic_barrier_does_not_cross_nearby_disconnected_surface(self):
        lower, faces = grid(side=7)
        upper = lower.copy()
        upper[:, 2] = .00015
        vertices = np.concatenate([lower, upper])
        polygons = faces + [[index+len(lower) for index in face] for face in faces]
        pin = LandmarkPin("muzzle.root", 24, (0.0, 0.0, .004), .04)
        after, report = fit_landmark_surface(
            vertices, polygons, [pin],
            max_displacement_m=.006, max_relative_edge_change=.5)
        np.testing.assert_array_equal(after[len(lower):], upper)
        self.assertGreater(report["changed_vertices"], 1)
        self.assertTrue(np.any(after[:len(lower), 2] != 0))

    def test_protected_vertices_never_move(self):
        original, faces = grid()
        pin = LandmarkPin("brow.outer", 40, (0, 0, .002), .045)
        after, _ = fit_landmark_surface(
            original, faces, [pin], protected_vertices=(41, 39, 31, 49),
            max_displacement_m=.006, max_relative_edge_change=.4)
        np.testing.assert_array_equal(after[[41,39,31,49]], original[[41,39,31,49]])

    def test_unsafe_fit_is_rejected_and_inputs_unmodified(self):
        vertices, faces = grid()
        before = vertices.copy()
        with self.assertRaisesRegex(ValueError, "exceeds safe plan"):
            fit_landmark_surface(vertices, faces,
                [LandmarkPin("eye", 40, (0, 0, .04), .02)],
                max_displacement_m=.005, max_relative_edge_change=.2)
        with self.assertRaisesRegex(ValueError, "overstretched"):
            fit_landmark_surface(vertices, faces,
                [LandmarkPin("eye", 40, (0, 0, .02), .04)],
                max_displacement_m=.03, max_relative_edge_change=.03)
        with self.assertRaisesRegex(ValueError, "protected vertex"):
            fit_landmark_surface(vertices, faces,
                [LandmarkPin("eye", 40, (0, 0, .001), .04)],
                protected_vertices=(40,), max_displacement_m=.01,
                max_relative_edge_change=.3)
        np.testing.assert_array_equal(vertices, before)

    def test_rejects_duplicate_anchors_and_bad_faces(self):
        vertices, faces = grid()
        p = LandmarkPin("eye", 40, (0, 0, .001), .02)
        with self.assertRaisesRegex(ValueError, "distinct"):
            fit_landmark_surface(vertices, faces, [p, LandmarkPin("another", 40, (0,0,.001), .02)],
                                  max_displacement_m=.01, max_relative_edge_change=.2)
        with self.assertRaisesRegex(ValueError, "repeated face"):
            fit_landmark_surface(vertices, [[0,1,1]],
                                 [p], max_displacement_m=.01,
                                 max_relative_edge_change=.2)


class LandmarkFitPlanTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root/"refs").mkdir()
        for view in ("front","right"):
            Image.new("RGB", (1600,1200)).save(self.root/"refs"/f"{view}.png")
        target = [.015, -.035, .255]
        source = {"schema_version":1, "character_id":"synthetic", "height_m":.35,
                  "registered_views":{
                      view:{"file":f"refs/{view}.png","registration":"original-aligned"}
                      for view in ("front","right")
                  },
                  "landmarks":[{
                      "id":"eye.L.center", "max_reprojection_error_px":.4,
                      "observations":[{"view":view,"xy_px":project(target,view,.35),"confidence":1}
                                      for view in ("front","right")]
                  }]}
        input_path=self.root/"source-landmarks.json"
        input_path.write_text(json.dumps(source))
        self.landmark_folder=self.root/"landmarks"
        triangulate_landmarks(input_path,self.landmark_folder)
        self.landmarks=self.landmark_folder/"landmarks.json"
        self.blend=self.root/"approved.blend"
        self.blend.write_bytes(b"BLENDER-fixture-for-sha")
        self.spec_file=self.root/"fit.json"
        self.spec = {
            "schema_version":1, "mesh":"Eye.L", "expected_vertex_count":81,
            "max_displacement_m":.01, "max_relative_edge_change":.4,
            "protected_vertices":[0,8],
            "pins":[{"id":"eye.L.center","source_vertex":40,"radius_m":.026}],
        }
        self.write_spec()

    def write_spec(self):
        self.spec_file.write_text(json.dumps(self.spec))

    def test_immutable_pinned_sources_and_materializable_coordinates(self):
        plan=save_landmark_fit_plan(self.blend,self.landmarks,self.spec_file,self.root/"plan")
        self.assertEqual(plan["pins"][0]["target_world_m"],[.015,-.035,.255])
        self.assertEqual(plan["pins"][0]["source_vertex"],40)
        self.assertEqual(load_landmark_fit_plan(self.root/"plan/landmark-fit-plan.json"),plan)
        with self.assertRaises(FileExistsError):
            save_landmark_fit_plan(self.blend,self.landmarks,self.spec_file,self.root/"plan")
        self.blend.write_bytes(b"BLENDER-changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_landmark_fit_plan(self.root/"plan/landmark-fit-plan.json")

    def test_forged_pin_and_image_modification_rejected(self):
        save_landmark_fit_plan(self.blend,self.landmarks,self.spec_file,self.root/"plan")
        record_path = self.root/"plan/landmark-fit-plan.json"
        forged=json.loads(record_path.read_text())
        forged["pins"][0]["target_world_m"]=[0,0,0]
        record_path.write_text(json.dumps(forged))
        with self.assertRaisesRegex(ValueError, "changed"):
            load_landmark_fit_plan(record_path)
        Image.new("RGB",(1600,1200),(1,2,3)).save(self.root/"refs/front.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_landmark_fit_plan(record_path)

    def test_bad_vertex_correspondence_and_unsafe_values(self):
        self.spec["pins"][0]["source_vertex"]=0
        self.write_spec()
        with self.assertRaisesRegex(ValueError, "unprotected"):
            validate_fit_spec(self.spec)
        self.spec["pins"][0]["source_vertex"]=40
        self.spec["pins"][0]["radius_m"]=-1
        self.write_spec()
        with self.assertRaisesRegex(ValueError, "radius"):
            validate_fit_spec(self.spec)


if __name__=="__main__":
    unittest.main()
