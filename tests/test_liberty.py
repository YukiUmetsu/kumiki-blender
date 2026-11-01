"""Reference-driven checks for the procedural, multi-view Liberty sculpt."""

import contextlib
import io
import json
import unittest

from kumiki_blender.architecture_geometry import audit_mesh
from kumiki_blender.cli import main
from kumiki_blender.liberty import LibertyError, make_liberty_study


class LibertyTests(unittest.TestCase):
    def test_statue_has_readable_silhouette_and_anatomical_parts(self):
        study = make_liberty_study()
        pieces = {part.name: part for part in study.instances}
        self.assertEqual(len(pieces), len(study.instances))
        for landmark in (
            "liberty-head-face", "liberty-head-nose", "liberty-head-left-eye",
            "liberty-head-right-eye", "liberty-head-chin", "liberty-head-left-ear",
            "liberty-head-right-ear", "liberty-hair-bun", "liberty-torch-handle",
            "liberty-torch-flame-core", "liberty-tablet-slab",
            "liberty-gown-underrobe", "liberty-gown-front-overfold",
            "liberty-pedestal-core",
        ):
            self.assertIn(landmark, pieces)
        self.assertEqual(sum("crown-ray-" in n for n in pieces), 7)
        self.assertGreater(sum("gown-pleat-" in n for n in pieces), 18)
        self.assertGreater(sum("hair-strand-" in n for n in pieces), 22)
        self.assertGreater(sum("torch-flame-tongue-" in n for n in pieces), 4)
        self.assertGreater(sum("pedestal-pilaster-" in n for n in pieces), 12)
        self.assertGreater(study.bounds_max[2], 34.0)
        self.assertLess(study.bounds_min[2], 0.1)

    def test_distinct_face_features_and_right_hand_torch_are_not_flat(self):
        pieces = {part.name: part for part in make_liberty_study().instances}
        head = pieces["liberty-head-face"].mesh
        nose = pieces["liberty-head-nose"].mesh
        self.assertLess(min(v[1] for v in nose.vertices),
                        min(v[1] for v in head.vertices) + 0.25)
        self.assertGreater(max(v[2] for v in nose.vertices) - min(v[2] for v in nose.vertices), .45)
        self.assertGreater(max(v[2] for v in pieces["liberty-torch-handle"].mesh.vertices), 32)
        self.assertGreater(max(v[2] for v in pieces["liberty-tablet-slab"].mesh.vertices), 22)
        self.assertGreater(sum("torch-grip-finger-" in n for n in pieces), 3)

    def test_torch_fingers_wrap_around_the_grip_with_tapered_knuckle_geometry(self):
        pieces = {part.name: part for part in make_liberty_study().instances}
        for i in range(4):
            mesh = pieces[f"liberty-torch-grip-finger-{i}"].mesh
            self.assertGreater(len(mesh.vertices), 120)
            ys = [p[1] for p in mesh.vertices]
            self.assertLess(min(ys), -.34)
            self.assertGreater(max(ys), 0.0)

    def test_export_parts_are_closed_meshes_and_have_unique_semantic_names(self):
        study = make_liberty_study()
        self.assertLess(len(study.instances), 700)
        for part in study.instances:
            with self.subTest(part=part.name):
                report = audit_mesh(part.mesh)
                self.assertEqual(report.boundary_edges, 0)
                self.assertEqual(report.nonmanifold_edges, 0)
                self.assertGreater(report.volume_m3, 0)

    def test_scale_and_optional_pedestal_affect_scene_without_touching_statue(self):
        full = make_liberty_study()
        no_pedestal = make_liberty_study(include_pedestal=False)
        self.assertTrue(all("pedestal" not in p.name for p in no_pedestal.instances))
        self.assertLess(no_pedestal.bounds_min[2], 0.1)
        self.assertGreater(full.bounds_max[2], no_pedestal.bounds_max[2])
        with self.assertRaises(LibertyError):
            make_liberty_study(scale=-1)
        with self.assertRaises(LibertyError):
            make_liberty_study(scale=float("nan"))

    def test_cli_plans_statue_without_blender(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["plan-liberty", "--no-pedestal"])
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["generator"], "procedural-statue-of-liberty")
        self.assertEqual(report["crown_rays"], 7)
        self.assertGreater(report["instances"], 80)


if __name__ == "__main__":
    unittest.main()
