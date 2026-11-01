"""Manifold, dimensional and compositional checks for Egyptian monument kit."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.architecture_geometry import (
    GeometryError, audit_mesh, loft_elliptical_sections, world_vertices,
)
from kumiki_blender.blender_monuments import plan_monument_shots
from kumiki_blender.cli import main
from kumiki_blender.egyptian_monuments import (
    MonumentError, make_monument_study, pyramid_assembly, pyramid_casing_stone,
    pyramid_frustum, sphinx_assembly,
)


class EgyptianMonumentTests(unittest.TestCase):
    def test_elliptical_loft_is_watertight_for_all_three_axes(self):
        rings = ((0, 0, 0, 0.5, 0.3), (0.6, 0.1, 0.2, 0.7, 0.5),
                 (1.2, 0.2, 0.1, 0.3, 0.25))
        for axis in "xyz":
            with self.subTest(axis=axis):
                mesh = loft_elliptical_sections(rings, axis=axis)
                report = audit_mesh(mesh)
                self.assertEqual(report.boundary_edges, 0)
                self.assertEqual(report.nonmanifold_edges, 0)
                self.assertGreater(report.volume_m3, 0)

    def test_elliptical_loft_rejects_degenerate_profiles(self):
        good = ((0, 0, 0, .5, .3), (1, 0, 0, .5, .3))
        invalid = (dict(sections=good, axis="q"),
                   dict(sections=((0, 0, 0, .1, .1), (0, 0, 0, .2, .2))),
                   dict(sections=((0, 0, 0, 0, .1), (1, 0, 0, .2, .2))),
                   dict(sections=good, radial_segments=7))
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(GeometryError):
                loft_elliptical_sections(**kwargs)

    def test_pyramid_stones_are_wedges_with_real_positive_thickness(self):
        for mesh in (pyramid_frustum(36, 1.1, 22),
                     pyramid_casing_stone(36, 35.1, 1.0, 0, .15)):
            report = audit_mesh(mesh)
            self.assertEqual(report.boundary_edges, 0)
            self.assertEqual(report.nonmanifold_edges, 0)
            self.assertGreater(report.volume_m3, 0)
        stone = pyramid_casing_stone(36, 35.1, 1.0, 0, .15, inset=.2)
        self.assertGreater(stone.vertices[0][1]-stone.vertices[4][1], .19)
        self.assertLess(abs(stone.vertices[2][0]), abs(stone.vertices[1][0]))

    def test_pyramid_layers_have_real_block_and_course_joints(self):
        parts = pyramid_assembly(prefix="khufu-study", base_width=36, height=23,
                                 courses=18, casing_fraction=.2)
        self.assertGreater(len(parts), 250)
        self.assertEqual(parts[0].name, "khufu-study-core")
        self.assertEqual(parts[-1].name, "khufu-study-capstone")
        self.assertTrue(any(i.material == "casing-limestone" for i in parts))
        self.assertTrue(any(i.material == "weathered-limestone" for i in parts))
        self.assertEqual(len({i.name for i in parts}), len(parts))
        for item in parts:
            with self.subTest(name=item.name):
                report = audit_mesh(item.mesh)
                self.assertEqual(report.boundary_edges, 0)
                self.assertEqual(report.nonmanifold_edges, 0)
                self.assertGreater(report.volume_m3, 0)
        low = next(p for p in parts if "course-000-face-0-stone-" in p.name)
        high = next(p for p in parts if "course-001-face-0-stone-" in p.name)
        self.assertLess(max(v[2] for v in world_vertices(low)),
                        min(v[2] for v in world_vertices(high)))

    def test_sphinx_is_organic_and_has_independent_anatomy(self):
        parts = sphinx_assembly(prefix="sphinx")
        self.assertGreater(len(parts), 20)
        names = {i.name for i in parts}
        for role in ("lion-torso", "left-forepaw", "right-forepaw", "face",
                     "nemes-hood", "left-haunch", "right-haunch", "carved-neck"):
            self.assertIn("sphinx-"+role, names)
        self.assertFalse(any("-eye" in name or "-brow" in name or
                             "hood-stripe" in name for name in names))
        self.assertFalse(any("restored-nose" in name for name in names))
        self.assertEqual(len(names), len(parts))
        self.assertGreater(sum(len(p.mesh.vertices) for p in parts), 12000)
        for part in parts:
            with self.subTest(name=part.name):
                report = audit_mesh(part.mesh)
                self.assertEqual(report.boundary_edges, 0)
                self.assertEqual(report.nonmanifold_edges, 0)
                self.assertGreater(report.volume_m3, 0)
        restored = sphinx_assembly(prefix="sphinx", restored_nose=True)
        self.assertEqual(len(restored), len(parts)+1)
        self.assertIn("sphinx-optional-restored-nose", [p.name for p in restored])

    def test_sphinx_scaling_is_parametric(self):
        small = sphinx_assembly(prefix="small", length=10.5)
        large = sphinx_assembly(prefix="large", length=21)
        width = lambda items: max(v[0] for p in items for v in world_vertices(p)) - min(
            v[0] for p in items for v in world_vertices(p))
        # Weathering is sampled in physical meters rather than normalized
        # object coordinates, so the sculpted silhouette has a small relief
        # tolerance rather than an exactly scaled vertex-for-vertex copy.
        self.assertAlmostEqual(width(large)/width(small), 2, delta=.03)

    def test_studies_have_correct_counts_bounds_and_camera_views(self):
        for kind in ("pyramid", "sphinx", "giza-study"):
            study = make_monument_study(kind=kind)
            with self.subTest(kind=kind):
                self.assertEqual(len(study.instances), sum(dict(study.part_counts).values()))
                self.assertEqual(len({p.name for p in study.instances}), len(study.instances))
                self.assertGreater(study.bounds_max[2], 4.0)
                shots = plan_monument_shots(study)
                self.assertEqual(shots[0].name, "hero")
                self.assertEqual(len(shots), {"giza-study": 6, "sphinx": 5,
                                              "pyramid": 3}[kind])
                self.assertEqual(len({s.name for s in shots}), len(shots))
                if kind != "pyramid":
                    self.assertIn("sphinx-profile", [s.name for s in shots])
                    self.assertIn("sphinx-face", [s.name for s in shots])
                self.assertGreater(shots[0].ortho_scale/1.6,
                                   (study.bounds_max[2]-study.bounds_min[2])*1.1)

    def test_invalid_settings_are_rejected_without_truncation(self):
        for kwargs in ({"kind": "obelisks"}, {"pyramid_base": -1},
                       {"pyramid_height": 100, "pyramid_base": 8},
                       {"pyramid_courses": True}, {"casing_fraction": 2},
                       {"sphinx_length": float("nan")}, {"restored_nose": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((MonumentError, GeometryError)):
                make_monument_study(**kwargs)
        for kwargs in ({"kind": "sphinx", "pyramid_base": float("nan")},
                       {"kind": "sphinx", "pyramid_courses": True},
                       {"kind": "pyramid", "sphinx_length": float("inf")},
                       {"kind": "pyramid", "restored_nose": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((MonumentError, GeometryError)):
                make_monument_study(**kwargs)

    def test_plan_cli_runs_without_blender_and_build_is_non_destructive(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = main(["plan-monument", "--kind", "giza-study", "--pyramid-courses", "18"])
        self.assertEqual(status, 0)
        data = json.loads(stream.getvalue())
        self.assertGreater(data["named_parts"]["pyramid_parts"], 300)
        self.assertGreater(data["named_parts"]["sphinx_parts"], 20)
        self.assertEqual(data["instances"], sum(data["named_parts"].values()))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"new"
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                status = main(["build-monument", "--kind", "sphinx", "--output", str(target)])
            self.assertEqual(status, 2)
            self.assertIn("Blender", errors.getvalue())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
