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
    MonumentError, khufu_petrie_assembly, make_monument_study, pyramid_assembly, pyramid_casing_stone,
    pyramid_frustum, sphinx_assembly,
)
from kumiki_blender.pyramid_survey import (
    KHUFU_ORIGINAL_BASE_M, KHUFU_ORIGINAL_HEIGHT_M,
    PETRIE_NE_COURSE_TOP_IN, petrie_course_tops_m,
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

    def test_default_pyramid_exposes_weathered_stone_instead_of_a_broad_white_band(self):
        parts = pyramid_assembly(prefix="pyramid")
        outer = [part for part in parts if "-course-" in part.name]
        polished = [part for part in outer if part.material == "casing-limestone"]
        self.assertLess(len(polished)/len(outer), .10)
        self.assertGreater(len(outer)-len(polished), 300)

    def test_petrie_survey_has_203_measured_levels_and_observed_course_height_variation(self):
        levels = petrie_course_tops_m()
        self.assertEqual(len(PETRIE_NE_COURSE_TOP_IN), 203)
        self.assertEqual(len(levels), 201)
        self.assertAlmostEqual(levels[0], 58.6 * .0254)
        self.assertAlmostEqual(levels[-1], 5407.9 * .0254)
        self.assertAlmostEqual(petrie_course_tops_m(last_course=203)[-1], 5451.8 * .0254)
        heights = [levels[0], *(b-a for a, b in zip(levels, levels[1:]))]
        self.assertGreater(max(heights)/min(heights), 2.5)
        for invalid in (True, 0, 204, 5.0):
            with self.subTest(invalid=invalid), self.assertRaises(GeometryError):
                petrie_course_tops_m(last_course=invalid)

    def test_khufu_petrie_preset_batches_real_course_heights_and_open_stone_joints(self):
        parts = khufu_petrie_assembly(prefix="khufu")
        self.assertEqual(len(parts), 1+201*4)
        self.assertEqual(parts[0].name, "khufu-core")
        self.assertNotIn("capstone", {part.name for part in parts})
        first = parts[1]
        last = parts[-1]
        self.assertEqual(first.name, "khufu-course-001-face-0-batch")
        self.assertEqual(last.name, "khufu-course-201-face-3-batch")
        self.assertGreater(len(first.mesh.vertices), 1000)
        for item in (first, last):
            report = audit_mesh(item.mesh)
            self.assertEqual(report.boundary_edges, 0)
            self.assertEqual(report.nonmanifold_edges, 0)
            self.assertGreater(report.volume_m3, 0)
        second_course = parts[5]
        self.assertLess(max(vertex[2] for vertex in world_vertices(first)),
                        min(vertex[2] for vertex in world_vertices(second_course)))

        study = make_monument_study(kind="pyramid", pyramid_preset="khufu-petrie")
        self.assertEqual(dict(study.part_counts)["pyramid_parts"], 805)
        self.assertEqual(dict(study.parameters)["pyramid_courses"], 201)
        self.assertEqual(dict(study.parameters)["pyramid_preset"], "khufu-petrie")
        self.assertAlmostEqual(study.bounds_max[2], petrie_course_tops_m()[-1])
        self.assertAlmostEqual(dict(study.parameters)["pyramid_base_m"], KHUFU_ORIGINAL_BASE_M)
        self.assertAlmostEqual(dict(study.parameters)["pyramid_height_m"], KHUFU_ORIGINAL_HEIGHT_M)

    def test_khufu_petrie_rejects_misleading_custom_parameters_or_unsurveyed_site_layout(self):
        for changes in ({"kind": "giza-study"}, {"kind": "sphinx"},
                        {"kind": "pyramid", "pyramid_courses": 201},
                        {"kind": "pyramid", "pyramid_height": 138.0}):
            with self.subTest(changes=changes), self.assertRaises(MonumentError):
                make_monument_study(pyramid_preset="khufu-petrie", **changes)

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = main(["plan-monument", "--kind", "pyramid",
                           "--pyramid-preset", "khufu-petrie"])
        self.assertEqual(status, 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["named_parts"], {"pyramid_parts": 805, "sphinx_parts": 0})

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

    def test_sphinx_reference_has_four_distinct_toes_per_forepaw(self):
        parts = {part.name: part for part in sphinx_assembly(prefix="sphinx")}
        for side in ("left", "right"):
            toes = [parts[f"sphinx-{side}-toe-{index}"] for index in range(4)]
            centers = [sum(v[1] for v in toe.mesh.vertices)/len(toe.mesh.vertices)
                       for toe in toes]
            self.assertEqual(centers, sorted(centers))
            self.assertGreater(centers[-1]-centers[0], .47)
            self.assertLess(centers[-1]-centers[0], .61)
            for toe in toes:
                self.assertLess(min(v[0] for v in toe.mesh.vertices), -5.42)
                self.assertGreater(max(v[0] for v in toe.mesh.vertices), -5.05)

    def test_face_relief_preserves_smooth_back_of_head(self):
        # Shape details belong on the anatomical anterior face, not repeated
        # around the back of the nemes as a texture-like displacement.
        from kumiki_blender.egyptian_monuments import _sculpt_face_relief
        source = loft_elliptical_sections(
            ((3.1, -3.9, 0, .34, .35),
             (3.75, -3.9, 0, .5, .5),
             (4.4, -3.9, 0, .33, .4)),
            axis="z", radial_segments=96, subdivisions=8)
        carved = _sculpt_face_relief(source, scale_x=1, scale_y=1, scale_z=1)
        self.assertEqual(carved.faces, source.faces)
        rear_indices = [i for i, v in enumerate(source.vertices) if v[0] > -3.85]
        self.assertTrue(rear_indices)
        self.assertTrue(all(carved.vertices[i] == source.vertices[i]
                            for i in rear_indices))
        anterior = [i for i, v in enumerate(source.vertices) if v[0] < -4.19]
        self.assertGreater(max(abs(carved.vertices[i][0]-source.vertices[i][0])
                               for i in anterior), .09)
        self.assertEqual((audit_mesh(carved).boundary_edges,
                          audit_mesh(carved).nonmanifold_edges), (0, 0))

    def test_sphinx_scaling_is_parametric(self):
        small = sphinx_assembly(prefix="small", length=10.5)
        large = sphinx_assembly(prefix="large", length=21)
        width = lambda items: max(v[0] for p in items for v in world_vertices(p)) - min(
            v[0] for p in items for v in world_vertices(p))
        # Weathering is sampled in physical meters rather than normalized
        # object coordinates, so the sculpted silhouette has a small relief
        # tolerance rather than an exactly scaled vertex-for-vertex copy.
        self.assertAlmostEqual(width(large)/width(small), 2, delta=.03)

    def test_rounded_lion_silhouette_and_head_are_not_cartoon_proportioned(self):
        parts = {part.name: part.mesh for part in sphinx_assembly(prefix="sphinx")}
        torso = parts["sphinx-lion-torso"]
        ring = torso.vertices[20*96:21*96]
        y_mid = sum(vertex[1] for vertex in ring)/len(ring)
        radius_y = max(abs(vertex[1]-y_mid) for vertex in ring)
        # A rectangular superellipse once had diagonal ratio >.83: its
        # planar side read as a giant sandstone beam even after voxel union.
        self.assertLess(abs(ring[12][1]-y_mid)/radius_y, .80)
        # The back cannot be an even-height slab: a large shoulder crest,
        # slightly lower saddle and raised hindquarters form a feline profile.
        shoulder_top = max(point[2] for point in torso.vertices[5*96:6*96])
        mid_back_top = max(point[2] for point in torso.vertices[20*96:21*96])
        self.assertGreater(shoulder_top-mid_back_top, .45)
        neck = parts["sphinx-carved-neck"]
        lower_neck = neck.vertices[:96]
        upper_neck = neck.vertices[-96:]
        self.assertGreater(sum(point[0] for point in lower_neck)/96 -
                           sum(point[0] for point in upper_neck)/96, .45)
        hood = parts["sphinx-nemes-hood"]
        high_ring = hood.vertices[-96:]
        previous_ring = hood.vertices[25*96:26*96]  # Last loft control point before crown cap.
        width = lambda points: max(point[1] for point in points)-min(point[1] for point in points)
        # Preserve a tapered headdress crown: an oversized flat cap reads
        # like a cylinder, while a fully hemispherical cap reads like an egg.
        self.assertGreater(width(high_ring)/width(previous_ring), .70)
        self.assertLess(width(high_ring)/width(previous_ring), .90)
        total_x = [point[0] for part in parts.values() for point in part.vertices]
        top_z = max(point[2] for part in parts.values() for point in part.vertices)
        self.assertLess(top_z/(max(total_x)-min(total_x)), .34)
        self.assertGreater(top_z/(max(total_x)-min(total_x)), .26)

    def test_studies_have_correct_counts_bounds_and_camera_views(self):
        for kind in ("pyramid", "sphinx", "giza-study"):
            study = make_monument_study(kind=kind)
            with self.subTest(kind=kind):
                self.assertEqual(len(study.instances), sum(dict(study.part_counts).values()))
                self.assertEqual(len({p.name for p in study.instances}), len(study.instances))
                self.assertGreater(study.bounds_max[2], 3.4 if kind == "sphinx" else 4.0)
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
