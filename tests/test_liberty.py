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

    def test_opt_in_tablet_keystone_and_hand_contact(self):
        from kumiki_blender.liberty_tablet import tablet_front_y, tablet_right_edge

        original = {part.name for part in make_liberty_study().instances}
        self.assertIn("liberty-tablet-top-bead", original)
        self.assertNotIn("liberty-tablet-hand-palm", original)
        pieces = {part.name: part for part in make_liberty_study(detailed_tablet=True).instances}
        self.assertNotIn("liberty-tablet-top-bead", pieces)
        self.assertIn("liberty-tablet-hand-palm", pieces)
        slab = pieces["liberty-tablet-slab"].mesh
        zs = [p[2] for p in slab.vertices]
        self.assertGreater(max(zs)-min(zs), 4.0)
        # A keystone widens toward the top; this study also adds thick chamfers.
        upper = [p[0] for p in slab.vertices if p[2] > 24.2]
        lower = [p[0] for p in slab.vertices if p[2] < 20.5]
        self.assertGreater(max(upper)-min(upper), max(lower)-min(lower)+.35)
        self.assertGreater(len(slab.vertices), 8)
        self.assertGreater(tablet_right_edge(22.3), 4.2)
        self.assertLess(tablet_front_y(22.3), -1.)
        for i in range(4):
            finger = pieces[f"liberty-tablet-grip-finger-{i}"].mesh
            self.assertGreater(len(finger.vertices), 100)
            self.assertLess(min(p[0] for p in finger.vertices), tablet_right_edge(22.3))
            self.assertLess(min(p[1] for p in finger.vertices), tablet_front_y(22.3))

    def test_detailed_tablet_respects_scaled_grounded_scene(self):
        full = make_liberty_study(scale=.5, include_pedestal=False, detailed_tablet=True)
        pieces = {part.name: part for part in full.instances}
        tablet = pieces["liberty-tablet-slab"].mesh
        self.assertAlmostEqual(max(p[2] for p in tablet.vertices), (24.4-9.30)*.5,
                               delta=.08)
        self.assertTrue(dict(full.parameters)["detailed_tablet"])
        for part in full.instances:
            if "tablet-" in part.name:
                with self.subTest(part=part.name):
                    report = audit_mesh(part.mesh)
                    self.assertEqual(report.boundary_edges, 0)
                    self.assertEqual(report.nonmanifold_edges, 0)
                    self.assertGreater(report.volume_m3, 0)
        with self.assertRaises(LibertyError):
            make_liberty_study(detailed_tablet="yes")

    def test_detailed_crown_has_real_openings_and_blade_rays(self):
        from kumiki_blender.liberty_crown import crown_band, crown_ray, crown_fasteners

        # The perforated band is one connected closed surface, not dark tubes
        # painted into an opaque helmet. Each missing window cell is an opening.
        band = crown_band()
        report = audit_mesh(band)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(report.nonmanifold_edges, 0)
        self.assertGreater(report.volume_m3, 0)
        self.assertGreater(len(band.faces), 300)
        edges = {tuple(sorted((a, b))) for face in band.faces
                 for a, b in zip(face, (*face[1:], face[0]))}
        # The annulus has one base handle; each true through-window adds one.
        genus = (2-(len(band.vertices)-len(edges)+len(band.faces)))//2
        self.assertEqual(genus, 26)
        base = {part.name for part in make_liberty_study().instances}
        self.assertIn("liberty-crown-window-dark-00", base)
        pieces = {part.name: part for part in make_liberty_study(detailed_crown=True).instances}
        self.assertNotIn("liberty-crown-window-dark-00", pieces)
        self.assertIn("liberty-crown-diadem-perforated", pieces)
        self.assertIn("liberty-crown-fasteners", pieces)
        hardware = crown_fasteners()
        # 50 mullion heads and 14 ray-socket heads, all closed rather than
        # open cylinders or texture-only circles.
        self.assertEqual(len(hardware.vertices), 64*62)
        self.assertEqual(len(hardware.faces), 64*72)
        self.assertGreater(max(p[2] for p in hardware.vertices), 29.40)
        self.assertEqual(audit_mesh(hardware).boundary_edges, 0)
        self.assertGreater(audit_mesh(hardware).volume_m3, 0.)
        self.assertIn("liberty-hair-forelock-sculpt", pieces)
        self.assertNotIn("liberty-hair-strand-00", pieces)
        self.assertNotIn("liberty-hair-left-temple-wave-00", pieces)
        self.assertGreater(len(pieces["liberty-hair-forelock-sculpt"].mesh.vertices), 3000)
        self.assertEqual(sum(name.startswith("liberty-crown-ray-") for name in pieces), 7)
        for i in range(7):
            ray = pieces[f"liberty-crown-ray-{i}"].mesh
            result = audit_mesh(ray)
            self.assertEqual(result.boundary_edges, 0)
            self.assertEqual(result.nonmanifold_edges, 0)
            self.assertGreater(result.volume_m3, 0)
            self.assertGreater(len(ray.vertices), 16)
        self.assertGreater(len(crown_ray(3).vertices), 16)
        for name, piece in pieces.items():
            if name.startswith("liberty-crown-") or name == "liberty-hair-forelock-sculpt":
                with self.subTest(part=name):
                    result = audit_mesh(piece.mesh)
                    self.assertEqual(result.boundary_edges, 0)
                    self.assertEqual(result.nonmanifold_edges, 0)
                    self.assertGreater(result.volume_m3, 0)
        scaled = make_liberty_study(scale=.5, include_pedestal=False, detailed_crown=True)
        self.assertTrue(dict(scaled.parameters)["detailed_crown"])
        with self.assertRaises(LibertyError):
            make_liberty_study(detailed_crown=1)

    def test_detailed_face_is_one_closed_anatomical_skin_without_feature_rods(self):
        from kumiki_blender.liberty_face_anatomy import face_projection, sculpted_ear

        study = make_liberty_study(detailed_face=True, include_pedestal=False)
        parts = {part.name: part for part in study.instances}
        face = parts["liberty-head-face"].mesh
        self.assertGreater(len(face.vertices), 48_000)
        report = audit_mesh(face)
        self.assertEqual((report.boundary_edges, report.nonmanifold_edges), (0, 0))
        self.assertGreater(report.volume_m3, 0.)
        for redundant in ("liberty-head-nose", "liberty-head-left-eye",
                          "liberty-head-left-upper-lid", "liberty-head-upper-lip",
                          "liberty-head-mouth-crease", "liberty-head-chin"):
            self.assertNotIn(redundant, parts)
        self.assertTrue(dict(study.parameters)["detailed_face"])
        self.assertNotIn("liberty-head-right-ear-helix", parts)
        for sign in (-1, 1):
            ear = sculpted_ear(sign)
            ear_report = audit_mesh(ear)
            self.assertEqual((ear_report.boundary_edges, ear_report.nonmanifold_edges), (0, 0))
            self.assertGreater(ear_report.volume_m3, 0.)
        self.assertGreater(face_projection(0., 26.51), .27)
        self.assertLess(face_projection(.46, 27.22), -.10)
        self.assertGreater(face_projection(.46, 27.55), .07)
        self.assertLess(face_projection(0., 25.92), 0.)
        self.assertAlmostEqual(face_projection(.46, 27.22),
                               face_projection(-.46, 27.22), places=5)
        with self.assertRaises(ValueError):
            face_projection(float("nan"), 27.)
        with self.assertRaises(LibertyError):
            make_liberty_study(detailed_face=1)

    def test_detailed_feet_have_open_broken_shackle_and_both_foot_chains(self):
        """Opt-in sculptural details follow NPS right/left spatial semantics."""
        original = {part.name for part in make_liberty_study().instances}
        self.assertNotIn("liberty-feet-right-broken-shackle", original)
        pieces = {part.name: part for part in make_liberty_study(
            detailed_feet=True).instances}
        self.assertIn("liberty-feet-right-broken-shackle", pieces)
        self.assertIn("liberty-feet-left-chain-01-broken", pieces)
        self.assertEqual(sum("-toe-" in name for name in pieces), 10)
        self.assertEqual(sum("-chain-" in name for name in pieces), 4)
        self.assertEqual(dict(make_liberty_study(detailed_feet=True).part_counts)["feet"], 19)
        shackle = pieces["liberty-feet-right-broken-shackle"].mesh
        left_end = pieces["liberty-feet-left-chain-01-broken"].mesh
        self.assertLess(sum(p[0] for p in shackle.vertices)/len(shackle.vertices), 0.)
        self.assertGreater(sum(p[0] for p in left_end.vertices)/len(left_end.vertices), 0.)
        # Exposed rings have no opening in their *surface* manifold: a split
        # link is a curved, capped open-ended bar, not a torn open mesh.
        for name, part in pieces.items():
            if "liberty-feet-" in name:
                with self.subTest(name=name):
                    report = audit_mesh(part.mesh)
                    self.assertEqual(report.boundary_edges, 0)
                    self.assertEqual(report.nonmanifold_edges, 0)
                    self.assertGreater(report.volume_m3, 0.)
        smaller = make_liberty_study(scale=.5, include_pedestal=False,
                                     detailed_feet=True)
        self.assertTrue(dict(smaller.parameters)["detailed_feet"])
        self.assertLess(smaller.bounds_min[2], .15)
        with self.assertRaises(LibertyError):
            make_liberty_study(detailed_feet=1)

    def test_detailed_drapery_is_integrated_sculptural_geometry(self):
        from kumiki_blender.liberty_drapery import (sculpted_front_overfold,
                                                   sculpted_torch_sleeve,
                                                   sculpted_underrobe)

        base = {part.name for part in make_liberty_study().instances}
        self.assertIn("liberty-gown-pleat-00", base)
        self.assertIn("liberty-sleeve-torch-wrinkle-00", base)
        study = make_liberty_study(detailed_drapery=True)
        parts = {part.name: part for part in study.instances}
        self.assertTrue(dict(study.parameters)["detailed_drapery"])
        self.assertNotIn("liberty-gown-pleat-00", parts)
        self.assertNotIn("liberty-sleeve-torch-wrinkle-00", parts)
        self.assertNotIn("liberty-gown-overfold-hem", parts)
        self.assertEqual(sum("gown-pleat-" in name for name in parts), 0)
        gown = parts["liberty-gown-underrobe"].mesh
        sleeve = parts["liberty-sleeve-torch"].mesh
        overfold = parts["liberty-gown-front-overfold"].mesh
        self.assertGreater(len(gown.vertices), 15_000)
        self.assertGreater(len(sleeve.vertices), 5_000)
        self.assertGreater(len(overfold.vertices), 10_000)
        # Its lower edge rises diagonally; it is not a straight skirt fringe.
        hem = overfold.vertices[68*73:69*73]
        self.assertGreater(hem[-1][2]-hem[0][2], 2.5)
        for mesh in (gown, sleeve, overfold, sculpted_underrobe(),
                     sculpted_torch_sleeve(), sculpted_front_overfold()):
            report = audit_mesh(mesh)
            self.assertEqual(report.boundary_edges, 0)
            self.assertEqual(report.nonmanifold_edges, 0)
            self.assertGreater(report.volume_m3, 0)
        # Independent dimensions and a scaled, grounded alternative are kept.
        reduced = make_liberty_study(scale=.5, include_pedestal=False,
                                     detailed_drapery=True)
        self.assertLess(reduced.bounds_min[2], .05)
        with self.assertRaises(LibertyError):
            make_liberty_study(detailed_drapery=1)

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
