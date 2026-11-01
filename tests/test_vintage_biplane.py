"""Reference-based geometry and useful control rig contracts for the red biplane."""

import contextlib
import io
import json
import math
import unittest

from kumiki_blender.architecture_geometry import audit_mesh
from kumiki_blender.cli import main
from kumiki_blender.vintage_biplane import make_biplane, animation_keyframes


class VintageBiplaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_biplane()

    def test_two_full_staggered_wings_and_reference_silhouette(self):
        model = self.model
        names = {part.name for part in model.parts}
        self.assertEqual(len(names), len(model.parts))
        self.assertTrue({"upper-left-wing", "upper-right-wing",
                         "lower-left-wing", "lower-right-wing"} <= names)
        self.assertGreater(len(model.parts), 130)
        self.assertAlmostEqual(model.bounds_min[0], -4.85, delta=.22)
        self.assertAlmostEqual(model.bounds_max[0], 4.85, delta=.22)
        self.assertLess(model.bounds_min[1], -3.4)
        self.assertGreater(model.bounds_max[1], 3.1)
        self.assertGreater(model.bounds_min[2], -.01, "aircraft cannot sink below its wheels")
        fuselage = next(p for p in model.parts if p.name == "fuselage-forward")
        self.assertGreater(max(v[2] for v in fuselage.mesh.vertices), 1.85)
        self.assertGreater(min(v[2] for v in fuselage.mesh.vertices), .90)
        upper = next(p for p in model.parts if p.name == "upper-left-wing")
        lower = next(p for p in model.parts if p.name == "lower-left-wing")
        self.assertLess(min(v[0] for v in upper.mesh.vertices), -4.65,
                        "fixed leading edge must run continuously to the rounded tip")
        self.assertLess(min(v[0] for v in lower.mesh.vertices), -4.55)
        movable = next(p for p in model.parts if p.name == "upper-left-aileron")
        self.assertGreater(min(v[0] for v in movable.mesh.vertices), -4.30,
                           "the rounded outer wing tip must remain fixed, not hinged")
        outboard_ribs = [p for p in model.parts if p.rig == "upper-left-aileron"
                         and p.group == "wing-rib"]
        self.assertTrue(outboard_ribs)
        self.assertTrue(all(min(v[0] for v in p.mesh.vertices) >=
                            min(v[0] for v in movable.mesh.vertices)-.03
                            for p in outboard_ribs), "no free-floating aileron rib after its tip")
        hinge = next(p for p in model.parts if p.name == "upper-left-aileron-hinge")
        self.assertGreater(min(v[0] for v in hinge.mesh.vertices), -4.30,
                           "an exposed hinge pin must stop at its actual control surface")
        self.assertGreater(min(v[2] for v in upper.mesh.vertices) -
                           max(v[2] for v in lower.mesh.vertices), .72)
        tail = next(p for p in model.parts if p.name == "tail-left-stabilizer")
        root_leading = min(v[1] for v in tail.mesh.vertices if -.2 < v[0] < -.09)
        tip_leading = min(v[1] for v in tail.mesh.vertices if v[0] < -1.65)
        self.assertGreater(tip_leading-root_leading, .22,
                           "reference tailplane has a visibly swept leading edge")

    def test_can_animate_aircraft_without_deforming_fixed_wings(self):
        model = self.model
        expected = {"propeller", "upper-left-aileron", "upper-right-aileron",
                    "lower-left-aileron", "lower-right-aileron",
                    "elevator", "rudder", "left-wheel", "right-wheel"}
        self.assertTrue(expected <= set(model.pivots))
        attached = {part.rig for part in model.parts if part.rig}
        self.assertTrue(expected <= attached)
        fixed = {part.rig for part in model.parts if part.name.endswith("-wing")}
        self.assertEqual(fixed, {None})
        for channel in expected:
            self.assertEqual(len(model.pivots[channel]), 3)
            self.assertTrue(all(math.isfinite(n) for n in model.pivots[channel]))
        self.assertAlmostEqual(model.pivots["upper-left-aileron"][1], -1.215, delta=.03)
        self.assertAlmostEqual(model.pivots["lower-left-aileron"][1], -1.067, delta=.03)
        self.assertAlmostEqual(model.pivots["upper-left-aileron"][2], 2.65)
        self.assertAlmostEqual(model.pivots["lower-left-aileron"][2], 1.55)
        self.assertAlmostEqual(model.pivots["rudder"][1], 3.49)

    def test_closed_3d_meshes_with_no_boundary_or_nonmanifold_edges(self):
        for part in self.model.parts:
            with self.subTest(name=part.name):
                report = audit_mesh(part.mesh)
                self.assertEqual(report.boundary_edges, 0)
                self.assertEqual(report.nonmanifold_edges, 0)
                self.assertGreater(report.volume_m3, 0)

    def test_actual_aircraft_details_not_just_primitive_silhouette(self):
        groups = self.model.part_counts
        self.assertGreaterEqual(groups["wing-rib"], 40)
        self.assertGreaterEqual(groups["flying-wire"], 16)
        self.assertGreaterEqual(groups["strut"], 12)
        self.assertEqual(groups["propeller-blade"], 2)
        self.assertGreaterEqual(groups["engine"], 6)
        self.assertGreaterEqual(groups["cockpit"], 5)
        names = {p.name for p in self.model.parts}
        self.assertIn("cockpit-left-side-fairing", names)
        self.assertIn("cockpit-right-side-fairing", names)
        self.assertIn("windscreen-glass", names)
        self.assertIn("engine-left-access-panel", names)
        self.assertIn("engine-right-access-panel", names)
        self.assertIn("engine-left-louver-00", names)
        self.assertIn("cockpit-instrument-dial-0", names)
        self.assertEqual(groups["wheel"], 4)  # tires and red hubs

    def test_loopable_rig_animation_uses_opposing_ailerons_and_prop_spin(self):
        frames = animation_keyframes()
        self.assertEqual(frames["propeller"][0], (1, 0.0))
        self.assertEqual(frames["propeller"][-1], (97, math.tau * 8))
        self.assertEqual(frames["upper-left-aileron"][1][0], 25)
        self.assertAlmostEqual(frames["upper-left-aileron"][1][1],
                               -frames["upper-right-aileron"][1][1])
        self.assertEqual(frames["lower-left-aileron"], frames["upper-left-aileron"])
        self.assertEqual(frames["rudder"][0][1], 0.0)

    def test_cli_gives_model_inventory_without_blender(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["plan-biplane"])
        self.assertEqual(result, 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["asset_id"], "red-vintage-biplane")
        self.assertGreaterEqual(plan["named_parts"]["flying-wire"], 16)
        self.assertIn("rudder", plan["rig_controls"])


if __name__ == "__main__":
    unittest.main()
