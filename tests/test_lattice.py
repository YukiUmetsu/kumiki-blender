import math
import unittest
from pathlib import Path

from kumiki_blender.lattice import make_lattice_tower
from kumiki_blender.recipe import load_recipe


FIXTURE = Path(__file__).resolve().parents[1] / "examples/lattice-observation-tower.json"


class LatticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = load_recipe(FIXTURE)

    def test_generation_is_deterministic(self):
        first = make_lattice_tower(self.recipe)
        self.assertEqual(first, make_lattice_tower(self.recipe))
        self.assertEqual(len(first.beams), len(set(beam.name for beam in first.beams)))

    def test_structural_member_counts(self):
        plan = make_lattice_tower(self.recipe)
        groups = [beam.group for beam in plan.beams]
        self.assertEqual(groups.count("leg"), 4 * self.recipe.levels)
        self.assertEqual(groups.count("brace"), 8 * self.recipe.levels)
        self.assertEqual(groups.count("ring"), 4 * (self.recipe.levels + 1))
        self.assertEqual(len(plan.decks), 2)

    def test_primary_legs_are_tapered_and_symmetric(self):
        plan = make_lattice_tower(self.recipe)
        bases = [beam.start for beam in plan.beams if beam.group == "leg" and beam.name.endswith("-00")]
        tops = [beam.end for beam in plan.beams if beam.group == "leg" and beam.name.endswith(f"-{self.recipe.levels - 1:02d}")]
        self.assertEqual(len(bases), 4)
        self.assertEqual({(round(p[0], 3), round(p[1], 3)) for p in bases},
                         {(x, y) for x in (-21.0, 21.0) for y in (-21.0, 21.0)})
        self.assertEqual({(round(p[0], 3), round(p[1], 3)) for p in tops},
                         {(x, y) for x in (-5.0, 5.0) for y in (-5.0, 5.0)})

    def test_all_beams_have_finite_nonzero_geometry(self):
        plan = make_lattice_tower(self.recipe)
        for beam in plan.beams:
            with self.subTest(beam=beam.name):
                self.assertGreater(math.dist(beam.start, beam.end), 0.01)
                self.assertGreater(beam.radius, 0)
                self.assertTrue(all(math.isfinite(v) for v in beam.start + beam.end))

    def test_bounds_cover_observation_decks_and_antenna(self):
        plan = make_lattice_tower(self.recipe)
        self.assertLessEqual(plan.bounds_min[2], 0)
        self.assertGreater(plan.bounds_max[2], self.recipe.height_m)
        self.assertTrue(any(beam.group == "antenna" for beam in plan.beams))
        for deck in plan.decks:
            self.assertTrue(plan.bounds_min[2] <= deck.z - deck.thickness <= plan.bounds_max[2])
            self.assertLessEqual(deck.span_m / 2, plan.bounds_max[0])

    def test_each_cross_brace_bay_has_a_real_plate_and_fasteners(self):
        plan = make_lattice_tower(self.recipe)
        self.assertEqual(len(plan.gussets), 4 * self.recipe.levels)
        self.assertEqual(len(plan.fasteners), 4 * len(plan.gussets))
        self.assertEqual(len(set(gusset.name for gusset in plan.gussets)), len(plan.gussets))
        for gusset in plan.gussets:
            self.assertGreater(gusset.width, 0)
            self.assertGreater(gusset.height, 0)
            self.assertGreater(gusset.thickness, 0)
            u, v, n = gusset.axis_u, gusset.axis_v, gusset.normal
            self.assertAlmostEqual(sum(u[i] * v[i] for i in range(3)), 0, places=8)
            self.assertAlmostEqual(sum(u[i] * n[i] for i in range(3)), 0, places=8)
            self.assertAlmostEqual(sum(n[i] * v[i] for i in range(3)), 0, places=8)

    def test_gusset_fasteners_are_outside_the_plate_surface(self):
        plan = make_lattice_tower(self.recipe)
        for gusset in plan.gussets:
            bolts = [bolt for bolt in plan.fasteners if bolt.name.startswith(gusset.name + "-bolt-")]
            self.assertEqual(len(bolts), 4)
            for bolt in bolts:
                offset = tuple(bolt.start[i] - gusset.center[i] for i in range(3))
                normal_depth = sum(offset[i] * gusset.normal[i] for i in range(3))
                self.assertGreaterEqual(normal_depth, gusset.thickness / 2 - 0.001)

    def test_bounds_include_connection_geometry_and_beam_thickness(self):
        plan = make_lattice_tower(self.recipe)
        for beam in (*plan.beams, *plan.fasteners):
            for point in (beam.start, beam.end):
                for axis in range(3):
                    self.assertLessEqual(plan.bounds_min[axis], point[axis] - beam.radius + 1e-9)
                    self.assertGreaterEqual(plan.bounds_max[axis], point[axis] + beam.radius - 1e-9)
        for plate in plan.gussets:
            for axis in range(3):
                reach = sum(abs(direction[axis]) * half for direction, half in (
                    (plate.axis_u, plate.width / 2), (plate.axis_v, plate.height / 2),
                    (plate.normal, plate.thickness / 2)))
                self.assertLessEqual(plan.bounds_min[axis], plate.center[axis] - reach + 1e-9)
                self.assertGreaterEqual(plan.bounds_max[axis], plate.center[axis] + reach - 1e-9)


if __name__ == "__main__":
    unittest.main()
