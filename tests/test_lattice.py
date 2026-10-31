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


if __name__ == "__main__":
    unittest.main()
