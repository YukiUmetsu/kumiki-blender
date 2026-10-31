import unittest
from pathlib import Path

from kumiki_blender.recipe import load_recipe
from kumiki_blender.shots import plan_quality_shots


FIXTURE = Path(__file__).resolve().parents[1] / "examples/lattice-observation-tower.json"


class ShotTests(unittest.TestCase):
    def test_four_named_views_include_a_close_up(self):
        shots = plan_quality_shots(load_recipe(FIXTURE))
        self.assertEqual([shot.name for shot in shots], ["hero", "front", "side", "detail"])
        self.assertLess(shots[-1].ortho_scale, shots[0].ortho_scale / 2)
        self.assertGreater(shots[-1].target[2], shots[1].target[2])

    def test_shot_plan_is_deterministic(self):
        recipe = load_recipe(FIXTURE)
        self.assertEqual(plan_quality_shots(recipe), plan_quality_shots(recipe))


if __name__ == "__main__":
    unittest.main()
