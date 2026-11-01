import unittest
from pathlib import Path

from kumiki_blender.recipe import load_recipe
from kumiki_blender.portico import make_portico_study
from kumiki_blender.shots import plan_capital_shot, plan_portico_shots, plan_quality_shots


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

    def test_hero_and_elevations_have_full_height_for_wide_previews(self):
        recipe = load_recipe(FIXTURE)
        # Conservative vertical coverage, including ~15% antenna above the
        # nominal tower and a below-ground structural tolerance.
        top = recipe.height_m * 1.15
        for shot in plan_quality_shots(recipe)[:3]:
            vertical_span = shot.ortho_scale / (16 / 9)
            self.assertGreater(shot.target[2] + vertical_span / 2, top)
            self.assertLess(shot.target[2] - vertical_span / 2, -1.0)

    def test_portico_wide_views_cover_entire_facade(self):
        study = make_portico_study(bays=3)
        shots = plan_portico_shots(study.bounds_min, study.bounds_max)
        self.assertEqual([s.name for s in shots], ["hero", "front", "detail"])
        width = study.bounds_max[0] - study.bounds_min[0]
        height = study.bounds_max[2] - study.bounds_min[2]
        for shot in shots[:2]:
            self.assertGreater(shot.ortho_scale, width * 1.2)
            self.assertGreater(shot.ortho_scale / (16 / 9), height * 1.2)

    def test_pointed_arch_shots_cover_elevated_cornice_even_for_one_bay(self):
        for bays in (1, 2, 3):
            study = make_portico_study(bays=bays, arch_style="pointed", balustrade=True)
            shots = plan_portico_shots(study.bounds_min, study.bounds_max)
            for shot in shots[:2]:
                visible_half_height = shot.ortho_scale / (16/9) / 2
                self.assertGreater(shot.target[2] + visible_half_height,
                                   study.bounds_max[2])
                self.assertLess(shot.target[2] - visible_half_height,
                                study.bounds_min[2])

    def test_capital_closeup_tracks_a_real_center_column_for_odd_and_even_bays(self):
        for bays, expected_x in ((2, 0.0), (3, -2.1), (5, -2.1)):
            shot = plan_capital_shot(bays)
            self.assertEqual(shot.name, "capital")
            self.assertAlmostEqual(shot.target[0], expected_x)
            self.assertLess(shot.ortho_scale, 3.0)
        with self.assertRaises(ValueError):
            plan_capital_shot(0)


if __name__ == "__main__":
    unittest.main()
