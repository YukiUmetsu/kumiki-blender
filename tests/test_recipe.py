import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.recipe import RecipeError, load_recipe


FIXTURE = Path(__file__).resolve().parents[1] / "examples/lattice-observation-tower.json"


class RecipeTests(unittest.TestCase):
    def test_valid_example(self):
        recipe = load_recipe(FIXTURE)
        self.assertEqual(recipe.asset_id, "lattice-observation-tower")
        self.assertEqual(recipe.levels, 16)
        self.assertEqual(recipe.height_m, 120.0)
        self.assertEqual(recipe.preview_size, (960, 540))

    def write_recipe(self, change):
        raw = json.loads(FIXTURE.read_text())
        change(raw)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        destination = Path(temp.name) / "recipe.json"
        destination.write_text(json.dumps(raw))
        return destination

    def test_unknown_field_is_rejected(self):
        path = self.write_recipe(lambda obj: obj.update({"surprise": "ignored?"}))
        with self.assertRaisesRegex(RecipeError, "surprise"):
            load_recipe(path)

    def test_non_finite_height_is_rejected(self):
        path = self.write_recipe(lambda obj: obj["geometry"].update({"height_m": float("nan")}))
        with self.assertRaisesRegex(RecipeError, "finite"):
            load_recipe(path)

    def test_negative_and_bool_dimensions_are_rejected(self):
        for value in (-1, True, 0):
            with self.subTest(value=value):
                path = self.write_recipe(lambda obj: obj["geometry"].update({"base_span_m": value}))
                with self.assertRaises(RecipeError):
                    load_recipe(path)

    def test_too_many_levels_are_rejected(self):
        path = self.write_recipe(lambda obj: obj["geometry"].update({"levels": 10000}))
        with self.assertRaisesRegex(RecipeError, "levels"):
            load_recipe(path)

    def test_top_span_must_be_smaller_than_base(self):
        path = self.write_recipe(lambda obj: obj["geometry"].update({"top_span_m": 45}))
        with self.assertRaisesRegex(RecipeError, "top_span_m"):
            load_recipe(path)

    def test_unsafe_asset_id_is_rejected(self):
        path = self.write_recipe(lambda obj: obj.update({"asset_id": "../escape"}))
        with self.assertRaisesRegex(RecipeError, "asset_id"):
            load_recipe(path)

    def test_invalid_preview_resolution_is_rejected(self):
        path = self.write_recipe(lambda obj: obj["preview"].update({"width": 20000}))
        with self.assertRaisesRegex(RecipeError, "width"):
            load_recipe(path)

    def test_invalid_hex_color_is_rejected(self):
        path = self.write_recipe(lambda obj: obj["material"].update({"steel_color": "red"}))
        with self.assertRaisesRegex(RecipeError, "steel_color"):
            load_recipe(path)


if __name__ == "__main__":
    unittest.main()
