import unittest

from kumiki_blender.render_profiles import select_render_profile


class RenderProfileTests(unittest.TestCase):
    def test_production_preserves_author_specifications(self):
        p = select_render_profile(1440, 900, 48, mode="production")
        self.assertEqual((p.width, p.height, p.samples), (1440, 900, 48))
        self.assertEqual(p.name, "production")

    def test_smoke_cuts_render_cost_while_preserving_aspect_ratio(self):
        wide = select_render_profile(960, 540, 24, mode="smoke")
        tall = select_render_profile(1440, 900, 48, mode="smoke")
        self.assertEqual((wide.width, wide.height, wide.samples), (640, 360, 8))
        self.assertEqual((tall.width, tall.height, tall.samples), (576, 360, 8))
        self.assertAlmostEqual(wide.width / wide.height, 16 / 9)

    def test_invalid_mode_and_dimensions_fail(self):
        for mode in ("surprise", "", "SMOKE "):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                select_render_profile(640, 360, 32, mode=mode)
        with self.assertRaises(ValueError):
            select_render_profile(-1, 360, 32, mode="smoke")
        with self.assertRaises(ValueError):
            select_render_profile(640, 360, 0, mode="production")


if __name__ == "__main__":
    unittest.main()
