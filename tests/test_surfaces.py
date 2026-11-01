import unittest

from kumiki_blender.surfaces import PBRSurface, SurfaceError, get_surface


class SurfaceTests(unittest.TestCase):
    def test_painted_metal_and_uncoated_metal_have_distinct_physical_response(self):
        painted = get_surface("painted-steel")
        raw = get_surface("galvanized-steel")
        self.assertLess(painted.metallic, raw.metallic)
        self.assertNotEqual(painted.roughness, raw.roughness)

    def test_glass_has_transmission_and_realistic_ior(self):
        glass = get_surface("optical-glass")
        self.assertGreater(glass.transmission, 0.8)
        self.assertTrue(1.3 < glass.ior < 1.7)

    def test_override_does_not_mutate_shared_preset(self):
        original = get_surface("painted-steel")
        changed = get_surface("painted-steel", color="#AA4433", roughness=0.66)
        self.assertEqual(changed.color, "#AA4433")
        self.assertEqual(changed.roughness, 0.66)
        self.assertEqual(get_surface("painted-steel"), original)

    def test_unknown_material_and_invalid_override_fail(self):
        with self.assertRaises(SurfaceError):
            get_surface("alien-metal")
        with self.assertRaises(SurfaceError):
            get_surface("painted-steel", roughness=float("nan"))

    def test_weathered_stone_strength_is_configurable_and_presets_are_immutable(self):
        base = get_surface("weathered-stone")
        adjusted = get_surface("weathered-stone", weathering_strength=0.83)
        self.assertEqual(adjusted.weathering_strength, 0.83)
        self.assertEqual(get_surface("weathered-stone"), base)
        self.assertNotEqual(adjusted, base)
        for spec in (-0.1, 1.2, True, float("inf")):
            with self.subTest(spec=spec), self.assertRaises(SurfaceError):
                get_surface("weathered-stone", weathering_strength=spec)
        with self.assertRaises(SurfaceError):
            get_surface("natural-stone", weathering_strength=0.5)


if __name__ == "__main__":
    unittest.main()
