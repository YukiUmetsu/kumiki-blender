"""Physical limestone relief is deterministic, manifold and reusable."""

import math
import unittest

from kumiki_blender.architecture_geometry import (
    GeometryError, audit_mesh, loft_elliptical_sections,
)
from kumiki_blender.egyptian_monuments import pyramid_casing_stone, sphinx_assembly
from kumiki_blender.rock_weathering import (
    erode_loft, rock_height_field, rough_bedrock_slab,
)


class RockWeatheringTests(unittest.TestCase):
    def test_geological_height_field_is_repeatable_and_stratified(self):
        positions = [rock_height_field(1.2, -0.5, i * .07, seed=42)
                     for i in range(40)]
        self.assertEqual(positions,
                         [rock_height_field(1.2, -0.5, i * .07, seed=42)
                          for i in range(40)])
        self.assertGreater(max(positions)-min(positions), .25)
        self.assertNotEqual(positions,
                            [rock_height_field(1.2, -0.5, i * .07, seed=43)
                             for i in range(40)])
        self.assertTrue(all(math.isfinite(p) and abs(p) < 1.1 for p in positions))

    def test_eroded_loft_is_closed_and_has_different_real_vertices(self):
        rings = ((0, 0, 0, .5, .4), (.7, 0, 0, .65, .5),
                 (1.4, 0, 0, .56, .44), (2.1, 0, 0, .3, .2))
        for axis in "xyz":
            with self.subTest(axis=axis):
                original = loft_elliptical_sections(
                    rings, axis=axis, radial_segments=48, subdivisions=4,
                    cross_section_exponent=3.5)
                changed = erode_loft(original, axis=axis, radial_segments=48,
                                     strength=.045, seed=6)
                self.assertEqual(changed,
                                 erode_loft(original, axis=axis, radial_segments=48,
                                            strength=.045, seed=6))
                self.assertEqual(changed.vertices[:48], original.vertices[:48])
                self.assertEqual(changed.vertices[-48:], original.vertices[-48:])
                self.assertNotEqual(changed.vertices[48:-48],
                                    original.vertices[48:-48])
                report = audit_mesh(changed)
                self.assertEqual((report.boundary_edges, report.nonmanifold_edges),
                                 (0, 0))
                self.assertGreater(report.volume_m3, 0)

    def test_topographic_bedrock_and_chipped_casing_are_watertight(self):
        rock = rough_bedrock_slab(12, 4, .6, roughness=.16, seed=31)
        self.assertGreater(len(rock.vertices), 500)
        top = [p[2] for p in rock.vertices if p[2] > .2]
        self.assertGreater(max(top)-min(top), .035)
        stone = pyramid_casing_stone(36, 35.1, 1, .2, .4,
                                     inset=.2, face_grid=5,
                                     erosion_strength=.10, seed=32)
        self.assertGreater(len(stone.vertices), 100)
        for mesh in (rock, stone):
            report = audit_mesh(mesh)
            self.assertEqual((report.boundary_edges, report.nonmanifold_edges),
                             (0, 0))
            self.assertGreater(report.volume_m3, 0)

    def test_sphinx_has_rough_anatomical_surfaces_not_smooth_loft(self):
        parts = {part.name: part.mesh for part in sphinx_assembly(prefix="rock")}
        torso = parts["rock-lion-torso"]
        head = parts["rock-face"]
        bedrock = parts["rock-bedrock"]
        self.assertGreater(len(torso.vertices), 3000)
        self.assertGreater(len(head.vertices), 2000)
        self.assertGreater(len(bedrock.vertices), 600)
        # No two adjacent strata share the precise same longitudinal profile.
        self.assertNotEqual(torso.vertices[96:192], torso.vertices[192:288])

    def test_invalid_erosion_settings_rejected(self):
        loft = loft_elliptical_sections(((0, 0, 0, .5, .4),
                                         (1, 0, 0, .4, .3)), radial_segments=16)
        for kwargs in ({"axis": "q", "radial_segments": 16, "strength": .1},
                       {"axis": "z", "radial_segments": 7, "strength": .1},
                       {"axis": "z", "radial_segments": 16, "strength": -1},
                       {"axis": "z", "radial_segments": 16,
                        "strength": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(GeometryError):
                erode_loft(loft, **kwargs)
        for kwargs in ({"roughness": -.1}, {"roughness": .4},
                       {"x_segments": True}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(GeometryError):
                rough_bedrock_slab(6, 3, .7, **kwargs)
        with self.assertRaises(GeometryError):
            pyramid_casing_stone(36, 35.1, 1, .1, .2, face_grid=True)
        with self.assertRaises(GeometryError):
            loft_elliptical_sections(((0, 0, 0, .5, .4),
                                      (1, 0, 0, .4, .3)),
                                     cross_section_exponent=1.0)


if __name__ == "__main__":
    unittest.main()
