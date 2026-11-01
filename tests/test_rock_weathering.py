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

    def test_differential_bedding_and_mineral_pits_are_real_closed_relief(self):
        source = loft_elliptical_sections(
            ((0, 0, 1.1, 1.1, .78), (.8, 0, 1.1, 1.1, .78),
             (1.6, 0, 1.1, 1.1, .78), (2.4, 0, 1.1, 1.1, .78)),
            axis="x", radial_segments=64, subdivisions=8)
        kwargs = dict(axis="x", radial_segments=64, strength=.025,
                      seed=40, strata_spacing=.27)
        gentle = erode_loft(source, **kwargs)
        fractured = erode_loft(source, bedding_strength=.075,
                               pitting_strength=.045, **kwargs)
        self.assertEqual(fractured,
                         erode_loft(source, bedding_strength=.075,
                                    pitting_strength=.045, **kwargs))
        self.assertEqual(fractured.vertices[:64], source.vertices[:64])
        self.assertEqual(fractured.vertices[-64:], source.vertices[-64:])
        differences = [math.dist(a, b) for a, b in zip(
            gentle.vertices[64:-64], fractured.vertices[64:-64])]
        self.assertGreater(max(differences), .045)
        self.assertGreater(sum(d > .004 for d in differences), 80)
        report = audit_mesh(fractured)
        self.assertEqual((report.boundary_edges, report.nonmanifold_edges), (0, 0))
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
        front_paw = parts["rock-left-forepaw"]
        self.assertGreater(len(torso.vertices), 3000)
        self.assertGreater(len(head.vertices), 2000)
        self.assertGreater(len(bedrock.vertices), 600)
        self.assertIn("rock-carved-neck", parts)
        # Keep the recumbent forelegs horizontal and the rock grounding low:
        # a tall pedestal and swollen triangular forelegs read like a toy.
        self.assertLess(max(v[2] for v in front_paw.vertices), 1.55)
        self.assertLess(max(v[2] for v in bedrock.vertices), .35)
        eye_z = 4.12 * (3.6/4.6)
        eyes = [v[0] for v in head.vertices if abs(v[2]-eye_z)<.052
                and .21<abs(v[1])<.34 and v[0]<-4.0]
        bridge = [v[0] for v in head.vertices if abs(v[2]-eye_z)<.052
                  and abs(v[1])<.07 and v[0]<-4.0]
        # Eye sockets need actual inward depth relative to the surviving
        # nasal bridge, not floating eyeball objects or shader-only darkness.
        socket_depth = min(eyes)-min(bridge)
        self.assertGreater(socket_depth, .08)
        # The revised taller anatomy has a more projecting surviving nasal
        # bridge, which increases bridge-to-orbit depth; keep the recess
        # bounded to avoid the old oversized punched-out eye cavities.
        self.assertLess(socket_depth, .24)
        # No two adjacent strata share the precise same longitudinal profile.
        self.assertNotEqual(torso.vertices[96:192], torso.vertices[192:288])

    def test_pyramid_stone_has_physical_pits_and_broken_arrises(self):
        stone = pyramid_casing_stone(36, 35.1, 1, .2, .4,
                                     inset=.27, face_grid=8,
                                     erosion_strength=.18, seed=31)
        cols, rows = 17, 9
        outside = stone.vertices[:cols*rows]
        inside = stone.vertices[cols*rows:]
        face_departures = [
            outside[j*cols+i][1] - (18*(1-j/8) + 17.55*j/8)
            for j in range(rows) for i in range(cols)
        ]
        self.assertGreater(max(face_departures)-min(face_departures), .09)
        # Outer boundary is not identical to the perfectly straight inner
        # quarry wedge. This is actual geometry, not a normal-map illusion.
        self.assertGreater(max(outside[j*cols][0] - inside[j*cols][0]
                               for j in range(rows)), .025)
        report = audit_mesh(stone)
        self.assertEqual((report.boundary_edges, report.nonmanifold_edges), (0, 0))

    def test_invalid_erosion_settings_rejected(self):
        loft = loft_elliptical_sections(((0, 0, 0, .5, .4),
                                         (1, 0, 0, .4, .3)), radial_segments=16)
        for kwargs in ({"axis": "q", "radial_segments": 16, "strength": .1},
                       {"axis": "z", "radial_segments": 7, "strength": .1},
                       {"axis": "z", "radial_segments": 16, "strength": -1},
                       {"axis": "z", "radial_segments": 16,
                        "strength": float("nan")},
                       {"axis": "z", "radial_segments": 16,
                        "strength": .1, "bedding_strength": -.01},
                       {"axis": "z", "radial_segments": 16,
                        "strength": .1, "pitting_strength": float("inf")}):
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
