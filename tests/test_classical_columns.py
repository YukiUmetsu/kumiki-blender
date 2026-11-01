import math
import unittest

from kumiki_blender.architecture_details import (
    classical_column, fluted_column_shaft, volute_scroll,
)
from kumiki_blender.architecture_geometry import GeometryError, audit_mesh, world_vertices


class ClassicalColumnTests(unittest.TestCase):
    def test_true_fluted_geometry_and_subtle_entasis(self):
        mesh = fluted_column_shaft(height=3.5, base_radius=0.36, top_radius=0.31,
                                   flutes=24, flute_depth=0.020, entasis=0.012)
        report = audit_mesh(mesh)
        self.assertGreater(report.volume_m3, 0.7)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(report.nonmanifold_edges, 0)
        ring = 24 * 8
        base = [math.hypot(x, y) for x, y, _ in mesh.vertices[:ring]]
        self.assertGreater(max(base)-min(base), 0.019)
        # A shaft with no flutes should be circular at any one Z ring.
        smooth = fluted_column_shaft(flutes=0, flute_depth=0)
        self.assertAlmostEqual(math.hypot(*smooth.vertices[0][:2]),
                               math.hypot(*smooth.vertices[13][:2]), places=9)
        # Entasis is a real bulge relative to the linear taper at mid-height.
        entasis_mesh = fluted_column_shaft(flutes=0, flute_depth=0,
                                           base_radius=0.36, top_radius=0.31, entasis=0.014)
        mid = entasis_mesh.vertices[4*48]
        self.assertGreater(math.hypot(mid[0], mid[1]), 0.336)

    def test_three_orders_have_distinct_editable_components(self):
        for order, base_count, volutes in (("tuscan", 2, 0), ("doric", 0, 0), ("ionic", 2, 4)):
            with self.subTest(order=order):
                pieces = classical_column(prefix="col1", order=order)
                self.assertEqual(sum("-base-" in item.name for item in pieces), base_count)
                self.assertEqual(sum("-volute-front-" in item.name or "-volute-back-" in item.name
                                     for item in pieces), volutes)
                self.assertEqual(sum("-volute-backing-" in item.name for item in pieces), volutes)
                self.assertEqual(sum(item.name.endswith("-shaft") for item in pieces), 1)
                self.assertEqual(len({item.name for item in pieces}), len(pieces))
                for item in pieces:
                    report = audit_mesh(item.mesh)
                    self.assertEqual(report.boundary_edges, 0)
                    self.assertEqual(report.nonmanifold_edges, 0)
                    self.assertGreater(report.volume_m3, 0)
                    self.assertTrue(all(-1.5 < vertex[0] < 1.5 for vertex in world_vertices(item)))
                xs = [vertex[0] for item in pieces for vertex in world_vertices(item)]
                self.assertAlmostEqual(max(xs), -min(xs), places=7)

    def test_volute_is_closed_3d_and_thinner_than_its_turn_spacing(self):
        scroll = volute_scroll(radius=0.18, tube_radius=0.022)
        report = audit_mesh(scroll)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(report.nonmanifold_edges, 0)
        self.assertGreater(report.volume_m3, 0)
        self.assertGreater(max(vertex[1] for vertex in scroll.vertices), 0.02)
        self.assertLess(min(vertex[1] for vertex in scroll.vertices), -0.02)

    def test_rejects_implausible_and_unsafe_settings(self):
        for options in ({"flutes": True}, {"flutes": 7}, {"flutes": 33},
                        {"flute_depth": 0.1}, {"height": float("nan")},
                        {"entasis": 0.18}):
            with self.subTest(options=options), self.assertRaises(GeometryError):
                fluted_column_shaft(**options)
        for options in ({"tube_radius": 0.079, "radius": 0.08},
                        {"segments": 14}, {"turns": 0.1}):
            with self.subTest(options=options), self.assertRaises(GeometryError):
                volute_scroll(**options)
        for options in ({"order": "corinthian"}, {"height": 1.0},
                        {"base_radius": 0.9, "height": 4.0}):
            with self.subTest(options=options), self.assertRaises(GeometryError):
                classical_column(prefix="col1", **options)

    def test_large_but_valid_order_scales_without_exceeding_volute_bounds(self):
        parts = classical_column(prefix="large-order", order="ionic",
                                 height=24.0, base_radius=2.5)
        self.assertEqual(sum("-volute-front-" in part.name for part in parts), 2)
        self.assertTrue(all(audit_mesh(part.mesh).volume_m3 > 0 for part in parts))


if __name__ == "__main__":
    unittest.main()
