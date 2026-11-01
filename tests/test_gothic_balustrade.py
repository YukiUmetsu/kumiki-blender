"""Geometry regression checks for pointed masonry and turned stone railings."""

import unittest

from kumiki_blender.architecture_details import balustrade_assembly, turned_stone_baluster
from kumiki_blender.architecture_geometry import (
    GeometryError, audit_mesh, pointed_arch_voussoirs, world_vertices,
)
from kumiki_blender.portico import PorticoError, make_portico_study


class GothicBalustradeTests(unittest.TestCase):
    def test_pointed_stones_are_discrete_solid_masonry_with_actual_apex(self):
        pieces = pointed_arch_voussoirs(1.64, 2.55, 0.32, 0.6)
        self.assertEqual(len(pieces), 11)
        for piece in pieces:
            report = audit_mesh(piece)
            self.assertEqual(report.boundary_edges, 0)
            self.assertEqual(report.nonmanifold_edges, 0)
            self.assertGreater(report.volume_m3, 0.05)
        apex = pieces[5]
        apex_y = max(v[1] for v in apex.vertices)
        self.assertGreater(apex_y, 2.55)
        self.assertTrue(any(abs(v[0]) < 1e-9 and abs(v[1] - apex_y) < 1e-7
                            for v in apex.vertices))
        # Nothing closes the opening below the apex, and the wide lower
        # space remains empty instead of using a planar arch-shaped decal.
        self.assertLess(min(v[0] for v in pieces[0].vertices), -1.9)
        self.assertGreater(max(v[0] for v in pieces[-1].vertices), 1.9)
        self.assertTrue(all(abs(x) > 1.0 for block in pieces
                            for x, y, _ in block.vertices if y < 1.0))

    def test_pointed_arch_even_count_and_without_joint_gaps(self):
        pieces = pointed_arch_voussoirs(1.2, 2.1, 0.24, 0.45,
                                        stone_count=10, joint_gap_m=0)
        self.assertEqual(len(pieces), 10)
        self.assertTrue(all(audit_mesh(piece).volume_m3 > 0 for piece in pieces))

    def test_pointed_arch_rejects_nonphysical_or_unsafe_parameters(self):
        for kwargs in ({"half_span": 0}, {"rise": 1.0}, {"rise": 30},
                       {"band_width": 2}, {"depth": -1}, {"stone_count": True},
                       {"stone_count": 65}, {"joint_gap_m": 0.2},
                       {"segments_per_stone": 1}):
            settings = dict(half_span=1.64, rise=2.55, band_width=0.32, depth=0.6)
            settings.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(GeometryError):
                pointed_arch_voussoirs(**settings)

    def test_lathed_balusters_have_curved_profiles_and_reused_geometry(self):
        mesh = turned_stone_baluster()
        self.assertGreater(len(mesh.vertices), 48 * 10)
        radii = {round((v[0]**2 + v[1]**2)**0.5, 4) for v in mesh.vertices}
        self.assertGreater(len(radii), 6)
        report = audit_mesh(mesh)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(report.nonmanifold_edges, 0)
        self.assertGreater(report.volume_m3, 0)
        items = balustrade_assembly(prefix="east-rail", length=3.3, baluster_count=7)
        self.assertEqual(len(items), 13)
        self.assertEqual(sum("-baluster-" in p.name for p in items), 7)
        self.assertEqual(len({id(p.mesh) for p in items if "-baluster-" in p.name}), 1)
        self.assertEqual(len({p.name for p in items}), len(items))
        x_positions = [p.origin[0] for p in items if "-baluster-" in p.name]
        self.assertEqual(x_positions, sorted(x_positions))
        self.assertAlmostEqual(x_positions[0], -x_positions[-1])
        for part in items:
            self.assertGreater(audit_mesh(part.mesh).volume_m3, 0)
            self.assertGreaterEqual(min(v[2] for v in world_vertices(part)), -0.01)

    def test_balustrade_validates_spacing(self):
        for kwargs in ({"length": 0}, {"height": 0}, {"depth": 0.03},
                       {"baluster_count": 100}, {"baluster_count": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(GeometryError):
                balustrade_assembly(prefix="test", **kwargs)
        for kwargs in ({"height": 0.1}, {"radius": 0},
                       {"height": 0.3, "radius": 0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(GeometryError):
                turned_stone_baluster(**kwargs)

    def test_assembly_switches_preserve_mesh_counts_and_style_distinction(self):
        default = make_portico_study(bays=2)
        gothic = make_portico_study(bays=2, arch_style="pointed", balustrade=True)
        counts = dict(gothic.part_counts)
        self.assertEqual(counts["arch_stones"], 22)
        self.assertEqual(counts["balusters"], 14)
        self.assertEqual(counts["balustrade_rails"], 4)
        self.assertEqual(counts["balustrade_posts"], 4)
        self.assertEqual(counts["balustrade_caps"], 4)
        self.assertEqual(len(gothic.instances), sum(counts.values()))
        self.assertGreater(gothic.bounds_max[2], default.bounds_max[2])
        self.assertIn("pointed-balustrade", gothic.asset_id)
        self.assertEqual(len({p.name for p in gothic.instances}), len(gothic.instances))
        self.assertTrue(all(audit_mesh(p.mesh).volume_m3 > 0 for p in gothic.instances))
        with self.assertRaises(PorticoError):
            make_portico_study(arch_style="baroque")
        with self.assertRaises(PorticoError):
            make_portico_study(balustrade=1)


if __name__ == "__main__":
    unittest.main()
