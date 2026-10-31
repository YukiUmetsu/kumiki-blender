"""Geometry-kernel contracts that run without Blender."""

import math
import unittest

from kumiki_blender.architecture_geometry import (
    GeometryError,
    arch_voussoirs,
    audit_mesh,
    extrude_profile,
    place_mesh,
    repeat_linear,
    revolve_profile,
    semicircular_arch,
    world_vertices,
)


class ArchitectureGeometryTests(unittest.TestCase):
    def test_concave_cornice_extrudes_as_a_closed_outward_mesh(self):
        # A stepped, concave stone molding profile, not a simple rectangle.
        profile = ((0, 0), (2, 0), (2, 0.2), (1.6, 0.2),
                   (1.6, 0.4), (1.2, 0.4), (1.2, 0.6), (0, 0.6))
        mesh = extrude_profile(profile, 0.75)
        report = audit_mesh(mesh)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(report.nonmanifold_edges, 0)
        self.assertAlmostEqual(report.volume_m3, 0.75 * 0.96, places=6)
        self.assertEqual(len(mesh.vertices), 16)
        # Changing winding must not change the world-space result.
        reversed_mesh = extrude_profile(tuple(reversed(profile)), 0.75)
        self.assertAlmostEqual(audit_mesh(reversed_mesh).volume_m3, report.volume_m3, places=6)

    def test_invalid_profiles_are_rejected_instead_of_creating_bad_meshes(self):
        for points in (
            ((0, 0), (1, 1), (0, 1), (1, 0)),  # Self-intersection.
            ((0, 0), (1, 0), (1, 0), (0, 1)),  # Duplicate edge.
            ((0, 0), (1, 0), (2, 0)),          # Collinear polygon.
            ((0, 0), (1, 0), (float("nan"), 1)),
        ):
            with self.subTest(points=points), self.assertRaises(GeometryError):
                extrude_profile(points, 1)
        with self.assertRaises(GeometryError):
            extrude_profile(((0, 0), (1, 0), (0, 1)), -2)

    def test_semicircular_arch_has_open_center_and_physical_depth(self):
        mesh = semicircular_arch(inner_radius=1.25, outer_radius=1.55, depth=0.4, segments=32)
        report = audit_mesh(mesh)
        self.assertEqual(report.boundary_edges, 0)
        self.assertAlmostEqual(report.volume_m3, math.pi * (1.55 ** 2 - 1.25 ** 2) * 0.5 * 0.4, delta=0.001)
        self.assertAlmostEqual(max(v[1] for v in mesh.vertices), 1.55)
        self.assertAlmostEqual(min(v[0] for v in mesh.vertices), -1.55)
        # No cap triangle should fill the void directly beneath the arch's apex.
        self.assertFalse(any(min(mesh.vertices[index][1] for index in f) < 0.01 and
                             max(mesh.vertices[index][1] for index in f) > 0.8 and
                             all(abs(mesh.vertices[index][0]) < 1 for index in f)
                             for f in mesh.faces))
        with self.assertRaises(GeometryError):
            semicircular_arch(inner_radius=2, outer_radius=1.5, depth=0.2)

    def test_lathed_column_detail_is_watertight_and_outward(self):
        # Radius/height points create stepped plinth, shaft and capital.
        profile = ((0.38, 0), (0.38, 0.1), (0.31, 0.2),
                   (0.22, 0.35), (0.22, 2.4), (0.32, 2.58), (0.39, 2.7))
        mesh = revolve_profile(profile, radial_segments=32)
        report = audit_mesh(mesh)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(report.nonmanifold_edges, 0)
        self.assertGreater(report.volume_m3, 0)
        self.assertAlmostEqual(max(p[2] for p in mesh.vertices), 2.7)
        self.assertEqual(len(mesh.vertices), 32 * len(profile))
        for invalid in (((0, 0), (0.2, 1)), ((1, 0), (1, 0)), ((1, 1), (1, 0))):
            with self.subTest(profile=invalid), self.assertRaises(GeometryError):
                revolve_profile(invalid)

    def test_voussoirs_have_real_seams_and_are_individually_closed(self):
        blocks = arch_voussoirs(1.25, 1.6, 0.55, stone_count=11, joint_gap_m=0.006)
        self.assertEqual(len(blocks), 11)
        self.assertTrue(all(audit_mesh(block).boundary_edges == 0 for block in blocks))
        total_volume = sum(audit_mesh(block).volume_m3 for block in blocks)
        full_volume = audit_mesh(semicircular_arch(1.25, 1.6, 0.55)).volume_m3
        self.assertLess(total_volume, full_volume)
        self.assertGreater(total_volume, full_volume * 0.97)
        with self.assertRaises(GeometryError):
            arch_voussoirs(1, 1.2, .3, stone_count=12, joint_gap_m=1.0)

    def test_repeated_mesh_instances_keep_shared_geometry_and_correct_poses(self):
        mesh = extrude_profile(((0, 0), (1, 0), (1, 1), (0, 1)), 0.2)
        instances = repeat_linear(mesh, prefix="window", count=3, step=(2.4, 0, 0),
                                  origin=(10, 0, 4), material="limestone")
        self.assertEqual([i.name for i in instances], ["window-000", "window-001", "window-002"])
        self.assertTrue(all(i.mesh is mesh for i in instances))
        self.assertAlmostEqual(world_vertices(instances[2])[0][0], 14.8)
        tilted = place_mesh(mesh, name="standing", origin=(0, 2, 0),
                            axis_x=(1, 0, 0), axis_y=(0, 0, 1), axis_z=(0, -1, 0))
        self.assertEqual(world_vertices(tilted)[1], (1, 2, 0))
        with self.assertRaises(GeometryError):
            place_mesh(mesh, name="bad", axis_x=(0, 0, 1), axis_y=(0, 0, 1))


if __name__ == "__main__":
    unittest.main()
