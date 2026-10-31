import unittest

from kumiki_blender.architecture_details import (
    Opening, box_mesh, cornice_extrusion, coursed_wall,
    dentil_row, door_assembly, gable_roof, window_assembly,
)
from kumiki_blender.architecture_geometry import GeometryError, audit_mesh, world_vertices


class ArchitecturalDetailTests(unittest.TestCase):
    def test_basic_solids_and_molding_are_closed_with_positive_volume(self):
        for mesh in (box_mesh(2, 0.4, 3), cornice_extrusion(12)):
            report = audit_mesh(mesh)
            self.assertEqual((report.boundary_edges, report.nonmanifold_edges), (0, 0))
            self.assertGreater(report.volume_m3, 0)

    def test_coursed_masonry_respects_a_real_opening_and_staggered_joints(self):
        hole = Opening(1.2, 2.8, 1.0, 3.0)
        stones = coursed_wall(width=4, height=4, thickness=0.4,
                              openings=(hole,), block_width=0.8, course_height=0.5)
        self.assertGreater(len(stones), 35)
        self.assertEqual(len(set(s.name for s in stones)), len(stones))
        x_by_course = {}
        for item in stones:
            corners = world_vertices(item)
            x0, x1 = min(v[0] for v in corners), max(v[0] for v in corners)
            z0, z1 = min(v[2] for v in corners), max(v[2] for v in corners)
            self.assertFalse(x0 < hole.right and x1 > hole.left
                             and z0 < hole.top and z1 > hole.bottom)
            x_by_course.setdefault(item.name.split('-')[1], []).append(x0)
            report = audit_mesh(item.mesh)
            self.assertEqual((report.boundary_edges, report.nonmanifold_edges), (0, 0))
        self.assertNotEqual(sorted(x_by_course['00']), sorted(x_by_course['01']))

    def test_rejects_invalid_masonry_openings_and_resource_limits(self):
        for holes in ((Opening(2, 1, 0, 1),), (Opening(-1, 2, 0, 1),),
                      (Opening(1, 2, 1, 2), Opening(1.5, 3, 1.5, 2.5))):
            with self.subTest(holes=holes), self.assertRaises(GeometryError):
                coursed_wall(width=4, height=4, thickness=.4, openings=holes)
        with self.assertRaises(GeometryError):
            coursed_wall(width=100, height=60, thickness=.4, block_width=.2, course_height=.15)

    def test_window_and_door_are_modular_and_manifold(self):
        window = window_assembly(prefix="window-test", columns=3, rows=2)
        door = door_assembly(prefix="door-test")
        self.assertGreaterEqual(len(window), 13)
        self.assertGreater(len(door), 20)
        self.assertIn('window-glass', {part.material for part in window})
        self.assertIn('door-timber', {part.material for part in door})
        self.assertIn('bronze-metal', {part.material for part in door})
        for part in window + door:
            report = audit_mesh(part.mesh)
            self.assertEqual((report.boundary_edges, report.nonmanifold_edges), (0, 0))
            self.assertGreater(report.volume_m3, 0)

    def test_roof_tiles_follow_real_slope_and_are_closed(self):
        parts = gable_roof(width=12, depth=7, eave_z=8, rise=2,
                           tile_rows=4, tile_columns=6)
        self.assertGreater(len(parts), 50)
        self.assertEqual(len([x for x in parts if 'tile' in x.name]), 48)
        front = next(x for x in parts if x.name == 'roof-front-tile-00-00')
        back = next(x for x in parts if x.name == 'roof-back-tile-00-00')
        self.assertGreater(front.axis_y[2], 0)
        self.assertLess(back.axis_y[2], 0)
        for part in parts:
            report = audit_mesh(part.mesh)
            self.assertEqual((report.boundary_edges, report.nonmanifold_edges), (0, 0))
            self.assertGreater(report.volume_m3, 0)

    def test_repetition_details_fail_on_colliding_teeth(self):
        self.assertEqual(len(dentil_row(prefix='dentil-test', length=4, count=8)), 8)
        with self.assertRaises(GeometryError):
            dentil_row(prefix='dentil-test', length=1, count=20, tooth_width=.2)
        with self.assertRaises(GeometryError):
            window_assembly(prefix='window-invalid', columns=7)
        with self.assertRaises(GeometryError):
            gable_roof(width=12, depth=7, eave_z=8, rise=2, tile_rows=30, tile_columns=100)


if __name__ == '__main__':
    unittest.main()
