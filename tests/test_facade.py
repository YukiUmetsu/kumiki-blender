import unittest

from kumiki_blender.architecture_geometry import audit_mesh, world_vertices
from kumiki_blender.facade import FacadeError, make_facade_study


class FacadeStudyTests(unittest.TestCase):
    def test_default_study_contains_separate_craft_and_portable_names(self):
        study = make_facade_study()
        counts = dict(study.part_counts)
        self.assertEqual(study.bays, 3)
        self.assertGreater(counts['masonry'], 600)
        self.assertGreater(counts['roof'], 400)
        self.assertGreater(counts['door'], 20)
        self.assertEqual(counts['gutter'], 12)
        self.assertEqual(counts['window'], 60)
        self.assertEqual(sum(counts.values()), len(study.instances))
        self.assertEqual(len({part.name for part in study.instances}), len(study.instances))
        self.assertGreater(study.bounds_max[2], 10)
        self.assertLess(study.bounds_min[2], 0)

    def test_studies_scale_by_bay_without_scaling_detail_geometry(self):
        three, five = make_facade_study(bays=3), make_facade_study(bays=5)
        self.assertGreater(len(five.instances), len(three.instances))
        self.assertAlmostEqual(five.bounds_max[0] - three.bounds_max[0], 7.6)
        self.assertEqual(dict(three.part_counts)['door'], dict(five.part_counts)['door'])

    def test_sampled_parts_have_valid_manifold_geometry_inside_bounds(self):
        study = make_facade_study()
        for part in study.instances[::17]:
            report = audit_mesh(part.mesh)
            self.assertEqual((report.boundary_edges, report.nonmanifold_edges), (0, 0))
            self.assertGreater(report.volume_m3, 0)
            for point in world_vertices(part):
                for i in range(3):
                    self.assertGreaterEqual(point[i], study.bounds_min[i] - 1e-9)
                    self.assertLessEqual(point[i], study.bounds_max[i] + 1e-9)

    def test_invalid_bay_counts_fail(self):
        for count in (0, 2, 4, 8, True, -1, 3.5):
            with self.subTest(count=count), self.assertRaises(FacadeError):
                make_facade_study(bays=count)


if __name__ == '__main__':
    unittest.main()
