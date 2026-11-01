import unittest

from kumiki_blender.architecture_geometry import audit_mesh
from kumiki_blender.portico import PorticoError, make_portico_study


class PorticoTests(unittest.TestCase):
    def test_modular_portico_has_actual_fluted_shafts_and_ionic_capitals(self):
        study = make_portico_study(bays=3)
        counts = dict(study.part_counts)
        self.assertEqual(counts["arch_stones"], 33)
        self.assertEqual(counts["columns"], 4)
        self.assertEqual(counts["volutes"], 32)
        self.assertEqual(counts["column_bases"], 8)
        self.assertEqual(counts["column_capitals"], 16)
        self.assertEqual(sum(instance.name.endswith("-shaft") for instance in study.instances), 4)
        self.assertFalse(any("-flute-" in instance.name for instance in study.instances))
        shafts = [item.mesh for item in study.instances if item.name.endswith("-shaft")]
        capitals = [item.mesh for item in study.instances if item.name.endswith("-capital-echinus")]
        self.assertEqual(len({id(mesh) for mesh in shafts}), 1)
        self.assertEqual(len({id(mesh) for mesh in capitals}), 1)
        self.assertEqual(counts["cornices"], 1)
        self.assertEqual(counts["plinths"], 1)
        self.assertEqual(len(study.instances), sum(counts.values()))
        self.assertEqual(len({i.name for i in study.instances}), len(study.instances))
        self.assertGreater(study.bounds_max[2], 7)
        self.assertLess(study.bounds_min[0], -6)
        self.assertGreater(study.bounds_max[0], 6)

    def test_all_unique_meshes_are_closed_and_have_positive_volume(self):
        study = make_portico_study()
        unique_meshes = {id(i.mesh): i.mesh for i in study.instances}
        for mesh in unique_meshes.values():
            with self.subTest(mesh_vertices=len(mesh.vertices)):
                report = audit_mesh(mesh)
                self.assertEqual(report.boundary_edges, 0)
                self.assertEqual(report.nonmanifold_edges, 0)
                self.assertGreater(report.volume_m3, 0)

    def test_bay_count_changes_the_assembly_without_changing_primitives(self):
        few = make_portico_study(bays=2)
        many = make_portico_study(bays=5)
        self.assertEqual(dict(many.part_counts)["arch_stones"], 55)
        self.assertEqual(dict(few.part_counts)["columns"], 3)
        self.assertEqual(dict(many.part_counts)["columns"], 6)
        self.assertEqual(dict(many.part_counts)["volutes"], 48)
        self.assertGreater(many.bounds_max[0] - many.bounds_min[0],
                           few.bounds_max[0] - few.bounds_min[0])

    def test_invalid_bay_configurations_are_rejected(self):
        for invalid in (0, -1, True, 21):
            with self.subTest(invalid=invalid), self.assertRaises(PorticoError):
                make_portico_study(bays=invalid)

    def test_portico_order_selects_distinct_column_geometry(self):
        doric = make_portico_study(bays=2, order="doric")
        tuscan = make_portico_study(bays=2, order="tuscan")
        ionic = make_portico_study(bays=2, order="ionic")
        self.assertEqual(dict(doric.part_counts)["column_bases"], 0)
        self.assertEqual(dict(tuscan.part_counts)["column_bases"], 6)
        self.assertEqual(dict(ionic.part_counts)["volutes"], 24)
        self.assertEqual(dict(doric.part_counts)["volutes"], 0)
        self.assertLess(len(doric.instances), len(ionic.instances))
        self.assertNotEqual(doric.asset_id, ionic.asset_id)
        with self.assertRaises(PorticoError):
            make_portico_study(order="corinthian")


if __name__ == "__main__":
    unittest.main()
