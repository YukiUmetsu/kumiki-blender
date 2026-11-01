"""Geometry QA for the sculpted, layered Liberty torch flame."""

import unittest

from kumiki_blender.architecture_geometry import audit_mesh
from kumiki_blender.liberty_torch_flame import make_layered_flame


class TorchFlameTests(unittest.TestCase):
    def test_flame_has_full_core_and_multiple_asymmetric_closed_tongues(self):
        parts = make_layered_flame()
        self.assertGreaterEqual(len(parts), 10)
        self.assertIn("Flame / swelling gilded core", parts)
        peaks = []
        for name, mesh in parts.items():
            with self.subTest(name=name):
                report = audit_mesh(mesh)
                self.assertEqual(report.boundary_edges, 0)
                self.assertEqual(report.nonmanifold_edges, 0)
                self.assertGreater(report.volume_m3, 0)
                peaks.append(max(v[2] for v in mesh.vertices))
        self.assertGreater(max(peaks)-min(peaks), .10)

    def test_gold_flame_fits_the_existing_torch_pose_not_spindly_rods(self):
        parts = make_layered_flame()
        all_vertices = [p for mesh in parts.values() for p in mesh.vertices]
        self.assertGreater(max(p[2] for p in all_vertices), 1.82)
        self.assertLess(max(p[2] for p in all_vertices), 1.88)
        self.assertGreater(max(p[0] for p in all_vertices)-min(p[0] for p in all_vertices), .24)
        self.assertGreater(max(p[1] for p in all_vertices)-min(p[1] for p in all_vertices), .24)


if __name__ == "__main__":
    unittest.main()
