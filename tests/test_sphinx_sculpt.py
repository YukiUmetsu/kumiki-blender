"""Sculpture body-source selection and transformation must stay predictable."""

import unittest

from kumiki_blender.architecture_geometry import (
    GeometryError, MeshInstance, audit_mesh, world_vertices,
)
from kumiki_blender.egyptian_monuments import make_monument_study, sphinx_assembly
from kumiki_blender.sphinx_sculpt import (
    combine_sculpture_sources, sphinx_integral_parts,
)


class SphinxSculptTests(unittest.TestCase):
    def test_selects_only_eight_lion_body_volumes(self):
        parts = sphinx_assembly(prefix="sphinx")
        sources = sphinx_integral_parts(parts)
        self.assertEqual(len(sources), 8)
        self.assertIn("sphinx-lion-torso", [part.name for part in sources])
        self.assertIn("sphinx-carved-neck", [part.name for part in sources])
        self.assertFalse(any("face" in part.name or "bedrock" in part.name
                             or "toe" in part.name for part in sources))
        self.assertTrue(all(part.material == "sphinx-limestone" for part in sources))

    def test_combined_world_coordinates_preserve_each_source_shell(self):
        scene = make_monument_study(kind="giza-study")
        sources = sphinx_integral_parts(scene.instances)
        combined = combine_sculpture_sources(sources)
        self.assertEqual(len(combined.vertices), sum(len(part.mesh.vertices) for part in sources))
        self.assertEqual(len(combined.faces), sum(len(part.mesh.faces) for part in sources))
        self.assertEqual(combined.vertices[0], world_vertices(sources[0])[0])
        report = audit_mesh(combined)
        # Topologically closed source shells, not yet a fused physical solid.
        self.assertEqual((report.boundary_edges, report.nonmanifold_edges), (0, 0))
        self.assertGreater(report.volume_m3, 10)
        self.assertTrue(all(vertex[1] < -20 for vertex in combined.vertices[:96]))

    def test_malformed_or_incomplete_sculpture_inputs_fail_closed(self):
        parts = sphinx_assembly(prefix="sphinx")
        with self.assertRaisesRegex(GeometryError, "Incomplete Sphinx"):
            sphinx_integral_parts(parts[:-1] if parts[-1].name.endswith("curved-tail")
                                   else tuple(part for part in parts
                                              if part.name != "sphinx-carved-neck"))
        with self.assertRaises(GeometryError):
            sphinx_integral_parts(parts, prefix="sphinx")
        with self.assertRaises(GeometryError):
            combine_sculpture_sources(())
        with self.assertRaises(GeometryError):
            sphinx_integral_parts(["not a mesh"])


if __name__ == "__main__":
    unittest.main()
