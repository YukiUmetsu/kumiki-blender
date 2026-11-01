"""Regression coverage for the headless sculpture generator.

Set KUMIKI_BLENDER_BINARY to a Blender 4.2+ executable for the live GLB test.
The Python-only tests remain usable in regular CI.
"""

import ast
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from kumiki_blender.glb_audit import audit_glb


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/landmarks/christ_redeemer.py"


class ChristRedeemerTests(unittest.TestCase):
    def test_sculpture_source_is_valid_python_with_required_workflows(self):
        syntax = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        definitions = {node.name for node in syntax.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertTrue({"build", "face", "robe", "arms", "verify",
                         "views", "export", "main"} <= definitions)

    @unittest.skipUnless(os.environ.get("KUMIKI_BLENDER_BINARY"),
                         "Set KUMIKI_BLENDER_BINARY for the live Blender QA test")
    def test_headless_build_dimensions_normals_and_portable_glb(self):
        executable = os.environ["KUMIKI_BLENDER_BINARY"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "model.glb"
            metrics = root / "metrics.json"
            cmd = [executable, "--background", "--factory-startup",
                   "--python-exit-code", "1", "--python", str(SCRIPT), "--",
                   "--verify", "--output", str(output), "--metrics", str(metrics)]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stdout[-3000:] +
                             process.stderr[-3000:])
            self.assertIn("SCULPTURE_AUDIT", process.stdout)
            report = json.loads(metrics.read_text())
            self.assertAlmostEqual(report["pedestal_top"], report["robe_bottom"], places=3)
            self.assertAlmostEqual(report["bounds_max"][2] - report["bounds_min"][2],
                                   38.0, delta=.05)
            self.assertAlmostEqual(report["bounds_max"][0] - report["bounds_min"][0],
                                   28.0, delta=.7)
            self.assertTrue(all(value >= .90
                                for value in report["sleeve_outward_fraction"].values()))
            self.assertEqual(report["sleeve_nonmanifold_edges"], {"-1": 0, "1": 0})
            self.assertGreater(report["mesh_count"], 60)
            self.assertGreater(report["head_surface_vertices"], 5000)
            self.assertGreater(report["vertex_count"], 15000)
            self.assertEqual(report["pedestal_sides"], 6)
            self.assertEqual(report["mosaic_images"], 1)
            self.assertGreater(report["uv_mesh_count"], 70)
            portable = audit_glb(output)
            self.assertEqual(portable["external_references"], 0)
            self.assertGreater(portable["meshes"], 60)
            with output.open("rb") as glb:
                glb.seek(12)
                json_size, kind = struct.unpack("<I4s", glb.read(8))
                self.assertEqual(kind, b"JSON")
                document = json.loads(glb.read(json_size).decode("utf-8"))
            self.assertGreaterEqual(len(document.get("images", [])), 1)
            self.assertGreaterEqual(len(document.get("textures", [])), 1)
            self.assertTrue(any("baseColorTexture" in mat.get("pbrMetallicRoughness", {})
                                for mat in document.get("materials", [])))
            self.assertTrue(any("TEXCOORD_0" in primitive.get("attributes", {})
                                for mesh in document["meshes"]
                                for primitive in mesh["primitives"]))


if __name__ == "__main__":
    unittest.main()
