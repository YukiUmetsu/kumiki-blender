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
ANATOMY_BASE = (Path(__file__).resolve().parents[1] /
                "assets/landmarks/christ_redeemer/head_anatomical_base_cc0.obj.gz")


class ChristRedeemerTests(unittest.TestCase):
    def test_sculpture_source_is_valid_python_with_required_workflows(self):
        syntax = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        definitions = {node.name for node in syntax.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertTrue({"build", "build_head_study", "face", "robe", "arms", "verify",
                         "views", "export", "main"} <= definitions)

    @unittest.skipUnless(os.environ.get("KUMIKI_BLENDER_BINARY"),
                         "Set KUMIKI_BLENDER_BINARY for the Blender-only face study")
    def test_head_anatomy_clay_study_is_independent_of_full_body(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "head-study.json"
            cmd = [os.environ["KUMIKI_BLENDER_BINARY"], "--background",
                   "--factory-startup", "--python-exit-code", "1",
                   "--python", str(SCRIPT), "--", "--head-study",
                   "--head-tilt", "34", "--verify", "--metrics", str(metrics)]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=75)
            self.assertEqual(process.returncode, 0, process.stdout[-1800:] +
                             process.stderr[-1800:])
            report = json.loads(metrics.read_text())
            self.assertEqual(report["study"], "head")
            self.assertEqual(report["head_tilt_deg"], 34)
            self.assertGreater(report["head_surface_vertices"], 30000)
            self.assertLess(report["mesh_count"], 30)

    @unittest.skipUnless(os.environ.get("KUMIKI_BLENDER_BINARY"),
                         "Set KUMIKI_BLENDER_BINARY for the Blender anatomical sculpt")
    def test_cc0_anatomical_quad_sculpt_stays_clay_only_and_multiview_ready(self):
        self.assertTrue(ANATOMY_BASE.is_file())
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "anatomical-study.json"
            command = [os.environ["KUMIKI_BLENDER_BINARY"], "--background",
                       "--factory-startup", "--python-exit-code", "1",
                       "--python", str(SCRIPT), "--", "--head-study",
                       "--face-method", "anatomical", "--verify",
                       "--metrics", str(metrics)]
            process = subprocess.run(command, capture_output=True, text=True, timeout=75)
            self.assertEqual(process.returncode, 0, process.stdout[-1800:] +
                             process.stderr[-1800:])
            report = json.loads(metrics.read_text())
            self.assertEqual(report["study"], "head")
            self.assertEqual(report["face_method"], "anatomical")
            self.assertGreater(report["head_surface_vertices"], 50000)
            self.assertGreater(report["head_width_m"], 2.5)
            self.assertGreater(report["head_depth_m"], 1.0)
            self.assertEqual(report["mesh_count"], 3)
            # A full 3D nose projects in front of the mid-cheek on the real
            # anatomical control cage. This is a topology regression bound,
            # not a claim of survey-fitted likeness.
            self.assertGreater(report["nose_advance_vs_cheek_mm"], 350)
            self.assertLess(report["nose_advance_vs_cheek_mm"], 650)

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
                                   29.4, delta=.015)
            self.assertTrue(all(value >= .90
                                for value in report["sleeve_outward_fraction"].values()))
            self.assertEqual(report["sleeve_nonmanifold_edges"], {"-1": 0, "1": 0})
            self.assertEqual(report["hand_nonmanifold_edges"], {"-1": 0, "1": 0})
            self.assertTrue(all(value > 1500
                                for value in report["fused_hand_vertices"].values()))
            self.assertTrue(all(3.10 < reach < 3.30
                                for reach in report["hand_reach_m"].values()))
            self.assertGreater(report["mesh_count"], 60)
            self.assertGreater(report["head_surface_vertices"], 5000)
            self.assertEqual(report["head_tilt_deg"], 22)
            self.assertGreater(report["vertex_count"], 15000)
            self.assertEqual(report["pedestal_sides"], 8)
            self.assertEqual(report["mosaic_images"], 3)
            self.assertGreaterEqual(report["uv_mesh_count"], 60)
            self.assertEqual(report["uv_mesh_count"], report["mesh_count"])
            portable = audit_glb(output, require_pbr_textures=True)
            self.assertEqual(portable["external_references"], 0)
            self.assertGreater(portable["meshes"], 60)
            with output.open("rb") as glb:
                glb.seek(12)
                json_size, kind = struct.unpack("<I4s", glb.read(8))
                self.assertEqual(kind, b"JSON")
                document = json.loads(glb.read(json_size).decode("utf-8"))
            self.assertGreaterEqual(len(document.get("images", [])), 3)
            self.assertGreaterEqual(len(document.get("textures", [])), 3)
            self.assertTrue(any("baseColorTexture" in mat.get("pbrMetallicRoughness", {})
                                for mat in document.get("materials", [])))
            self.assertTrue(any("metallicRoughnessTexture" in mat.get("pbrMetallicRoughness", {})
                                for mat in document.get("materials", [])))
            self.assertTrue(any("normalTexture" in mat for mat in document.get("materials", [])))
            self.assertTrue(any("TEXCOORD_0" in primitive.get("attributes", {})
                                for mesh in document["meshes"]
                                for primitive in mesh["primitives"]))


if __name__ == "__main__":
    unittest.main()
