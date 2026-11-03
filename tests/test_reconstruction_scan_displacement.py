"""Verified high-density PLY surface relief projected onto UV texture sheets."""

import contextlib
import hashlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.scan_displacement import plan_scan_displacement
from kumiki_blender.reconstruction.texture_finish import TextureFinishSettings, finish_dense_textures
from kumiki_blender.reconstruction.udim import package_finished_udim


class ScanDisplacementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dense = self.root / "dense"
        workspace = self.dense / "workspace"
        workspace.mkdir(parents=True)
        self.scan = workspace / "high-surface.ply"
        vertices = []
        faces = []
        side = 13
        for y in range(side):
            for x in range(side):
                px, py = x / (side-1), y / (side-1)
                relief = .012 * math.exp(-((px-.5)**2 + (py-.5)**2)/.045)
                vertices.append((px, py, relief))
        for y in range(side-1):
            for x in range(side-1):
                i = y*side+x
                faces.append((i, i+1, i+side+1))
                faces.append((i, i+side+1, i+side))
        self.scan.write_text(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_indices\nend_header\n" +
            "".join(f"{x:.8f} {y:.8f} {z:.8f}\n" for x, y, z in vertices) +
            "".join(f"3 {a} {b} {c}\n" for a, b, c in faces))
        self.obj = workspace / "low-uv.obj"
        self.mtl = workspace / "low-uv.mtl"
        self.atlas = workspace / "low-uv_map_Kd.png"
        self.obj.write_text(
            f"mtllib {self.mtl.name}\n"
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "usemtl stone\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
        self.mtl.write_text(f"newmtl stone\nmap_Kd {self.atlas.name}\n")
        Image.new("RGB", (128, 128), (100, 120, 120)).save(self.atlas)
        self.report = self.dense / "dense-report.json"
        self.report.write_text(json.dumps({
            "status": "dense-mesh-candidate / artistic-and-rights-review-required",
            "mesh": {"ply": "workspace/high-surface.ply", "sha256": self.digest(self.scan),
                     "vertices": len(vertices), "faces": len(faces)},
            "textured_obj": {"path": "workspace/low-uv.obj", "sha256": self.digest(self.obj)},
            "texture_atlases": [{"path": "workspace/low-uv_map_Kd.png",
                                 "sha256": self.digest(self.atlas)}],
        }))

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_plan_bounds_coordinates_scale_and_source_hashes(self):
        target = self.root / "scan-bake-plan"
        plan = plan_scan_displacement(self.dense, target,
                                      meters_per_unit=1, max_offset_m=.02,
                                      max_side=128)
        self.assertFalse(target.exists())
        self.assertEqual(plan["displacement_scale_m"], .04)
        self.assertEqual(plan["uv_triangle_count"], 2)
        self.assertEqual(plan["atlas_dimensions"][self.atlas.name]["original"], [128, 128])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["plan-scan-displacement", "--dense-revision", str(self.dense),
                                   "--output", str(target), "--meters-per-unit", "1",
                                   "--max-side", "128"]), 0)
        with self.assertRaisesRegex(ReferenceError, "outside the dense revision"):
            plan_scan_displacement(self.dense, self.dense / "scan-bake",
                                   meters_per_unit=1)
        with self.assertRaisesRegex(ReferenceError, "measured conversion"):
            plan_scan_displacement(self.dense, target, meters_per_unit=0)
        self.scan.write_text(self.scan.read_text().replace("0.00000000 0.00000000", ".5 .5", 1))
        with self.assertRaisesRegex(ReferenceError, "changed after reconstruction"):
            plan_scan_displacement(self.dense, target, meters_per_unit=1)

    @unittest.skipUnless(os.environ.get("KUMIKI_BLENDER_EXECUTABLE"), "requires Blender 4.2+")
    def test_real_scan_bake_produces_16bit_measured_relief_and_pbr_consumes_it(self):
        target = self.root / "scan-bake"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["bake-scan-displacement", "--dense-revision", str(self.dense),
                                   "--output", str(target), "--meters-per-unit", "1",
                                   "--max-offset-m", ".02", "--max-side", "128",
                                   "--blender", os.environ["KUMIKI_BLENDER_EXECUTABLE"]]), 0)
        result = json.loads((target / "scan-displacement-report.json").read_text())
        self.assertFalse(result["distribution_allowed"])
        self.assertEqual(result["source_scan_sha256"], self.digest(self.scan))
        detail = result["atlases"][self.atlas.name]
        self.assertGreater(detail["valid_fraction"], .9)
        self.assertGreater(detail["signed_offset_m"]["p95_absolute"], .002)
        self.assertEqual(detail["inspection_flags"], [])
        self.assertEqual(self.digest(target / detail["height"]["path"]),
                         detail["height"]["sha256"])
        with Image.open(target / detail["height"]["path"]) as height:
            self.assertEqual(height.size, (128, 128))
            self.assertEqual(height.mode, "I;16")
            self.assertGreater(height.getpixel((64, 64)), 42000)
            self.assertLess(height.getpixel((3, 3)), 33600)
        with Image.open(target / detail["coverage"]["path"]) as mask:
            self.assertGreater(mask.getpixel((64, 64)), 0)
        with Image.open(target / detail["signed_preview"]["path"]) as preview:
            self.assertGreater(preview.getpixel((64, 64)), preview.getpixel((3, 3)))
        finished = self.root / "textured-height"
        record = finish_dense_textures(self.dense, finished,
                                       TextureFinishSettings(surface_recipe=str(
                                           target / "surface-recipe.example.json")))
        self.assertEqual(record["surface_maps"][0]["displacement_scale_m"], .04)
        tiled = package_finished_udim(finished, self.root / "tiled-height")
        self.assertEqual(tiled["material_tiles"]["stone"], 1001)
        self.assertTrue((self.root / "tiled-height" / "height.1001.png").is_file())

    @unittest.skipUnless(os.environ.get("KUMIKI_BLENDER_EXECUTABLE"), "requires Blender 4.2+")
    def test_disconnected_uv_surface_fails_without_creating_output(self):
        # Move the low mesh well outside the allowed physical offset. A wrong
        # scan must NOT become a fake neutral displacement surface.
        self.obj.write_text(self.obj.read_text().replace(" 0\nv ", " 1\nv ")
                            .replace("v 0 1 0\n", "v 0 1 1\n"))
        report = json.loads(self.report.read_text())
        report["textured_obj"]["sha256"] = self.digest(self.obj)
        self.report.write_text(json.dumps(report))
        target = self.root / "unregistered-scan-output"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["bake-scan-displacement", "--dense-revision", str(self.dense),
                                   "--output", str(target), "--meters-per-unit", "1",
                                   "--max-offset-m", ".02", "--max-side", "128",
                                   "--blender", os.environ["KUMIKI_BLENDER_EXECUTABLE"]]), 2)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob(".scan-displacement-*")), [])
