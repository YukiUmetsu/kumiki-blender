"""Controlled polarization pairs are diagnostic specular evidence, not PBR roughness."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.polarized_qa import (
    PolarizedQAOptions, inspect_polarized_uv,
)
from kumiki_blender.reconstruction.references import ReferenceError


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PolarizedMaterialTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dense = self.root / "dense"
        folder = self.dense / "workspace"
        folder.mkdir(parents=True)
        obj, mtl, self.atlas = folder / "mesh.obj", folder / "mesh.mtl", folder / "stone.png"
        obj.write_text(
            "mtllib mesh.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl surface\nf 1/1 2/2 3/3 4/4\n")
        mtl.write_text("newmtl surface\nmap_Kd stone.png\n")
        Image.new("RGB", (96, 64), (110, 105, 100)).save(self.atlas)
        (self.dense / "dense-report.json").write_text(json.dumps({
            "status": "dense-mesh-candidate / review-required",
            "textured_obj": {"path": "workspace/mesh.obj", "sha256": _sha(obj)},
            "texture_atlases": [{"path": "workspace/stone.png", "sha256": _sha(self.atlas)}],
        }))
        self.captures = self.root / "captures"
        self.captures.mkdir()
        Image.new("L", (96, 64), 255).save(self.captures / "mask.png")
        self.recipe = self.captures / "surface-recipe.json"
        self.recipe.write_text(json.dumps({"schema_version": 1, "atlases": {
            self.atlas.name: {"regions": [{"id": "stone", "mask": "mask.png", "roughness": .8}]}
        }}))
        self.manifest = self.captures / "polarized.json"
        frames = []
        for i in range(3):
            parallel = Image.new("RGB", (96, 64), (100, 100, 100))
            cross = Image.new("RGB", (96, 64), (100, 100, 100))
            ImageDraw.Draw(parallel).rectangle((10, 8, 49, 55), fill=(190, 190, 190))
            # No reliable UV pair coverage in left strip.
            confidence = Image.new("L", (96, 64), 255)
            ImageDraw.Draw(confidence).rectangle((0, 0, 6, 63), fill=0)
            p, c, conf = f"parallel-{i}.png", f"cross-{i}.png", f"confidence-{i}.png"
            parallel.save(self.captures / p)
            cross.save(self.captures / c)
            confidence.save(self.captures / conf)
            frames.append({"id": f"view-{i}", "parallel": p, "cross": c,
                           "confidence": conf, "same_exposure": True,
                           "unchanged_lighting": True, "analyzer_rotation_degrees": 90})
        self.manifest.write_text(json.dumps({"schema_version": 1, "atlases": {
            self.atlas.name: frames}}))

    def test_polarized_candidate_is_diagnostic_and_source_is_unchanged(self):
        source_sha = _sha(self.atlas)
        target = self.root / "qa"
        result = inspect_polarized_uv(self.dense, self.recipe, self.manifest, target)
        self.assertFalse(result["roughness_measured"])
        self.assertFalse(result["distribution_allowed"])
        atlas = result["atlases"][self.atlas.name]
        self.assertEqual(atlas["capture_pair_count"], 3)
        self.assertGreater(atlas["p95_relative_specular_contrast"], .1)
        self.assertGreater(atlas["supported_fraction_of_mapped_material_uv"], .85)
        with Image.open(target / "atlas-01/specular-contrast.png") as contrast:
            self.assertGreater(contrast.getpixel((30, 30)), 80)
            self.assertEqual(contrast.getpixel((80, 30)), 0)
            self.assertEqual(contrast.getpixel((0, 30)), 0)
        with Image.open(target / "atlas-01/supported-uv.png") as coverage:
            self.assertEqual(coverage.getpixel((0, 30)), 0)
            self.assertEqual(coverage.getpixel((30, 30)), 255)
        self.assertEqual(_sha(self.atlas), source_sha)
        self.assertFalse((target / ".pairs-01.mmap").exists())
        for item in result["files"]:
            self.assertEqual(_sha(target / item["path"]), item["sha256"])

    def test_rejects_unsupported_capture_conditions_and_tampering(self):
        destination = self.root / "bad"
        data = json.loads(self.manifest.read_text())
        data["atlases"][self.atlas.name][0]["same_exposure"] = False
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ReferenceError, "equal exposure"):
            inspect_polarized_uv(self.dense, self.recipe, self.manifest, destination)
        self.assertFalse(destination.exists())
        data["atlases"][self.atlas.name][0]["same_exposure"] = True
        data["atlases"][self.atlas.name][1]["parallel"] = "parallel-0.png"
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ReferenceError, "distinct independent"):
            inspect_polarized_uv(self.dense, self.recipe, self.manifest, destination)
        data["atlases"][self.atlas.name][1]["parallel"] = "parallel-1.png"
        self.manifest.write_text(json.dumps(data))
        Image.new("RGB", (96, 64), (5, 6, 7)).save(self.atlas)
        with self.assertRaisesRegex(ReferenceError, "changed"):
            inspect_polarized_uv(self.dense, self.recipe, self.manifest, destination)

    def test_cli_limits_and_output_collision(self):
        with self.assertRaisesRegex(ReferenceError, "max_side"):
            inspect_polarized_uv(self.dense, self.recipe, self.manifest, self.root / "x",
                                 options=PolarizedQAOptions(max_side=30))
        output = self.root / "cli-result"
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["inspect-polarized-uv", "--dense-revision", str(self.dense),
                                   "--surface-recipe", str(self.recipe),
                                   "--captures", str(self.manifest),
                                   "--output", str(output)]), 0)
        self.assertIn("unmeasured-roughness", stream.getvalue())
        with self.assertRaisesRegex(ReferenceError, "unused output"):
            inspect_polarized_uv(self.dense, self.recipe, self.manifest, output)

    def test_invalid_negative_and_saturated_pairs_are_excluded(self):
        parallel = self.captures / "parallel-0.png"
        cross = self.captures / "cross-1.png"
        with Image.open(parallel) as image:
            mutable = image.copy()
        ImageDraw.Draw(mutable).rectangle((70, 5, 94, 60), fill=(255, 255, 255))
        mutable.save(parallel)
        with Image.open(cross) as image:
            mutable = image.copy()
        ImageDraw.Draw(mutable).rectangle((10, 8, 49, 55), fill=(245, 245, 245))
        mutable.save(cross)
        result = inspect_polarized_uv(self.dense, self.recipe, self.manifest,
                                      self.root / "rejected")
        atlas = result["atlases"][self.atlas.name]
        self.assertGreater(atlas["rejected_negative_pair_texels"], 0)
        self.assertGreater(atlas["rejected_saturated_pair_texels"], 0)
        with Image.open(self.root / "rejected/atlas-01/supported-uv.png") as mask:
            self.assertEqual(mask.getpixel((30, 30)), 0)
            self.assertEqual(mask.getpixel((80, 30)), 0)
            self.assertEqual(mask.getpixel((55, 30)), 255)

    def test_rejects_path_traversal_and_duplicate_json_keys(self):
        data = json.loads(self.manifest.read_text())
        data["atlases"][self.atlas.name][0]["cross"] = "../outside.png"
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ReferenceError, "relative"):
            inspect_polarized_uv(self.dense, self.recipe, self.manifest, self.root / "unsafe")
        self.assertFalse((self.root / "unsafe").exists())
        self.manifest.write_text('{"schema_version":1,"schema_version":1,"atlases":{}}')
        with self.assertRaisesRegex(ReferenceError, "Duplicate"):
            inspect_polarized_uv(self.dense, self.recipe, self.manifest, self.root / "dupe")


if __name__ == "__main__":
    unittest.main()
