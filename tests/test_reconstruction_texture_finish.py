"""Reproducible UV texture corrections without altering geometry or source photographs."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.blender_dense import plan_dense_blender_review
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.texture_finish import (
    TextureFinishSettings, _atlas_correction, _sample_luminance, finish_dense_textures,
)


class TextureFinishTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dense = self.root / "dense"
        workspace = self.dense / "workspace"
        workspace.mkdir(parents=True)
        self.ply = workspace / "mesh.ply"
        self.ply.write_bytes(
            b"ply\nformat ascii 1.0\nelement vertex 101\n"
            b"element face 120\nend_header\n" + b"first party scan"
        )
        base = "scene_dense_mesh_refined_texture"
        self.obj = workspace / (base + ".obj")
        self.mtl = workspace / (base + ".mtl")
        self.atlas = workspace / (base + "_material_00_map_Kd.png")
        self.obj.write_text(f"mtllib {self.mtl.name}\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
                            "vt 0 0\nvt 1 0\nvt 0 1\nusemtl m0\nf 1/1 2/2 3/3\n")
        self.mtl.write_text(f"newmtl m0\nmap_Kd {self.atlas.name}\n")
        with Image.new("RGB", (48, 32), (42, 70, 91)) as img:
            img.paste((0, 0, 0), (0, 0, 12, 32))
            img.save(self.atlas)
        report = {
            "status": "dense-mesh-candidate / artistic-and-rights-review-required",
            "mesh": {"ply": str(self.ply.relative_to(self.dense)),
                     "sha256": self._digest(self.ply), "vertices": 101, "faces": 120},
            "textured_obj": {"path": str(self.obj.relative_to(self.dense)),
                             "sha256": self._digest(self.obj)},
            "texture_atlases": [{"path": str(self.atlas.relative_to(self.dense)),
                                 "sha256": self._digest(self.atlas)}],
        }
        (self.dense / "dense-report.json").write_text(json.dumps(report))

    @staticmethod
    def _digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_corrected_atlas_remains_pinned_importable_and_original_is_unchanged(self):
        original = {path: path.read_bytes() for path in (self.obj, self.mtl, self.atlas, self.ply)}
        output = self.root / "finished"
        report = finish_dense_textures(self.dense, output,
                                        TextureFinishSettings(exposure_ev=0.85, gamma=1.2,
                                                              saturation=1.1, shadow_lift=0.05,
                                                              neutralize_cast=True))
        self.assertEqual(report["status"], "candidate / corrected-atlas-artistic-review-required")
        self.assertFalse(report["distribution_allowed"])
        self.assertEqual(len(report["texture_atlases"]), 1)
        self.assertTrue((output / report["texture_atlases"][0]["comparison_preview"]).is_file())
        self.assertEqual(self._digest(output / self.obj.name), self._digest(self.obj))
        self.assertEqual(self._digest(output / self.mtl.name), self._digest(self.mtl))
        with Image.open(self.atlas) as raw, Image.open(output / self.atlas.name) as modified:
            self.assertEqual(modified.size, raw.size)
            self.assertEqual(modified.getpixel((3, 3)), (0, 0, 0))
            self.assertGreater(sum(modified.getpixel((30, 20))), sum(raw.getpixel((30, 20))))
        for path, data in original.items():
            self.assertEqual(path.read_bytes(), data)
        plan = plan_dense_blender_review(self.dense, self.root / "blender", textured=True,
                                         texture_finish=output)
        self.assertEqual(plan["source_obj"], str(output / self.obj.name))
        self.assertEqual(plan["texture_finish_revision"], str(output))
        self.assertFalse(plan["texture_quality_verified"])
        with self.assertRaisesRegex(ReferenceError, "requires --textured"):
            plan_dense_blender_review(self.dense, self.root / "blender", texture_finish=output)
        (output / self.atlas.name).write_bytes(b"untrusted altered texture")
        with self.assertRaisesRegex(ReferenceError, "Texture atlas changed"):
            plan_dense_blender_review(self.dense, self.root / "blender", textured=True,
                                      texture_finish=output)

    def test_source_mutations_and_settings_are_rejected_before_output(self):
        for settings in (TextureFinishSettings(gamma=0),
                         TextureFinishSettings(exposure_ev=float("nan")),
                         TextureFinishSettings(protect_black=True),
                         TextureFinishSettings(max_balance_ev=-1)):
            with self.assertRaises(ReferenceError):
                finish_dense_textures(self.dense, self.root / "unfinished", settings)
            self.assertFalse((self.root / "unfinished").exists())
        source_report = self.dense / "dense-report.json"
        record = json.loads(source_report.read_text())
        record["texture_atlases"][0]["sha256"] = "a" * 64
        source_report.write_text(json.dumps(record))
        with self.assertRaisesRegex(ReferenceError, "Texture atlas changed"):
            finish_dense_textures(self.dense, self.root / "unfinished", TextureFinishSettings())

    def test_cli_color_correction_arguments(self):
        target = self.root / "cli-finished"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["finish-dense-textures", "--dense-revision", str(self.dense),
                                   "--output", str(target), "--exposure-ev", "0.75",
                                   "--shadow-lift", "0.12", "--neutralize-cast"]), 0)
        settings = json.loads((target / "texture-finish-report.json").read_text())["settings"]
        self.assertEqual(settings["exposure_ev"], 0.75)
        self.assertEqual(settings["shadow_lift"], 0.12)
        self.assertTrue(settings["neutralize_cast"])
        with patch("kumiki_blender.cli.shutil.which", return_value="/bin/blender"), \
             patch("kumiki_blender.cli.subprocess.run") as run:
            run.return_value.returncode = 0
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["render-dense-blender-review", "--dense-revision", str(self.dense),
                                       "--texture-finish", str(target), "--textured",
                                       "--output", str(self.root / "review"), "--no-renders"]), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--texture-finish") + 1], str(target))

    def test_transparent_png_atlas_alpha_is_preserved(self):
        with Image.new("RGBA", (48, 32), (50, 60, 70, 255)) as original:
            original.putpixel((2, 3), (0, 0, 0, 0))
            original.save(self.atlas)
        report_path = self.dense / "dense-report.json"
        report = json.loads(report_path.read_text())
        report["texture_atlases"][0]["sha256"] = self._digest(self.atlas)
        report_path.write_text(json.dumps(report))
        output = self.root / "alpha-finished"
        finish_dense_textures(self.dense, output, TextureFinishSettings(gamma=1.2))
        with Image.open(output / self.atlas.name) as corrected:
            self.assertEqual(corrected.mode, "RGBA")
            self.assertEqual(corrected.getpixel((2, 3))[3], 0)
            self.assertEqual(corrected.getpixel((30, 10))[3], 255)

    def test_multiple_atlas_pages_are_exposure_balanced_without_changing_uvs(self):
        second = self.atlas.parent / "scene_dense_mesh_refined_texture_material_01_map_Kd.png"
        Image.new("RGB", (48, 32), (155, 155, 155)).save(second)
        self.mtl.write_text(self.mtl.read_text() +
                            f"newmtl m1\nmap_Kd {second.name}\n")
        report_path = self.dense / "dense-report.json"
        report = json.loads(report_path.read_text())
        report["texture_atlases"].append({"path": str(second.relative_to(self.dense)),
                                         "sha256": self._digest(second)})
        report_path.write_text(json.dumps(report))
        destination = self.root / "balanced"
        result = finish_dense_textures(self.dense, destination,
                                       TextureFinishSettings(balance_atlases=True,
                                                             max_balance_ev=0.9))
        self.assertEqual(len(result["texture_atlases"]), 2)
        values = {item["path"]: item["atlas_balance_ev"] for item in result["texture_atlases"]}
        self.assertGreater(values[self.atlas.name], 0)
        self.assertLess(values[second.name], 0)
        with Image.open(self.atlas) as dark_raw, Image.open(second) as bright_raw, \
             Image.open(destination / self.atlas.name) as dark_finished, \
             Image.open(destination / second.name) as bright_finished:
            raw_gap = abs(sum(dark_raw.getpixel((30, 12))) - sum(bright_raw.getpixel((30, 12))))
            balanced_gap = abs(sum(dark_finished.getpixel((30, 12))) -
                               sum(bright_finished.getpixel((30, 12))))
            self.assertLess(balanced_gap, raw_gap)
        planned = plan_dense_blender_review(self.dense, self.root / "review", textured=True,
                                            texture_finish=destination)
        self.assertEqual(len(planned["texture_atlases"]), 2)

    def test_atlas_balance_excludes_transparent_white_padding(self):
        transparent = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        transparent.paste((80, 80, 80, 255), (16, 16, 48, 48))
        self.assertAlmostEqual(_sample_luminance(transparent, 12), 80.0, places=2)
        _result, diagnostics = _atlas_correction(transparent, TextureFinishSettings())
        self.assertEqual(diagnostics["approx_highlight_clip_fraction_after"], 0)
        blank = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        self.assertIsNone(_sample_luminance(blank, 12))


if __name__ == "__main__":
    unittest.main()
