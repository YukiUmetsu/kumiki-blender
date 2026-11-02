"""Reproducible UV texture corrections without altering geometry or source photographs."""

import contextlib
import hashlib
import io
import json
import os
import struct
import subprocess
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
from kumiki_blender.reconstruction.udim import package_finished_udim
from kumiki_blender.reconstruction.uv_seams import _solve_page_offsets


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

    def test_guided_surface_recipe_exports_hash_pinned_pbr_and_16bit_height(self):
        region = self.root / "stone-mask.png"
        guidance = self.root / "lighting.png"
        height = self.root / "height.png"
        Image.new("L", (48, 32), 255).save(region)
        Image.new("L", (48, 32), 128).save(guidance)
        Image.new("I;16", (48, 32), 32768).save(height)
        recipe = self.root / "surface.json"
        recipe.write_text(json.dumps({
            "schema_version": 1,
            "atlases": {self.atlas.name: {
                "regions": [{"id": "limestone", "mask": region.name, "roughness": 0.88}],
                "illumination_map": guidance.name,
                "height_map": height.name,
                "displacement_scale_m": 0.012,
            }},
        }))
        original = self.atlas.read_bytes()
        output = self.root / "advanced-finished"
        report = finish_dense_textures(self.dense, output, TextureFinishSettings(
            surface_recipe=str(recipe), surface_delight_strength=0.6))
        self.assertEqual(self.atlas.read_bytes(), original)
        self.assertEqual(report["surface_recipe_sha256"], self._digest(recipe))
        entry = report["surface_maps"][0]
        self.assertEqual(entry["atlas"], self.atlas.name)
        self.assertEqual(entry["displacement_scale_m"], 0.012)
        self.assertEqual(entry["quality"]["roughness_source"], "material-priors-not-measured")
        self.assertEqual(self._digest(output / entry["roughness"]["path"]),
                         entry["roughness"]["sha256"])
        self.assertEqual((output / entry["height"]["path"]).read_bytes(), height.read_bytes())
        plan = plan_dense_blender_review(self.dense, self.root / "advanced-render",
                                         textured=True, texture_finish=output)
        self.assertEqual(plan["surface_maps"][0]["roughness"]["path"],
                         str(output / entry["roughness"]["path"]))
        report_path = output / "texture-finish-report.json"
        original_report = report_path.read_text()
        altered = json.loads(original_report)
        altered["surface_maps"].append(dict(altered["surface_maps"][0]))
        report_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ReferenceError, "duplicate|exactly one"):
            plan_dense_blender_review(self.dense, self.root / "duplicate-render",
                                      textured=True, texture_finish=output)
        altered["surface_maps"] = []
        report_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ReferenceError, "missing|exactly one"):
            plan_dense_blender_review(self.dense, self.root / "missing-render",
                                      textured=True, texture_finish=output)
        report_path.write_text(original_report)
        (output / entry["roughness"]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ReferenceError, "Surface map changed"):
            plan_dense_blender_review(self.dense, self.root / "tampered-render",
                                      textured=True, texture_finish=output)

    def test_invalid_surface_recipe_fails_before_output_is_created(self):
        region = self.root / "region-bad.png"
        Image.new("L", (2, 2), 255).save(region)
        recipe = self.root / "bad-surface.json"
        recipe.write_text(json.dumps({"schema_version": 1, "atlases": {
            self.atlas.name: {"regions": [{"id": "stone", "mask": region.name,
                                            "roughness": .8}]}}}))
        destination = self.root / "invalid-finished"
        with self.assertRaisesRegex(ReferenceError, "dimensions"):
            finish_dense_textures(self.dense, destination,
                                  TextureFinishSettings(surface_recipe=str(recipe)))
        self.assertFalse(destination.exists())

    @unittest.skipUnless(os.environ.get("KUMIKI_BLENDER_EXECUTABLE"), "requires real Blender 4.2+")
    def test_real_blender_packs_roughness_and_height_and_exports_glb(self):
        region = self.root / "render-region.png"
        height = self.root / "render-height.png"
        Image.new("L", (48, 32), 255).save(region)
        Image.new("I;16", (48, 32), 32768).save(height)
        recipe = self.root / "render-surface.json"
        recipe.write_text(json.dumps({"schema_version": 1, "atlases": {
            self.atlas.name: {
                "regions": [{"id": "limestone", "mask": region.name, "roughness": .84}],
                "height_map": height.name, "displacement_scale_m": .008,
            }}}))
        finished = self.root / "render-finish"
        finish_dense_textures(self.dense, finished,
                              TextureFinishSettings(surface_recipe=str(recipe)))
        render = self.root / "render-output"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["render-dense-blender-review",
                                   "--dense-revision", str(self.dense),
                                   "--texture-finish", str(finished),
                                   "--output", str(render), "--textured", "--no-renders",
                                   "--blender", os.environ["KUMIKI_BLENDER_EXECUTABLE"]]), 0)
        report = json.loads((render / "blender-review-report.json").read_text())
        self.assertTrue(report["status"].startswith("private-Blender-mesh-candidate"))
        glb = render / report["review_glb"]
        self.assertTrue(glb.is_file())
        with glb.open("rb") as stream:
            self.assertEqual(stream.read(4), b"glTF")
            stream.read(8)
            length, kind = struct.unpack("<I4s", stream.read(8))
            self.assertEqual(kind, b"JSON")
            gltf = json.loads(stream.read(length))
        materials = gltf["materials"]
        self.assertTrue(any("metallicRoughnessTexture" in m.get("pbrMetallicRoughness", {})
                            for m in materials))
        self.assertGreaterEqual(len(gltf.get("images", [])), 2)
        # Inspect the actual editable .blend as well: GLB cannot carry height
        # displacement, but Cycles must retain the packed 16-bit source.
        expression = (
            "import bpy; "
            "m = next(m for m in bpy.data.materials if m.use_nodes); "
            "nodes = list(m.node_tree.nodes); "
            "shader = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED'); "
            "output = next(n for n in nodes if n.type == 'OUTPUT_MATERIAL'); "
            "assert shader.inputs['Roughness'].is_linked; "
            "assert output.inputs['Displacement'].is_linked; "
            "assert m.displacement_method == 'BOTH'; "
            "assert len([n for n in nodes if n.type == 'TEX_IMAGE' and "
            "n.image and n.image.packed_file]) >= 3"
        )
        verified = subprocess.run([os.environ["KUMIKI_BLENDER_EXECUTABLE"],
                                   "--background", str(render / report["review_blend"]),
                                   "--python-exit-code", "1", "--python-expr", expression],
                                  capture_output=True, text=True, check=False)
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)

    def test_udim_package_remaps_shared_uvs_and_keeps_sources_unchanged(self):
        second = self.atlas.with_name("another-page.png")
        Image.new("RGB", (48, 32), (110, 110, 110)).save(second)
        self.mtl.write_text(self.mtl.read_text() + f"newmtl m1\nmap_Kd {second.name}\n")
        self.obj.write_text(self.obj.read_text() + "usemtl m1\nf 1/1 2/2 3/3\n")
        record_path = self.dense / "dense-report.json"
        record = json.loads(record_path.read_text())
        record["textured_obj"]["sha256"] = self._digest(self.obj)
        record["texture_atlases"].append({
            "path": str(second.relative_to(self.dense)), "sha256": self._digest(second)})
        record_path.write_text(json.dumps(record))
        src_before = (self.obj.read_bytes(), self.mtl.read_bytes(), second.read_bytes())
        revision = self.root / "udim-finish"
        finish_dense_textures(self.dense, revision, TextureFinishSettings())
        destination = self.root / "udim-package"
        outcome = package_finished_udim(revision, destination)
        self.assertEqual([item["tile"] for item in outcome["tiles"]], [1001, 1002])
        self.assertEqual(outcome["uv_remap"]["remapped_faces"], 2)
        self.assertIn("<UDIM>", (destination / "udim.mtl").read_text())
        self.assertIn("vt 1", (destination / "udim.obj").read_text())
        self.assertTrue((destination / "albedo.1001.png").is_file())
        self.assertTrue((destination / "albedo.1002.png").is_file())
        self.assertEqual((self.obj.read_bytes(), self.mtl.read_bytes(), second.read_bytes()),
                         src_before)

    @unittest.skipUnless(os.environ.get("KUMIKI_BLENDER_EXECUTABLE"), "requires real Blender 4.2+")
    def test_udim_package_opens_as_tiled_material_in_blender(self):
        second = self._set_up_cross_page_seam()
        recipe = self.root / "udim-surface.json"
        atlases = {}
        for index, path in enumerate((self.atlas, second)):
            mask = self.root / f"udim-mask-{index}.png"
            height = self.root / f"udim-height-{index}.png"
            Image.new("L", (96, 96), 255).save(mask)
            Image.new("I;16", (96, 96), 32768).save(height)
            atlases[path.name] = {
                "regions": [{"id": "stone", "mask": mask.name, "roughness": .84}],
                "height_map": height.name,
                "displacement_scale_m": .004 if index == 0 else .012,
            }
        recipe.write_text(json.dumps({"schema_version": 1, "atlases": atlases}))
        finished = self.root / "udim-blender-finish"
        finish_dense_textures(self.dense, finished,
                              TextureFinishSettings(surface_recipe=str(recipe)))
        target = self.root / "udim-blender-package"
        package_finished_udim(finished, target)
        expression = (
            "import bpy, sys; "
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'src')!r}); "
            "from kumiki_blender.reconstruction.blender_udim import import_finished_udim_package; "
            f"import_finished_udim_package(bpy, {str(target)!r}); "
            "images = [n.image for m in bpy.data.materials if m.use_nodes "
            "for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]; "
            "assert images, 'UDIM OBJ lost diffuse images'; "
            "assert len(set(i.name for i in images)) == 3, 'UDIM PBR/height maps missing'; "
            "assert all(i.source == 'TILED' and len(i.tiles) >= 2 for i in images), "
            "'UDIM map not configured as a tiled image'; "
            "assert all(i.has_data and tuple(i.size) == (96, 96) for i in images), "
            "'UDIM image tile pixel data was not loaded'; "
            "scales = sorted(n.inputs['Scale'].default_value for m in bpy.data.materials "
            "if m.use_nodes for n in m.node_tree.nodes if n.type == 'DISPLACEMENT'); "
            "assert len(scales) == 2 and abs(scales[0] - .004) < 1e-6 "
            "and abs(scales[1] - .012) < 1e-6, 'UDIM displacement scale lost'"
        )
        process = subprocess.run([os.environ["KUMIKI_BLENDER_EXECUTABLE"],
                                  "--background", "--factory-startup",
                                  "--python-exit-code", "1", "--python-expr", expression],
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_blender_udim_rejects_tampered_tiles_and_duplicate_assignments(self):
        from kumiki_blender.reconstruction.blender_udim import _validated_package

        self._set_up_cross_page_seam()
        finished = self.root / "tamper-finish"
        finish_dense_textures(self.dense, finished, TextureFinishSettings())
        target = self.root / "tamper-udim"
        package_finished_udim(finished, target)
        report_file = target / "udim-report.json"
        original_report = report_file.read_text()
        report = json.loads(original_report)
        report["tiles"][1]["tile"] = 1001
        report_file.write_text(json.dumps(report))
        with self.assertRaisesRegex(ReferenceError, "unique"):
            _validated_package(target)
        report_file.write_text(original_report)
        (target / "albedo.1001.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ReferenceError, "changed"):
            _validated_package(target)

    def test_unpinned_mtl_resource_directives_cannot_reach_blender(self):
        self.mtl.write_text(self.mtl.read_text() + "bump ../../untrusted-normal.png\n")
        with self.assertRaisesRegex(ReferenceError, "unsupported or unpinned"):
            plan_dense_blender_review(self.dense, self.root / "unsafe-review", textured=True)
        self.assertFalse((self.root / "unsafe-review").exists())

    def test_cli_surface_and_udim_commands(self):
        mask = self.root / "cli-mask.png"
        Image.new("L", (48, 32), 255).save(mask)
        recipe = self.root / "cli-surface.json"
        recipe.write_text(json.dumps({"schema_version": 1, "atlases": {
            self.atlas.name: {"regions": [{"id": "stone", "mask": mask.name,
                                           "roughness": 0.9}]}}}))
        finished = self.root / "cli-surface-finish"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["finish-dense-textures", "--dense-revision", str(self.dense),
                                   "--output", str(finished), "--surface-recipe", str(recipe),
                                   "--surface-delight-strength", "0.45",
                                   "--surface-low-frequency-radius", "48"]), 0)
            self.assertEqual(main(["package-finished-udim", "--texture-finish", str(finished),
                                   "--output", str(self.root / "cli-udim")]), 0)
        self.assertTrue((self.root / "cli-udim" / "roughness.1001.png").is_file())

    def test_source_mutations_and_settings_are_rejected_before_output(self):
        for settings in (TextureFinishSettings(gamma=0),
                         TextureFinishSettings(exposure_ev=float("nan")),
                         TextureFinishSettings(protect_black=True),
                         TextureFinishSettings(max_balance_ev=-1),
                         TextureFinishSettings(auto_exposure_luma=0),
                         TextureFinishSettings(auto_exposure_luma=float("nan")),
                         TextureFinishSettings(max_auto_exposure_ev=3),
                         TextureFinishSettings(uv_aware_stats="yes"),
                         TextureFinishSettings(balance_uv_seams="yes"),
                         TextureFinishSettings(max_seam_ev=-0.01),
                         TextureFinishSettings(min_seam_anchors=3),
                         TextureFinishSettings(balance_uv_seams=True,
                                               balance_atlases=True)):
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
                                   "--shadow-lift", "0.12", "--neutralize-cast",
                                   "--uv-aware-stats",
                                   "--auto-exposure-luma", "115",
                                   "--max-auto-exposure-ev", "0.5"]), 0)
        settings = json.loads((target / "texture-finish-report.json").read_text())["settings"]
        self.assertEqual(settings["exposure_ev"], 0.75)
        self.assertEqual(settings["shadow_lift"], 0.12)
        self.assertTrue(settings["neutralize_cast"])
        self.assertTrue(settings["uv_aware_stats"])
        self.assertEqual(settings["auto_exposure_luma"], 115.0)
        self.assertEqual(settings["max_auto_exposure_ev"], 0.5)
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

    def test_opt_in_auto_exposure_brightens_dark_atlas_without_fabricating_empty_texels(self):
        target = self.root / "auto-exposed"
        report = finish_dense_textures(
            self.dense, target,
            TextureFinishSettings(auto_exposure_luma=125, max_auto_exposure_ev=1.0))
        atlas_report = report["texture_atlases"][0]
        self.assertGreater(atlas_report["auto_exposure_ev"], 0)
        self.assertLessEqual(atlas_report["auto_exposure_ev"], 1.0)
        self.assertIsNotNone(atlas_report["source_median_luma"])
        with Image.open(self.atlas) as original, Image.open(target / self.atlas.name) as corrected:
            self.assertGreater(sum(corrected.getpixel((30, 12))),
                               sum(original.getpixel((30, 12))))
            self.assertEqual(corrected.getpixel((2, 12)), (0, 0, 0))
        planned = plan_dense_blender_review(self.dense, self.root / "review-auto",
                                            textured=True, texture_finish=target)
        self.assertEqual(len(planned["texture_atlases"]), 1)

    def test_atlas_balance_excludes_transparent_white_padding(self):
        transparent = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        transparent.paste((80, 80, 80, 255), (16, 16, 48, 48))
        self.assertAlmostEqual(_sample_luminance(transparent, 12), 80.0, places=2)
        _result, diagnostics = _atlas_correction(transparent, TextureFinishSettings())
        self.assertEqual(diagnostics["approx_highlight_clip_fraction_after"], 0)
        blank = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        self.assertIsNone(_sample_luminance(blank, 12))

    def test_uv_aware_exposure_ignores_bright_unused_atlas_padding(self):
        # Our OBJ triangle occupies the bottom-left half (v=0 is the bottom).
        # White outside its UV polygon must not bias an exposure estimate.
        atlas = Image.new("RGB", (48, 32), (240, 240, 240))
        for y in range(32):
            for x in range(48):
                if x/47 < y/31 - 0.06:
                    atlas.putpixel((x, y), (60, 60, 60))
        atlas.save(self.atlas)
        report_path = self.dense / "dense-report.json"
        record = json.loads(report_path.read_text())
        record["texture_atlases"][0]["sha256"] = self._digest(self.atlas)
        report_path.write_text(json.dumps(record))
        baseline = finish_dense_textures(
            self.dense, self.root / "global-luma",
            TextureFinishSettings(auto_exposure_luma=120, max_auto_exposure_ev=1.0))
        aware_path = self.root / "uv-aware-luma"
        aware = finish_dense_textures(
            self.dense, aware_path,
            TextureFinishSettings(auto_exposure_luma=120, max_auto_exposure_ev=1.0,
                                  uv_aware_stats=True))
        global_item = baseline["texture_atlases"][0]
        uv_item = aware["texture_atlases"][0]
        self.assertLess(global_item["auto_exposure_ev"], 0)
        self.assertGreater(uv_item["auto_exposure_ev"], 0)
        self.assertEqual(uv_item["uv_mapped_faces"], 1)
        self.assertGreater(uv_item["uv_mapped_atlas_fraction"], 0.4)
        self.assertLess(uv_item["uv_mapped_atlas_fraction"], 0.6)
        self.assertEqual(uv_item["source_median_luma"], 60)
        with Image.open(self.atlas) as original, Image.open(aware_path / self.atlas.name) as fixed:
            self.assertGreater(fixed.getpixel((6, 25))[0], original.getpixel((6, 25))[0])
        plan = plan_dense_blender_review(self.dense, self.root / "uv-aware-review",
                                         textured=True, texture_finish=aware_path)
        self.assertEqual(plan["source_obj"], str(aware_path / self.obj.name))

    def test_uv_aware_stats_reject_invalid_uv_indices_and_unused_atlas(self):
        source_report = self.dense / "dense-report.json"
        original = self.obj.read_text()
        self.obj.write_text(original.replace("f 1/1 2/2 3/3", "f 1//1 2//2 3//3"))
        report = json.loads(source_report.read_text())
        report["textured_obj"]["sha256"] = self._digest(self.obj)
        source_report.write_text(json.dumps(report))
        target = self.root / "invalid-uv"
        with self.assertRaisesRegex(ReferenceError, "missing a UV index"):
            finish_dense_textures(self.dense, target,
                                  TextureFinishSettings(uv_aware_stats=True))
        self.assertFalse(target.exists())
        # Negative relative OBJ indices are valid and should select the same UVs.
        self.obj.write_text(original.replace("f 1/1 2/2 3/3", "f 1/-3 2/-2 3/-1"))
        report["textured_obj"]["sha256"] = self._digest(self.obj)
        source_report.write_text(json.dumps(report))
        valid = finish_dense_textures(self.dense, self.root / "negative-uv",
                                      TextureFinishSettings(uv_aware_stats=True))
        self.assertEqual(valid["texture_atlases"][0]["uv_mapped_faces"], 1)
        # An atlas listed in the source report but never bound to a polygon
        # cannot provide meaningful surface coverage statistics.
        second = self.atlas.parent / "unused-map.png"
        Image.new("RGB", (48, 32), (120, 120, 120)).save(second)
        self.mtl.write_text(self.mtl.read_text() + f"newmtl spare\nmap_Kd {second.name}\n")
        report["texture_atlases"].append({"path": str(second.relative_to(self.dense)),
                                          "sha256": self._digest(second)})
        source_report.write_text(json.dumps(report))
        with self.assertRaisesRegex(ReferenceError, "mapped UV face"):
            finish_dense_textures(self.dense, self.root / "unused-uv",
                                  TextureFinishSettings(uv_aware_stats=True))

    def test_uv_aware_balancing_handles_two_materials_and_detects_missing_texels(self):
        second = self.atlas.parent / "second-map.png"
        Image.new("RGB", (48, 32), (155, 155, 155)).save(second)
        self.mtl.write_text(self.mtl.read_text() + f"newmtl m1\nmap_Kd {second.name}\n")
        self.obj.write_text(self.obj.read_text() + "usemtl m1\nf 1/1 2/2 3/3\n")
        report_path = self.dense / "dense-report.json"
        report = json.loads(report_path.read_text())
        report["textured_obj"]["sha256"] = self._digest(self.obj)
        report["texture_atlases"].append({"path": str(second.relative_to(self.dense)),
                                          "sha256": self._digest(second)})
        report_path.write_text(json.dumps(report))
        result = finish_dense_textures(self.dense, self.root / "multi-uv",
                                       TextureFinishSettings(uv_aware_stats=True,
                                                             balance_atlases=True))
        self.assertEqual(len(result["texture_atlases"]), 2)
        by_name = {item["path"]: item for item in result["texture_atlases"]}
        self.assertEqual(by_name[self.atlas.name]["uv_mapped_faces"], 1)
        self.assertEqual(by_name[second.name]["uv_mapped_faces"], 1)
        self.assertGreater(by_name[self.atlas.name]["atlas_balance_ev"], 0)
        self.assertLess(by_name[second.name]["atlas_balance_ev"], 0)
        # The first source page has a black patch that overlaps *actual* UVs.
        self.assertGreater(by_name[self.atlas.name]["approx_uncovered_dark_fraction"], 0)
        self.assertLess(by_name[second.name]["approx_uncovered_dark_fraction"], 0.01)
        for page in by_name.values():
            self.assertTrue((self.root / "multi-uv" / page["uv_dark_review"]).is_file())

    def test_uv_aware_rejects_tiled_coordinates_without_touching_sources(self):
        self.obj.write_text(self.obj.read_text().replace("vt 1 0", "vt 2 0"))
        record_path = self.dense / "dense-report.json"
        record = json.loads(record_path.read_text())
        record["textured_obj"]["sha256"] = self._digest(self.obj)
        record_path.write_text(json.dumps(record))
        output = self.root / "tiled-uv"
        with self.assertRaisesRegex(ReferenceError, "tiled/out-of-range"):
            finish_dense_textures(self.dense, output,
                                  TextureFinishSettings(uv_aware_stats=True))
        self.assertFalse(output.exists())

    def _set_up_cross_page_seam(self):
        """Two uniform atlas pages meet along eleven topologically shared vertices."""
        second = self.atlas.with_name("second-seam-page.png")
        Image.new("RGB", (96, 96), (160, 160, 160)).save(second)
        Image.new("RGB", (96, 96), (80, 80, 80)).save(self.atlas)
        self.mtl.write_text(self.mtl.read_text() + f"newmtl m1\nmap_Kd {second.name}\n")
        lines = [f"mtllib {self.mtl.name}"]
        # A shared row of 11 vertices and unique upper/lower triangle rows.
        for y in (0, 1, -1):
            for i in range(11):
                lines.append(f"v {i} {y} 0")
        for y in (0.48, 0.70, 0.52, 0.30):
            for i in range(11):
                lines.append(f"vt {0.25 + i * 0.05:.4f} {y}")
        lines.append("usemtl m0")
        for i in range(10):
            lines.append(f"f {i+1}/{i+1} {i+2}/{i+2} {12+i}/{12+i}")
        lines.append("usemtl m1")
        for i in range(10):
            lines.append(f"f {i+1}/{23+i} {23+i}/{34+i} {i+2}/{24+i}")
        self.obj.write_text("\n".join(lines) + "\n")
        report_path = self.dense / "dense-report.json"
        report = json.loads(report_path.read_text())
        report["textured_obj"]["sha256"] = self._digest(self.obj)
        report["texture_atlases"][0]["sha256"] = self._digest(self.atlas)
        report["texture_atlases"].append({"path": str(second.relative_to(self.dense)),
                                          "sha256": self._digest(second)})
        report_path.write_text(json.dumps(report))
        return second

    def test_cross_page_uv_seam_balancing_uses_shared_vertices_not_page_medians(self):
        second = self._set_up_cross_page_seam()
        # The page-wide medians intentionally point the WRONG way: most texels
        # outside the actual seam strip are brighter on page 0 and darker on
        # page 1. Only corresponding seam texels reveal the real page mismatch.
        for path, outside in ((self.atlas, 220), (second, 35)):
            with Image.open(path) as source:
                edited = source.copy()
            for y in list(range(35)) + list(range(61, 96)):
                for x in range(96):
                    edited.putpixel((x, y), (outside, outside, outside))
            edited.save(path)
        report_path = self.dense / "dense-report.json"
        report = json.loads(report_path.read_text())
        report["texture_atlases"][0]["sha256"] = self._digest(self.atlas)
        report["texture_atlases"][1]["sha256"] = self._digest(second)
        report_path.write_text(json.dumps(report))
        originals = {path: path.read_bytes() for path in (self.obj, self.mtl, self.atlas, second)}
        median_result = finish_dense_textures(
            self.dense, self.root / "wrong-page-medians",
            TextureFinishSettings(balance_atlases=True, max_balance_ev=0.6))
        self.assertLess(median_result["texture_atlases"][0]["atlas_balance_ev"], 0)
        self.assertGreater(median_result["texture_atlases"][1]["atlas_balance_ev"], 0)
        target = self.root / "seam-finished"
        result = finish_dense_textures(
            self.dense, target, TextureFinishSettings(balance_uv_seams=True,
                                                      min_seam_anchors=8,
                                                      max_seam_ev=0.6))
        seam = result["uv_seam_balance"]
        self.assertEqual(seam["candidate_vertices"], 11)
        self.assertEqual(seam["valid_vertex_samples"], 11)
        self.assertEqual(len(seam["pairs"]), 1)
        self.assertEqual(seam["pairs"][0]["valid_anchors"], 11)
        self.assertTrue(seam["pairs"][0]["used"])
        self.assertAlmostEqual(seam["pairs"][0]["median_gap_ev_before"], 1.0, places=3)
        self.assertAlmostEqual(seam["pairs"][0]["estimated_gap_ev_after"], 0, places=3)
        by_name = {item["path"]: item for item in result["texture_atlases"]}
        self.assertAlmostEqual(by_name[self.atlas.name]["uv_seam_balance_ev"], 0.5, places=3)
        self.assertAlmostEqual(by_name[second.name]["uv_seam_balance_ev"], -0.5, places=3)
        with Image.open(target / self.atlas.name) as first, Image.open(target / second.name) as last:
            self.assertLess(abs(first.getpixel((50, 50))[0] - last.getpixel((50, 50))[0]), 3)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        planned = plan_dense_blender_review(self.dense, self.root / "seam-render", textured=True,
                                            texture_finish=target)
        self.assertEqual(len(planned["texture_atlases"]), 2)

    def test_cli_seam_gain_is_bounded_and_settings_are_recorded(self):
        self._set_up_cross_page_seam()
        destination = self.root / "cli-seam"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["finish-dense-textures", "--dense-revision", str(self.dense),
                                   "--output", str(destination), "--balance-uv-seams",
                                   "--max-seam-ev", "0.3", "--min-seam-anchors", "10"]), 0)
        result = json.loads((destination / "texture-finish-report.json").read_text())
        self.assertTrue(result["settings"]["balance_uv_seams"])
        self.assertEqual(result["settings"]["min_seam_anchors"], 10)
        self.assertEqual(result["settings"]["max_seam_ev"], 0.3)
        applied = [item["uv_seam_balance_ev"] for item in result["texture_atlases"]]
        self.assertEqual(sorted(applied), [-0.3, 0.3])
        self.assertAlmostEqual(result["uv_seam_balance"]["pairs"][0]["estimated_gap_ev_after"],
                               0.4, places=3)

    def test_cross_page_seam_fails_safe_when_anchors_sparse_or_inconsistent(self):
        second = self._set_up_cross_page_seam()
        # Too few confirmed correspondences: do not invent a page-level gain.
        sparse = finish_dense_textures(
            self.dense, self.root / "sparse-seam",
            TextureFinishSettings(balance_uv_seams=True, min_seam_anchors=20))
        self.assertFalse(sparse["uv_seam_balance"]["pairs"][0]["used"])
        self.assertEqual(sparse["texture_atlases"][0]["uv_seam_balance_ev"], 0)
        self.assertIn("insufficient-shared-anchors",
                      sparse["uv_seam_balance"]["pairs"][0]["excluded_reason"])
        # Rapidly changing page color along the seam suggests lighting or
        # genuinely different material, rather than a single correctable EV.
        with Image.open(self.atlas) as source:
            modified = source.copy()
        for x in range(96):
            tone = 40 if x < 48 else 200
            for y in range(96):
                modified.putpixel((x, y), (tone, tone, tone))
        modified.save(self.atlas)
        report_path = self.dense / "dense-report.json"
        report = json.loads(report_path.read_text())
        report["texture_atlases"][0]["sha256"] = self._digest(self.atlas)
        report_path.write_text(json.dumps(report))
        inconsistent = finish_dense_textures(
            self.dense, self.root / "inconsistent-seam",
            TextureFinishSettings(balance_uv_seams=True, min_seam_anchors=8))
        self.assertFalse(inconsistent["uv_seam_balance"]["pairs"][0]["used"])
        self.assertEqual(inconsistent["uv_seam_balance"]["pairs"][0]["excluded_reason"],
                         "inconsistent-seam-brightness")
        self.assertEqual(inconsistent["texture_atlases"][0]["uv_seam_balance_ev"], 0)

    def test_cross_page_seam_no_shared_vertices_is_reported_without_changes(self):
        second = self._set_up_cross_page_seam()
        # Distinct vertex indices intentionally cannot prove adjacency.
        content = self.obj.read_text()
        extra = [f"v {i} 0 0" for i in range(11)]
        # Replace m1's shared vertex references with new vertex records.
        m0, m1 = content.split("usemtl m1\n")
        m0 = m0 + "\n".join(extra) + "\n"
        for i in range(10):
            m1 = m1.replace(f"f {i+1}/{23+i} ", f"f {34+i}/{23+i} ")
            m1 = m1.replace(f" {i+2}/{24+i}\n", f" {35+i}/{24+i}\n")
        self.obj.write_text(m0 + "usemtl m1\n" + m1)
        report_path = self.dense / "dense-report.json"
        report = json.loads(report_path.read_text())
        report["textured_obj"]["sha256"] = self._digest(self.obj)
        report_path.write_text(json.dumps(report))
        outcome = finish_dense_textures(
            self.dense, self.root / "unconnected-seam",
            TextureFinishSettings(balance_uv_seams=True))
        self.assertEqual(outcome["uv_seam_balance"]["candidate_vertices"], 0)
        self.assertEqual(outcome["uv_seam_balance"]["pairs"], [])
        self.assertTrue(outcome["uv_seam_balance"]["warnings"])
        with Image.open(self.root / "unconnected-seam" / second.name) as image:
            self.assertEqual(image.getpixel((50, 50))[0], 160)

    def test_seam_analysis_rejects_tiled_uv_before_writing_any_revision(self):
        self._set_up_cross_page_seam()
        self.obj.write_text(self.obj.read_text().replace("vt 0.2500 0.48", "vt 1.25 0.48"))
        report_path = self.dense / "dense-report.json"
        report = json.loads(report_path.read_text())
        report["textured_obj"]["sha256"] = self._digest(self.obj)
        report_path.write_text(json.dumps(report))
        destination = self.root / "invalid-seam"
        with self.assertRaisesRegex(ReferenceError, "tiled/out-of-range"):
            finish_dense_textures(self.dense, destination,
                                  TextureFinishSettings(balance_uv_seams=True))
        self.assertFalse(destination.exists())

    def test_seam_page_graph_corrects_three_connected_pages_without_global_shift(self):
        # Page brightness A=80, B=160, C=113: the differences must be
        # reconciled across the cycle rather than adjusted one pair at a time.
        solved = _solve_page_offsets([
            ("a", "b", 1.0, 25), ("b", "c", -0.5, 30),
            ("a", "c", 0.5, 40)], max_ev=0.75)
        self.assertEqual(set(solved), {"a", "b", "c"})
        self.assertAlmostEqual(sum(solved.values()), 0, places=4)
        self.assertAlmostEqual(solved["a"] - solved["b"], 1.0, places=3)
        self.assertAlmostEqual(solved["b"] - solved["c"], -0.5, places=3)
        capped = _solve_page_offsets([("a", "b", 3.0, 30)], max_ev=0.4)
        self.assertEqual(capped, {"a": 0.4, "b": -0.4})
        uneven = _solve_page_offsets([("a", "b", 3.0, 30),
                                      ("a", "c", 3.0, 30)], max_ev=0.4)
        self.assertLessEqual(max(map(abs, uneven.values())), 0.400001)
        self.assertAlmostEqual(sum(uneven.values()), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
