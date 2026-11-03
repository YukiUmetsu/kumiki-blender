"""UV illumination guidance keeps real pigment while reducing transient shadow."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.illumination_guides import (
    IlluminationGuideOptions, estimate_uv_illumination,
)
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.surface_finish import load_surface_recipe
from kumiki_blender.reconstruction.texture_finish import (
    TextureFinishSettings, finish_dense_textures,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UVIlluminationGuideTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dense = self.root / "dense"
        folder = self.dense / "workspace"
        folder.mkdir(parents=True)
        self.obj, self.mtl, self.atlas = folder / "mesh.obj", folder / "mesh.mtl", folder / "stone.png"
        self.obj.write_text(
            "mtllib mesh.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl surface\nf 1/1 2/2 3/3 4/4\n")
        self.mtl.write_text("newmtl surface\nmap_Kd stone.png\n")
        base = Image.new("RGB", (128, 96), (175, 175, 175))
        ImageDraw.Draw(base).rectangle((0, 0, 63, 95), fill=(65, 65, 65))
        base.save(self.atlas)
        (self.dense / "dense-report.json").write_text(json.dumps({
            "status": "dense-mesh-candidate / review-required",
            "textured_obj": {"path": "workspace/mesh.obj", "sha256": sha(self.obj)},
            "texture_atlases": [{"path": "workspace/stone.png", "sha256": sha(self.atlas)}],
        }))
        guide = self.root / "guidance"
        guide.mkdir()
        self.guide = guide
        Image.new("L", (128, 96), 255).save(guide / "stone-region.png")
        observations = []
        for i in range(3):
            photo = (base.copy() if i == 0 else Image.new("RGB", (128, 96), (126, 126, 126)))
            photo.save(guide / f"uv-{i}.png")
            visibility = Image.new("L", (128, 96), 255)
            ImageDraw.Draw(visibility).rectangle((0, 0, 4, 95), fill=0)
            visibility.save(guide / f"visible-{i}.png")
            observations.append({"id": f"camera-{i}", "photo": f"uv-{i}.png",
                                 "confidence": f"visible-{i}.png"})
        self.recipe = guide / "surface.json"
        self.recipe.write_text(json.dumps({"schema_version": 1, "atlases": {
            self.atlas.name: {
                "regions": [{"id": "limestone", "mask": "stone-region.png", "roughness": .84}],
                "observations": observations,
            }}}))

    def test_guidance_reduces_baked_shadow_and_remains_a_valid_surface_recipe(self):
        original = self.atlas.read_bytes()
        output = self.root / "illumination"
        result = estimate_uv_illumination(self.dense, self.recipe, output,
                                          options=IlluminationGuideOptions(smoothing_radius=10))
        self.assertFalse(result["distribution_allowed"])
        self.assertEqual(result["source_atlas_sha256"][self.atlas.name], sha(self.atlas))
        recipe = load_surface_recipe(output / "surface-recipe.json", {self.atlas.name: self.atlas})
        spec = recipe[self.atlas.name]
        self.assertEqual(len(spec.observations), 3)
        self.assertEqual(spec.regions[0].roughness, .84)
        field = spec.illumination_map.open_verified()
        self.assertLess(field.getpixel((25, 48)), 128)
        self.assertGreaterEqual(field.getpixel((100, 48)), 127)
        self.assertEqual(field.getpixel((0, 48)), 128)
        item = result["atlases"][self.atlas.name]
        self.assertGreater(item["guided_fraction_of_uv"], .75)
        for artifact in result["files"]:
            self.assertEqual(sha(output / artifact["path"]), artifact["sha256"])
        finished = self.root / "finished"
        finish_dense_textures(self.dense, finished, TextureFinishSettings(
            surface_recipe=str(output / "surface-recipe.json"),
            surface_delight_strength=1.0, surface_max_delight_ev=1.5))
        with Image.open(finished / self.atlas.name) as photo:
            old_gap = 175 - 65
            new_gap = photo.getpixel((100, 50))[0] - photo.getpixel((25, 50))[0]
            self.assertLess(new_gap, old_gap * .7)
        self.assertEqual(self.atlas.read_bytes(), original)

    def test_rejects_missing_registered_views_or_tampered_atlas(self):
        destination = self.root / "bad"
        data = json.loads(self.recipe.read_text())
        data["atlases"][self.atlas.name]["observations"] = data["atlases"][self.atlas.name]["observations"][:2]
        self.recipe.write_text(json.dumps(data))
        with self.assertRaises(ReferenceError):
            estimate_uv_illumination(self.dense, self.recipe, destination)
        self.assertFalse(destination.exists())
        # Restore third independent view; source hash pin must still hold.
        data["atlases"][self.atlas.name]["observations"].append(
            {"id": "camera-2", "photo": "uv-2.png", "confidence": "visible-2.png"})
        self.recipe.write_text(json.dumps(data))
        Image.new("RGB", (128, 96), (30, 30, 30)).save(self.atlas)
        with self.assertRaisesRegex(ReferenceError, "changed"):
            estimate_uv_illumination(self.dense, self.recipe, destination)
        self.assertFalse(destination.exists())

    def test_options_cli_and_output_collision(self):
        with self.assertRaisesRegex(ReferenceError, "smoothing_radius"):
            estimate_uv_illumination(self.dense, self.recipe, self.root / "bad",
                options=IlluminationGuideOptions(smoothing_radius=2))
        out = self.root / "cli-illumination"
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["estimate-uv-illumination",
                                   "--dense-revision", str(self.dense),
                                   "--surface-recipe", str(self.recipe),
                                   "--output", str(out)]), 0)
        self.assertIn("artistic-review-required", stream.getvalue())
        with self.assertRaisesRegex(ReferenceError, "unused output"):
            estimate_uv_illumination(self.dense, self.recipe, out)


if __name__ == "__main__":
    unittest.main()
