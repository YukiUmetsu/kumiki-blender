"""Deterministic seed-driven UV material separation and finishing integration."""

import hashlib
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.material_separation import (
    SeparationOptions, plan_material_separation, propose_material_regions,
)
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.surface_finish import load_surface_recipe
from kumiki_blender.reconstruction.texture_finish import (
    TextureFinishSettings, finish_dense_textures,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaterialSeparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dense = self.root / "dense"
        folder = self.dense / "workspace"
        folder.mkdir(parents=True)
        self.obj = folder / "mesh.obj"
        self.mtl = folder / "mesh.mtl"
        self.atlas = folder / "atlas.png"
        self.obj.write_text(
            "mtllib mesh.mtl\n"
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "usemtl surface\nf 1/1 2/2 3/3 4/4\n")
        self.mtl.write_text("newmtl surface\nmap_Kd atlas.png\n")
        self.size = (96, 64)
        atlas = Image.new("RGBA", self.size, (0, 0, 0, 0))
        drawing = ImageDraw.Draw(atlas)
        drawing.rectangle((2, 2, 45, 61), fill=(190, 70, 44, 255))
        drawing.rectangle((46, 2, 93, 61), fill=(40, 102, 195, 255))
        # The source atlas baked a transient deep shadow into part of red stone.
        drawing.rectangle((12, 27, 40, 55), fill=(56, 20, 13, 255))
        atlas.save(self.atlas)
        (self.dense / "dense-report.json").write_text(json.dumps({
            "status": "dense-mesh-candidate / artistic-review-required",
            "textured_obj": {"path": "workspace/mesh.obj", "sha256": digest(self.obj)},
            "texture_atlases": [{"path": "workspace/atlas.png", "sha256": digest(self.atlas)}],
        }))
        self.guidance = self.root / "guidance"
        self.guidance.mkdir()
        self.recipe = self.guidance / "seed-recipe.json"
        stone = Image.new("L", self.size, 0)
        metal = Image.new("L", self.size, 0)
        ImageDraw.Draw(stone).rectangle((5, 5, 10, 10), fill=255)
        ImageDraw.Draw(metal).rectangle((79, 5, 84, 10), fill=255)
        stone.save(self.guidance / "stone.png")
        metal.save(self.guidance / "metal.png")
        self.views = []
        for index in range(3):
            with Image.open(self.atlas) as image:
                view = image.copy()
            draw = ImageDraw.Draw(view)
            draw.rectangle((12, 27, 40, 55), fill=(190, 70, 44, 255))
            if index == 0:
                draw.rectangle((12, 27, 40, 55), fill=(56, 20, 13, 255))
            if index == 1:
                draw.rectangle((47, 38, 55, 50), fill=(10, 31, 62, 255))
            view.save(self.guidance / f"view-{index}.png")
            mask = Image.new("L", self.size, 255)
            ImageDraw.Draw(mask).rectangle((0, 0, 1, 63), fill=0)
            if index > 0:
                ImageDraw.Draw(mask).rectangle((50, 28, 60, 34), fill=0)
            mask.save(self.guidance / f"visibility-{index}.png")
            self.views.append({"id": f"view-{index}", "photo": f"view-{index}.png",
                               "confidence": f"visibility-{index}.png"})
        self._write_recipe()

    def _write_recipe(self, *, regions=None, observations=None):
        self.recipe.write_text(json.dumps({
            "schema_version": 1,
            "atlases": {self.atlas.name: {
                "regions": regions if regions is not None else [
                    {"id": "red-stone", "mask": "stone.png", "roughness": .88},
                    {"id": "blue-paint", "mask": "metal.png", "roughness": .32}],
                "observations": observations if observations is not None else self.views,
            }},
        }))

    def test_multiview_seed_proposals_preserve_padding_and_work_with_finishing(self):
        original = self.atlas.read_bytes()
        output = self.root / "proposed"
        plan = plan_material_separation(self.dense, self.recipe, output)
        self.assertEqual(len(plan["atlases"][self.atlas.name]["views"]), 3)
        self.assertFalse(output.exists())
        result = propose_material_regions(self.dense, self.recipe, output)
        self.assertTrue(result["review_required"])
        self.assertFalse(result["distribution_allowed"])
        self.assertEqual(original, self.atlas.read_bytes())
        recipe = load_surface_recipe(output / "surface-recipe.json", {self.atlas.name: self.atlas})
        self.assertEqual(len(recipe[self.atlas.name].observations), 3)
        regions = recipe[self.atlas.name].regions
        red = regions[0].mask.open_verified()
        blue = regions[1].mask.open_verified()
        # Both labeled materials remain distinct under independently shadowed views.
        self.assertEqual(red.getpixel((26, 40)), 255)
        self.assertEqual(blue.getpixel((26, 40)), 0)
        self.assertEqual(red.getpixel((72, 40)), 0)
        self.assertEqual(blue.getpixel((72, 40)), 255)
        self.assertEqual(red.getpixel((0, 0)), 0)
        self.assertEqual(blue.getpixel((0, 0)), 0)
        self.assertEqual(Image.open(output / "atlas-01/unassigned.png").getpixel((0, 0)), 255)
        for entry in result["files"]:
            self.assertEqual(digest(output / entry["path"]), entry["sha256"])
        finished = self.root / "finished"
        completion = finish_dense_textures(self.dense, finished, TextureFinishSettings(
            surface_recipe=str(output / "surface-recipe.json"),
            surface_multiview_strength=.75))
        self.assertTrue((finished / (self.atlas.name + ".roughness.png")).is_file())
        self.assertEqual(completion["surface_maps"][0]["quality"]["roughness_source"],
                         "material-priors-not-measured")

    def test_seed_mask_always_locked_and_unobserved_is_unassigned(self):
        output = self.root / "proposed"
        propose_material_regions(self.dense, self.recipe, output)
        recipe = load_surface_recipe(output / "surface-recipe.json", {self.atlas.name: self.atlas})
        red, blue = recipe[self.atlas.name].regions
        self.assertEqual(red.mask.open_verified().getpixel((5, 5)), 255)
        self.assertEqual(blue.mask.open_verified().getpixel((80, 6)), 255)
        self.assertEqual(red.mask.open_verified().getpixel((0, 40)), 0)
        self.assertEqual(blue.mask.open_verified().getpixel((0, 40)), 0)
        # A valid textured pixel with only one visible registered camera is
        # unknown even though its atlas RGB happens to match a seed class.
        self.assertEqual(blue.mask.open_verified().getpixel((55, 30)), 0)
        self.assertEqual(Image.open(output / "atlas-01/unassigned.png").getpixel((55, 30)), 255)

    def test_two_disagreeing_illuminations_remain_unassigned(self):
        confidence_path = self.guidance / "visibility-2.png"
        with Image.open(confidence_path) as original:
            confidence = original.copy()
        ImageDraw.Draw(confidence).rectangle((20, 38, 27, 46), fill=0)
        confidence.save(confidence_path)
        output = self.root / "ambiguous"
        propose_material_regions(self.dense, self.recipe, output)
        recipe = load_surface_recipe(output / "surface-recipe.json", {self.atlas.name: self.atlas})
        for region in recipe[self.atlas.name].regions:
            self.assertEqual(region.mask.open_verified().getpixel((23, 42)), 0)
        self.assertEqual(Image.open(output / "atlas-01/unassigned.png").getpixel((23, 42)), 255)

    def test_rejects_tampered_atlas_before_any_output(self):
        output = self.root / "proposed"
        with Image.open(self.atlas) as original:
            image = original.copy()
        image.putpixel((3, 3), (222, 222, 222, 255))
        image.save(self.atlas)
        with self.assertRaisesRegex(ReferenceError, "changed"):
            plan_material_separation(self.dense, self.recipe, output)
        self.assertFalse(output.exists())

    def test_rejects_mutated_guidance_and_mislabeled_padding(self):
        output = self.root / "proposed"
        specs = load_surface_recipe(self.recipe, {self.atlas.name: self.atlas})
        specs[self.atlas.name].observations[0].photo.path.write_bytes(b"modified")
        with self.assertRaisesRegex(ReferenceError, "changed"):
            specs[self.atlas.name].observations[0].photo.open_verified()
        with self.assertRaises(ReferenceError):
            propose_material_regions(self.dense, self.recipe, output)
        self.assertFalse(output.exists())

        # Restore that view and reject seed strokes over transparent atlas padding.
        self.guidance.joinpath("view-0.png").write_bytes(self.atlas.read_bytes())
        with Image.open(self.guidance / "stone.png") as raw:
            seed = raw.copy()
        seed.putpixel((0, 0), 255)
        seed.save(self.guidance / "stone.png")
        with self.assertRaisesRegex(ReferenceError, "valid photographed"):
            plan_material_separation(self.dense, self.recipe, output)

    def test_cli_preflight_and_refuses_reused_output(self):
        output = self.root / "cli-proposals"
        command = ["--dense-revision", str(self.dense),
                   "--seed-recipe", str(self.recipe), "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(main(["plan-material-separation", *command]), 0)
        self.assertIn("seed-guided-material-proposal", capture.getvalue())
        self.assertFalse(output.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["propose-material-separation", *command]), 0)
        before = (output / "surface-recipe.json").read_bytes()
        with self.assertRaisesRegex(ReferenceError, "new external output"):
            propose_material_regions(self.dense, self.recipe, output)
        self.assertEqual((output / "surface-recipe.json").read_bytes(), before)

    def test_rejects_overlapping_masks_insufficient_labels_and_excess_limits(self):
        output = self.root / "proposed"
        self._write_recipe(regions=[{"id": "a", "mask": "stone.png", "roughness": .8}])
        with self.assertRaisesRegex(ReferenceError, "2–8"):
            plan_material_separation(self.dense, self.recipe, output)
        self._write_recipe(regions=[{"id": "a", "mask": "stone.png", "roughness": .8},
                                    {"id": "b", "mask": "stone.png", "roughness": .6}])
        with self.assertRaisesRegex(ReferenceError, "overlap"):
            plan_material_separation(self.dense, self.recipe, output)
        self._write_recipe()
        with self.assertRaisesRegex(ReferenceError, "max_side"):
            plan_material_separation(self.dense, self.recipe, output,
                                     options=SeparationOptions(max_side=64))
        with self.assertRaisesRegex(ReferenceError, "min_margin"):
            plan_material_separation(self.dense, self.recipe, output,
                                     options=SeparationOptions(min_margin=2))

    def test_atlas_only_outputs_an_honest_warning(self):
        self._write_recipe(observations=[])
        # An empty observations key is intentionally invalid; omit it instead.
        data = json.loads(self.recipe.read_text())
        del data["atlases"][self.atlas.name]["observations"]
        self.recipe.write_text(json.dumps(data))
        result = propose_material_regions(self.dense, self.recipe, self.root / "atlas-only")
        self.assertTrue(any("baked sunlight" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
