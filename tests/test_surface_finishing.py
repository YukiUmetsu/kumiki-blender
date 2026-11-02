"""Synthetic photographic finishing checks, with no Blender or network dependency."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.surface_finish import (
    SurfaceOptions, finish_surface_atlas, load_surface_recipe,
)
from kumiki_blender.reconstruction.udim import remap_obj_uv_for_udim


class GuidedSurfaceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.atlas = self.root / "atlas.png"
        self.size = (48, 32)
        self.manifest = self.root / "surface.json"

    def _image(self, name, color, mode="L"):
        path = self.root / name
        Image.new(mode, self.size, color).save(path)
        return path

    def _recipe(self, options):
        self.manifest.write_text(json.dumps({
            "schema_version": 1, "atlases": {self.atlas.name: options},
        }))
        return load_surface_recipe(self.manifest, {self.atlas.name: self.atlas})[self.atlas.name]

    def test_guided_linear_light_delighting_preserves_regions_and_alpha(self):
        photo = Image.new("RGBA", self.size, (90, 90, 90, 255))
        for y in range(32):
            for x in range(48):
                value = 180 if x < 24 else 80
                photo.putpixel((x, y), (value, value, value, 255))
        photo.putpixel((0, 0), (0, 0, 0, 0))
        photo.save(self.atlas)
        mask_a = Image.new("L", self.size, 0)
        mask_b = Image.new("L", self.size, 0)
        for y in range(32):
            for x in range(48):
                (mask_a if x < 24 else mask_b).putpixel((x, y), 255)
        mask_a.save(self.root / "a.png")
        mask_b.save(self.root / "b.png")
        illum = Image.new("L", self.size, 0)
        for y in range(32):
            for x in range(48):
                illum.putpixel((x, y), 220 if x < 24 else 50)
        illum.save(self.root / "illum.png")
        spec = self._recipe({
            "regions": [{"id": "marble", "mask": "a.png", "roughness": 0.76},
                        {"id": "metal", "mask": "b.png", "roughness": 0.3}],
            "illumination_map": "illum.png",
        })
        corrected, roughness, diagnostics = finish_surface_atlas(
            photo, spec, SurfaceOptions(delight_strength=1.0, max_delight_ev=1.5))
        self.assertLess(abs(corrected.getpixel((8, 8))[0] - corrected.getpixel((38, 8))[0]), 35)
        self.assertEqual(corrected.getpixel((0, 0)), (0, 0, 0, 0))
        self.assertAlmostEqual(roughness.getpixel((8, 8)) / 255, 0.76, delta=0.01)
        self.assertAlmostEqual(roughness.getpixel((38, 8)) / 255, 0.3, delta=0.01)
        self.assertEqual(diagnostics["delighting_method"], "guided-illumination")
        self.assertEqual(diagnostics["roughness_source"], "material-priors-not-measured")

    def test_cleanup_affects_only_explicit_small_defect_mask(self):
        photo = Image.new("RGB", self.size, (110, 90, 70))
        photo.putpixel((10, 10), (245, 0, 0))
        photo.putpixel((34, 10), (245, 0, 0))
        photo.save(self.atlas)
        region = self._image("region.png", 255)
        defect = self._image("defect.png", 0)
        with Image.open(defect) as item:
            mask = item.copy()
        mask.putpixel((10, 10), 255)
        mask.save(defect)
        spec = self._recipe({"regions": [{"id": "stone", "mask": region.name,
                                            "roughness": 0.87}],
                             "damage_mask": defect.name})
        result, _, diagnostics = finish_surface_atlas(
            photo, spec, SurfaceOptions(cleanup_radius=3))
        self.assertLess(result.getpixel((10, 10))[0], 150)
        self.assertEqual(result.getpixel((34, 10)), (245, 0, 0))
        self.assertGreater(diagnostics["cleanup_pixels"], 0)

    def test_cleanup_rejects_large_contiguous_hole_even_below_five_percent(self):
        photo = Image.new("RGB", (128, 128), (115, 100, 80))
        atlas = self.root / "wide-atlas.png"
        photo.save(atlas)
        region = Image.new("L", (128, 128), 255)
        region.save(self.root / "wide-region.png")
        damage = Image.new("L", (128, 128), 0)
        for y in range(50, 62):
            for x in range(50, 62):
                damage.putpixel((x, y), 255)
        damage.save(self.root / "wide-damage.png")
        self.manifest.write_text(json.dumps({"schema_version": 1, "atlases": {
            atlas.name: {"regions": [{"id": "stone", "mask": "wide-region.png",
                                        "roughness": .85}],
                         "damage_mask": "wide-damage.png"}}}))
        spec = load_surface_recipe(self.manifest, {atlas.name: atlas})[atlas.name]
        with self.assertRaisesRegex(ReferenceError, "beyond cleanup radius"):
            finish_surface_atlas(photo, spec, SurfaceOptions(cleanup_radius=1))

    def test_strict_path_overlap_and_dimension_validation(self):
        self._image("atlas.png", 120, "RGB")
        self._image("region.png", 255)
        for invalid in (
            {"regions": [{"id": "stone", "mask": "../outside.png", "roughness": .8}]},
            {"regions": [{"id": "stone", "mask": "region.png", "roughness": 1.5}]},
            {"regions": [{"id": "a", "mask": "region.png", "roughness": .5},
                         {"id": "b", "mask": "region.png", "roughness": .8}]},
            {"regions": [{"id": "a", "mask": "region.png", "roughness": .5}],
             "height_map": "region.png"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ReferenceError):
                self._recipe(invalid)
        Image.new("L", (8, 8), 255).save(self.root / "wrong.png")
        with self.assertRaisesRegex(ReferenceError, "dimensions"):
            self._recipe({"regions": [{"id": "a", "mask": "wrong.png", "roughness": .6}]})
        # Do not trust a source whose bytes changed after preflight.
        valid = self._recipe({"regions": [{"id": "a", "mask": "region.png", "roughness": .6}]})
        self._image("region.png", 0)
        with Image.open(self.atlas) as photo, self.assertRaisesRegex(ReferenceError, "changed"):
            finish_surface_atlas(photo, valid, SurfaceOptions())

    def test_height_map_must_be_16bit_and_tied_to_scale(self):
        self._image("atlas.png", 150, "RGB")
        self._image("region.png", 255)
        Image.new("I;16", self.size, 32768).save(self.root / "height.png")
        spec = self._recipe({"regions": [{"id": "a", "mask": "region.png",
                                           "roughness": .8}],
                             "height_map": "height.png", "displacement_scale_m": .012})
        self.assertEqual(spec.height_map.path.name, "height.png")
        self.assertEqual(spec.displacement_scale_m, .012)
        with self.assertRaisesRegex(ReferenceError, "displacement_scale_m"):
            self._recipe({"regions": [{"id": "a", "mask": "region.png",
                                       "roughness": .8}], "height_map": "height.png"})
        self._image("height-8bit.png", 127)
        with self.assertRaisesRegex(ReferenceError, "16-bit"):
            self._recipe({"regions": [{"id": "a", "mask": "region.png",
                                       "roughness": .8}], "height_map": "height-8bit.png",
                          "displacement_scale_m": .01})

    def test_authored_roughness_map_overrides_material_prior(self):
        self._image("atlas.png", 150, "RGB")
        self._image("region.png", 255)
        self._image("roughness.png", 64)
        spec = self._recipe({"regions": [{"id": "polished-stone", "mask": "region.png",
                                           "roughness": 0.85, "roughness_map": "roughness.png"}]})
        with Image.open(self.atlas) as photo:
            _albedo, roughness, report = finish_surface_atlas(photo, spec, SurfaceOptions())
        self.assertEqual(roughness.getpixel((20, 12)), 64)
        self.assertIn("authored", report["roughness_source"])

    def test_explicit_low_frequency_fallback_reports_heuristic_and_reduces_gradient(self):
        photo = Image.new("RGB", self.size, (80, 80, 80))
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                value = 80 + int(100 * x / (self.size[0] - 1))
                photo.putpixel((x, y), (value,) * 3)
        photo.save(self.atlas)
        self._image("region.png", 255)
        spec = self._recipe({"regions": [{"id": "stone", "mask": "region.png", "roughness": .8}]})
        corrected, _, diagnostics = finish_surface_atlas(
            photo, spec, SurfaceOptions(delight_strength=1.0,
                                        low_frequency_radius=16))
        original_gap = photo.getpixel((43, 15))[0] - photo.getpixel((4, 15))[0]
        corrected_gap = corrected.getpixel((43, 15))[0] - corrected.getpixel((4, 15))[0]
        self.assertLess(corrected_gap, original_gap)
        self.assertEqual(diagnostics["delighting_method"], "low-frequency-heuristic")


class UDIMTests(unittest.TestCase):
    def test_shared_uvs_are_split_per_material_without_changing_vertex_positions(self):
        obj = (
            "mtllib old.mtl\n"
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "vt 0.25 0.5\nvt 0.75 0.5\nvt 0.25 0.75\n"
            "usemtl stone\nf 1/1 2/2 3/3\n"
            "usemtl copper\nf 1/1 2/2 3/3\n"
        )
        converted, stats = remap_obj_uv_for_udim(
            obj, {"stone": 1001, "copper": 1002})
        self.assertIn("mtllib udim.mtl", converted)
        self.assertIn("v 0 0 0", converted)
        self.assertIn("vt 1.25 0.5", converted)
        self.assertIn("vt 1.75 0.5", converted)
        self.assertEqual(stats["remapped_faces"], 2)
        lines = [line for line in converted.splitlines() if line.startswith("f ")]
        self.assertNotEqual(lines[0], lines[1])

    def test_udim_rejects_missing_uvs_unknown_materials_and_out_of_tile_coords(self):
        for obj, message in (
            ("v 0 0 0\nusemtl stone\nf 1 2 3\n", "UV"),
            ("vt 0.2 0.4\nusemtl copper\nf 1/1 2/1 3/1\n", "material"),
            ("vt 2.0 0.4\nusemtl stone\nf 1/1 2/1 3/1\n", "outside"),
        ):
            with self.subTest(obj=obj), self.assertRaisesRegex(ReferenceError, message):
                remap_obj_uv_for_udim(obj, {"stone": 1001})


if __name__ == "__main__":
    unittest.main()
