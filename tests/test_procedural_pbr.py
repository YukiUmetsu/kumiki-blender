import struct
import unittest
import zlib

from kumiki_blender.procedural_pbr import _stone_weathering, _tile_noise, make_pbr_maps
from kumiki_blender.surfaces import get_surface


def decode_png_pixels(payload):
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    offset, chunks, size = 8, [], None
    while offset < len(payload):
        length = struct.unpack_from(">I", payload, offset)[0]
        kind = payload[offset + 4:offset + 8]
        content = payload[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            size = struct.unpack_from(">II", content)
        elif kind == b"IDAT":
            chunks.append(content)
        offset += 12 + length
    raw = zlib.decompress(b"".join(chunks))
    width, height = size
    rows = [raw[row * (width * 3 + 1) + 1:(row + 1) * (width * 3 + 1)] for row in range(height)]
    return size, b"".join(rows)


class ProceduralPBRTests(unittest.TestCase):
    def test_generates_valid_distinct_embeddable_png_maps(self):
        maps = make_pbr_maps("natural-stone", "#BDB4A5", 0.76, resolution=32)
        self.assertEqual(maps.resolution, 32)
        base_size, base = decode_png_pixels(maps.base_color_png)
        rough_size, rough = decode_png_pixels(maps.roughness_png)
        normal_size, normal = decode_png_pixels(maps.normal_png)
        self.assertEqual(base_size, (32, 32))
        self.assertEqual(rough_size, base_size)
        self.assertEqual(normal_size, base_size)
        self.assertEqual(len(base), 32 * 32 * 3)
        self.assertGreater(len(set(base[::3])), 4, "Stone albedo should have non-uniform grain")
        self.assertGreater(len(set(rough[::3])), 4, "Roughness should have physical variation")
        self.assertGreater(min(normal[2::3]), 180, "Normal map should not introduce inverted surfaces")
        self.assertNotEqual(maps.base_color_png, maps.roughness_png)

    def test_repeatable_and_periodic(self):
        a = make_pbr_maps("cast-concrete", "#888888", 0.8, resolution=32)
        b = make_pbr_maps("cast-concrete", "#888888", 0.8, resolution=32)
        c = make_pbr_maps("cast-concrete", "#888888", 0.7, resolution=32)
        self.assertEqual(a, b)
        self.assertNotEqual(a.roughness_png, c.roughness_png)
        for v in (0, 0.15, 0.74):
            self.assertAlmostEqual(_tile_noise(0, v, 11, 42), _tile_noise(1, v, 11, 42))
            self.assertAlmostEqual(_tile_noise(v, 0, 11, 42), _tile_noise(v, 1, 11, 42))

    def test_materials_retain_procedural_texture_identity_after_override(self):
        stone = get_surface("natural-stone", color="#CCBB99")
        self.assertEqual(stone.texture_kind, "natural-stone")
        self.assertEqual(get_surface("optical-glass").texture_kind, "")

    def test_weathered_masonry_has_distinct_streaked_and_rough_pbr_maps(self):
        weathered = make_pbr_maps("weathered-stone", "#B7AD9D", 0.81, resolution=64)
        clean = make_pbr_maps("natural-stone", "#B7AD9D", 0.81, resolution=64)
        self.assertNotEqual(weathered.base_color_png, clean.base_color_png)
        _, color = decode_png_pixels(weathered.base_color_png)
        _, rough = decode_png_pixels(weathered.roughness_png)
        self.assertGreater(len(set(color[::3])), 12)
        self.assertGreater(len(set(rough[::3])), 9)
        self.assertEqual(weathered, make_pbr_maps("weathered-stone", "#B7AD9D", 0.81, resolution=64))
        self.assertEqual(get_surface("weathered-stone").texture_kind, "weathered-stone")
        clean_weathered = make_pbr_maps("weathered-stone", "#B7AD9D", 0.81,
                                       resolution=64, weathering_strength=0)
        strongly_weathered = make_pbr_maps("weathered-stone", "#B7AD9D", 0.81,
                                           resolution=64, weathering_strength=1)
        self.assertNotEqual(clean_weathered.base_color_png, strongly_weathered.base_color_png)
        self.assertNotEqual(clean_weathered.roughness_png, strongly_weathered.roughness_png)
        for spec in (-0.1, 1.1, True, float("nan")):
            with self.subTest(weathering_strength=spec), self.assertRaises(ValueError):
                make_pbr_maps("weathered-stone", "#B7AD9D", 0.81,
                              resolution=32, weathering_strength=spec)
        for value in (0, 0.13, 0.59, 0.94):
            self.assertAlmostEqual(_stone_weathering(0, value, 81),
                                   _stone_weathering(1, value, 81))
            self.assertAlmostEqual(_stone_weathering(value, 0, 81),
                                   _stone_weathering(value, 1, 81))

    def test_rejects_unsupported_or_excessive_texture_specs(self):
        for spec in (("unknown", "#888888", 0.5, 32),
                     ("natural-stone", "red", 0.5, 32),
                     ("natural-stone", "#888888", 2.0, 32),
                     ("natural-stone", "#888888", 0.5, 300)):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                make_pbr_maps(spec[0], spec[1], spec[2], resolution=spec[3])


if __name__ == "__main__":
    unittest.main()
