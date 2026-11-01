"""Geographic shape, mesh, and Blender-adapter contracts for the Egypt diorama."""

import contextlib
import io
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.asset_catalog import find_asset, load_catalog, revision_path
from kumiki_blender.cli import main
from kumiki_blender.egypt_details import MeshSketch, district_layout
from kumiki_blender.egypt_map import (
    EgyptMapSpec, BORDER_LINES, CITY_MARKERS, LABELS, RIVER_CHANNELS, geo_to_xy,
    sample_terrain, terrain_mesh, xy_to_geo,
)
from kumiki_blender.blender_egypt_map import (
    _color_atlas_png, _label_hover_height, _river_ribbon,
    _terrain_contact_z, _vertex_islands,
    build_egypt_map_scene, plan_egypt_shots,
)


class EgyptMapTests(unittest.TestCase):
    def test_legacy_entry_points_delegate_to_categorized_geography_modules(self):
        from kumiki_blender.geography.egypt.terrain import sample_terrain as geographic_sample
        from kumiki_blender.geography.egypt.geometry import MeshSketch as geometry_sketch
        from kumiki_blender.geography.egypt.blender_scene import (
            build_egypt_map_scene as categorized_builder,
        )
        self.assertIs(sample_terrain, geographic_sample)
        self.assertIs(MeshSketch, geometry_sketch)
        self.assertIs(build_egypt_map_scene, categorized_builder)

    def test_atlas_is_discoverable_as_geography_map_in_new_asset_catalog(self):
        asset = find_asset(load_catalog(), "egypt-relief-atlas")
        self.assertEqual((asset.category, asset.subcategory), ("geography", "maps"))
        self.assertEqual(asset.plan_command, ("plan-egypt-map",))
        self.assertEqual(asset.build_command, ("build-egypt-map",))
        self.assertTrue(any(path.endswith("geography/egypt/blender_scene.py")
                            for path in asset.sources))
        self.assertEqual(str(revision_path(asset, "r01")),
                         ".review/geography/maps/egypt-relief-atlas/r01")

    def test_reference_geography_preserves_seas_sinai_delta_and_lake(self):
        locations = (
            ((29.0, 32.0), "mediterranean"),
            ((35.5, 25.4), "red-sea"),
            ((37.5, 26.1), "red-sea"),
            ((32.88, 29.4), "red-sea"),  # Gulf of Suez
            ((35.10, 29.1), "red-sea"),  # Gulf of Aqaba
            ((33.85, 28.9), "highlands"),  # Sinai, between the gulfs
            ((32.9, 23.0), "lake-nasser"),
            ((30.64, 29.49), "lake-qarun"),
            ((31.22, 30.4), "delta"),
            ((27.1, 26.0), "desert"),
        )
        for (lon, lat), expected in locations:
            with self.subTest(location=(lon, lat)):
                self.assertEqual(sample_terrain(lon, lat).kind, expected)

    def test_qattara_is_below_its_raised_rim_and_eastern_desert_is_rugged(self):
        basin = sample_terrain(28.4, 29.45).height_m
        edge = sample_terrain(28.5, 28.65).height_m
        west = sample_terrain(27.0, 26.0).height_m
        east = sample_terrain(34.2, 26.0).height_m
        self.assertLess(basin + .20, edge)
        self.assertGreater(sample_terrain(28.4, 29.45).rgb[1],
                           sample_terrain(28.5, 28.65).rgb[1] + .04)
        self.assertGreater(east, west + .14)

    def test_oblique_map_has_readable_peaks_and_exaggerated_western_escarpments(self):
        self.assertGreater(sample_terrain(33.85, 28.9).height_m, .45)
        self.assertGreater(sample_terrain(34.2, 26).height_m, .85)
        self.assertGreater(sample_terrain(28.5, 28.65).height_m,
                           sample_terrain(28.4, 29.45).height_m + .15)
        self.assertGreater(sample_terrain(27, 26).height_m, .28)

    def test_western_desert_has_distinct_rock_scarps_not_flat_sand_plane(self):
        northern_scarp = sample_terrain(28.5, 27.8).height_m
        adjacent_plain = sample_terrain(28.5, 26.8).height_m
        southern_scarp = sample_terrain(29.0, 25.6).height_m
        southern_plain = sample_terrain(29.0, 24.6).height_m
        self.assertGreater(northern_scarp, adjacent_plain + .08)
        self.assertGreater(southern_scarp, southern_plain + .06)

    def test_red_sea_shore_has_coastal_shelf_not_sawtooth_cliffs(self):
        coast = sample_terrain(34.98, 25.35)
        self.assertNotEqual(coast.kind, "red-sea")
        self.assertLess(coast.height_m, .35)

    def test_lake_nasser_has_low_erosion_banks_instead_of_artificial_spires(self):
        bank = sample_terrain(33.25, 23.50)
        self.assertEqual(bank.kind, "desert")
        self.assertLess(bank.height_m, .20)

    def test_nile_valley_is_green_and_lower_than_desert(self):
        luxor = sample_terrain(32.65, 25.7)
        desert = sample_terrain(30.2, 25.7)
        self.assertIn(luxor.kind, ("nile", "river"))
        self.assertGreater(luxor.vegetation, .6)
        self.assertLess(luxor.height_m, desert.height_m + .12)
        self.assertLess(desert.vegetation, .2)

    def test_geo_projection_is_east_positive_north_positive(self):
        west = geo_to_xy(24.0, 27)
        east = geo_to_xy(38.0, 27)
        north = geo_to_xy(31, 32.6)
        south = geo_to_xy(31, 21.7)
        self.assertAlmostEqual(east[0] - west[0], 16.0)
        self.assertAlmostEqual(north[1] - south[1], 11.4)
        self.assertAlmostEqual(west[1], east[1])
        longitude, latitude = 32.35, 26.4
        back = xy_to_geo(*geo_to_xy(longitude, latitude))
        self.assertAlmostEqual(back[0], longitude)
        self.assertAlmostEqual(back[1], latitude)

    def test_height_and_colors_are_deterministic_finite_and_bounded(self):
        for lon, lat in ((24, 22), (31.2, 30), (35.1, 28.4), (38, 32.6), (28.5, 29.4)):
            a = sample_terrain(lon, lat)
            self.assertEqual(a, sample_terrain(lon, lat))
            self.assertTrue(math.isfinite(a.height_m))
            self.assertGreaterEqual(a.height_m, .015)
            self.assertLess(a.height_m, 2.0)
            self.assertEqual(len(a.rgb), 3)
            self.assertTrue(all(0.0 <= channel <= 1.0 for channel in a.rgb))
        for invalid in ((float("nan"), 25), (31, float("inf")), (100, 30)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sample_terrain(*invalid)

    def test_small_grid_has_stitched_edges_and_outward_faces(self):
        mesh = terrain_mesh(EgyptMapSpec(columns=24, rows=18))
        self.assertEqual(len(mesh.vertices), 25 * 19)
        self.assertEqual(len(mesh.faces), 24 * 18)
        self.assertEqual(len(mesh.face_kinds), len(mesh.faces))
        self.assertTrue(all(len(set(face)) == 4 for face in mesh.faces))
        for face in mesh.faces:
            a, b, c = [mesh.vertices[index] for index in face[:3]]
            normal_z = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            self.assertGreater(normal_z, 0)
        self.assertIn("mediterranean", mesh.face_kinds)
        self.assertIn("desert", mesh.face_kinds)

    def test_water_quads_have_low_vertices_no_painted_mountain_spires(self):
        mesh = terrain_mesh(EgyptMapSpec(columns=48, rows=36))
        water = {"mediterranean", "red-sea", "lake-nasser", "lake-qarun"}
        heights = [max(mesh.vertices[index][2] for index in face)
                   for face, kind in zip(mesh.faces, mesh.face_kinds) if kind in water]
        self.assertGreater(len(heights), 80)
        self.assertLess(max(heights), .12)

    def test_design_has_multiple_waterways_and_legible_cinematic_shots(self):
        self.assertIn("main-nile", RIVER_CHANNELS)
        self.assertIn("rosetta", RIVER_CHANNELS)
        self.assertIn("damietta", RIVER_CHANNELS)
        self.assertIn("suez-canal", RIVER_CHANNELS)
        self.assertIn("libya-egypt", BORDER_LINES)
        self.assertGreaterEqual(len(CITY_MARKERS), 5)
        self.assertIn("Qattara Depression", {entry.text for entry in LABELS})
        self.assertIn("Sinai", {entry.text for entry in LABELS})
        shots = plan_egypt_shots()
        self.assertEqual(shots[0].name, "hero")
        self.assertEqual({shot.name for shot in shots}, {"hero", "atlas", "delta", "red-sea",
                                                         "cairo", "luxor", "alexandria", "aswan",
                                                         "sharm", "qattara", "lake-nasser",
                                                         "eastern-peaks"})
        self.assertTrue(all(shot.ortho_scale > 1 for shot in shots))

    def test_city_district_is_dense_but_street_spaced_and_reproducible(self):
        first = district_layout(10, 9, .078, 18)
        self.assertEqual(first, district_layout(10, 9, .078, 18))
        self.assertGreater(len(first), 55)
        self.assertLess(len(first), 90)
        self.assertEqual({building.palette for building in first}, {0, 1, 2})
        self.assertGreater(len({round(building.height, 3) for building in first}), 10)
        self.assertTrue(any(building.rooftop for building in first))
        self.assertTrue(all(building.width < .078 and building.depth < .078
                            and building.height > 0 for building in first))
        with self.assertRaises(ValueError):
            district_layout(1, 200, .01, 18)

    def test_masonry_dome_obelisk_and_palm_are_real_zoomable_meshes(self):
        solid = MeshSketch()
        solid.box(0, 0, 0, .2, .16, .12)
        solid.tapered_prism(0, 0, .12, .16, .12, .13, taper=.7)
        solid.lathe(0, 0, .25, ((0, .08), (.05, .06), (.10, .05)), sides=12)
        solid.hemisphere(0, 0, .35, .07)
        solid.obelisk(.2, .2, .1, .04, .2)
        self.assertGreater(len(solid.vertices), 150)
        self.assertGreater(len(solid.faces), 125)
        self.assertTrue(all(0 <= i < len(solid.vertices)
                            for face in solid.faces for i in face))
        trunks, leaves = MeshSketch(), MeshSketch()
        trunks.palm(.2, .1, .3, .15, trunk_only=True)
        leaves.palm(.2, .1, .3, .15, leaf_only=True)
        self.assertGreater(len(trunks.faces), 15)
        self.assertGreater(len(leaves.faces), 30)
        self.assertTrue(all(math.isfinite(coord)
                            for vertex in solid.vertices+trunks.vertices+leaves.vertices
                            for coord in vertex))

    def test_atlas_is_embeddable_png_and_river_has_a_real_strip_surface(self):
        image = _color_atlas_png(32, 24)
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(struct.unpack(">II", image[16:24]), (32, 24))
        self.assertIn(b"IDAT", image)
        vertices, faces = _river_ribbon("main-nile", RIVER_CHANNELS["main-nile"])
        self.assertGreater(len(vertices), 400)
        self.assertEqual(len(faces), len(vertices)//2-1)
        self.assertTrue(all(len(face) == 4 for face in faces))
        self.assertTrue(all(math.isfinite(n) for vertex in vertices for n in vertex))

    def test_label_contact_z_tracks_real_relief_instead_of_sinking_through_hills(self):
        low_x, low_y = geo_to_xy(32.65, 25.70)
        ridge_x, ridge_y = geo_to_xy(34.2, 26.0)
        self.assertAlmostEqual(_terrain_contact_z(low_x, low_y, .08),
                               sample_terrain(32.65, 25.70).height_m+.08)
        self.assertGreater(_terrain_contact_z(ridge_x, ridge_y, .08),
                           _terrain_contact_z(low_x, low_y, .08) + .2)

    def test_converted_glyph_islands_preserve_letter_boundaries(self):
        parts = _vertex_islands(8, ((0, 1), (1, 2), (3, 4), (4, 5), (5, 3), (6, 7)))
        self.assertEqual(parts, ((0, 1, 2), (3, 4, 5), (6, 7)))

    def test_entire_label_uses_one_hover_height_to_avoid_stacked_glyphs(self):
        low = geo_to_xy(32.65, 25.7)
        ridge = geo_to_xy(34.2, 26.0)
        lift = _label_hover_height((low, ridge), .08)
        self.assertAlmostEqual(lift, _terrain_contact_z(*ridge, .08))
        self.assertGreater(lift, _terrain_contact_z(*low, .08) + .2)

    def test_cli_plans_without_blender_and_rejects_missing_runtime(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(main(["plan-egypt-map"]), 0)
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan["generator"], "egypt-relief-map")
        self.assertEqual(plan["terrain_grid"], [480, 352])
        self.assertGreaterEqual(plan["terrain_vertices"], 50_000)
        self.assertGreaterEqual(plan["waterways"], 4)
        high_stream = io.StringIO()
        with contextlib.redirect_stdout(high_stream):
            self.assertEqual(main(["plan-egypt-map", "--detail", "high"]), 0)
        self.assertEqual(json.loads(high_stream.getvalue())["terrain_grid"], [700, 420])
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "revision"
            with self.assertRaisesRegex(RuntimeError, "Blender"):
                build_egypt_map_scene(destination, bpy_module=None)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
