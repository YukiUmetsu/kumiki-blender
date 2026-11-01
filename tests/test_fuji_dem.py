"""Reference-critical Mount Fuji topography and portable heightmap tests."""

from array import array
import io
import math
from pathlib import Path
import tempfile
import unittest

from kumiki_blender.cli import main
from kumiki_blender.geography.fuji.dem import FujiDem, FujiSpec, load_dem, save_dem
from kumiki_blender.geography.fuji.blender_scene import _value_noise, plan_fuji_shots, terrain_color
from kumiki_blender.geography.fuji.iconic_scene import iconic_spec, plan_iconic_shots, _snow_limit


class FujiTerrainTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> FujiDem:
        spec = FujiSpec(samples=33)
        values = array("f", (float(500+x*3+y*4) for y in range(33) for x in range(33)))
        return FujiDem(spec, values, "synthetic test fixture")

    def test_metric_extent_and_1_to_1_height_sampling(self):
        dem = self._fixture()
        self.assertAlmostEqual(dem.spec.width_km, 25.97, delta=.25)
        self.assertAlmostEqual(dem.spec.depth_km, 26.03, delta=.25)
        self.assertAlmostEqual(dem.height(dem.spec.longitude_min, dem.spec.latitude_max), 500)
        self.assertAlmostEqual(dem.height(dem.spec.longitude_max, dem.spec.latitude_min), 724)
        with self.assertRaises(ValueError):
            dem.height(140, 35.36)

    def test_checksum_roundtrip_and_truncation_rejected(self):
        original = self._fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fuji.bin"
            save_dem(path, original)
            decoded = load_dem(path)
            self.assertEqual(decoded.elevations, original.elevations)
            self.assertEqual(decoded.source, original.source)
            path.write_bytes(path.read_bytes()[:-11])
            with self.assertRaises(ValueError):
                load_dem(path)

    def test_invalid_elevation_or_size_rejected(self):
        spec = FujiSpec(samples=33)
        with self.assertRaises(ValueError):
            FujiDem(spec, array("f", [0]), "fixture")
        with self.assertRaises(ValueError):
            FujiDem(spec, array("f", [float("nan")]*33**2), "fixture")

    def test_camera_angles_and_snow_palette(self):
        shots = plan_fuji_shots()
        self.assertEqual(shots[0].name, "hero")
        self.assertEqual(len({s.name for s in shots}), 6)
        self.assertIn("hoei-crater", [s.name for s in shots])
        self.assertIn("hoei-context", [s.name for s in shots])
        dark = terrain_color(2300, .35, 0, 1)
        white = terrain_color(3750, .12, 0, 0)
        self.assertGreater(sum(white[:3]), sum(dark[:3])*2)
        forest = terrain_color(1350, .12, 3, 4)
        self.assertGreater(forest[1], forest[0])
        self.assertGreater(forest[1], forest[2])
        scree = terrain_color(2660, .31, 2, -2)
        self.assertGreater(scree[0], forest[0])
        self.assertGreater(scree[0], scree[2])
        self.assertTrue(all(0<=channel<=1 for channel in white))

    def test_forest_noise_deterministic_and_continuous(self):
        at_point = _value_noise(3.4321, -2.8719)
        self.assertAlmostEqual(at_point, _value_noise(3.4321, -2.8719))
        self.assertGreaterEqual(at_point, 0)
        self.assertLessEqual(at_point, 1)
        self.assertLess(abs(at_point-_value_noise(3.4322,-2.8718)),.002)
        # A repeat-free local palette is essential at 26-km view scale.
        self.assertNotAlmostEqual(at_point,_value_noise(4.4321,-2.8719),places=2)

    def test_geographic_samples_in_declared_bounds(self):
        spec = FujiSpec()
        for lon, lat in ((138.7274,35.3606), (138.7451,35.34098),
                         (138.75208,35.34294)):
            x,y = spec.xy(lon,lat)
            self.assertLess(abs(x), spec.width_km/2)
            self.assertLess(abs(y), spec.depth_km/2)

    def test_plan_command_with_fixture_and_no_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fuji.bin"
            save_dem(path, self._fixture())
            import contextlib
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["plan-fuji", "--dem", str(path), "--detail", "preview"]), 0)
            self.assertIn('"terrain_divisions": 256', output.getvalue())

    def test_iconic_does_not_need_dem_and_is_distinct_asset(self):
        import contextlib
        import json
        result=io.StringIO()
        with contextlib.redirect_stdout(result):
            self.assertEqual(main(["plan-fuji", "--style", "iconic", "--detail", "high",
                                   "--dem", "/not/a/real/dem.bin"]), 0)
        spec=json.loads(result.getvalue())
        self.assertEqual(spec["asset_id"],"mount-fuji-iconic-diorama")
        self.assertEqual(spec["footprint_units"],[10.2,8.6])
        self.assertGreater(spec["sakura_blossom_target"],8000)
        self.assertEqual(len(plan_iconic_shots()),7)
        self.assertIn("sakura-botanical",[shot.name for shot in plan_iconic_shots()])
        self.assertEqual(plan_iconic_shots()[0].name,"hero")
        self.assertGreater(_snow_limit(.2),0.25)
        self.assertLess(_snow_limit(.2),.43)
        with self.assertRaises(ValueError):
            iconic_spec("ultra")


if __name__ == "__main__":
    unittest.main()
