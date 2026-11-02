"""Reference-critical Mount Fuji topography and portable heightmap tests."""

from array import array
import io
import math
from pathlib import Path
import tempfile
import unittest

from kumiki_blender.cli import main
from kumiki_blender.geography.fuji.dem import FujiDem, FujiSpec, load_dem, save_dem
from kumiki_blender.geography.fuji.patches import (
    FujiDetailPatch, FujiPatchSpec, _base_sample, dem_fingerprint,
    load_patch, save_patch, validate_patches,
)
from kumiki_blender.geography.fuji.blender_scene import (ROCK_NORMAL_TILES, _value_noise,
                                                        plan_fuji_shots, terrain_color)
from kumiki_blender.geography.fuji.iconic_scene import (
    ICONIC_BLUE_BASALT, ICONIC_TREE_LINE, SAKURA_PETAL_PALETTE,
    iconic_spec, plan_iconic_shots, _snow_limit,
)


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
        self.assertLess(next(s.ortho_scale for s in shots if s.name == "summit-crater"), 2.6)
        self.assertLess(next(s.ortho_scale for s in shots if s.name == "hoei-crater"), 3.0)
        hoei_view=next(s for s in shots if s.name == "hoei-context")
        self.assertLess(hoei_view.position[2],6.0)  # Raking aerial, not a vertical map view.
        dark = terrain_color(2300, .35, 0, 1)
        white = terrain_color(3750, .12, 0, 0)
        self.assertGreater(sum(white[:3]), sum(dark[:3])*2)
        forest = terrain_color(1350, .12, 3, 4)
        self.assertGreater(forest[1], forest[0])
        self.assertGreater(forest[1], forest[2])
        scree = terrain_color(2660, .31, 2, -2, season="summer")
        self.assertGreater(scree[0], forest[0])
        self.assertGreater(scree[0], scree[2])
        # The winter reference photos instead call for blue-grey volcanic
        # slopes beneath the snow and a larger white snowcap.
        winter_scree = terrain_color(2660, .31, 2, -2, season="winter")
        self.assertGreater(winter_scree[2], winter_scree[0] * 1.12)
        self.assertGreater(winter_scree[0], scree[0] * 2.5)
        self.assertTrue(all(0<=channel<=1 for channel in white))

    def test_forest_noise_deterministic_and_continuous(self):
        self.assertEqual(ROCK_NORMAL_TILES,6)
        at_point = _value_noise(3.4321, -2.8719)
        self.assertAlmostEqual(at_point, _value_noise(3.4321, -2.8719))
        self.assertGreaterEqual(at_point, 0)
        self.assertLessEqual(at_point, 1)
        self.assertLess(abs(at_point-_value_noise(3.4322,-2.8718)),.002)
        # A repeat-free local palette is essential at 26-km view scale.
        self.assertNotAlmostEqual(at_point,_value_noise(4.4321,-2.8719),places=2)

    def test_photo_guided_palette_has_dark_scoria_not_a_uniform_brown_cone(self):
        """Ministry of Environment: black scoria above Fuji's fifth station."""
        trees=terrain_color(1450,.2,2,3)
        other_trees=terrain_color(1450,.2,-5,3)
        basalt=terrain_color(2780,.25,-1,2,season="summer")
        summit_ash=terrain_color(3450,.14,-.3,.4,season="summer")
        hoei=terrain_color(2660,.31,1.61,-2.18,season="summer")
        snow=terrain_color(3700,.12,0,0)
        self.assertGreater(trees[1],trees[0]*1.6)
        self.assertLess(trees[1],.15)  # Dense forest, not mint-green grass.
        # With much darker conifers, measure the spatial canopy contrast
        # relative to its new mean rather than an outdated bright-green delta.
        self.assertGreater(abs(trees[1]-other_trees[1])/max(trees[1],other_trees[1]),.16)
        self.assertLess(max(basalt[:3]),.16)  # Main cone is dark scoria.
        self.assertLess(max(basalt[:3])-min(basalt[:3]),.035)
        # A lit mountain should read as warm/neutral ash, not mauve or violet.
        self.assertGreaterEqual(summit_ash[1],summit_ash[2])
        self.assertLess(summit_ash[0]-summit_ash[1],.035)
        self.assertGreater(hoei[0],basalt[0]*1.2)  # Localized rusty Hōei deposits.
        self.assertGreater(min(snow[:3]),.70)

    def test_user_photo_winter_snow_reaches_lower_gullies_with_cool_shadows(self):
        """Photo direction: white snow, blue-grey flanks, dark conifer foot."""
        shoulder = terrain_color(2700,.4,-.1,.8,season="winter")
        bare_rock = terrain_color(2700,.4,-.1,.8,season="summer")
        lower_slate = terrain_color(2450,.2,.7,-.5,season="winter")
        forest = terrain_color(1800,.2,2,3,season="winter")
        snow = terrain_color(3700,.12,0,0,season="winter")
        self.assertGreater(sum(shoulder[:3]),sum(bare_rock[:3])*2)
        self.assertGreater(lower_slate[2],lower_slate[0]*1.2)
        self.assertLess(max(forest[:3]),.07)
        self.assertGreater(min(snow[:3]),.80)
        self.assertGreater(snow[2],snow[0])
        # Photo close-ups show interrupted snow by exposed volcanic rock ribs,
        # not a single white cone or evenly painted circular snowline.
        upper_ring=[terrain_color(3150,.28,1.4*math.cos(i*math.pi/10),
                                  1.4*math.sin(i*math.pi/10),season="winter")
                    for i in range(20)]
        self.assertGreater(max(c[0] for c in upper_ring)
                           -min(c[0] for c in upper_ring),.35)

    def test_summer_fuji_has_no_snow_at_summit_or_in_hoei_gullies(self):
        """Both seasonal GLBs use measured terrain; only baked albedo differs."""
        locations=((3770,.04,0,0),(3700,.70,.21,-.11),(3420,.16,-.5,.8),
                   (2900,.24,1.61,-2.18),(2240,.18,-2.4,-1.3))
        for height,slope,x,y in locations:
            with self.subTest(height=height):
                summer=terrain_color(height,slope,x,y,season="summer")
                self.assertLess(max(summer[:3]),.26)
                self.assertEqual(summer[3],1)
        winter=terrain_color(3770,.04,0,0)
        summer=terrain_color(3770,.04,0,0,season="summer")
        self.assertGreater(winter[0]-summer[0],.6)
        self.assertEqual(winter,terrain_color(3770,.04,0,0,season="winter"))
        with self.assertRaisesRegex(ValueError,"Unsupported Fuji"):
            terrain_color(3700,.1,0,0,season="monsoon")

    def test_summer_hoei_ash_shoulder_is_localized_and_forest_has_canopy_texture(self):
        # The shoulder center sits on ~2,660 m surveyed terrain; an earlier
        # candidate below 2,250 m never received the intended ochre color.
        shoulder=terrain_color(2660,.25,2.30,-1.84,season="summer")
        remote=terrain_color(2800,.25,3.8,-4.0,season="summer")
        self.assertGreater(shoulder[0],remote[0]*2)
        self.assertGreater(shoulder[1],remote[1]*1.6)
        self.assertGreater(shoulder[0],shoulder[2]*1.6)
        self.assertLess(shoulder[0]/shoulder[1],1.45)  # No saturated yellow-orange blob.
        # These locations are only 60 m apart: high-frequency crown pigment
        # must vary within one forest, not just between broad green regions.
        a=terrain_color(1800,.16,4.2,2.8,season="summer")
        b=terrain_color(1800,.16,4.26,2.8,season="summer")
        self.assertGreater(sum(abs(a[i]-b[i]) for i in range(3)), .002)

    def test_geographic_samples_in_declared_bounds(self):
        spec = FujiSpec()
        for lon, lat in ((138.7274,35.3606), (138.7451,35.34098),
                         (138.75208,35.34294)):
            x,y = spec.xy(lon,lat)
            self.assertLess(abs(x), spec.width_km/2)
            self.assertLess(abs(y), spec.depth_km/2)

    def test_survey_patch_checksums_exact_boundary_and_no_overlap(self):
        dem=self._fixture()
        spec=FujiPatchSpec("summit",6,14,7,15,4,33)
        values=array("f")
        for r in range(spec.rows):
            for c in range(spec.cols):
                h=_base_sample(dem,spec,r,c)
                if 0<r<spec.rows-1 and 0<c<spec.cols-1:
                    h+=12*math.sin(math.pi*r/(spec.rows-1))*math.sin(math.pi*c/(spec.cols-1))
                values.append(h)
        patch=FujiDetailPatch(spec,values,"GSI test fixture",dem_fingerprint(dem),.99)
        validate_patches((patch,),dem,32)
        self.assertGreater(patch.elevations[16*spec.cols+16],_base_sample(dem,spec,16,16))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"summit.bin"
            save_patch(path,patch)
            decoded=load_patch(path)
            self.assertEqual(decoded.elevations,patch.elevations)
            self.assertEqual(decoded.spec,patch.spec)
            path.write_bytes(path.read_bytes()[:-5])
            with self.assertRaises(ValueError):
                load_patch(path)
        overlap_spec=FujiPatchSpec("other",12,18,13,19,2,33)
        other=FujiDetailPatch(overlap_spec,array("f",(_base_sample(dem,overlap_spec,r,c)
                                  for r in range(overlap_spec.rows)
                                  for c in range(overlap_spec.cols))),"fixture",dem_fingerprint(dem),1)
        with self.assertRaisesRegex(ValueError,"Overlapping"):
            validate_patches((patch,other),dem,32)
        duplicate_spec=FujiPatchSpec("summit",18,22,18,22,2,33)
        duplicated=FujiDetailPatch(duplicate_spec,
                                   array("f",(_base_sample(dem,duplicate_spec,r,c)
                                   for r in range(duplicate_spec.rows)
                                   for c in range(duplicate_spec.cols))),"fixture",dem_fingerprint(dem),1)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            validate_patches((patch,duplicated),dem,32)
        wrong=FujiDetailPatch(spec,values,"fixture",123,.99)
        with self.assertRaisesRegex(ValueError,"does not match"):
            validate_patches((wrong,),dem,32)
        values[0]+=1
        with self.assertRaisesRegex(ValueError,"Seam"):
            validate_patches((FujiDetailPatch(spec,values,"fixture",dem_fingerprint(dem),.99),),dem,32)

    def test_plan_command_with_fixture_and_no_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fuji.bin"
            save_dem(path, self._fixture())
            import contextlib
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["plan-fuji", "--dem", str(path), "--detail", "preview"]), 0)
            self.assertIn('"terrain_divisions": 256', output.getvalue())
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["plan-fuji", "--dem", str(path),
                                       "--detail", "preview", "--season", "summer"]), 0)
            self.assertIn('"snow_cover": "none"',output.getvalue())

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
        self.assertGreater(ICONIC_BLUE_BASALT[2],ICONIC_BLUE_BASALT[0]*2)
        self.assertGreater(ICONIC_TREE_LINE[1],ICONIC_TREE_LINE[0]*2)
        self.assertGreater(len(SAKURA_PETAL_PALETTE),4)
        self.assertTrue(all(pink[0]>pink[2] and pink[2]>pink[1]
                            for pink in SAKURA_PETAL_PALETTE))
        with self.assertRaises(ValueError):
            iconic_spec("ultra")


if __name__ == "__main__":
    unittest.main()
