import json
import math
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.asset_import import AssetImportError
from kumiki_blender.landmark_scan import (
    fit_sphinx_scan, prepare_sphinx_scan, sphinx_proportion_review,
    sphinx_source_readiness_review,
)
from tests.test_asset_import import fixture_glb, provenance_json


class LandmarkScanTests(unittest.TestCase):
    def test_all_front_axes_fit_the_same_cropped_statue(self):
        for front_axis, min_corner, max_corner in (
            ("-X", (-5, -1.5, 2), (5, 1.5, 6)),
            ("+X", (-5, -1.5, 2), (5, 1.5, 6)),
            ("-Y", (-1.5, -5, 2), (1.5, 5, 6)),
            ("+Y", (-1.5, -5, 2), (1.5, 5, 6)),
        ):
            with self.subTest(front_axis=front_axis):
                alignment = fit_sphinx_scan(
                    min_corner, max_corner, front_axis=front_axis,
                    target_length_m=20, target_center_xy=(0, -30))
                actual_min, actual_max = alignment.aligned_bounds_m
                self.assertAlmostEqual(actual_max[0]-actual_min[0], 20)
                self.assertAlmostEqual(actual_max[1]-actual_min[1], 6)
                self.assertAlmostEqual(actual_max[2]-actual_min[2], 8)
                self.assertAlmostEqual((actual_max[1]+actual_min[1])/2, -30)
                self.assertAlmostEqual(actual_min[2], .02)
                self.assertEqual(alignment.uniform_scale, 2)
                self.assertTrue(math.isfinite(alignment.to_manifest()["rotation_z_deg"]))

    def test_asymmetric_bounds_align_center_and_ground(self):
        alignment = fit_sphinx_scan(
            (1, 4, -3), (8, 6, .5), front_axis="+X",
            target_length_m=10.5, target_center_xy=(2, -11))
        lo, hi = alignment.aligned_bounds_m
        self.assertAlmostEqual((lo[0]+hi[0])/2, 2)
        self.assertAlmostEqual((lo[1]+hi[1])/2, -11)
        self.assertAlmostEqual(lo[2], .02)

    def test_source_geometry_review_detects_replica_like_wrong_ratios(self):
        real_ratio = fit_sphinx_scan(
            (0, 0, 0), (73.5, 19.3, 20.0),
            front_axis="-X", target_length_m=10.5, target_center_xy=(0, 0))
        reviewed = sphinx_proportion_review(real_ratio)
        self.assertTrue(reviewed["passes_bounding_envelope_screen"])
        self.assertIn("anatomical", reviewed["status"])
        self.assertIn("replica", reviewed["unverified_details"][-1])

        # Identical scan length, but an over-wide headdress or extensive site
        # mesh raises the bbox ratio. This screen does not certify details.
        inflated = fit_sphinx_scan(
            (0, 0, 0), (73.5, 35.0, 28.0),
            front_axis="-X", target_length_m=10.5, target_center_xy=(0, 0))
        warning = sphinx_proportion_review(inflated)
        self.assertFalse(warning["passes_bounding_envelope_screen"])
        self.assertGreater(warning["relative_deviation"]["width"], .5)
        with self.assertRaises(AssetImportError):
            sphinx_proportion_review(real_ratio, tolerance=float("nan"))

    def test_low_poly_illustrative_proxy_cannot_be_mistaken_for_cinematic_sphinx(self):
        alignment = fit_sphinx_scan(
            (0, 0, 0), (73.5, 19.3, 20.0),
            front_axis="-X", target_length_m=73.5,
            target_center_xy=(0, 0))
        toy = sphinx_source_readiness_review(
            alignment, mesh_objects=4, vertex_count=1000,
            triangle_count=1800, uv_mesh_objects=4, image_textures=0)
        self.assertFalse(toy["technical_screen_passed"])
        self.assertFalse(toy["monument_source_authenticated"])
        self.assertIn("closeup_triangle_floor", toy["checks"])
        self.assertFalse(toy["checks"]["has_image_textures"])
        self.assertEqual(len(toy["issues"]), 2)

        plausible = sphinx_source_readiness_review(
            alignment, mesh_objects=2, vertex_count=80_000,
            triangle_count=150_000, uv_mesh_objects=2, image_textures=4)
        self.assertTrue(plausible["technical_screen_passed"])
        self.assertFalse(plausible["monument_source_authenticated"])
        self.assertIn("review", plausible["status"])
        self.assertEqual(plausible["issues"], [])

    def test_source_screen_rejects_inconsistent_mesh_counts(self):
        alignment = fit_sphinx_scan(
            (0, 0, 0), (73.5, 19.3, 20.0),
            front_axis="-X", target_length_m=73.5,
            target_center_xy=(0, 0))
        normal = dict(mesh_objects=2, vertex_count=100, triangle_count=100,
                      uv_mesh_objects=2, image_textures=1)
        for change in ({"mesh_objects": True}, {"triangle_count": -1},
                       {"uv_mesh_objects": 3}, {"mesh_objects": 0},
                       {"triangle_count": 1}, {"image_textures": 1.5}):
            with self.subTest(change=change), self.assertRaises(AssetImportError):
                sphinx_source_readiness_review(alignment, **(normal | change))

    def test_bad_axis_dims_and_silhouette_are_rejected(self):
        arguments = dict(bounds_min=(-5., -1., 0.), bounds_max=(5., 1., 4.),
                         front_axis="-X", target_length_m=10.5, target_center_xy=(0., 0.))
        for changes in (
            {"front_axis": "Z"}, {"target_length_m": float("nan")},
            {"target_center_xy": (0., float("inf"))},
            {"bounds_max": (5., 1., 0.)},
            {"bounds_min": (5., -1., 0.)},
            {"bounds_max": (5., 20., 4.)},
            {"bounds_max": (5., 1., .1)},
        ):
            with self.subTest(changes=changes), self.assertRaises(AssetImportError):
                fit_sphinx_scan(**(arguments | changes))

    def test_provenance_and_commercial_rights_are_checked_before_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, record = root / "statue.glb", root / "source.json"
            fixture_glb(source, materials=["Weathered limestone"])
            for changes in (
                {}, {"commercial_use_allowed": False},
                {"commercial_use_allowed": "true"},
                {"commercial_use_allowed": True, "license": "CC-BY-NC-4.0"},
                {"commercial_use_allowed": True, "license": "CC-BY-ND-4.0"},
            ):
                record.write_text(json.dumps(provenance_json(**changes)))
                with self.subTest(changes=changes), self.assertRaises(AssetImportError):
                    prepare_sphinx_scan(source, record)
            record.write_text(json.dumps(provenance_json(commercial_use_allowed=True)))
            scan = prepare_sphinx_scan(source, record, front_axis="+Y")
            self.assertEqual(scan.front_axis, "+Y")
            self.assertEqual(scan.inspection["material_names"], ["Weathered limestone"])
            self.assertTrue(scan.provenance.to_manifest(scan.inspection["input_sha256"])
                            ["commercial_use_allowed"])


if __name__ == "__main__":
    unittest.main()
