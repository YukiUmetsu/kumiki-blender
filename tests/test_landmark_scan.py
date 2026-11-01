import json
import math
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.asset_import import AssetImportError
from kumiki_blender.landmark_scan import fit_sphinx_scan, prepare_sphinx_scan
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
