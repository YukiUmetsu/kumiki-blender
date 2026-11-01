"""Contract for replacing only the old torch with the detailed assembly."""

import unittest

from kumiki_blender.liberty import make_liberty_study
from kumiki_blender.liberty_torch_placement import (
    detailed_torch_placement, detailed_torch_shots, replace_legacy_torch_part,
)


class LibertyTorchPlacementTests(unittest.TestCase):
    def test_replace_gallery_flutes_flame_but_preserve_gripping_hand(self):
        study = make_liberty_study()
        removed = [p.name for p in study.instances if replace_legacy_torch_part(p.name)]
        kept = [p.name for p in study.instances if not replace_legacy_torch_part(p.name)]
        self.assertIn("liberty-torch-handle", removed)
        self.assertIn("liberty-torch-flame-core", removed)
        self.assertTrue(any("gallery-baluster" in name for name in removed))
        self.assertEqual(sum("torch-grip-finger" in name for name in kept), 4)
        self.assertIn("liberty-torch-grip-thumb", kept)
        self.assertIn("liberty-sleeve-torch", kept)

    def test_new_shaft_gauge_gallery_and_flame_fit_old_gripped_pose(self):
        study = make_liberty_study()
        placement = detailed_torch_placement(study)
        self.assertAlmostEqual(placement.origin[0], -3.45, places=2)
        self.assertAlmostEqual(placement.origin[1], 0.0, places=2)
        self.assertAlmostEqual(placement.origin[2], 32.7, places=2)
        self.assertGreater(placement.radial_scale, 2.5)
        self.assertLess(placement.radial_scale, 2.9)
        self.assertGreater(placement.height_scale, 1.6)
        self.assertLess(placement.height_scale, 1.9)

    def test_torch_adapts_to_scaled_and_pedestalless_statue(self):
        full = detailed_torch_placement(make_liberty_study())
        half = detailed_torch_placement(make_liberty_study(scale=.5))
        naked = detailed_torch_placement(make_liberty_study(include_pedestal=False))
        self.assertAlmostEqual(half.origin[0], full.origin[0]*.5)
        self.assertAlmostEqual(half.radial_scale, full.radial_scale*.5)
        self.assertAlmostEqual(half.height_scale, full.height_scale*.5)
        self.assertLess(naked.origin[2], full.origin[2] - 8)

    def test_detailed_gallery_flame_and_grip_get_individual_qa(self):
        shots = detailed_torch_shots(make_liberty_study())
        self.assertEqual([shot.name for shot in shots],
                         ["torch-detail-front", "torch-detail-profile", "torch-detail-low",
                          "torch-flame-front", "torch-grip-front", "torch-grip-profile"])
        self.assertTrue(all(abs(shot.target[0]+3.45) < .01 for shot in shots))
        self.assertTrue(all(shot.ortho_scale < 4.0 for shot in shots))
        small = detailed_torch_shots(make_liberty_study(scale=.5))
        self.assertAlmostEqual(small[0].ortho_scale, shots[0].ortho_scale*.5)


if __name__ == "__main__":
    unittest.main()
