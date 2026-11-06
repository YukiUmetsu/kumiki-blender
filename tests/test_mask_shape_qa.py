"""Mask silhouette diagnostics are conservative and read-only."""
import unittest

from PIL import Image

from kumiki_blender.reconstruction.mask_shape_qa import inspect_mask_shape


class MaskShapeTests(unittest.TestCase):
    def test_enclosed_background_holes_are_reported_but_not_filled(self):
        mask = Image.new('L', (200, 200), 0)
        mask.paste(255, (30, 20, 170, 185))
        mask.paste(0, (85, 75, 115, 105))
        before = mask.tobytes()
        result = inspect_mask_shape(mask)
        self.assertEqual(mask.tobytes(), before)
        self.assertEqual(result['interior_background_regions'], 1)
        self.assertGreater(result['interior_background_fraction'], 0.02)
        self.assertIn('interior-background-review', result['review_flags'])
        self.assertNotIn('bottom-border-foreground-review', result['review_flags'])

    def test_detached_foreground_and_bottom_border_are_review_only(self):
        mask = Image.new('L', (200, 200), 0)
        mask.paste(255, (40, 20, 120, 120))
        mask.paste(255, (170, 30, 190, 70))
        mask.paste(255, (0, 190, 200, 200))
        result = inspect_mask_shape(mask)
        self.assertGreaterEqual(result['significant_foreground_components'], 3)
        self.assertIn('detached-foreground-review', result['review_flags'])
        self.assertIn('bottom-border-foreground-review', result['review_flags'])
        self.assertIn('can be', result['limitations'])

    def test_legitimate_open_to_background_gap_is_not_an_enclosed_hole(self):
        mask = Image.new('L', (200, 200), 0)
        mask.paste(255, (20, 20, 180, 180))
        mask.paste(0, (85, 80, 115, 200))
        result = inspect_mask_shape(mask)
        self.assertEqual(result['interior_background_regions'], 0)
        self.assertNotIn('interior-background-review', result['review_flags'])

    def test_bounded_resampling_does_not_mutate_large_input(self):
        mask = Image.new('L', (1600, 2400), 0)
        mask.paste(255, (200, 400, 1400, 2000))
        original = mask.getpixel((500, 500))
        result = inspect_mask_shape(mask)
        self.assertEqual(max(result['sampled_dimensions']), 384)
        self.assertEqual(mask.getpixel((500, 500)), original)
        self.assertGreater(result['foreground_fraction'], 0)


if __name__ == '__main__':
    unittest.main()
