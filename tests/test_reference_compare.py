"""Photo/render QA sheets are honest about unmatched camera projections."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.reference_compare import comparison_sheet


class ReferenceComparisonTests(unittest.TestCase):
    def test_preserves_aspect_ratio_and_labels_unregistered_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            photo=path / "photo.jpg"
            render=path / "render.png"
            out=path / "qa" / "sheet.png"
            Image.new("RGB", (1200, 700), (90, 40, 15)).save(photo)
            Image.new("RGB", (800, 600), (15, 60, 95)).save(render)
            comparison_sheet(photo,render,out,panel_width=320,panel_height=240)
            with Image.open(out) as sheet:
                self.assertEqual(sheet.size,(700,392))
                self.assertEqual(sheet.mode,"RGB")
            with self.assertRaisesRegex(ValueError,"at least 128"):
                comparison_sheet(photo,render,out,panel_width=80)


if __name__ == "__main__":
    unittest.main()
