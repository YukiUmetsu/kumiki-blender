"""Photo/render QA sheets are honest about unmatched camera projections."""

import tempfile
import importlib.util
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



SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reference_compare.py"


class ReferenceCompareTests(unittest.TestCase):
    def test_side_by_side_and_registered_overlay_dimensions(self):
        spec = importlib.util.spec_from_file_location("reference_compare", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref, rendered = root / "ref.png", root / "render.png"
            Image.new("RGB", (100, 50), "#ffffff").save(ref)
            Image.new("RGB", (50, 100), "#333333").save(rendered)
            for aligned, expected_width in ((False, 188), (True, 272)):
                out = root / ("aligned.png" if aligned else "unaligned.png")
                module.compare(ref, rendered, out, size=64, aligned=aligned)
                with Image.open(out) as result:
                    self.assertEqual(result.size, (expected_width, 154))


if __name__ == "__main__":
    unittest.main()
