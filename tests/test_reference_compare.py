"""Small deterministic regression for honest multi-view reference sheets."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    from PIL import Image
except ImportError:
    Image = None


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reference_compare.py"


@unittest.skipUnless(Image, "Pillow is required for image comparison sheets")
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
