"""A fixed visual-comparison sheet makes failed sculpt attempts reproducible."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LibertyReferenceComparisonTests(unittest.TestCase):
    def test_sheet_keeps_actual_reference_pixels_and_model_crops_separate(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is optional; comparison tool is not required for core builds")

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            reference = folder / "reference.png"
            render = folder / "render.png"
            output = folder / "sheet.png"
            Image.new("RGB", (500, 500), (245, 245, 245)).save(reference)
            Image.new("RGB", (256, 256), (22, 70, 64)).save(render)
            command = [sys.executable, str(root / "scripts" / "reference_compare.py"),
                       str(reference), str(render), "--ref-crop",
                       "50", "50", "450", "450", "--size", "256",
                       "--output", str(output)]
            complete = subprocess.run(command, capture_output=True, text=True,
                                      timeout=15, check=False)
            self.assertEqual(complete.returncode, 0, complete.stderr)
            with Image.open(output) as image:
                self.assertEqual(image.size, (548, 342))
                self.assertEqual(image.getpixel((120, 140)), (245, 245, 245))
                self.assertEqual(image.getpixel((410, 140)), (22, 70, 64))

            invalid = command.copy()
            invalid[invalid.index("450")] = "550"
            failed = subprocess.run(invalid, capture_output=True, text=True,
                                    timeout=15, check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("invalid crop", failed.stderr)
