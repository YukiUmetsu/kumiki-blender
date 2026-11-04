"""Public CLI boundary for source-neutral reconstruction evidence."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from kumiki_blender.cli import main
from tests.test_reconstruction_colmap import FAKE_COLMAP


class ReconstructionCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "photos").mkdir()
        entries = []
        for index, name in enumerate(("front", "left", "right")):
            image = Image.new("RGB", (120, 90), "white")
            ImageDraw.Draw(image).rectangle((index * 20 + 3, 12, index * 20 + 14, 79), fill="black")
            image.save(self.root / "photos" / f"{name}.png")
            entry = {"id": name, "file": f"photos/{name}.png",
                     "source_url": f"https://example.org/{name}",
                     "view": ("front", "left-profile", "right-profile")[index],
                     "regions": ["face"]}
            if index == 0:
                Image.new("L", (120, 90), 255).save(self.root / "photos" / "mask.png")
                entry["mask"] = "photos/mask.png"
            entries.append(entry)
        self.manifest = self.root / "set.json"
        self.manifest.write_text(json.dumps({"schema_version": 1, "asset_id": "monument-test",
                                            "subject": "Monument reference",
                                            "required_views": ["front", "left-profile", "right-profile"],
                                            "required_regions": ["face", "mouth"],
                                            "images": entries}))
        self.fake = self.root / "colmap-fixture"
        self.fake.write_text(FAKE_COLMAP)
        self.fake.chmod(0o755)

    def test_reference_audit_returns_structured_missing_coverage(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["reference-audit", "--manifest", str(self.manifest)])
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["missing_regions"], ["mouth"])
        self.assertEqual(report["missing_views"], [])
        self.assertFalse(report["historical_fidelity_verified"])

    def test_reconstruction_outputs_candidate_without_calling_blender(self):
        destination = self.root / "r01"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["reconstruct-photos", "--manifest", str(self.manifest),
                         "--colmap", str(self.fake), "--output", str(destination)])
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["registered_images"], 3)
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertTrue((destination / "reconstruction-report.json").exists())

    def test_invalid_manifest_returns_error_without_output_creation(self):
        destination = self.root / "r01"
        self.manifest.write_text("{}")
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            code = main(["reconstruct-photos", "--manifest", str(self.manifest),
                         "--colmap", str(self.fake), "--output", str(destination)])
        self.assertEqual(code, 2)
        self.assertIn("reference", errors.getvalue().lower())
        self.assertFalse(destination.exists())

    def test_photo_preparation_prints_per_photo_qa_progress(self):
        for index in range(8):
            Image.new("RGB", (120, 90), (20 + index, 80, 40)).save(
                self.root / "photos" / f"extra-{index}.png")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["prepare-statue-photos", "--photos", str(self.root / "photos"),
                         "--asset-id", "monument-test", "--output", str(self.root / "prepared"),
                         "--cache-dir", str(self.root / "cache"), "--no-foreground"])
        self.assertEqual(code, 0)
        self.assertIn("Photo QA 1/", err.getvalue())
        self.assertIn("photo-qa-contact-sheet.jpg", out.getvalue())


if __name__ == "__main__":
    unittest.main()
