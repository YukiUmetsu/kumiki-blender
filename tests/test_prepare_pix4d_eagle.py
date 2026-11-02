"""No Pix4D photographs are stored or redistributed by these tests."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_pix4d_eagle import prepare


class EaglePhotoIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "README.txt").write_text("synthetic test only")
        for index in range(44):
            (self.root / "images" / f"IMG_{index:04d}.JPG").write_bytes(str(index).encode())

    def test_manifest_has_original_per_image_hash_and_restrictive_provenance(self):
        result = prepare(self.root)
        self.assertEqual(result["images"], 44)
        self.assertFalse(result["distribution_allowed"])
        manifest = json.loads((self.root / "reference-set.json").read_text())
        record = json.loads((self.root / "reference-source.json").read_text())
        self.assertEqual(len(manifest["images"]), 44)
        self.assertEqual(manifest["images"][0]["view"], "unknown")
        self.assertEqual(len(manifest["images"][0]["expected_sha256"]), 64)
        self.assertFalse(record["distribution_allowed"])
        self.assertIn("Pix4D", record["provider"])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare(self.root)

    def test_missing_photo_and_changed_archive_do_not_write_dossier(self):
        (self.root / "images/IMG_0000.JPG").unlink()
        with self.assertRaisesRegex(ValueError, "44 original"):
            prepare(self.root)
        (self.root / "images/IMG_0000.JPG").write_bytes(b"restored")
        archive = self.root / "fake.zip"
        archive.write_bytes(b"wrong zip")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare(self.root, archive=archive)
        self.assertFalse((self.root / "reference-set.json").exists())


if __name__ == "__main__":
    unittest.main()
