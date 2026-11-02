"""Rights-aware CC BY 4.0 photogrammetry intake; synthetic files only."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_cycladic_figurine import PHOTO_COUNT, SOURCE_PAGE, prepare


class CycladicPhotoIntakeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "original/photos").mkdir(parents=True)
        for index in range(PHOTO_COUNT):
            (self.root / "original/photos" / f"photo_{index:03d}.JPG").write_bytes(
                f"test photo {index}".encode())

    def test_72_images_are_sha_pinned_and_require_attribution(self):
        result = prepare(self.root)
        self.assertEqual(result["images"], 72)
        self.assertFalse(result["distribution_allowed"])
        manifest = json.loads((self.root / "reference-set.json").read_text())
        rights = json.loads((self.root / "reference-source.json").read_text())
        self.assertEqual(manifest["asset_id"], "wyoming-cycladic-figurine-cc-by-4")
        self.assertEqual(len(manifest["images"]), 72)
        self.assertEqual(manifest["images"][0]["view"], "unknown")
        first = self.root / manifest["images"][0]["file"]
        self.assertEqual(manifest["images"][0]["expected_sha256"],
                         hashlib.sha256(first.read_bytes()).hexdigest())
        self.assertEqual(manifest["images"][0]["source_url"], SOURCE_PAGE)
        self.assertEqual(rights["license"], "CC BY 4.0")
        self.assertIn("Hutchens", rights["attribution"])
        self.assertFalse(rights["distribution_allowed"])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare(self.root)

    def test_rejects_incomplete_source_symlink_and_modified_archive(self):
        photo = self.root / "original/photos/photo_000.JPG"
        photo.unlink()
        with self.assertRaisesRegex(ValueError, "exactly 72"):
            prepare(self.root)
        photo.write_bytes(b"restored")
        archive = self.root / "cycladic_figurine_photo_dataset.zip"
        archive.write_bytes(b"unverified archive")
        with self.assertRaisesRegex(ValueError, "publisher's recorded MD5"):
            prepare(self.root, archive=archive)
        archive.unlink()
        photo.unlink()
        photo.symlink_to(self.root / "original/photos/photo_001.JPG")
        with self.assertRaisesRegex(ValueError, "exactly 72"):
            prepare(self.root)
        self.assertFalse((self.root / "reference-set.json").exists())


if __name__ == "__main__":
    unittest.main()
