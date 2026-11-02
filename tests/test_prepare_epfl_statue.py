"""EPFL reference intake tests use synthetic bytes, never restricted photos."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_epfl_statue import SETS, prepare


class EpflPhotoIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def populate(self, scene):
        config = SETS[scene]
        folder = self.root / config["folder"]
        folder.mkdir()
        for index in range(config["count"]):
            (folder / f"EPFL photo {index:03d}{config['extension']}").write_bytes(bytes([index]))
        return folder

    def test_exact_counts_and_rights_for_both_sculpture_datasets(self):
        for scene in SETS:
            with self.subTest(scene=scene), tempfile.TemporaryDirectory() as location:
                self.root = Path(location)
                self.populate(scene)
                report = prepare(self.root, scene)
                self.assertFalse(report["distribution_allowed"])
                manifest = json.loads((self.root / "reference-set.json").read_text())
                provenance = json.loads((self.root / "reference-source.json").read_text())
                self.assertEqual(len(manifest["images"]), SETS[scene]["count"])
                self.assertEqual(manifest["images"][0]["view"], "unknown")
                self.assertFalse(provenance["distribution_allowed"])
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    prepare(self.root, scene)

    def test_rejects_missing_photo_and_wrong_archive(self):
        folder = self.populate("sculpted-mask")
        next(folder.glob("*.jpg")).unlink()
        with self.assertRaisesRegex(ValueError, "Expected 32"):
            prepare(self.root, "sculpted-mask")
        (folder / "restored.jpg").write_bytes(b"photo")
        archive = self.root / "unexpected.zip"
        archive.write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare(self.root, "sculpted-mask", archive=archive)
        self.assertFalse((self.root / "reference-set.json").exists())


if __name__ == "__main__":
    unittest.main()
