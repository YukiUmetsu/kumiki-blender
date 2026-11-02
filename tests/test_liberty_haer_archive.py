"""Offline integrity and conservative photogrammetry-sufficiency checks."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


studio = load("liberty_haer_ref_for_archive_test", "blender_liberty_reference_studio.py")
fetcher = load("liberty_haer_fetch_test", "fetch_liberty_haer.py")
overlap = load("liberty_haer_overlap_test", "liberty_haer_photo_overlap.py")


class LibertyHaerArchiveTests(unittest.TestCase):
    def test_reviewed_source_set_matches_checksum_manifest(self):
        expected = {int(v.survey_id.rsplit("-", 1)[-1]) for v in studio.ALL_VIEWS}
        self.assertEqual(expected, set(fetcher.EXPECTED_SHA256))
        self.assertEqual(expected, set(overlap.HEAD_ROIS) | set(range(177, 182)))
        self.assertTrue(all(len(digest) == 64 for digest in fetcher.EXPECTED_SHA256.values()))

    def test_integrity_fails_closed_for_invalid_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.jpg"
            path.write_bytes(b"not a jpeg")
            with self.assertRaisesRegex(ValueError, "not a JPEG"):
                fetcher.verify(path, 236)
            path.write_bytes(b"\xff\xd8\xff" + b"tampered")
            with self.assertRaisesRegex(ValueError, "reviewed SHA-256"):
                fetcher.verify(path, 236)

    def test_matching_is_only_exploratory(self):
        self.assertEqual(overlap.classify_overlap(6, 0, 0), "weak_correspondence")
        self.assertEqual(overlap.classify_overlap(23, 7, 10), "weak_correspondence")
        self.assertEqual(overlap.classify_overlap(88, 79, 76), "homography_dominated")
        self.assertEqual(overlap.classify_overlap(45, 2, 25), "unverified_candidate")


if __name__ == "__main__":
    unittest.main()
