"""Licensing and integrity contracts for the separate CC0 facial donor."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.liberty_anatomy_source import (
    AnatomySourceError, load_anatomy_source, verify_anatomy_source,
)


class LibertyAnatomySourceTests(unittest.TestCase):
    def test_only_reviewed_cc0_blender_studio_asset_is_used(self):
        record = load_anatomy_source()
        self.assertEqual(record["license"], "CC0-1.0")
        self.assertIn("Dan Ulrich", record["author"])
        self.assertIn("download.blender.org", record["download_url"])
        self.assertEqual(record["member"], "human_base_meshes_bundle.blend")
        self.assertEqual(len(record["sha256"]), 64)
        self.assertEqual(len(record["archive_sha256"]), 64)

    def test_rejects_mismatched_exact_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "head.blend"
            candidate.write_bytes(b"BLENDER" + bytes(20))
            record = {**load_anatomy_source(), "size_bytes": candidate.stat().st_size,
                      "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}
            self.assertEqual(verify_anatomy_source(candidate, record), record)
            candidate.write_bytes(b"BLENDER" + bytes(19) + b"\x01")
            with self.assertRaisesRegex(AnatomySourceError, "SHA-256"):
                verify_anatomy_source(candidate, record)

    def test_zstd_blend_header_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "head.blend"
            candidate.write_bytes(b"\x28\xb5\x2f\xfd" + bytes(20))
            record = {**load_anatomy_source(), "size_bytes": candidate.stat().st_size,
                      "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}
            self.assertEqual(verify_anatomy_source(candidate, record), record)
            alias = Path(tmp) / "alias.blend"
            alias.symlink_to(candidate)
            with self.assertRaisesRegex(AnatomySourceError, "symlink"):
                verify_anatomy_source(alias, record)

    def test_missing_binary_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(AnatomySourceError, "missing"):
                verify_anatomy_source(Path(tmp) / "not-found.blend")


if __name__ == "__main__":
    unittest.main()
