"""Pinned upstream topology is never confused with rig or clothing helpers."""

import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kumiki_blender.head_sources import makehuman


FIXTURE = ("# This is a mesh file for MakeHuman\n"
           "# This asset was explicitly released as CC0 in september 2020.\n"
           "v 0 7 0\n"
           "v 1 7 0\n"
           "v 0 8 1\n"
           "v 1 8 1\n"
           "v 100 100 100\n"
           "g body\n"
           "f 1/1 2/2 4/3 3/4\n"
           "g helper-hair\n"
           "f 2 4 5\n")


class MakeHumanSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.obj = Path(self.temp.name) / "base.obj"
        self.obj.write_text(FIXTURE, encoding="utf-8")

    def verified(self):
        expected = hashlib.sha256(self.obj.read_bytes()).hexdigest()
        with patch.object(makehuman, "SOURCE_SHA256", expected):
            return makehuman.verify_makehuman_obj(self.obj)

    def test_official_pin_is_fixed_to_commit_and_canonical_digest(self):
        self.assertIn("a8bc2d54ff0ac92e78ff71431b1023eda42bf482", makehuman.SOURCE_URL)
        self.assertEqual(makehuman.SOURCE_SHA256,
                         "8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c")

    def test_modified_or_unlicensed_source_is_rejected(self):
        with self.assertRaisesRegex(makehuman.MakeHumanSourceError, "SHA-256"):
            makehuman.verify_makehuman_obj(self.obj)
        invalid = FIXTURE.replace("released as CC0", "not CC0")
        self.obj.write_text(invalid, encoding="utf-8")
        with patch.object(makehuman, "SOURCE_SHA256", hashlib.sha256(invalid.encode()).hexdigest()):
            with self.assertRaisesRegex(makehuman.MakeHumanSourceError, "CC0"):
                makehuman.verify_makehuman_obj(self.obj)

    def test_parser_keeps_only_body_faces_and_compacts_unused_vertices(self):
        vertices, faces = makehuman.load_body_mesh(self.verified())
        self.assertEqual(len(vertices), 4)
        self.assertEqual(faces, ((0, 1, 3, 2),))
        self.assertEqual(vertices[2], (0.0, 8.0, 1.0))

    def test_rejects_bad_topology_and_nonfinite_coordinates(self):
        for data in (FIXTURE.replace("v 0 8 1", "v nan 8 1"),
                     FIXTURE.replace("f 1/1 2/2 4/3 3/4", "f 1 2 999"),
                     FIXTURE.replace("g body", "g not-body")):
            with self.subTest(data=data[-80:]):
                self.obj.write_text(data, encoding="utf-8")
                source = self.verified()
                with self.assertRaises(makehuman.MakeHumanSourceError):
                    makehuman.load_body_mesh(source)

    def test_explicit_download_verifies_before_caching_then_reuses_local_file(self):
        cache = Path(self.temp.name) / "source-cache" / "base.obj"
        with (patch.object(makehuman, "SOURCE_SHA256", hashlib.sha256(FIXTURE.encode()).hexdigest()),
              patch.object(makehuman, "urlopen", return_value=io.BytesIO(FIXTURE.encode())) as download):
            source = makehuman.fetch_makehuman_obj(cache)
            self.assertEqual(source.sha256, hashlib.sha256(FIXTURE.encode()).hexdigest())
            self.assertEqual(cache.read_text(), FIXTURE)
            makehuman.fetch_makehuman_obj(cache)
            download.assert_called_once()

    def test_unapproved_download_does_not_create_a_cache_file(self):
        cache = Path(self.temp.name) / "source-cache" / "base.obj"
        with patch.object(makehuman, "urlopen", return_value=io.BytesIO(FIXTURE.encode())):
            with self.assertRaisesRegex(makehuman.MakeHumanSourceError, "SHA-256"):
                makehuman.fetch_makehuman_obj(cache)
        self.assertFalse(cache.exists())


if __name__ == "__main__":
    unittest.main()
