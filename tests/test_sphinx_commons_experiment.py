"""Source-grounded Sphinx photo experiment: offline provenance and safety tests."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiment_sphinx_commons_2010.py"
SPEC = importlib.util.spec_from_file_location("experiment_sphinx_commons_2010", SCRIPT)
assert SPEC and SPEC.loader
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class SphinxCommonsExperimentTests(unittest.TestCase):
    def test_wikimedia_original_urls_are_stable_and_attributed(self):
        self.assertEqual(
            experiment.original_url("Great_Sphinx_2010.jpg"),
            "https://upload.wikimedia.org/wikipedia/commons/0/03/Great_Sphinx_2010.jpg",
        )
        self.assertEqual(len(experiment.SOURCES), 5)
        self.assertEqual(len(set(experiment.SOURCES)), 5)
        self.assertTrue(all(experiment.commons_page(name).startswith(
            "https://commons.wikimedia.org/wiki/File:") for name in experiment.SOURCES))

    def test_local_photos_bypass_network_and_keep_provenance_unverified(self):
        from unittest.mock import patch
        from kumiki_blender.reconstruction.references import load_reference_set

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            originals = root / "local"
            originals.mkdir()
            for index, extension in enumerate(("jpg", "png", "webp"), 1):
                Image.new("RGB", (960, 720), (30 * index, 70, 90)).save(
                    originals / f"photo-{index:04d}.{extension}"
                )
            target = root / "revision"
            with patch.object(experiment.urllib.request, "urlopen",
                              side_effect=AssertionError("network should not be called")):
                manifest, metadata = experiment.prepare_local_sources(target, originals)
            self.assertEqual(len(metadata), 3)
            self.assertEqual(len(load_reference_set(manifest).images), 3)
            entries = json.loads(manifest.read_text())["images"]
            self.assertTrue(all(entry["source_kind"] == "user-upload" for entry in entries))
            self.assertTrue(all("source_url" not in entry for entry in entries))
            self.assertTrue(all(entry["expected_sha256"] for entry in entries))
            self.assertIn("not", (target / "CREDITS.md").read_text().lower())
            self.assertFalse((target / "sparse-points.glb").exists())

    def test_local_photos_reject_insufficient_views(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            originals = root / "local"
            originals.mkdir()
            Image.new("RGB", (960, 720)).save(originals / "one.jpg")
            with self.assertRaisesRegex(ValueError, "Need 3"):
                experiment.prepare_local_sources(root / "revision", originals)

    def test_wikimedia_429_is_not_retried(self):
        from unittest.mock import patch
        from urllib.error import HTTPError
        with tempfile.TemporaryDirectory() as folder:
            error = HTTPError("https://upload.wikimedia.org/", 429, "Too Many Requests",
                              {"Retry-After": "120"}, None)
            with patch.object(experiment.urllib.request, "urlopen", side_effect=error) as mocked:
                with patch.object(experiment.time, "sleep") as sleep:
                    with self.assertRaisesRegex(RuntimeError, "Use --photos-dir"):
                        experiment.download_photo(experiment.SOURCES[0],
                                                  Path(folder) / "test.jpg")
                    mocked.assert_called_once()
                    sleep.assert_not_called()

    def test_offline_sources_generate_valid_provenance_manifest_without_mesh(self):
        from kumiki_blender.reconstruction.references import load_reference_set
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            photos = root / "originals"
            photos.mkdir()
            for index, filename in enumerate(experiment.SOURCES):
                Image.new("RGB", (960, 720), (40 + 30 * index, 80, 100)).save(
                    photos / filename, format="JPEG"
                )
            revision = root / "revision"
            manifest, metadata = experiment.prepare_sources(revision, photos)
            self.assertEqual(len(metadata), 5)
            self.assertEqual(len(load_reference_set(manifest).images), 5)
            self.assertTrue(all(item["sha256"] for item in metadata))
            self.assertIn("Wknight94", (revision / "CREDITS.md").read_text())
            self.assertIn("ShareAlike", (revision / "CREDITS.md").read_text())
            self.assertFalse((revision / "sparse-points.glb").exists())
            entries = json.loads(manifest.read_text())["images"]
            self.assertTrue(all(item["source_url"].startswith(
                "https://commons.wikimedia.org/wiki/File:") for item in entries))


if __name__ == "__main__":
    unittest.main()
