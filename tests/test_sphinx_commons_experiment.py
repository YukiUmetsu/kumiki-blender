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
