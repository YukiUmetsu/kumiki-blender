"""Local asset defaults never overwrite previous revisions or use unsafe paths."""

import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.statue_asset_registry import (
    remember_statue_inputs, resolve_statue_inputs,
)


class StatueAssetRegistryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.photos = self.root / "my-sphinx-photos"
        self.photos.mkdir()

    def test_output_automatically_increments_and_recalls_photos(self):
        photos, first = resolve_statue_inputs("great-sphinx", self.photos, None,
                                              workspace=self.root)
        self.assertEqual(first, self.root / ".review/great-sphinx/statue-hq-r001")
        remember_statue_inputs("great-sphinx", photos, first, workspace=self.root)
        first.mkdir(parents=True)
        reused, second = resolve_statue_inputs("great-sphinx", None, None,
                                               workspace=self.root)
        self.assertEqual(reused, self.photos.resolve())
        self.assertEqual(second, self.root / ".review/great-sphinx/statue-hq-r002")
        self.assertTrue(first.is_dir())
        self.assertEqual(json.loads((first.parent / "asset-state.json").read_text())[
            "photos_dir"], str(self.photos.resolve()))
        resumed_photos, resumed_output = resolve_statue_inputs(
            "great-sphinx", None, None, workspace=self.root, resume=True)
        self.assertEqual((resumed_photos, resumed_output), (self.photos.resolve(), first))

    def test_explicit_photo_folder_updates_saved_asset(self):
        replacement = self.root / "replacement"
        replacement.mkdir()
        original, first = resolve_statue_inputs("sphinx", self.photos, None,
                                                workspace=self.root)
        remember_statue_inputs("sphinx", original, first, workspace=self.root)
        changed, override = resolve_statue_inputs("sphinx", replacement,
                                                  self.root / "custom-run", workspace=self.root)
        remember_statue_inputs("sphinx", changed, override, workspace=self.root)
        found, _ = resolve_statue_inputs("sphinx", None, None, workspace=self.root)
        self.assertEqual(found, replacement.resolve())

    def test_missing_history_requires_photos_once_and_rejects_traversal(self):
        with self.assertRaisesRegex(ReferenceError, "supply --photos once"):
            resolve_statue_inputs("new-asset", None, None, workspace=self.root)
        with self.assertRaisesRegex(ReferenceError, "Asset ID"):
            resolve_statue_inputs("../outside", self.photos, None, workspace=self.root)
        with self.assertRaisesRegex(ReferenceError, "No saved output revision"):
            resolve_statue_inputs("new-asset", self.photos, None,
                                  workspace=self.root, resume=True)

    def test_does_not_follow_existing_registry_symlink(self):
        review = self.root / ".review"
        review.symlink_to(self.photos, target_is_directory=True)
        with self.assertRaisesRegex(ReferenceError, "symbolic links"):
            resolve_statue_inputs("sphinx", self.photos, None, workspace=self.root)

    def test_cli_remembers_photos_and_allocates_next_revision(self):
        captured = []

        def mock_build(photos, asset_id, output, settings, **kwargs):
            captured.append((photos, asset_id, output))
            return {"status": "private-GLB-candidate / review-required"}

        with patch("pathlib.Path.cwd", return_value=self.root), patch(
                "kumiki_blender.reconstruction.statue_pipeline.build_statue_glb",
                side_effect=mock_build), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["build-statue-glb", "--asset-id", "sphinx",
                                   "--photos", str(self.photos)]), 0)
            captured[0][2].mkdir(parents=True)
            self.assertEqual(main(["build-statue-glb", "--asset-id", "sphinx"]), 0)
        self.assertEqual(captured[0][2].name, "statue-hq-r001")
        self.assertEqual(captured[1][2].name, "statue-hq-r002")
        self.assertEqual(captured[0][0], captured[1][0])


if __name__ == "__main__":
    unittest.main()
