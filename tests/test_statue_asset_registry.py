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
from kumiki_blender.reconstruction.statue_pipeline import RESUME_INPUTS_CHANGED_ERROR
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

    def test_implicit_resume_starts_new_revision_when_inputs_change(self):
        calls = []

        def mock_build(photos, asset_id, output, settings, *, resume=False, **kwargs):
            calls.append((output.name, resume))
            if resume and any(self.photos.iterdir()):
                raise ReferenceError(RESUME_INPUTS_CHANGED_ERROR)
            if not resume:
                output.mkdir(parents=True)
            return {"status": "private-GLB-candidate / review-required",
                    "output": str(output)}

        with patch("pathlib.Path.cwd", return_value=self.root), patch(
                "kumiki_blender.reconstruction.statue_pipeline.build_statue_glb",
                side_effect=mock_build), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["build-statue-glb", "--asset-id", "sphinx",
                                   "--photos", str(self.photos), "--no-progress"]), 0)
            # Unchanged sources genuinely resume the existing revision.
            self.assertEqual(main(["build-statue-glb", "--asset-id", "sphinx",
                                   "--resume", "--no-progress"]), 0)
            (self.photos / "new-elevated-view.jpg").write_bytes(b"new photo")
            self.assertEqual(main(["build-statue-glb", "--asset-id", "sphinx",
                                   "--resume", "--no-progress"]), 0)

        self.assertEqual(calls, [("statue-hq-r001", False),
                                 ("statue-hq-r001", True),
                                 ("statue-hq-r001", True),
                                 ("statue-hq-r002", False)])
        root = self.root / ".review/sphinx"
        self.assertTrue((root / "statue-hq-r001").is_dir())
        self.assertTrue((root / "statue-hq-r002").is_dir())
        state = json.loads((root / "asset-state.json").read_text())
        self.assertEqual(Path(state["last_output"]), root / "statue-hq-r002")
        self.assertEqual(Path(state["photos_dir"]), self.photos.resolve())

    def test_explicit_output_still_refuses_incompatible_resume(self):
        old = self.root / "manual-revision"
        old.mkdir()
        calls = []

        def mock_build(photos, asset_id, output, settings, *, resume=False, **kwargs):
            calls.append((output, resume))
            raise ReferenceError(RESUME_INPUTS_CHANGED_ERROR)

        with patch("pathlib.Path.cwd", return_value=self.root), patch(
                "kumiki_blender.reconstruction.statue_pipeline.build_statue_glb",
                side_effect=mock_build), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["build-statue-glb", "--asset-id", "sphinx",
                                   "--photos", str(self.photos), "--output", str(old),
                                   "--resume", "--no-progress"]), 2)

        self.assertEqual(calls, [(old, True)])
        self.assertFalse((self.root / ".review/sphinx/statue-hq-r001").exists())


if __name__ == "__main__":
    unittest.main()
