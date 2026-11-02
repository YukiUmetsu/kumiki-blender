"""A verified CC0 head source is ready for the existing catalog workflow."""

import contextlib
import hashlib
import io
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kumiki_blender.asset_catalog import find_asset, load_catalog, validate_sources
from kumiki_blender.cli import main
from kumiki_blender.head_sources import makehuman
from tests.test_makehuman_source import FIXTURE


class MakeHumanCLITests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "base.obj"
        self.source.write_text(FIXTURE, encoding="utf-8")
        self.output = self.root / "r01"

    def test_catalog_includes_buildable_official_anatomical_head_family(self):
        entries = load_catalog()
        entry = find_asset(entries, "makehuman-neutral-head")
        self.assertEqual(entry.kind, "imported")
        self.assertEqual(entry.build_command, ("prepare-makehuman-head", "--download"))
        validate_sources(entries, Path(__file__).resolve().parents[1])

    def test_wrong_source_never_spawns_blender_or_creates_revision(self):
        message = io.StringIO()
        with patch("subprocess.run") as run, contextlib.redirect_stderr(message):
            result = main(["prepare-makehuman-head", "--source-obj", str(self.source),
                           "--output", str(self.output)])
        self.assertEqual(result, 2)
        self.assertIn("SHA-256", message.getvalue())
        run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_verified_local_source_is_forwarded_to_blender_worker(self):
        approved_digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with (patch.object(makehuman, "SOURCE_SHA256", approved_digest),
              patch.object(makehuman, "load_body_mesh", return_value=((0,) * 10001, (0,) * 10001)),
              patch("shutil.which", return_value="/usr/bin/blender"),
              patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run):
            result = main(["prepare-makehuman-head", "--source-obj", str(self.source),
                           "--output", str(self.output)])
        self.assertEqual(result, 0)
        args = run.call_args.args[0]
        self.assertIn("--factory-startup", args)
        self.assertEqual(args[-4:], ["--source-obj", str(self.source),
                                     "--output", str(self.output)])
        self.assertFalse(self.output.exists())

    def test_existing_output_is_rejected_before_download(self):
        self.output.mkdir()
        with patch.object(makehuman, "fetch_makehuman_obj") as fetch:
            result = main(["prepare-makehuman-head", "--download", "--output", str(self.output)])
        self.assertEqual(result, 2)
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
