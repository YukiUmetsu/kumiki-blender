"""Stable categorized asset discovery without Blender or a hand-built index."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from kumiki_blender.asset_catalog import (CATALOG_ROOT, CatalogError,
                                         find_asset, load_catalog, revision_path,
                                         search_assets, validate_sources)
from kumiki_blender.cli import main


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = load_catalog()

    def test_existing_model_families_are_registered_once_and_have_real_sources(self):
        ids = {entry.asset_id for entry in self.entries}
        self.assertEqual(len(ids), len(self.entries))
        # The discovery registry is intentionally open-ended: adding a new
        # categorized model must not break tests for existing families.
        self.assertGreaterEqual(len(self.entries), 10)
        self.assertTrue({"red-vintage-biplane", "lattice-observation-tower",
                         "stone-facade", "classical-column", "modular-portico",
                         "giza-pyramid-study", "giza-sphinx-study",
                         "giza-ensemble-study", "egypt-relief-atlas",
                         "statue-of-liberty"} <= ids)
        validate_sources(self.entries, Path(__file__).resolve().parents[1])
        for entry in self.entries:
            self.assertTrue(entry.manifest.startswith("src/kumiki_blender/assets/"))
            self.assertEqual((entry.category, entry.subcategory, entry.asset_id),
                             tuple(Path(entry.manifest).parts[-4:-1]))

    def test_discovery_by_alias_tag_and_category(self):
        planes = search_assets(self.entries, "OLD red PLANE")
        self.assertEqual([p.asset_id for p in planes], ["red-vintage-biplane"])
        self.assertEqual(len(search_assets(self.entries, "giza")), 3)
        self.assertEqual([entry.asset_id for entry in search_assets(
            self.entries, "Egypt terrain")], ["egypt-relief-atlas"])
        self.assertEqual(len(search_assets(self.entries, category="landmarks")), 4)
        liberty = search_assets(self.entries, "lady liberty")
        self.assertEqual([item.asset_id for item in liberty], ["statue-of-liberty"])
        self.assertEqual(liberty[0].subcategory, "statues")
        self.assertEqual(len(search_assets(self.entries, rigged=True)), 1)
        self.assertEqual([entry.asset_id for entry in search_assets(
            self.entries, tag="column")], ["classical-column"])
        self.assertEqual(search_assets(self.entries, "nonexistentquery"), ())
        self.assertEqual(find_asset(self.entries, "red-vintage-biplane").animated, True)
        with self.assertRaises(CatalogError):
            find_asset(self.entries, "missing-model")

    def test_output_revisions_have_stable_categorized_paths_and_no_traversal(self):
        plane = find_asset(self.entries, "red-vintage-biplane")
        self.assertEqual(revision_path(plane, "r002").as_posix(),
                         ".review/vehicles/aircraft/red-vintage-biplane/r002")
        liberty = find_asset(self.entries, "statue-of-liberty")
        self.assertEqual(revision_path(liberty, "r02").as_posix(),
                         ".review/landmarks/statues/statue-of-liberty/r02")
        for invalid in ("../x", "../../r01", "R01", ".hidden", "r1", "r01/other"):
            with self.subTest(invalid=invalid), self.assertRaises(CatalogError):
                revision_path(plane, invalid)

    def test_cli_browse_show_and_validate_emit_json(self):
        for args in (("catalog", "categories"), ("catalog", "list", "--rigged"),
                     ("catalog", "search", "ancient", "pyramid"),
                     ("catalog", "show", "red-vintage-biplane"),
                     ("catalog", "show", "statue-of-liberty"),
                     ("catalog", "validate")):
            with self.subTest(args=args):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(main(list(args)), 0)
                self.assertIsInstance(json.loads(stream.getvalue()), dict)

    def test_catalog_build_preflights_bpy_and_does_not_create_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "vehicles" / "aircraft" / "red-vintage-biplane" / "r01"
            stream = io.StringIO()
            # A Python-only invocation gives an actionable error, not an
            # incomplete directory or an unintentional Blender invocation.
            with mock.patch.dict("sys.modules", {"bpy": None}):
                with contextlib.redirect_stderr(stream):
                    code = main(["catalog", "build", "red-vintage-biplane",
                                 "--root", str(root)])
            self.assertEqual(code, 2)
            self.assertIn("requires Blender", stream.getvalue())
            self.assertFalse(target.exists())
            stream = io.StringIO()
            with contextlib.redirect_stderr(stream):
                self.assertEqual(main(["catalog", "build", "classical-column",
                                       "--root", str(root)]), 2)
            self.assertIn("reusable component", stream.getvalue())
            liberty_target = root / "landmarks" / "statues" / "statue-of-liberty" / "r01"
            stream = io.StringIO()
            with contextlib.redirect_stderr(stream):
                self.assertEqual(main(["catalog", "build", "statue-of-liberty",
                                       "--root", str(root)]), 2)
            self.assertIn("requires Blender", stream.getvalue())
            self.assertFalse(liberty_target.exists())

    def test_malformed_manifests_fail_instead_of_being_silently_indexed(self):
        original = json.loads((CATALOG_ROOT / "vehicles" / "aircraft" /
                               "red-vintage-biplane" / "asset.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "vehicles" / "aircraft" / "red-vintage-biplane" / "asset.json"
            file.parent.mkdir(parents=True)
            for change in ({"id": "different-id"}, {"schema_version": 2},
                           {"tags": ["red", "red"]}, {"sources": ["src/../secret.py"]},
                           {"build": ["rm", "-rf", "/tmp"]},
                           {"unknown_field": "typo"}):
                with self.subTest(change=change):
                    file.write_text(json.dumps({**original, **change}))
                    with self.assertRaises(CatalogError):
                        load_catalog(root)
            file.write_text(json.dumps(original))
            another = root / "vehicles" / "historic" / "red-vintage-biplane" / "asset.json"
            another.parent.mkdir(parents=True)
            shutil.copyfile(file, another)
            with self.assertRaisesRegex(CatalogError, "Duplicate catalog ID"):
                load_catalog(root)


if __name__ == "__main__":
    unittest.main()
