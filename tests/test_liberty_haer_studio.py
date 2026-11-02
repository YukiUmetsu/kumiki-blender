"""Offline safety checks for the authentic installed-head reference studio."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "blender_liberty_reference_studio.py"
spec = importlib.util.spec_from_file_location("liberty_haer_reference_studio", SCRIPT)
studio = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = studio
spec.loader.exec_module(studio)


class LibertyHaerStudioTests(unittest.TestCase):
    def test_sources_are_distinct_installed_head_photographs(self):
        self.assertEqual([view.survey_id for view in studio.ARCHIVED_VIEWS], [
            f"HAER NY-138-{n}" for n in range(236, 242)])
        self.assertEqual([view.survey_id for view in studio.DETAIL_VIEWS], [
            f"HAER NY-138-{n}" for n in range(177, 182)])
        self.assertEqual(len(set(view.filename for view in studio.ALL_VIEWS)), 11)
        self.assertEqual([view.loc_url.rsplit("/", 2)[-2] for view in studio.ARCHIVED_VIEWS], [
            f"ny1251.color.{n:06}c" for n in range(570005, 570011)])
        for view in studio.ARCHIVED_VIEWS:
            self.assertTrue(view.jpeg_url.startswith(
                "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/"))
            self.assertEqual(view.group, "pose")
        for offset, view in enumerate(studio.DETAIL_VIEWS):
            self.assertIn(f"photos/120{435+offset}pv.jpg", view.jpeg_url)
            self.assertEqual(view.group, "detail")

    def test_missing_photos_fail_closed_before_blender_opens(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference_dir = Path(tmp)
            with self.assertRaisesRegex(ValueError, "missing/empty HAER NY-138-236"):
                studio.resolve_photos(reference_dir)
            for view in studio.ALL_VIEWS:
                (reference_dir / view.filename).write_bytes(b"\xff\xd8\xff" + b"fixture" * 200)
            resolved = studio.resolve_photos(reference_dir)
            self.assertEqual([photo for photo, _ in resolved], list(studio.ALL_VIEWS))
            self.assertTrue(all(path.is_absolute() for _, path in resolved))
            (reference_dir / studio.ARCHIVED_VIEWS[0].filename).write_bytes(b"not JPEG" * 200)
            with self.assertRaisesRegex(ValueError, "not a JPEG photo"):
                studio.resolve_photos(reference_dir)

    def test_nonexistent_source_directory_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "missing HAER directory"):
                studio.resolve_photos(Path(tmp) / "not-yet-downloaded")


if __name__ == "__main__":
    unittest.main()
