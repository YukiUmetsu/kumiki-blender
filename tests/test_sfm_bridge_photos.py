"""New bridge-photo intake must remain generic, immutable, and provenance-aware."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from kumiki_blender.reconstruction.sfm_bridge_photos import prepare_bridge_photos
from kumiki_blender.reconstruction.splat_input import load_photo_folder
from kumiki_blender.cli import main as cli_main
from tests.test_sfm_cohort_recovery import fixture


class BridgePhotosTests(unittest.TestCase):
    def test_combines_original_and_new_photos_without_changing_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            # Exercise the exact photo-folder loader used by reconstruct-photos,
            # not just file-copy logic on the synthetic binary test fixture.
            sparse = revision / "sparse-reconstruction"
            index_path = sparse / "staged-reference-index.json"
            index = json.loads(index_path.read_text())
            for number, entry in enumerate(index["images"], 1):
                original = sparse / "images" / entry["filename"]
                Image.new("RGB", (256, 256), (number * 23, number * 17, 50)).save(original)
                entry["sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
            index_path.write_text(json.dumps(index))
            source = revision / "sparse-reconstruction" / "images" / "a.png"
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            new = root / "bridges"
            new.mkdir()
            Image.new("RGB", (300, 300), "olive").save(new / "angle-left.jpg")
            Image.new("RGB", (320, 260), "silver").save(new / "angle-right.png")
            metadata = root / "sources.json"
            metadata.write_text(json.dumps({"images": [
                {"filename": "angle-left.jpg", "source_url": "https://example.com/photo1",
                 "license": "reference-only", "view": "oblique"},
                {"filename": "angle-right.png", "source_url": "local-unattributed"},
            ]}))
            out = root / "combined"
            report = prepare_bridge_photos(revision, new, out, source_manifest=metadata)
            self.assertEqual(report["total_images"], 9)
            self.assertEqual(report["new_bridge_images"], 2)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)
            self.assertEqual(hashlib.sha256((out / "photos" / "a.png").read_bytes()).hexdigest(), original_hash)
            self.assertTrue((out / "photos" / "bridge-0001.jpg").is_file())
            self.assertTrue((out / "photos" / "bridge-0002.png").is_file())
            self.assertEqual(len(load_photo_folder(out / "photos", "any-object").images), 9)
            manifest = json.loads((out / "bridge-photo-manifest.json").read_text())
            self.assertFalse(manifest["distribution_allowed"])
            new_entries = [entry for entry in manifest["images"] if entry["kind"] == "bridge"]
            self.assertEqual(new_entries[0]["source_filename"], "angle-left.jpg")
            self.assertEqual(new_entries[0]["source_url"], "https://example.com/photo1")
            self.assertEqual(new_entries[0]["view"], "oblique")
            with self.assertRaises(FileExistsError):
                prepare_bridge_photos(revision, new, out)

    def test_rejects_identical_photographs_and_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            new = root / "bridges"
            new.mkdir()
            image = Image.new("RGB", (256, 256), "blue")
            image.save(new / "one.png")
            image.save(new / "two.png")
            out = root / "combined"
            with self.assertRaisesRegex(ValueError, "Duplicate photo"):
                prepare_bridge_photos(revision, new, out)
            self.assertFalse(out.exists())

    def test_rejects_symlink_and_bad_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            new = root / "bridges"
            new.mkdir()
            photo = new / "one.png"
            Image.new("RGB", (256, 256), "blue").save(photo)
            (new / "linked.png").symlink_to(photo)
            with self.assertRaisesRegex(ValueError, "Unsafe or oversized"):
                prepare_bridge_photos(revision, new, root / "out")
            (new / "linked.png").unlink()
            metadata = root / "meta.json"
            metadata.write_text(json.dumps({"images": [{"filename": "../one.png"}]}))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                prepare_bridge_photos(revision, new, root / "out", source_manifest=metadata)
            self.assertFalse((root / "out").exists())

    def test_cli_stages_private_bridge_photos_for_generic_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            new = root / "bridges"
            new.mkdir()
            Image.new("RGB", (256, 256), "blue").save(new / "new-view.png")
            output = root / "new-dataset"
            capture = io.StringIO()
            with redirect_stdout(capture):
                rc = cli_main(["prepare-sfm-bridge-photos", "--revision", str(revision),
                               "--photos", str(new), "--output", str(output)])
            self.assertEqual(rc, 0)
            report = json.loads(capture.getvalue())
            self.assertEqual(report["total_images"], 8)
            self.assertTrue((output / "bridge-photo-manifest.json").is_file())

    def test_refuses_to_stage_under_source_sparse_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = fixture(root)
            new = root / "bridges"
            new.mkdir()
            Image.new("RGB", (256, 256), "blue").save(new / "one.png")
            dest = revision / "sparse-reconstruction" / "output"
            with self.assertRaisesRegex(ValueError, "outside the original"):
                prepare_bridge_photos(revision, new, dest)
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
