"""Immutable hybrid mask revisions with per-view provenance and CLI coverage."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.dense import DenseSettings, plan_dense_reconstruction
from kumiki_blender.reconstruction.mask_selection import compose_rectified_masks
from kumiki_blender.reconstruction.references import ReferenceError


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HybridMaskTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sparse = self.root / "sparse"
        photos = self.sparse / "undistorted/images"
        photos.mkdir(parents=True)
        model = self.sparse / "undistorted/sparse"
        model.mkdir()
        for stem in ("cameras", "images", "points3D"):
            (model / (stem + ".bin")).write_bytes(b"first-party SfM fixture")
        self.primary = self.root / "rembg-r01"
        self.alternate = self.root / "grabcut-r01"
        images, mask_entries = [], {self.primary: [], self.alternate: []}
        for index in range(3):
            name = f"photo-{index+1:04d}.png"
            photo = photos / name
            Image.new("RGB", (100, 80), (125, 135, 145)).save(photo)
            images.append({"filename": name, "sha256": digest(photo),
                           "dimensions": [100, 80]})
            for revision in (self.primary, self.alternate):
                (revision / "masks").mkdir(parents=True, exist_ok=True)
                mask = Image.new("L", (100, 80), 0)
                # Different selected areas: first mask is intentionally tiny
                # for view 2; alternate recovers the missing subject but is
                # never substituted unless explicitly requested.
                if revision == self.primary and index == 1:
                    mask.paste(255, (45, 30, 48, 40))
                elif revision == self.alternate:
                    mask.paste(255, (16, 11, 83, 73))
                else:
                    mask.paste(255, (22, 15, 75, 70))
                path = revision / "masks" / (name + ".mask.png")
                mask.save(path)
                mask_entries[revision].append({
                    "filename": name, "source_photo_sha256": digest(photo),
                    "mask": "masks/" + name + ".mask.png", "mask_sha256": digest(path),
                    "warnings": []})
        (self.sparse / "undistorted-reference-index.json").write_text(
            json.dumps({"images": images}))
        (self.sparse / "reconstruction-report.json").write_text(json.dumps({
            "status": "candidate / visual-review-required", "registered_images": 3,
            "staged_independent_images": 3, "undistorted_images": 3}))
        for revision in (self.primary, self.alternate):
            (revision / "mask-report.json").write_text(json.dumps({
                "status": "candidate / manual-review-required",
                "source_revision": str(self.sparse), "images": mask_entries[revision]}))

    def compose(self, output=None, photos=None):
        return compose_rectified_masks(
            self.sparse, self.primary, self.alternate,
            output or self.root / "hybrid-r01",
            alternate_images=photos if photos is not None else ["photo-0002.png"])

    def test_selected_masks_preserve_pinned_photo_hashes_and_work_in_dense_or_splats(self):
        source_report = self.alternate / "mask-report.json"
        source_data = json.loads(source_report.read_text())
        source_data["images"][1]["guide"] = {"part_points": {"torso": [[0.5, 0.5]]},
                                             "background_points": [[0.02, 0.02]]}
        source_report.write_text(json.dumps(source_data))
        source_hash = digest(self.alternate / "masks/photo-0002.png.mask.png")
        report = self.compose()
        target = self.root / "hybrid-r01"
        self.assertEqual(report["status"], "candidate / hybrid-mask-review-required")
        self.assertFalse(report["approved_for_reconstruction"])
        self.assertEqual(len(report["images"]), 3)
        self.assertEqual([entry["selected_source"] for entry in report["images"]],
                         ["primary", "alternate", "primary"])
        self.assertEqual(report["images"][1]["guide"], source_data["images"][1]["guide"])
        self.assertEqual(digest(target / "masks/photo-0002.png.mask.png"), source_hash)
        self.assertEqual(digest(self.alternate / "masks/photo-0002.png.mask.png"), source_hash)
        self.assertTrue((target / "mask-contact-sheet.jpg").is_file())
        self.assertTrue(all((target / entry["preview"]).is_file() for entry in report["images"]))
        self.assertGreater(report["images"][1]["foreground_fraction"], .4)
        self.assertTrue(any("scenery" in w for w in report["images"][1]["warnings"]))
        plan = plan_dense_reconstruction(self.sparse, self.root / "dense-r02",
                                         DenseSettings(min_registered_images=3),
                                         rectified_masks=target / "masks", executables={})
        self.assertEqual(len(plan["rectified_masks"]), 3)

    def test_bad_selection_fails_before_creating_output(self):
        for selected in ([], ["unknown.jpg"], ["photo-0002.png", "photo-0002.png"]):
            with self.subTest(selected=selected), self.assertRaises(ReferenceError):
                self.compose(photos=selected)
            self.assertFalse((self.root / "hybrid-r01").exists())
        with self.assertRaisesRegex(ReferenceError, "distinct"):
            compose_rectified_masks(self.sparse, self.primary, self.primary,
                                    self.root / "hybrid-r01", alternate_images=["photo-0002.png"])

    def test_reports_and_files_are_verified_before_any_output_is_created(self):
        origin = self.alternate / "masks/photo-0002.png.mask.png"
        origin.write_bytes(b"tampered")
        with self.assertRaises(ReferenceError):
            self.compose()
        self.assertFalse((self.root / "hybrid-r01").exists())
        origin.unlink()
        Image.new("L", (100, 80), 255).save(origin)  # Incorrect 100% foreground
        with self.assertRaises(ReferenceError):
            self.compose()
        self.assertFalse((self.root / "hybrid-r01").exists())

    def test_wrong_source_and_mask_symlink_are_rejected(self):
        path = self.alternate / "mask-report.json"
        document = json.loads(path.read_text())
        document["source_revision"] = "/different/reconstruction"
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ReferenceError, "another reconstruction"):
            self.compose()
        self.assertFalse((self.root / "hybrid-r01").exists())
        document["source_revision"] = str(self.sparse)
        path.write_text(json.dumps(document))
        source = self.alternate / "masks/photo-0002.png.mask.png"
        source.rename(self.root / "original-mask.png")
        source.symlink_to(self.root / "original-mask.png")
        with self.assertRaises(ReferenceError):
            self.compose()

    def test_cli_entrypoint_creates_revision_and_rejects_existing_output(self):
        args = ["compose-dense-masks", "--reconstruction", str(self.sparse),
                "--primary-mask-revision", str(self.primary),
                "--alternate-mask-revision", str(self.alternate),
                "--output", str(self.root / "hybrid-r01"),
                "--use-alternate", "photo-0002.png"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertTrue((self.root / "hybrid-r01/mask-report.json").is_file())
        with self.assertRaisesRegex(ReferenceError, "new output"):
            self.compose()


if __name__ == "__main__":
    unittest.main()
