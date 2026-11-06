"""Reviewed masks stay valid only for byte-identical rectified photos."""
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.mask_carry_forward import (
    GENERATED, REVIEWED, carry_masks_forward, choose_mask_origins)
from kumiki_blender.reconstruction.references import ReferenceError


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaskOriginTests(unittest.TestCase):
    def test_reviewed_mask_is_kept_only_for_an_identical_photo(self):
        photos = [{"filename": name, "sha256": name + "-new"} for name in ("a", "b", "c", "d")]
        reviewed = {"a": {"source_photo_sha256": "a-new"}, "b": {"source_photo_sha256": "b-old"}}
        generated = {"a": {"source_photo_sha256": "a-new"}, "b": {"source_photo_sha256": "b-new"},
                     "c": {"source_photo_sha256": "c-other"}}
        origins, missing = choose_mask_origins(photos, reviewed, generated)
        self.assertEqual(origins, {"a": REVIEWED, "b": GENERATED})
        self.assertEqual(missing, ["c", "d"])


class CarryMasksTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.sparse = self.root / "recovered-cameras"
        self.sparse.mkdir()
        self.photos = [{"filename": f"view-{i}.png", "sha256": f"{i}" * 64, "dimensions": [40, 30]}
                       for i in range(3)]
        (self.sparse / "undistorted-reference-index.json").write_text(
            json.dumps({"images": self.photos}))
        # Reviewed masks came from an earlier revision: view-0 is unchanged,
        # view-1 was rectified again with other pixels, view-2 did not exist.
        self.reviewed = self.mask_revision("reviewed", "/earlier-cameras", {
            "view-0.png": ("0" * 64, 10), "view-1.png": ("f" * 64, 12)})
        self.generated = self.mask_revision("generated", str(self.sparse), {
            photo["filename"]: (photo["sha256"], 20 + i) for i, photo in enumerate(self.photos)})

    def mask_revision(self, name, source_revision, entries):
        root = self.root / name
        (root / "masks").mkdir(parents=True)
        images = []
        for filename, (photo_sha, right) in entries.items():
            mask = Image.new("L", (40, 30), 0)
            mask.paste(255, (5, 5, right, 25))
            path = root / "masks" / (filename + ".mask.png")
            mask.save(path, format="PNG")
            images.append({"filename": filename, "source_photo_sha256": photo_sha,
                           "mask": "masks/" + path.name, "mask_sha256": digest(path)})
        (root / "mask-report.json").write_text(json.dumps({
            "status": "candidate / test", "source_revision": source_revision, "images": images}))
        return root

    def test_keeps_reviewed_masks_and_fills_the_rest_from_generated_masks(self):
        output = self.root / "carried"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["carry-dense-masks", "--reconstruction", str(self.sparse),
                                   "--reviewed-masks", str(self.reviewed),
                                   "--generated-masks", str(self.generated),
                                   "--output", str(output)]), 0)
        report = json.loads((output / "mask-report.json").read_text())
        self.assertEqual([item["origin"] for item in report["images"]],
                         [REVIEWED, GENERATED, GENERATED])
        self.assertEqual(report["summary"]["reviewed_masks_not_reusable"], ["view-1.png"])
        self.assertFalse(report["approved_for_reconstruction"])
        self.assertEqual(report["source_revision"], str(self.sparse))
        self.assertEqual(digest(output / "masks/view-0.png.mask.png"),
                         digest(self.reviewed / "masks/view-0.png.mask.png"))
        self.assertEqual(digest(output / "masks/view-1.png.mask.png"),
                         digest(self.generated / "masks/view-1.png.mask.png"))

    def test_missing_masks_name_the_photos_and_the_fix(self):
        output = self.root / "carried"
        with self.assertRaisesRegex(ReferenceError, "view-1.png, view-2.png.*prepare-dense-masks"):
            carry_masks_forward(self.sparse, self.reviewed, output)
        self.assertFalse(output.exists())

    def test_rejects_generated_masks_of_another_reconstruction(self):
        other = self.mask_revision("other", "/another-revision", {"view-2.png": ("2" * 64, 9)})
        with self.assertRaisesRegex(ReferenceError, "different reconstruction"):
            carry_masks_forward(self.sparse, self.reviewed, self.root / "carried",
                                generated_masks=other)

    def test_rejects_a_changed_or_wrong_sized_mask_before_writing(self):
        output = self.root / "carried"
        path = self.reviewed / "masks/view-0.png.mask.png"
        Image.new("L", (40, 30), 255).save(path, format="PNG")
        with self.assertRaisesRegex(ReferenceError, "modified"):
            carry_masks_forward(self.sparse, self.reviewed, output, generated_masks=self.generated)
        self.photos[0]["dimensions"] = [80, 60]
        (self.sparse / "undistorted-reference-index.json").write_text(
            json.dumps({"images": self.photos}))
        with self.assertRaisesRegex(ReferenceError, "modified|rectified"):
            carry_masks_forward(self.sparse, self.reviewed, output, generated_masks=self.generated)
        self.assertFalse(output.exists())

    def test_output_cannot_be_inside_an_input(self):
        with self.assertRaisesRegex(ReferenceError, "unused output"):
            carry_masks_forward(self.sparse, self.reviewed, self.reviewed / "carried",
                                generated_masks=self.generated)


if __name__ == "__main__":
    unittest.main()
