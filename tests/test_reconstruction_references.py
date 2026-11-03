"""Contract tests for photographic evidence used by monument reconstruction."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from kumiki_blender.reconstruction.references import (
    ReferenceError, audit_reference_set, load_reference_set,
)


class ReferenceSetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "photos").mkdir()

    def image(self, name, *, style=0, dimensions=(120, 90)):
        image = Image.new("RGB", dimensions, "#d7c3a3")
        draw = ImageDraw.Draw(image)
        if style == 0:
            draw.rectangle((5, 8, 44, 80), fill="black")
        elif style == 1:
            draw.rectangle((70, 8, 115, 80), fill="black")
        else:
            draw.polygon([(5, 80), (55, 5), (115, 80)], fill="black")
        file = self.root / "photos" / name
        image.save(file)
        return file

    def record(self, name, view, *, regions=("face",), mask=None):
        value = {"id": name.rsplit(".", 1)[0], "file": "photos/" + name,
                 "source_url": "https://example.org/archive/" + name,
                 "view": view, "regions": list(regions)}
        if mask:
            value["mask"] = mask
        return value

    def manifest(self, images, *, required_views=None, required_regions=None):
        content = {"schema_version": 1, "asset_id": "great-sphinx",
                   "subject": "The Great Sphinx at Giza",
                   "required_views": required_views or ["front", "left-profile", "right-profile"],
                   "required_regions": required_regions or ["face", "headdress"],
                   "images": images}
        file = self.root / "references.json"
        file.write_text(json.dumps(content))
        return file

    def test_validated_images_produce_coverage_and_immutable_digests(self):
        a = self.image("front.png", style=0)
        self.image("left.png", style=1)
        self.image("right.png", style=2)
        path = self.manifest([
            self.record("front.png", "front", regions=("face", "headdress")),
            self.record("left.png", "left-profile", regions=("face",)),
            self.record("right.png", "right-profile", regions=("face",)),
        ])
        set_ = load_reference_set(path)
        self.assertEqual(set_.images[0].sha256, hashlib.sha256(a.read_bytes()).hexdigest())
        self.assertEqual(set_.images[0].dimensions, (120, 90))
        report = audit_reference_set(set_)
        self.assertEqual(report["source_images"], 3)
        self.assertEqual(report["independent_images"], 3)
        self.assertEqual(report["missing_views"], [])
        self.assertEqual(report["missing_regions"], [])
        self.assertEqual(report["status"], "review-required")
        self.assertFalse(report["historical_fidelity_verified"])

    def test_user_uploaded_private_photos_need_no_fabricated_url(self):
        self.image("front.png")
        user_photo = self.record("front.png", "front")
        user_photo.pop("source_url")
        user_photo["source_kind"] = "user-upload"
        reference = load_reference_set(self.manifest([user_photo]))
        self.assertEqual(reference.images[0].source_url, "local-unattributed")
        result = audit_reference_set(reference)
        self.assertEqual(result["user_upload_images_without_verified_public_provenance"], 1)
        self.assertFalse(result["historical_fidelity_verified"])
        for bad in (
            {**user_photo, "source_url": "https://example.org/photo"},
            {**user_photo, "source_kind": "website"},
            {key: value for key, value in user_photo.items() if key != "source_kind"},
        ):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ReferenceError, "HTTPS source_url or source_kind"):
                    load_reference_set(self.manifest([bad]))

    def test_exact_and_resized_duplicates_do_not_inflate_view_coverage(self):
        original = self.image("first.png", style=0)
        (self.root / "photos" / "duplicate.png").write_bytes(original.read_bytes())
        with Image.open(original) as image:
            image.resize((240, 180)).save(self.root / "photos" / "resized.png")
        path = self.manifest([
            self.record("first.png", "front"),
            self.record("duplicate.png", "front"),
            self.record("resized.png", "front"),
        ], required_views=["front", "left-profile"], required_regions=["face", "chin"])
        report = audit_reference_set(load_reference_set(path))
        self.assertEqual(report["source_images"], 3)
        self.assertEqual(report["independent_images"], 2)
        self.assertEqual(report["conservative_coverage_images"], 1)
        self.assertEqual(report["view_counts"], {"front": 1})
        self.assertEqual(report["missing_views"], ["left-profile"])
        self.assertEqual(report["missing_regions"], ["chin"])
        self.assertEqual(len(report["duplicate_groups"]), 1)
        self.assertEqual(len(report["near_duplicate_groups"]), 1)

    def test_rejects_unexpected_fields_duplicate_ids_and_paths_that_escape(self):
        self.image("first.png")
        first = self.record("first.png", "front")
        for bad_image, reason in (
            ({**first, "surprise": True}, "unexpected"),
            ({**first, "file": "../outside.jpg"}, "relative"),
            ({**first, "file": "/tmp/example.jpg"}, "relative"),
            ({**first, "source_url": "http://example.org/p.jpg"}, "HTTPS"),
        ):
            with self.subTest(bad_image=bad_image):
                with self.assertRaisesRegex(ReferenceError, reason):
                    load_reference_set(self.manifest([bad_image]))
        with self.assertRaisesRegex(ReferenceError, "duplicate"):
            load_reference_set(self.manifest([first, first]))
        external = tempfile.TemporaryDirectory()
        self.addCleanup(external.cleanup)
        outside = Path(external.name) / "outside.png"
        self.image("first.png").replace(outside)
        (self.root / "photos" / "first.png").symlink_to(outside)
        with self.assertRaisesRegex(ReferenceError, "escapes"):
            load_reference_set(self.manifest([first]))

    def test_rejects_wrong_mask_dimensions_and_unpinned_source_changes(self):
        photo = self.image("front.png")
        mask = self.root / "photos" / "mask.png"
        Image.new("L", (100, 80), 255).save(mask)
        record = self.record("front.png", "front", mask="photos/mask.png")
        with self.assertRaisesRegex(ReferenceError, "mask dimensions"):
            load_reference_set(self.manifest([record]))
        Image.new("L", (120, 90), 255).save(mask)
        record["expected_sha256"] = "0" * 64
        with self.assertRaisesRegex(ReferenceError, "SHA-256"):
            load_reference_set(self.manifest([record]))
        record["expected_sha256"] = hashlib.sha256(photo.read_bytes()).hexdigest()
        self.assertEqual(len(load_reference_set(self.manifest([record])).images), 1)

    def test_rejects_duplicate_json_keys_and_invalid_image_content(self):
        path = self.root / "references.json"
        path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ReferenceError, "duplicate JSON key"):
            load_reference_set(path)
        (self.root / "photos" / "broken.png").write_bytes(b"not a real PNG")
        with self.assertRaisesRegex(ReferenceError, "decode"):
            load_reference_set(self.manifest([self.record("broken.png", "front")]))

    def test_untrusted_types_and_malformed_urls_raise_reference_error(self):
        self.image("front.png")
        record = self.record("front.png", "front")
        for changes in ({"view": []}, {"source_url": "https://[unbalanced/archive"},
                        {"regions": ["face", "face"]}):
            with self.subTest(changes=changes):
                with self.assertRaises(ReferenceError):
                    load_reference_set(self.manifest([{**record, **changes}]))


if __name__ == "__main__":
    unittest.main()
