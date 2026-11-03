"""Execute the real host-process boundary with a controlled local CLI fixture."""

import json
import io
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from kumiki_blender.reconstruction.colmap import (
    _prepare_images, _feature_import_report, _registered_image_count, _repair_jpeg_orientation, _sparse_point_metrics,
    build_sparse_reconstruction,
)
from kumiki_blender.reconstruction.references import ReferenceError, load_reference_set


FAKE_COLMAP = '''#!/usr/bin/env python3
import os
import sys
from pathlib import Path

task = sys.argv[1]
params = sys.argv[2:]
legacy = "FAKE_COLMAP_LEGACY" in os.environ
if task == "--version":
    print("COLMAP 3.12" if legacy else "COLMAP 3.13")
    sys.exit(0)
if "--help" in params:
    if task == "feature_extractor":
        print("--SiftExtraction.use_gpu" if legacy else "--FeatureExtraction.use_gpu")
    elif task == "exhaustive_matcher":
        print("--SiftMatching.use_gpu" if legacy else "--FeatureMatching.use_gpu")
    sys.exit(0)
def argument(key):
    return Path(params[params.index(key)+1])
if task == "feature_extractor":
    if ("--SiftExtraction.use_gpu" if legacy else "--FeatureExtraction.use_gpu") not in params:
        sys.exit(42)
    images = argument("--image_path")
    if len(list(images.iterdir())) != 3:
        sys.exit(11)
    masks = argument("--ImageReader.mask_path")
    if not (masks / "front.png.png").is_file():
        sys.exit(12)
    argument("--database_path").write_text("sample data")
elif task == "exhaustive_matcher":
    if ("--SiftMatching.use_gpu" if legacy else "--FeatureMatching.use_gpu") not in params:
        sys.exit(42)
    if "FAKE_FAIL_MATCHER" in __import__("os").environ:
        print("simulated incompatible feature matching")
        sys.exit(7)
elif task == "mapper":
    model = argument("--output_path") / "0"
    model.mkdir(parents=True)
    (model / "cameras.bin").write_bytes(b"synthetic test")
elif task == "image_undistorter":
    from shutil import copyfile
    target = argument("--output_path")
    (target / "images").mkdir()
    (target / "sparse").mkdir()
    for image in argument("--image_path").iterdir():
        copyfile(image, target / "images" / image.name)
elif task == "model_converter":
    output = argument("--output_path")
    if not output.is_dir():
        print("model_converter expects an existing output directory")
        sys.exit(49)
    (output / "images.txt").write_text("# Image list\\n1 1 0 0 0 0 0 0 1 front.png\\n\\n2 1 0 0 0 0 0 0 1 left.png\\n\\n3 1 0 0 0 0 0 0 1 right.png\\n\\n")
    (output / "cameras.txt").write_text("1 PINHOLE 120 90 100 100 60 45\\n")
else:
    sys.exit(99)
'''


class ColmapWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "photos"
        self.images.mkdir()
        self.fake = self.root / "colmap-fixture"
        self.fake.write_text(FAKE_COLMAP)
        self.fake.chmod(0o755)

    def references(self, *, count=3, masked=True):
        records = []
        names = ("front.png", "left.png", "right.png")[:count]
        for i, name in enumerate(names):
            image = Image.new("RGB", (120, 90), "#e2bd82")
            draw = ImageDraw.Draw(image)
            draw.rectangle((i * 22 + 3, 10, i * 22 + 19, 79), fill="black")
            image.save(self.images / name)
            record = {"id": name[:-4], "file": "photos/" + name,
                      "source_url": "https://example.org/" + name,
                      "view": ("front", "left-profile", "right-profile")[i],
                      "regions": ["face"]}
            if masked and i == 0:
                Image.new("L", (120, 90), 255).save(self.images / "front-mask.png")
                record["mask"] = "photos/front-mask.png"
            records.append(record)
        manifest = self.root / "reference-set.json"
        manifest.write_text(json.dumps({"schema_version": 1, "asset_id": "stone-monument",
                                        "subject": "A historic stone monument",
                                        "required_views": ["front", "left-profile", "right-profile"],
                                        "required_regions": ["face"], "images": records}))
        return load_reference_set(manifest)

    def test_stages_curated_photos_masks_and_records_sparse_candidate(self):
        output = self.root / "r01"
        report = build_sparse_reconstruction(self.references(), output, colmap=str(self.fake))
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertEqual(report["registered_images"], 3)
        self.assertFalse(report["historical_fidelity_verified"])
        self.assertEqual([item[0] for item in report["commands"]],
                         ["feature_extractor", "exhaustive_matcher", "mapper", "model_converter"])
        self.assertEqual(report["commands"][0][-2:], ["--FeatureExtraction.use_gpu", "0"])
        self.assertEqual(report["commands"][1][-2:], ["--FeatureMatching.use_gpu", "0"])
        self.assertEqual(report["colmap_info"]["version"], "COLMAP 3.13")
        self.assertEqual(report["colmap_info"]["cpu_flag_family"], "feature")
        self.assertTrue((output / "images/front.png").is_file())
        self.assertTrue((output / "masks/front.png.png").is_file())
        with Image.open(output / "masks/front.png.png") as mask:
            self.assertEqual(mask.mode, "L")
        self.assertEqual(json.loads((output / "reconstruction-report.json").read_text()), report)
        self.assertEqual(json.loads((output / "reference-audit.json").read_text())["source_images"], 3)
        self.assertEqual(len(list((output / "logs").glob("*.log"))), 4)

    def test_preflight_refuses_missing_executable_too_few_images_and_existing_output(self):
        source = self.references(count=2, masked=False)
        output = self.root / "r01"
        with self.assertRaisesRegex(ReferenceError, "three independent"):
            build_sparse_reconstruction(source, output, colmap=str(self.fake))
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ReferenceError, "COLMAP executable"):
            build_sparse_reconstruction(self.references(), output, colmap="/not/installed/colmap")
        self.assertFalse(output.exists())
        output.mkdir()
        (output / "keep.txt").write_text("preserve")
        with self.assertRaisesRegex(ReferenceError, "already exists"):
            build_sparse_reconstruction(self.references(), output, colmap=str(self.fake))
        self.assertEqual((output / "keep.txt").read_text(), "preserve")

    def test_rechecks_image_hash_before_output_created(self):
        source = self.references()
        (self.images / "front.png").write_bytes(b"changed after evidence validation")
        output = self.root / "r01"
        with self.assertRaisesRegex(ReferenceError, "changed since validation"):
            build_sparse_reconstruction(source, output, colmap=str(self.fake))
        self.assertFalse(output.exists())

    def test_rechecks_mask_hash_before_output_created(self):
        source = self.references()
        Image.new("L", (120, 90), 0).save(self.images / "front-mask.png")
        output = self.root / "r01"
        with self.assertRaisesRegex(ReferenceError, "mask changed since validation"):
            build_sparse_reconstruction(source, output, colmap=str(self.fake))
        self.assertFalse(output.exists())

    def test_legacy_colmap_flags_are_probed_before_reconstruction(self):
        with patch.dict(os.environ, {"FAKE_COLMAP_LEGACY": "1"}):
            report = build_sparse_reconstruction(self.references(), self.root / "r01",
                                                 colmap=str(self.fake))
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertEqual(report["commands"][0][-2:], ["--SiftExtraction.use_gpu", "0"])
        self.assertEqual(report["commands"][1][-2:], ["--SiftMatching.use_gpu", "0"])
        self.assertEqual(report["colmap_info"]["cpu_flag_family"], "sift")

    def test_similar_but_distinct_photos_are_not_excluded_from_colmap(self):
        self.references()
        with Image.open(self.images / "front.png") as photo:
            second = photo.copy()
        second.putpixel((2, 2), (255, 0, 0))
        second.save(self.images / "left.png")
        manifest = self.root / "reference-set.json"
        raw = json.loads(manifest.read_text())
        raw["images"][1]["view"] = "front"
        manifest.write_text(json.dumps(raw))
        source = load_reference_set(manifest)
        report = build_sparse_reconstruction(source, self.root / "r01", colmap=str(self.fake))
        self.assertEqual(report["staged_independent_images"], 3)
        self.assertEqual(report["status"], "candidate / visual-review-required")

    def test_optional_undistorter_preserves_rectified_camera_and_pixel_provenance(self):
        output = self.root / "r01"
        report = build_sparse_reconstruction(self.references(), output, colmap=str(self.fake),
                                             undistort=True, max_image_size=4000)
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertEqual(report["undistorted_model"], "undistorted/text-model")
        self.assertEqual(report["undistorted_images"], 3)
        self.assertEqual(report["source_sparse_model"], "0")
        self.assertEqual([command[0] for command in report["commands"]][-2:],
                         ["image_undistorter", "model_converter"])
        index = json.loads((output / "undistorted-reference-index.json").read_text())
        self.assertEqual(len(index["images"]), 3)
        self.assertTrue(all(item["mask_requires_rectification"] for item in index["images"]))

    def test_partial_staging_failure_persists_diagnostic_report(self):
        source = self.references()
        real_read = Path.read_bytes

        def fail_second_image(path):
            if path.name == "left.png":
                raise OSError("simulated exhausted storage")
            return real_read(path)

        output = self.root / "r01"
        with patch.object(Path, "read_bytes", fail_second_image):
            report = build_sparse_reconstruction(source, output, colmap=str(self.fake))
        self.assertEqual(report["status"], "review-required")
        self.assertEqual(report["failed_stage"], "staging")
        self.assertIn("simulated exhausted storage", report["error"])
        self.assertEqual(json.loads((output / "reconstruction-report.json").read_text()), report)

    def test_failed_external_stage_preserves_review_required_diagnostics(self):
        output = self.root / "r01"
        import os
        old_value = os.environ.get("FAKE_FAIL_MATCHER")
        os.environ["FAKE_FAIL_MATCHER"] = "1"
        try:
            report = build_sparse_reconstruction(self.references(), output, colmap=str(self.fake))
        finally:
            if old_value is None:
                os.environ.pop("FAKE_FAIL_MATCHER", None)
            else:
                os.environ["FAKE_FAIL_MATCHER"] = old_value
        self.assertEqual(report["status"], "review-required")
        self.assertEqual(report["failed_stage"], "exhaustive_matcher")
        self.assertEqual(report["exit_code"], 7)
        self.assertFalse(report["historical_fidelity_verified"])
        self.assertIn("simulated incompatible", (output / "logs/02-exhaustive_matcher.log").read_text())
        self.assertTrue((output / "reconstruction-report.json").exists())

    def test_registered_image_parser_never_counts_2d_observation_rows(self):
        model = self.root / "model"
        model.mkdir()
        observations = "10 20 3 40 50 6 70 80 9 100 110 12"
        (model / "images.txt").write_text(
            "# image poses and alternating observations\n"
            "1 1 0 0 0 0 0 0 1 front.png\n" + observations + "\n"
            "2 1 0 0 0 0 0 0 1 left.png\n" + observations + "\n")
        self.assertEqual(_registered_image_count(model), 2)

    def test_sparse_reprojection_evidence_is_not_mislabeled_as_mesh_accuracy(self):
        model = self.root / "quality"
        model.mkdir()
        (model / "points3D.txt").write_text(
            "# POINT3D_ID X Y Z R G B ERROR TRACK\n"
            "1 0 0 0 255 0 0 0.50 1 10 2 11\n"
            "2 0 0 0 255 0 0 1.50 1 10 2 11 3 12 4 13\n"
            "3 0 0 0 255 0 0 NaN 1 10 2 11\n")
        result = _sparse_point_metrics(model)
        self.assertEqual(result["triangulated_points"], 2)
        self.assertEqual(result["mean_track_length"], 3)
        self.assertEqual(result["median_reprojection_error_px"], 1)
        self.assertIn("not surface", result["interpretation"])

    def test_repairs_invalid_exif_orientation_losslessly_before_colmap(self):
        for old in (0, 9):
            with self.subTest(old=old):
                image = Image.new("RGB", (80, 70), "#cb9365")
                exif = Image.Exif()
                exif[274] = old
                stream = io.BytesIO()
                image.save(stream, format="JPEG", exif=exif)
                original = stream.getvalue()
                repaired, changed = _repair_jpeg_orientation(original)
                self.assertTrue(changed)
                self.assertEqual(len(repaired), len(original))
                self.assertEqual(sum(a != b for a, b in zip(repaired, original)), 1)
                self.assertEqual(Image.open(io.BytesIO(repaired)).getexif()[274], 1)
                self.assertEqual(_repair_jpeg_orientation(repaired), (repaired, False))
        self.assertEqual(_repair_jpeg_orientation(b"not a JPEG"), (b"not a JPEG", False))

    def test_normalizes_webp_and_mislabelled_jpeg_without_changing_sources(self):
        from kumiki_blender.reconstruction.references import ReferenceImage
        import hashlib

        source_webp = self.images / "photo.webp"
        source_jpg = self.images / "odd.jpg"
        source_png = self.images / "photo.png"
        Image.new("RGB", (96, 72), (40, 120, 210)).save(source_webp, format="WEBP")
        # A WebP payload with a JPEG extension reproduces decoder incompatibility.
        source_jpg.write_bytes(source_webp.read_bytes())
        Image.new("RGB", (96, 72), (100, 150, 210)).save(source_png, format="PNG")
        originals = {path: path.read_bytes() for path in (source_webp, source_jpg, source_png)}
        refs = [
            ReferenceImage(f"photo-{i:04d}", path, "local-unattributed", "unknown",
                           ("object",), None, None, hashlib.sha256(path.read_bytes()).hexdigest(),
                           (96, 72), 0)
            for i, path in enumerate(originals, 1)
        ]
        output = self.root / "normalized"
        output.mkdir()
        self.assertFalse(_prepare_images(refs, output, normalize_for_colmap=True))
        index = json.loads((output / "staged-reference-index.json").read_text())["images"]
        self.assertEqual(len(index), 3)
        self.assertTrue(all(item["normalized_for_colmap"] for item in index))
        self.assertEqual([item["source_format"] for item in index], ["WEBP", "WEBP", "PNG"])
        for item in index:
            self.assertEqual(item["dimensions"], [96, 72])
            self.assertTrue(item["filename"].endswith(".png"))
            with Image.open(output / "images" / item["filename"]) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "RGB")
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_normalization_applies_exif_orientation_to_image_and_mask(self):
        from kumiki_blender.reconstruction.references import ReferenceImage
        import hashlib

        photo = self.images / "rotated.jpg"
        mask = self.images / "rotated-mask.png"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (96, 72), (210, 140, 90)).save(photo, exif=exif)
        Image.new("L", (96, 72), 255).save(mask)
        source = ReferenceImage("rotated", photo, "local-unattributed", "unknown",
                                ("object",), mask, hashlib.sha256(mask.read_bytes()).hexdigest(),
                                hashlib.sha256(photo.read_bytes()).hexdigest(), (96, 72), 0)
        output = self.root / "oriented"
        output.mkdir()
        self.assertTrue(_prepare_images([source], output, normalize_for_colmap=True))
        index = json.loads((output / "staged-reference-index.json").read_text())["images"][0]
        self.assertEqual(index["original_dimensions"], [96, 72])
        self.assertEqual(index["dimensions"], [72, 96])
        self.assertEqual(index["exif_orientation_applied"], 6)
        with Image.open(output / "images/rotated.png") as staged:
            self.assertEqual(staged.size, (72, 96))
        with Image.open(output / "masks/rotated.png.png") as staged:
            self.assertEqual(staged.size, (72, 96))
        with Image.open(photo) as original:
            self.assertEqual(original.size, (96, 72))

    def test_feature_import_audit_detects_missing_images(self):
        directory = self.root / "imports"
        directory.mkdir()
        index = directory / "staged-reference-index.json"
        index.write_text(json.dumps({"images": [
            {"filename": "a.png"}, {"filename": "b.png"}]}))
        database = directory / "database.db"
        with sqlite3.connect(database) as db:
            db.execute("CREATE TABLE images (name TEXT)")
            db.execute("INSERT INTO images VALUES ('a.png')")
        report = _feature_import_report(database, index)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["imported_images"], 1)
        self.assertEqual(report["missing_images"], ["b.png"])
        with sqlite3.connect(database) as db:
            db.execute("INSERT INTO images VALUES ('b.png')")
        self.assertEqual(_feature_import_report(database, index)["status"], "complete")

    def test_staged_photos_record_original_and_repaired_hashes(self):
        photo = self.images / "bad-orientation.jpg"
        exif = Image.Exif()
        exif[274] = 0
        Image.new("RGB", (120, 90), "#a1b2c3").save(photo, exif=exif)
        manifest = self.root / "jpeg-reference.json"
        manifest.write_text(json.dumps({
            "schema_version": 1, "asset_id": "exif-statue", "subject": "EXIF test",
            "required_views": ["front"], "required_regions": ["face"],
            "images": [{"id": "exif-test", "file": "photos/bad-orientation.jpg",
                        "source_url": "https://example.org/exif.jpg", "view": "front",
                        "regions": ["face"]}],
        }))
        references = load_reference_set(manifest)
        output = self.root / "staged"
        output.mkdir()
        self.assertFalse(_prepare_images(list(references.images), output))
        index = json.loads((output / "staged-reference-index.json").read_text())["images"][0]
        self.assertTrue(index["invalid_exif_orientation_normalized"])
        self.assertNotEqual(index["original_sha256"], index["sha256"])
        with Image.open(photo) as original:
            self.assertEqual(original.getexif()[274], 0)
        with Image.open(output / "images/exif-test.jpg") as staged:
            self.assertEqual(staged.getexif()[274], 1)


if __name__ == "__main__":
    unittest.main()
