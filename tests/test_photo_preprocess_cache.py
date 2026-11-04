"""Per-photo content reuse must survive new images and filename renumbering."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from kumiki_blender.reconstruction.colmap import (
    _prepare_images, _stage_subject_qa_index, _verify_subject_qa_resume, _staged_images,
)
from kumiki_blender.reconstruction.references import ReferenceError, audit_reference_set
from kumiki_blender.reconstruction.photo_preprocess_cache import (
    prepare_statue_photo_cache, read_normalized_photo,
)
from kumiki_blender.reconstruction.photo_early_qa import analyze_cached_photo
from kumiki_blender.reconstruction.splat_input import load_photo_folder


class PhotoPreprocessCacheTests(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.cache = self.root / "source-cache"
        self.checkpoint = self.root / "sam2.pt"
        self.checkpoint.write_bytes(b"fake-checkpoint-v1")
        for i in range(8):
            Image.new("RGB", (96, 80), (10 * i, 40, 190 - i)).save(
                self.photos / f"photo-{i + 1:02}.jpg")

    @staticmethod
    def fake_mask(photo, _guide, _predictor):
        pixels = np.zeros((photo.height, photo.width), dtype=np.uint8)
        pixels[4:-4, 4:-4] = 255
        return pixels, {"components": ["subject"], "missing_parts": [],
                        "background_seed_violations": 0}

    def build(self, ordinal):
        source = load_photo_folder(self.photos, "great-sphinx")
        with patch("kumiki_blender.reconstruction.masking._sam2_mask", side_effect=self.fake_mask):
            return prepare_statue_photo_cache(
                source, self.cache, self.root / f"run-{ordinal}", checkpoint=self.checkpoint,
                predictor_loader=lambda *_: (object(), "mps"))

    def test_new_photo_only_misses_its_own_preprocessing_keys(self):
        initial = self.build(1)
        self.assertEqual((initial["normalized_misses"], initial["foreground_misses"]), (8, 8))
        old_sha = hashlib.sha256((self.photos / "photo-01.jpg").read_bytes()).hexdigest()
        Image.new("RGB", (96, 80), (150, 80, 30)).save(self.photos / "photo-00.jpg")
        second = self.build(2)
        self.assertEqual((second["normalized_hits"], second["normalized_misses"]), (8, 1))
        self.assertEqual((second["foreground_hits"], second["foreground_misses"]), (8, 1))
        photo = next(item for item in load_photo_folder(self.photos, "great-sphinx").images
                     if item.path.name == "photo-01.jpg")
        normalized, metadata = read_normalized_photo(self.cache, photo)
        self.assertEqual(metadata["source_sha256"], old_sha)
        with Image.open(normalized) as cached_image:
            self.assertEqual(cached_image.mode, "RGB")
        first = initial["images"][0]
        self.assertNotEqual(first["original_filename"], second["images"][0]["original_filename"])
        self.assertEqual(first["source_sha256"], second["images"][1]["source_sha256"])

        # COLMAP uses the exact cached RGB pixels, NOT candidate background cutouts.
        stage = self.root / "colmap-staging"
        stage.mkdir()
        self.assertFalse(_prepare_images(list(load_photo_folder(self.photos, "great-sphinx").images),
                                         stage, normalize_for_colmap=True,
                                         normalized_cache_dir=self.cache))
        index = json.loads((stage / "staged-reference-index.json").read_text())["images"]
        self.assertEqual(sum(bool(item["normalization_cache_hit"]) for item in index), 9)
        self.assertTrue(all(item["normalized_for_colmap"] for item in index))
        for item in index:
            with Image.open(stage / "images" / item["filename"]) as staged:
                self.assertEqual(staged.mode, "RGB")

    def test_changed_bytes_and_corruption_trigger_only_necessary_work(self):
        self.build(1)
        Image.new("RGB", (96, 80), (10, 20, 25)).save(self.photos / "photo-05.jpg")
        updated = self.build(2)
        self.assertEqual((updated["normalized_hits"], updated["normalized_misses"]), (7, 1))
        self.assertEqual((updated["foreground_hits"], updated["foreground_misses"]), (7, 1))
        photo = next(item for item in load_photo_folder(self.photos, "great-sphinx").images
                     if item.path.name == "photo-05.jpg")
        normalized, _ = read_normalized_photo(self.cache, photo)
        normalized.write_bytes(b"corrupt")
        refreshed = self.build(3)
        self.assertEqual((refreshed["normalized_hits"], refreshed["normalized_misses"]), (7, 1))
        # The normalized RGB recomputes to identical bytes, so its still-valid
        # foreground mask remains reusable without another neural inference.
        self.assertEqual((refreshed["foreground_hits"], refreshed["foreground_misses"]), (8, 0))

    def test_checkpoint_change_only_recomputes_foreground(self):
        self.build(1)
        self.checkpoint.write_bytes(b"fake-checkpoint-v2")
        updated = self.build(2)
        self.assertEqual((updated["normalized_hits"], updated["normalized_misses"]), (8, 0))
        self.assertEqual((updated["foreground_hits"], updated["foreground_misses"]), (0, 8))

    def test_resume_reuses_already_finished_source_image_work(self):
        self.build(1)
        source = load_photo_folder(self.photos, "great-sphinx")
        with patch("kumiki_blender.reconstruction.masking._sam2_mask", side_effect=self.fake_mask):
            result = prepare_statue_photo_cache(
                source, self.cache, self.root / "run-1", checkpoint=self.checkpoint,
                predictor_loader=lambda *_: (object(), "mps"), resume=True)
        self.assertEqual((result["normalized_hits"], result["foreground_hits"]), (8, 8))

    def test_resume_refuses_background_model_identity_change_without_touching_report(self):
        self.build(1)
        receipt = self.root / "run-1/photo-preprocess-report.json"
        original = receipt.read_bytes()
        source = load_photo_folder(self.photos, "great-sphinx")
        with self.assertRaisesRegex(ReferenceError, "Cannot resume preprocessing"):
            prepare_statue_photo_cache(
                source, self.cache, self.root / "run-1", checkpoint=self.checkpoint,
                predictor_loader=lambda *_: (object(), "mps"),
                background_proposer=lambda image: np.full(
                    (image.height, image.width), 255, dtype=np.uint8), resume=True)
        self.assertEqual(receipt.read_bytes(), original)

    def test_early_qa_marks_unreliable_foreground_as_uncertain_not_background(self):
        photo = self.photos / "photo-01.jpg"
        source = next(item for item in load_photo_folder(self.photos, "great-sphinx").images
                      if item.path == photo)
        from kumiki_blender.reconstruction.photo_preprocess_cache import normalize_cached_photo
        normalized, _, _ = normalize_cached_photo(source, self.cache)
        bad = self.root / "bad-mask.png"
        Image.new("L", (96, 80), 0).save(bad)
        receipt, hit = analyze_cached_photo(normalized, source.sha256, self.cache,
                                            mask_path=bad, model_sha="model-1")
        self.assertFalse(hit)
        self.assertIn("foreground-too-small", receipt["warnings"])
        with Image.open(receipt["confidence_path"]) as confidence:
            self.assertEqual(set(np.asarray(confidence).flat), {128})
        _, hit = analyze_cached_photo(normalized, source.sha256, self.cache,
                                      mask_path=bad, model_sha="model-1")
        self.assertTrue(hit)
        Image.new("L", (96, 80), 255).save(bad)
        changed, hit = analyze_cached_photo(normalized, source.sha256, self.cache,
                                             mask_path=bad, model_sha="model-1")
        self.assertFalse(hit)
        self.assertNotEqual(changed["key"], receipt["key"])

    def test_missing_anatomy_prompt_invalidates_mask_confidence(self):
        from kumiki_blender.reconstruction.photo_preprocess_cache import normalize_cached_photo
        image = load_photo_folder(self.photos, "great-sphinx").images[0]
        normalized, normalization, _ = normalize_cached_photo(image, self.cache)
        candidate = self.root / "partial.png"
        mask = Image.new("L", normalization["dimensions"], 0)
        from PIL import ImageDraw
        ImageDraw.Draw(mask).rectangle((15, 15, 75, 65), fill=255)
        mask.save(candidate)
        bad, _ = analyze_cached_photo(normalized, image.sha256, self.cache,
                                      mask_path=candidate, model_sha="sam2-test",
                                      mask_details={"missing_parts": ["tail"]})
        self.assertIn("candidate-missing-part:tail", bad["warnings"])
        with Image.open(bad["confidence_path"]) as confidence:
            self.assertEqual(set(np.asarray(confidence).flat), {128})
        self.assertEqual(normalization["exif_orientation_applied"], 1)

    def test_early_qa_cache_survives_new_photo_and_creates_contact_sheet(self):
        first = self.build(1)
        self.assertEqual((first["qa_hits"], first["qa_misses"]), (0, 8))
        self.assertTrue((self.root / "run-1" / "photo-qa-contact-sheet.jpg").is_file())
        detail = self.root / "run-1" / first["images"][0]["qa_visual_preview"]
        self.assertTrue(detail.is_file())
        with Image.open(detail) as panels:
            self.assertEqual(panels.size, (1536, 576))
        Image.new("RGB", (96, 80), (30, 70, 130)).save(self.photos / "added.jpg")
        second = self.build(2)
        self.assertEqual((second["qa_hits"], second["qa_misses"]), (8, 1))
        self.assertEqual(len(second["images"]), 9)
        self.assertTrue(all(Path(row["qa_confidence_path"]).is_file()
                            for row in second["images"]))

    def test_staged_feature_index_rejects_stale_image_or_modified_confidence(self):
        report = self.build(1)
        source = load_photo_folder(self.photos, "great-sphinx")
        stage = self.root / "stage-qa"
        stage.mkdir()
        _prepare_images(list(source.images), stage, normalize_for_colmap=True,
                        normalized_cache_dir=self.cache)
        report_path = self.root / "run-1" / "photo-preprocess-report.json"
        index = _stage_subject_qa_index(report_path, stage)
        self.assertEqual(len(index["images"]), 8)
        self.assertTrue((stage / "subject-feature-index.json").is_file())
        first = Path(report["images"][0]["qa_confidence_path"])
        first.write_bytes(b"corrupted")
        with self.assertRaisesRegex(Exception, "confidence.*hash|confidence.*changed"):
            _stage_subject_qa_index(report_path, stage)

    def test_duplicate_source_receipts_match_one_independent_camera(self):
        shutil.copyfile(self.photos / "photo-01.jpg", self.photos / "photo-copy.jpg")
        report = self.build(1)
        self.assertEqual(len(report["images"]), 9)
        sources = load_photo_folder(self.photos, "great-sphinx")
        selected = _staged_images(sources, audit_reference_set(sources))
        self.assertEqual(len(selected), 8)
        stage = self.root / "deduplicated-staging"
        stage.mkdir()
        _prepare_images(selected, stage, normalize_for_colmap=True,
                        normalized_cache_dir=self.cache)
        result = _stage_subject_qa_index(
            self.root / "run-1/photo-preprocess-report.json", stage)
        self.assertEqual(len(result["images"]), 8)

    def test_resume_qa_incompatibility_does_not_change_either_original_receipt(self):
        self.build(1)
        sources = load_photo_folder(self.photos, "great-sphinx")
        stage = self.root / "resume-source-qa"
        stage.mkdir()
        _prepare_images(list(sources.images), stage, normalize_for_colmap=True,
                        normalized_cache_dir=self.cache)
        receipt = self.root / "run-1/photo-preprocess-report.json"
        _stage_subject_qa_index(receipt, stage)
        baseline = (stage / "subject-feature-index.json").read_bytes()
        checksum = hashlib.sha256(baseline).hexdigest()
        data = json.loads(receipt.read_text())
        # Cache-hit counters are not semantic evidence and may change on rerun.
        data["qa_hits"] = 8
        data["qa_misses"] = 0
        receipt.write_text(json.dumps(data))
        _verify_subject_qa_resume(receipt, stage, checksum)
        data["images"][0]["qa_key"] = "different-setting"
        receipt.write_text(json.dumps(data))
        with self.assertRaisesRegex(Exception, "Cannot resume.*subject QA"):
            _verify_subject_qa_resume(receipt, stage, checksum)
        self.assertEqual((stage / "subject-feature-index.json").read_bytes(), baseline)
        data["images"][0]["qa_key"] = json.loads(baseline)["images"][
            list(json.loads(baseline)["images"])[0]]["qa_key"]
        receipt.write_text(json.dumps(data))
        _verify_subject_qa_resume(receipt, stage, checksum)

    def test_cached_independent_background_guides_sam2_without_erasing_uncertain_regions(self):
        for index, photo in enumerate(sorted(self.photos.iterdir())):
            Image.new("RGB", (96, 80), (125 + index * 6, 100, 70)).save(photo)

        def mask(image, guide, _predictor):
            pixels = np.zeros((image.height, image.width), dtype=np.uint8)
            pixels[15:65, 9:85] = 255
            if guide["foreground_points"]:
                pixels[20:60, 20:75] = 255
                pixels[15:20] = 0
                pixels[60:65] = 0
            return pixels, {"components": ["subject"], "missing_parts": [],
                            "background_seed_violations": 0}

        calls = []
        def independent(image):
            calls.append(image.size)
            pixels = np.zeros((image.height, image.width), dtype=np.uint8)
            pixels[20:60, 20:75] = 255
            return pixels

        source = load_photo_folder(self.photos, "great-sphinx")
        with patch("kumiki_blender.reconstruction.masking._sam2_mask", side_effect=mask):
            first = prepare_statue_photo_cache(
                source, self.cache, self.root / "consensus-1", checkpoint=self.checkpoint,
                predictor_loader=lambda *_: (object(), "mps"),
                background_proposer=independent)
        self.assertEqual(first["refinement_accepted"], 8)
        self.assertEqual(len(calls), 8)
        first_row = first["images"][0]
        self.assertTrue(Path(first_row["auxiliary_mask_path"]).is_file())
        with Image.open(first_row["qa_confidence_path"]) as confidence:
            data = np.asarray(confidence)
            self.assertEqual(int(data[30, 30]), 255)
            self.assertEqual(int(data[5, 5]), 0)
            self.assertEqual(int(data[17, 30]), 128)
        with patch("kumiki_blender.reconstruction.masking._sam2_mask", side_effect=AssertionError("cache miss")):
            second = prepare_statue_photo_cache(
                source, self.cache, self.root / "consensus-2", checkpoint=self.checkpoint,
                predictor_loader=lambda *_: (object(), "mps"),
                background_proposer=independent)
        self.assertEqual(second["foreground_hits"], 8)
        self.assertEqual(second["qa_hits"], 8)
        self.assertEqual(len(calls), 8)
        Path(first_row["auxiliary_mask_path"]).write_bytes(b"corrupted-independent-mask")
        with patch("kumiki_blender.reconstruction.masking._sam2_mask", side_effect=mask):
            repaired = prepare_statue_photo_cache(
                source, self.cache, self.root / "consensus-3", checkpoint=self.checkpoint,
                predictor_loader=lambda *_: (object(), "mps"),
                background_proposer=independent)
        self.assertEqual((repaired["foreground_hits"], repaired["foreground_misses"]), (7, 1))


if __name__ == "__main__":
    unittest.main()
