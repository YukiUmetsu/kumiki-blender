"""Execute the real host-process boundary with a controlled local CLI fixture."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from kumiki_blender.reconstruction.colmap import (
    _registered_image_count, build_sparse_reconstruction,
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
elif task == "model_converter":
    output = argument("--output_path")
    if not output.is_dir():
        print("model_converter expects an existing output directory")
        sys.exit(49)
    (output / "images.txt").write_text("# Image list\\n1 1 0 0 0 0 0 0 1 front.png\\n\\n2 1 0 0 0 0 0 0 1 left.png\\n\\n3 1 0 0 0 0 0 0 1 right.png\\n\\n")
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

    def test_partial_staging_failure_persists_diagnostic_report(self):
        source = self.references()
        real_copy = shutil.copyfile

        def fail_second_image(src, dst):
            if Path(src).name == "left.png":
                raise OSError("simulated exhausted storage")
            return real_copy(src, dst)

        output = self.root / "r01"
        with patch("kumiki_blender.reconstruction.colmap.shutil.copyfile",
                   side_effect=fail_second_image):
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


if __name__ == "__main__":
    unittest.main()
