"""Execute the real host-process boundary with a controlled local CLI fixture."""

import json
import io
import os
import shutil
import sqlite3
import struct
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
import sqlite3
import struct
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
        if "FAKE_LEARNED_COLMAP" in os.environ:
            print("--FeatureExtraction.type" if "FAKE_HELP_FLAGS_ONLY" in os.environ
                  else "--FeatureExtraction.type SIFT ALIKED_N16ROT LOMA_B LOMA_B128")
            print("--FeatureExtraction.max_image_size")
    elif task == "exhaustive_matcher":
        print("--SiftMatching.use_gpu" if legacy else "--FeatureMatching.use_gpu")
        if "FAKE_LEARNED_COLMAP" in os.environ:
            print("--FeatureMatching.type" if "FAKE_HELP_FLAGS_ONLY" in os.environ
                  else "--FeatureMatching.type SIFT_LIGHTGLUE ALIKED_LIGHTGLUE LOMA_B LOMA_B128")
    sys.exit(1 if "FAKE_HELP_NONZERO" in os.environ else 0)
def argument(key):
    return Path(params[params.index(key)+1])
if task == "feature_extractor":
    if "FAKE_SLOW_FEATURE_EXTRACTOR" in os.environ:
        import time
        time.sleep(3)
    if ("--SiftExtraction.use_gpu" if legacy else "--FeatureExtraction.use_gpu") not in params:
        sys.exit(42)
    images = argument("--image_path")
    if len(list(images.iterdir())) != 3:
        sys.exit(11)
    masks = argument("--ImageReader.mask_path")
    if not (masks / "front.png.png").is_file():
        sys.exit(12)
    with sqlite3.connect(argument("--database_path")) as db:
        db.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
        db.execute("CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, width INTEGER, height INTEGER)")
        db.execute("INSERT INTO cameras VALUES (1,120,90)")
        db.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        db.execute("CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        db.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        db.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        db.executemany("INSERT INTO images (name, camera_id) VALUES (?, 1)",
                       [(path.name,) for path in sorted(images.iterdir())])
        for (image_id,) in db.execute("SELECT image_id FROM images"):
            db.execute("INSERT INTO keypoints VALUES (?, 4, 4, ?)",
                       (image_id, b''.join(struct.pack('<ffff', 10.0+i*8, 12.0+i*7, 1.0+i, 0.0)
                                           for i in range(4))))
            db.execute("INSERT INTO descriptors VALUES (?, 4, 128, ?)", (image_id, bytes([image_id])*512))
elif task == "exhaustive_matcher":
    if ("--SiftMatching.use_gpu" if legacy else "--FeatureMatching.use_gpu") not in params:
        sys.exit(42)
    if "FAKE_FAIL_MATCHER" in __import__("os").environ:
        print("simulated incompatible feature matching")
        sys.exit(7)
elif task == "matches_importer":
    if "--match_type" not in params or params[params.index("--match_type") + 1] != "pairs":
        sys.exit(43)
    if "--FeatureMatching.type" not in params or "--ExhaustiveMatching.block_size" in params:
        sys.exit(44)
    pairs = argument("--match_list_path").read_text().splitlines()
    if not pairs or len(pairs) > 5:
        sys.exit(45)
    progress = argument("--database_path").parent / "fake-matched-pairs.txt"
    if ("FAKE_CHUNK_OOM" in os.environ and len(pairs) > 1 or
            "FAKE_SINGLE_PAIR_OOM" in os.environ):
        print("Memory guard stopped COLMAP: RSS 13000 MiB >= 12288 MiB")
        sys.exit(125)
    if "FAKE_CHUNK_FAIL_AFTER_FIRST" in os.environ and progress.exists():
        print("simulated native matching failure")
        sys.exit(7)
    with progress.open("a") as stream:
        for pair in pairs:
            stream.write(pair + "\\n")
elif task == "geometric_verifier":
    if "FAKE_FAIL_GEOMETRIC" in os.environ:
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
        # The sandbox process namespace may hide the Python test process from
        # ps. Keep native RSS sampling deterministic in worker fixture tests.
        rss_probe = patch("kumiki_blender.reconstruction.colmap._process_rss_kib", return_value=0)
        rss_probe.start()
        self.addCleanup(rss_probe.stop)
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

    def test_learned_matching_presets_pass_compatible_colmap_arguments(self):
        presets = {"sift-lightglue": ("SIFT", "SIFT_LIGHTGLUE"),
                   "aliked-lightglue": ("ALIKED_N16ROT", "ALIKED_LIGHTGLUE"),
                   "loma": ("LOMA_B", "LOMA_B"),
                   "loma128": ("LOMA_B128", "LOMA_B128")}
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1"}):
            for preset, (extractor, matcher) in presets.items():
                with self.subTest(preset=preset):
                    output = self.root / preset
                    report = build_sparse_reconstruction(
                        self.references(), output, colmap=str(self.fake), matching_preset=preset)
                    self.assertEqual(report["status"], "candidate / visual-review-required")
                    self.assertEqual(report["matching_preset"], preset)
                    self.assertEqual(report["registered_filenames"],
                                     ["front.png", "left.png", "right.png"])
                    self.assertEqual(report["unregistered_filenames"], [])
                    self.assertIn(["--FeatureExtraction.type", extractor],
                                  [report["commands"][0][index:index + 2]
                                   for index in range(len(report["commands"][0]) - 1)])
                    self.assertEqual(report["commands"][0][-2:],
                                     ["--FeatureExtraction.max_image_size", "4000"])
                    self.assertIn("--FeatureExtraction.num_threads", report["commands"][0])
                    self.assertIn("--FeatureMatching.num_threads", report["commands"][1])
                    self.assertIn("--FeatureMatching.max_num_matches", report["commands"][1])
                    self.assertIn("--ExhaustiveMatching.block_size", report["commands"][1])
                    self.assertEqual(report["memory_controls"]["feature_threads"], 1)
                    self.assertEqual(report["memory_controls"]["max_colmap_rss_mb"], 12288)
                    self.assertEqual(report["memory_controls"]["matching_max_features"], 4096)
                    self.assertEqual(report["memory_controls"]["matching_block_size"], 5)
                    if preset == "sift-lightglue":
                        self.assertIn("--SiftExtraction.first_octave", report["commands"][0])
                    self.assertEqual(report["commands"][1][-2:],
                                     ["--FeatureMatching.type", matcher])

    def test_learned_matching_preflight_fails_before_creating_revision(self):
        output = self.root / "unsupported-learned"
        with self.assertRaisesRegex(ReferenceError, "lacks learned feature-selection"):
            build_sparse_reconstruction(self.references(), output, colmap=str(self.fake),
                                        matching_preset="aliked-lightglue")
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ReferenceError, "requires --engine colmap"):
            build_sparse_reconstruction(self.references(), output, engine="pycolmap",
                                        matching_preset="loma")
        self.assertFalse(output.exists())

    def test_colmap_help_without_enum_values_and_nonzero_help_exit_is_supported(self):
        # Real packages can omit accepted enum names and exit 1 for --help.
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1",
                                  "FAKE_HELP_FLAGS_ONLY": "1", "FAKE_HELP_NONZERO": "1"}):
            report = build_sparse_reconstruction(self.references(), self.root / "flags-only",
                                                 colmap=str(self.fake), matching_preset="sift-lightglue")
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertEqual(report["matching_preset"], "sift-lightglue")

    def test_memory_guard_stops_native_extraction_and_keeps_diagnostic_log(self):
        output = self.root / "guarded"
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1", "FAKE_SLOW_FEATURE_EXTRACTOR": "1"}):
            with patch("kumiki_blender.reconstruction.colmap._process_rss_kib",
                       return_value=13 * 1024 * 1024):
                report = build_sparse_reconstruction(self.references(), output,
                                                     colmap=str(self.fake),
                                                     matching_preset="sift-lightglue",
                                                     max_image_size=4000)
        self.assertEqual(report["failed_stage"], "feature_extractor")
        self.assertEqual(report["exit_code"], 125)
        self.assertEqual(report["memory_guard_reason"], "RSS 13312 MiB >= 12288 MiB")
        self.assertIn(report["memory_guard_reason"], report["error"])
        self.assertIn("Memory guard stopped COLMAP", (output / "logs/01-feature_extractor.log").read_text())

    def test_memory_controls_reject_invalid_limits_before_creating_output(self):
        output = self.root / "invalid-controls"
        for controls in ({"feature_threads": 0}, {"matching_threads": 65},
                         {"max_colmap_rss_mb": 100}, {"sift_first_octave": -2},
                         {"matching_block_size": 0}, {"matching_max_features": 100}):
            with self.subTest(controls=controls):
                with self.assertRaises(ReferenceError):
                    build_sparse_reconstruction(self.references(), output, colmap=str(self.fake),
                                                **controls)
                self.assertFalse(output.exists())

    def test_matching_resource_controls_override_learned_defaults(self):
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1"}):
            report = build_sparse_reconstruction(
                self.references(), self.root / "custom-matching-bounds",
                colmap=str(self.fake), matching_preset="sift-lightglue",
                matching_block_size=3, matching_max_features=2048)
        args = report["commands"][1]
        self.assertEqual(args[args.index("--ExhaustiveMatching.block_size") + 1], "3")
        self.assertEqual(args[args.index("--FeatureMatching.max_num_matches") + 1], "2048")
        self.assertEqual(report["memory_controls"]["matching_block_size"], 3)
        self.assertEqual(report["memory_controls"]["matching_max_features"], 2048)

    def test_learned_matching_runs_in_separate_checkpointed_colmap_processes(self):
        output = self.root / "chunked"
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1"}):
            report = build_sparse_reconstruction(self.references(), output,
                                                 colmap=str(self.fake),
                                                 matching_preset="sift-lightglue",
                                                 matching_pairs_per_process=2)
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertEqual(report["matching_progress"]["total_pairs"], 3)
        self.assertEqual(report["matching_progress"]["completed_pairs"], 3)
        self.assertEqual(report["matching_progress"]["completed_batches"], 2)
        self.assertEqual(report["matching_progress"]["status"], "complete")
        self.assertEqual([c[0] for c in report["commands"]].count("matches_importer"), 2)
        self.assertNotIn("exhaustive_matcher", [c[0] for c in report["commands"]])
        self.assertEqual(len((output / "fake-matched-pairs.txt").read_text().splitlines()), 3)
        self.assertEqual(len(list((output / "pairs").glob("*.txt"))), 2)
        self.assertEqual(len(list((output / "logs").glob("*-matches_importer.log"))), 2)

    def test_mps_matching_subprocess_is_batched_and_verified_before_mapping(self):
        from kumiki_blender.reconstruction import colmap

        actual_run = colmap._run

        def stub_mps_only(executable, stage, arguments, output, number, timeout_seconds,
                          *, max_rss_mb=None):
            if stage == "match":
                self.assertEqual(executable[1:3], ["-m", "kumiki_blender.reconstruction.mps_lightglue_worker"])
                self.assertIn("--max-keypoints", arguments)
                self.assertIn("--device", arguments)
                self.assertEqual(arguments[arguments.index("--device") + 1], "mps")
                return 0, [stage, *arguments]
            return actual_run(executable, stage, arguments, output, number, timeout_seconds,
                              max_rss_mb=max_rss_mb)

        with (patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1"}),
              patch("kumiki_blender.reconstruction.colmap._run", side_effect=stub_mps_only)):
            report = build_sparse_reconstruction(
                self.references(), self.root / "mps", colmap=str(self.fake),
                matching_preset="sift-lightglue", matching_device="mps",
                matching_max_features=2048, matching_pairs_per_process=2)
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertEqual(report["matching_progress"]["completed_pairs"], 3)
        self.assertEqual(report["matching_device"], "mps")
        self.assertEqual([command[0] for command in report["commands"]],
                         ["feature_extractor", "match", "match", "geometric_verifier",
                          "mapper", "model_converter"])

    def test_resume_existing_cpu_batch_progress_with_new_mps_backend(self):
        from kumiki_blender.reconstruction import colmap

        output = self.root / "mps-resume"
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1", "FAKE_CHUNK_FAIL_AFTER_FIRST": "1"}):
            failed = build_sparse_reconstruction(
                self.references(), output, colmap=str(self.fake),
                matching_preset="sift-lightglue", matching_pairs_per_process=2)
        self.assertEqual(failed["matching_progress"]["completed_pairs"], 2)
        actual_run = colmap._run

        def stub_mps_only(executable, stage, arguments, output, number, timeout_seconds,
                          *, max_rss_mb=None):
            if stage == "match":
                return 0, [stage, *arguments]
            return actual_run(executable, stage, arguments, output, number, timeout_seconds,
                              max_rss_mb=max_rss_mb)

        with (patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1"}),
              patch("kumiki_blender.reconstruction.colmap._run", side_effect=stub_mps_only)):
            resumed = build_sparse_reconstruction(
                self.references(), output, colmap=str(self.fake),
                matching_preset="sift-lightglue", matching_device="mps",
                matching_pairs_per_process=8, matching_max_features=2048, resume=True)
        self.assertEqual(resumed["matching_progress"]["completed_pairs"], 3)
        self.assertEqual(resumed["matching_progress"]["batch_size"], 8)
        self.assertEqual(resumed["matching_progress"]["completed_batches"], 2)
        self.assertEqual([command[0] for command in resumed["commands"]].count("feature_extractor"), 1)
        self.assertEqual(resumed["matching_device_history"][-1],
                         {"from": "colmap-cpu", "to": "mps"})

    def test_mps_geometric_verification_failure_is_resumable_without_rematching(self):
        from kumiki_blender.reconstruction import colmap

        output = self.root / "geo-resume"
        actual_run = colmap._run

        def stub_mps_only(executable, stage, arguments, output, number, timeout_seconds,
                          *, max_rss_mb=None):
            if stage == "match":
                return 0, [stage, *arguments]
            return actual_run(executable, stage, arguments, output, number, timeout_seconds,
                              max_rss_mb=max_rss_mb)

        with (patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1", "FAKE_FAIL_GEOMETRIC": "1"}),
              patch("kumiki_blender.reconstruction.colmap._run", side_effect=stub_mps_only)):
            failed = build_sparse_reconstruction(
                self.references(), output, colmap=str(self.fake),
                matching_preset="sift-lightglue", matching_device="mps",
                matching_pairs_per_process=2)
        self.assertEqual(failed["failed_stage"], "geometric_verifier")
        self.assertEqual(failed["matching_progress"]["status"], "complete")
        with (patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1"}),
              patch("kumiki_blender.reconstruction.colmap._run", side_effect=stub_mps_only)):
            resumed = build_sparse_reconstruction(
                self.references(), output, colmap=str(self.fake),
                matching_preset="sift-lightglue", matching_device="mps",
                matching_pairs_per_process=2, resume=True)
        self.assertEqual(resumed["status"], "candidate / visual-review-required")
        self.assertEqual([command[0] for command in resumed["commands"]].count("match"), 2)
        self.assertEqual([command[0] for command in resumed["commands"]].count("geometric_verifier"), 2)

    def test_mps_preflight_requires_valid_preset_and_batched_matching(self):
        output = self.root / "invalid-mps"
        for values in ({"matching_device": "mps"},
                       {"matching_device": "mps", "matching_preset": "sift-lightglue"},
                       {"matching_device": "other"}):
            with self.subTest(values=values), self.assertRaises(ReferenceError):
                build_sparse_reconstruction(self.references(), output, colmap=str(self.fake), **values)
            self.assertFalse(output.exists())

    def test_curated_pairs_reduce_number_of_workers_and_preserve_checkpoint_identity(self):
        chosen = self.root / "pairs.txt"
        chosen.write_text("left.png front.png\nright.png left.png\n", encoding="utf-8")
        output = self.root / "selected-pairs"
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1"}):
            report = build_sparse_reconstruction(
                self.references(), output, colmap=str(self.fake),
                matching_preset="sift-lightglue", matching_pairs_per_process=2,
                matching_pair_list=chosen)
        self.assertEqual(report["matching_progress"]["total_pairs"], 2)
        self.assertEqual(report["matching_pair_selection"], "manual")
        self.assertEqual(len((output / "fake-matched-pairs.txt").read_text().splitlines()), 2)

    def test_curated_pair_list_rejects_unknown_names_before_matching(self):
        chosen = self.root / "bad-pairs.txt"
        chosen.write_text("left.png missing.png\n", encoding="utf-8")
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1"}):
            with self.assertRaisesRegex(ReferenceError, "known staged image"):
                build_sparse_reconstruction(
                    self.references(), self.root / "bad-pairs", colmap=str(self.fake),
                    matching_preset="sift-lightglue", matching_pairs_per_process=2,
                    matching_pair_list=chosen)

    def test_memory_exceeded_batch_halves_until_single_pair_succeeds(self):
        output = self.root / "adaptive-chunked"
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1", "FAKE_CHUNK_OOM": "1"}):
            report = build_sparse_reconstruction(self.references(), output,
                                                 colmap=str(self.fake),
                                                 matching_preset="sift-lightglue",
                                                 matching_pairs_per_process=3)
        self.assertEqual(report["matching_progress"]["completed_pairs"], 3)
        self.assertEqual(report["matching_progress"]["batch_size"], 1)
        self.assertEqual(report["matching_progress"]["completed_batches"], 3)
        self.assertEqual(report["matching_progress"]["last_failure"]["code"], 125)
        self.assertEqual(report["status"], "candidate / visual-review-required")
        self.assertEqual(len((output / "fake-matched-pairs.txt").read_text().splitlines()), 3)

    def test_single_pair_out_of_memory_stops_without_an_unbounded_retry_loop(self):
        output = self.root / "single-pair-limit"
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1", "FAKE_SINGLE_PAIR_OOM": "1"}):
            report = build_sparse_reconstruction(self.references(), output,
                                                 colmap=str(self.fake),
                                                 matching_preset="sift-lightglue",
                                                 matching_pairs_per_process=2)
        self.assertEqual(report["failed_stage"], "matches_importer")
        self.assertEqual(report["exit_code"], 125)
        self.assertEqual(report["matching_progress"]["next_pair_index"], 0)
        self.assertEqual(report["matching_progress"]["batch_size"], 1)
        self.assertEqual(len([c for c in report["commands"] if c[0] == "matches_importer"]), 2)

    def test_resume_reuses_successful_batches_and_skips_extraction(self):
        output = self.root / "resumed-matching"
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1", "FAKE_CHUNK_FAIL_AFTER_FIRST": "1"}):
            failed = build_sparse_reconstruction(self.references(), output,
                                                 colmap=str(self.fake),
                                                 matching_preset="sift-lightglue",
                                                 matching_pairs_per_process=2)
        self.assertEqual(failed["matching_progress"]["next_pair_index"], 2)
        self.assertEqual(failed["failed_stage"], "matches_importer")
        # Simulate interruption after writing the next pair list but before
        # launching its worker. Resume must allocate a fresh filename.
        (output / "pairs" / "0004-pairs.txt").write_text("orphaned file\n")
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1"}):
            recovered = build_sparse_reconstruction(self.references(), output,
                                                    colmap=str(self.fake),
                                                    matching_preset="sift-lightglue",
                                                    matching_pairs_per_process=1, resume=True)
        self.assertEqual(recovered["status"], "candidate / visual-review-required")
        self.assertEqual(recovered["resume_count"], 1)
        self.assertEqual(recovered["matching_progress"]["completed_pairs"], 3)
        self.assertEqual(recovered["matching_progress"]["batch_size"], 1)
        self.assertEqual([c[0] for c in recovered["commands"]].count("feature_extractor"), 1)
        self.assertEqual(len((output / "fake-matched-pairs.txt").read_text().splitlines()), 3)
        self.assertEqual(len(list((output / "logs").glob("*-matches_importer.log"))), 3)
        self.assertTrue((output / "pairs" / "0005-pairs.txt").is_file())

    def test_resume_existing_legacy_matcher_failure_without_reextracting(self):
        output = self.root / "legacy-failure"
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1", "FAKE_FAIL_MATCHER": "1"}):
            failed = build_sparse_reconstruction(self.references(), output,
                                                 colmap=str(self.fake),
                                                 matching_preset="sift-lightglue")
        self.assertEqual(failed["failed_stage"], "exhaustive_matcher")
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1"}):
            recovered = build_sparse_reconstruction(self.references(), output,
                                                    colmap=str(self.fake),
                                                    matching_preset="sift-lightglue",
                                                    matching_pairs_per_process=2, resume=True)
        self.assertEqual(recovered["status"], "candidate / visual-review-required")
        self.assertEqual(recovered["matching_progress"]["total_pairs"], 3)
        self.assertEqual([c[0] for c in recovered["commands"]].count("feature_extractor"), 1)

    def test_resume_rejects_incompatible_input_resolution_without_modifying_report(self):
        output = self.root / "unsafe-resume"
        with patch.dict(os.environ, {"FAKE_LEARNED_COLMAP": "1", "FAKE_FAIL_MATCHER": "1"}):
            build_sparse_reconstruction(self.references(), output, colmap=str(self.fake),
                                        matching_preset="sift-lightglue")
            before = (output / "reconstruction-report.json").read_bytes()
            with self.assertRaisesRegex(ReferenceError, "resolution has changed"):
                build_sparse_reconstruction(self.references(), output, colmap=str(self.fake),
                                            matching_preset="sift-lightglue", max_image_size=2048,
                                            matching_pairs_per_process=2, resume=True)
            self.assertEqual((output / "reconstruction-report.json").read_bytes(), before)

    def test_registration_names_use_normalized_staged_filenames_not_original_jpeg_extensions(self):
        self.references()
        manifest = self.root / "reference-set.json"
        raw = json.loads(manifest.read_text())
        for entry in raw["images"]:
            old = self.root / entry["file"]
            jpg = old.with_suffix(".jpg")
            with Image.open(old) as image:
                image.save(jpg)
            entry["file"] = "photos/" + jpg.name
        manifest.write_text(json.dumps(raw))
        source = load_reference_set(manifest)
        report = build_sparse_reconstruction(source, self.root / "jpeg-revision", colmap=str(self.fake))
        self.assertEqual(report["registered_filenames"], ["front.png", "left.png", "right.png"])
        self.assertEqual(report["unregistered_filenames"], [])

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
