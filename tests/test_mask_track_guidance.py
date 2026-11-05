"""Measured COLMAP tracks produce only reviewable, held-out-tested SAM2 prompts."""

import hashlib
import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.mask_track_guidance import (
    _seed_class, evaluate_track_resegmentation, extract_track_guidance,
)
from kumiki_blender.reconstruction.mask_ground_plane import evaluate_local_ground_spill
from kumiki_blender.reconstruction.mask_track_resegmentation import refine_masks_from_tracks
from kumiki_blender.reconstruction.mask_track_selection import select_track_mask_candidates
from kumiki_blender.reconstruction.references import ReferenceError


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrackGuidanceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sparse = self.root / "registered-cameras"
        self.photo_dir = self.sparse / "undistorted/images"
        self.model_dir = self.sparse / "undistorted/sparse"
        self.model_dir.mkdir(parents=True)
        self.photo_dir.mkdir()
        self.mask_revision = self.root / "old-masks"
        for name in ("masks", "alternatives", "review"):
            (self.mask_revision / name).mkdir(parents=True)
        self.weights = self.root / "sam2.pt"
        self.weights.write_bytes(b"test-checkpoint")
        self.features = [(x, y, 1000 + i) for i, (x, y) in enumerate(
            (x, y) for x in range(20, 81, 6) for y in range(20, 66, 5))]
        self.images = []
        camera_model = bytearray(struct.pack("<Q", 6))
        for camera in range(6):
            name = f"view-{camera:02d}.png"
            photo_path = self.photo_dir / name
            Image.new("RGB", (100, 80), (115, 112, 102)).save(photo_path)
            selected = Image.new("L", (100, 80), 0)
            selected.paste(255, (10, 10, 90, 75))
            if camera == 5:
                paint = ImageDraw.Draw(selected)
                for x, y, point_id in self.features:
                    if x >= 46:
                        paint.rectangle((x-1, y-1, x+1, y+1), fill=0)
            full = Image.new("L", (100, 80), 0)
            full.paste(255, (10, 10, 90, 75))
            base = self.mask_revision / f"masks/{name}.mask.png"
            broad = self.mask_revision / f"alternatives/{name}.broad.png"
            grounded = self.mask_revision / f"alternatives/{name}.grounded.png"
            selected.save(base)
            selected.save(broad)
            full.save(grounded)
            self.images.append({
                "filename": name, "source_photo_sha256": digest(photo_path),
                "mask": f"masks/{name}.mask.png", "mask_sha256": digest(base),
                "guide_source": "grounding-dino / candidate",
                "detector_guide": {"component_boxes": {"subject": [0.12, 0.10, 0.90, 0.94]}},
                "candidate_masks": {
                    "broad": {"path": f"alternatives/{name}.broad.png", "sha256": digest(broad)},
                    "grounded": {"path": f"alternatives/{name}.grounded.png", "sha256": digest(grounded)}},
                "candidate_comparison": {"status": "uncertain / competing-candidates"},
                "warnings": [], "foreground_fraction": .60,
            })
            camera_model.extend(struct.pack("<i7di", camera + 1, 1, 0, 0, 0, 0, 0, 0, 1))
            camera_model.extend(name.encode() + b"\0")
            camera_model.extend(struct.pack("<Q", len(self.features)))
            for x, y, point_id in self.features:
                camera_model.extend(struct.pack("<ddq", x, y, point_id))
        (self.model_dir / "images.bin").write_bytes(camera_model)
        self.manifest = [{"filename": item["filename"],
                          "sha256": item["source_photo_sha256"],
                          "dimensions": [100, 80]} for item in self.images]
        (self.sparse / "undistorted-reference-index.json").write_text(json.dumps(
            {"images": self.manifest}))
        self.report = {"status": "candidate / inspect-every-mask-before-dense-reconstruction",
                       "source_revision": str(self.sparse.resolve()), "images": self.images}
        (self.mask_revision / "mask-report.json").write_text(json.dumps(self.report))

    def test_cross_view_point_prompts_are_evidence_pinned_and_held_out(self):
        evidence = extract_track_guidance(self.sparse, self.mask_revision, self.report)
        entry = evidence["images"]["view-05.png"]
        self.assertGreaterEqual(len(entry["seed_points"]), 3)
        self.assertGreaterEqual(len(entry["holdout_points"]), 5)
        self.assertTrue(all(_seed_class(row["point3d_id"]) == "seed"
                            for row in entry["seed_points"]))
        self.assertTrue(all(_seed_class(row["point3d_id"]) == "holdout"
                            for row in entry["holdout_points"]))
        self.assertTrue(all(len(row["source_views"]) >= 3
                            for row in entry["seed_points"]))
        self.assertEqual(evidence["compared_track_ids"], len(self.features))

    def test_unrelated_background_features_and_disconnected_groups_do_not_seed(self):
        for item in self.images:
            item["guide_source"] = "grounding-dino / no-detection"
        evidence = extract_track_guidance(self.sparse, self.mask_revision, self.report)
        self.assertEqual(evidence["viable_track_ids"], 0)
        for item in self.images:
            item["guide_source"] = "grounding-dino / candidate"
        for item in self.images:
            item["detector_guide"] = {"component_boxes": {"subject": [0.0, 0.0, 0.1, 0.1]}}
        evidence = extract_track_guidance(self.sparse, self.mask_revision, self.report)
        self.assertEqual(evidence["viable_track_ids"], 0)
        self.assertEqual(evidence["images"]["view-05.png"]["seed_points"], [])
        # A two-camera island cannot satisfy 3 independent supporting views.
        for item in self.images:
            item["detector_guide"] = {"component_boxes": {"subject": [0.12, 0.10, 0.9, 0.94]}}
        self.features = self.features[:5]
        self._make_disconnected_groups()
        evidence = extract_track_guidance(self.sparse, self.mask_revision, self.report)
        self.assertEqual(evidence["viable_track_ids"], 0)

    def _make_disconnected_groups(self):
        model = bytearray(struct.pack("<Q", 6))
        for camera in range(6):
            name = f"view-{camera:02d}.png"
            model.extend(struct.pack("<i7di", camera+1, 1, 0, 0, 0, 0, 0, 0, 1))
            model.extend(name.encode() + b"\0")
            model.extend(struct.pack("<Q", len(self.features)))
            for x, y, point_id in self.features:
                model.extend(struct.pack("<ddq", x, y, point_id + 100_000*(camera//2)))
        (self.model_dir / "images.bin").write_bytes(model)

    def test_heldout_rejects_false_positive_mask_expansion(self):
        original = np.zeros((80, 100), dtype=np.uint8)
        original[10:75, 10:90] = 255
        leaked = np.ones_like(original) * 255
        anchors = [{"normalized_xy": [.5, .5]}] * 5
        receipt = evaluate_track_resegmentation(original, leaked, anchors, anchors)
        self.assertFalse(receipt["accepted"])
        self.assertIn("excessive-foreground-area-change", receipt["reasons"])

    def test_independent_background_holdouts_allow_supported_growth_but_veto_ground(self):
        old = np.zeros((100, 100), dtype=np.uint8)
        old[20:80, 20:70] = 255
        recovered = old.copy()
        recovered[20:80, 70:95] = 255  # 50% additional, track-supported body.
        seeds = [{"normalized_xy": [x, y]} for x, y in (
            (.75, .25), (.83, .3), (.89, .4), (.76, .55), (.84, .65), (.9, .75))]
        heldout = [{"normalized_xy": [x, y]} for x, y in (
            (.77, .35), (.85, .45), (.8, .6), (.88, .7),
            (.3, .3), (.4, .4), (.5, .5), (.6, .6), (.3, .7), (.6, .75))]
        negatives = [{"normalized_xy": [x, .05]} for x in (.1, .3, .5, .7)]
        bg_holdout = [{"normalized_xy": [x, y]} for x, y in (
            (.1, .15), (.3, .1), (.5, .1), (.7, .1),
            (.1, .9), (.3, .9), (.5, .9), (.7, .9), (.9, .9), (.95, .1))]
        old_guard = evaluate_track_resegmentation(old, recovered, seeds, heldout)
        self.assertFalse(old_guard["accepted"])
        evaluated = evaluate_track_resegmentation(
            old, recovered, seeds, heldout, negatives, bg_holdout)
        self.assertTrue(evaluated["accepted"])
        self.assertEqual(evaluated["adaptive_growth_limit"], 1.55)
        self.assertEqual(evaluated["background_holdout_new_foreground"], 0)
        leaked_ground = recovered.copy()
        leaked_ground[84:96, 20:95] = 255
        rejected = evaluate_track_resegmentation(
            old, leaked_ground, seeds, heldout, negatives, bg_holdout)
        self.assertFalse(rejected["accepted"])
        self.assertIn("independent-background-tracks-became-foreground", rejected["reasons"])

    def test_cross_view_negative_prompts_require_multiple_mask_and_detector_votes(self):
        # A consistently observed ground track must be clear of both masks
        # *and* of all DINO component boxes in other registered views.
        ground_id = next(point_id for point_id in range(200000, 200100)
                         if _seed_class(point_id) == "seed")
        for entry in self.images:
            smaller = Image.new("L", (100, 80), 0)
            smaller.paste(255, (20, 20, 80, 70))
            for filename in (entry["mask"], entry["candidate_masks"]["broad"]["path"],
                             entry["candidate_masks"]["grounded"]["path"]):
                smaller.save(self.mask_revision / filename)
            entry["mask_sha256"] = digest(self.mask_revision / entry["mask"])
            for alternative in entry["candidate_masks"].values():
                alternative["sha256"] = digest(self.mask_revision / alternative["path"])
        model = bytearray(struct.pack("<Q", 6))
        for camera in range(6):
            model.extend(struct.pack("<i7di", camera+1, 1, 0, 0, 0, 0, 0, 0, 1))
            model.extend(f"view-{camera:02d}.png".encode() + b"\0")
            model.extend(struct.pack("<Q", len(self.features) + 1))
            for x, y, point_id in [*self.features, (10, 60, ground_id)]:
                model.extend(struct.pack("<ddq", x, y, point_id))
        (self.model_dir / "images.bin").write_bytes(model)
        result = extract_track_guidance(self.sparse, self.mask_revision, self.report)
        self.assertEqual(result["viable_background_track_ids"], 1)
        self.assertEqual(result["images"]["view-05.png"]["background_seed_points"][0]
                         ["point3d_id"], ground_id)
        # Text detector must also exclude the track: a false broad box vetoes it.
        for entry in self.images:
            entry["detector_guide"]["component_boxes"]["subject"] = [0, 0, 1, 1]
        result = extract_track_guidance(self.sparse, self.mask_revision, self.report)
        self.assertEqual(result["viable_background_track_ids"], 0)

    def test_low_triangulated_ground_plane_identifies_leak_inside_target_mask(self):
        # In real images a DINO part box can cover the entire floor. Ground
        # geometry must still work when that box is too broad, and must catch
        # one image in which both SAM2 candidates mistakenly include ground.
        ground = [(x, y, 5000+i) for i, (x, y) in enumerate(
            (x, y) for x in range(15, 86, 5) for y in range(40, 66, 5))]
        for item in self.images:
            name = item["filename"]
            mask = Image.new("L", (100, 80), 0)
            mask.paste(255, (25, 10, 75, 35))
            if name == "view-05.png":
                mask.paste(255, (40, 50, 65, 64))  # false foreground on floor
            for filename in (item["mask"], item["candidate_masks"]["broad"]["path"],
                             item["candidate_masks"]["grounded"]["path"]):
                mask.save(self.mask_revision / filename)
            item["mask_sha256"] = digest(self.mask_revision / item["mask"])
            for alternative in item["candidate_masks"].values():
                alternative["sha256"] = digest(self.mask_revision / alternative["path"])
            item["detector_guide"] = {"component_boxes": {"part_01": [0, 0, 1, 1]}}

        images = bytearray(struct.pack("<Q", 6))
        for camera in range(6):
            name = f"view-{camera:02d}.png"
            images.extend(struct.pack("<i7di", camera+1, 1, 0, 0, 0,
                                      camera*.2, 0, 0, 1))
            images.extend(name.encode() + b"\0")
            images.extend(struct.pack("<Q", len(self.features) + len(ground)))
            for x, y, point_id in [*self.features, *ground]:
                images.extend(struct.pack("<ddq", x, y, point_id))
        (self.model_dir / "images.bin").write_bytes(images)
        points = bytearray(struct.pack("<Q", len(ground)))
        for index, (x, y, point_id) in enumerate(ground):
            # World up from identity camera rotation is -Y: floor Y=3.
            points.extend(struct.pack("<QdddBBBdQ", point_id, x*.05, 3.0,
                                      y*.05, 140, 120, 95, .1, 6))
            for camera in range(6):
                points.extend(struct.pack("<II", camera+1, len(self.features)+index))
        (self.model_dir / "points3D.bin").write_bytes(points)
        evidence = extract_track_guidance(self.sparse, self.mask_revision, self.report)
        self.assertEqual(evidence["points3d_model_sha256"], digest(self.model_dir / "points3D.bin"))
        self.assertEqual(evidence["geometric_ground_plane"]["status"],
                         "candidate / low-camera-up-aligned-3d-plane")
        self.assertGreater(evidence["geometric_ground_plane"]["heldout_inlier_tracks"], 0)
        self.assertGreater(evidence["viable_background_track_ids"], 20)
        negatives = (evidence["images"]["view-05.png"]["background_seed_points"] +
                     evidence["images"]["view-05.png"]["background_holdout_points"])
        self.assertTrue(any(.40 <= row["normalized_xy"][0] <= .65 and
                            .62 <= row["normalized_xy"][1] <= .80 for row in negatives))
        self.assertTrue(all(row["source"] == "triangulated-low-ground-plane"
                            for row in negatives))
        self.assertTrue(all("view-05.png" not in row["source_views"] for row in negatives))

    def test_existing_ground_leak_in_holdout_is_not_acceptable(self):
        old = np.zeros((100, 100), dtype=np.uint8)
        old[20:80, 20:70] = 255
        proposed = old.copy()
        proposed[20:80, 70:95] = 255
        negatives = [{"normalized_xy": [.25, .75]}]
        receipt = evaluate_track_resegmentation(
            old, proposed,
            [{"normalized_xy": [x, .4]} for x in (.74, .78, .82, .86, .9)],
            [{"normalized_xy": [x, .5]} for x in (.75, .8, .85, .9, .95)],
            background_seeds=[], background_holdouts=negatives)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["background_holdout_new_foreground"], 0)
        self.assertEqual(receipt["background_holdout_foreground"], 1)
        self.assertIn("independent-background-tracks-became-foreground", receipt["reasons"])

    def test_measured_ground_appearance_vetoes_spill_but_not_stone_body(self):
        photo = Image.new("RGB", (200, 160), (130, 130, 125))
        ImageDraw.Draw(photo).rectangle((0, 107, 199, 159), fill=(185, 158, 120))
        before = Image.new("L", photo.size, 0)
        before.paste(255, (35, 30, 160, 105))
        floor_spill = before.copy()
        floor_spill.paste(255, (20, 114, 195, 145))
        ground_seed = [{"source": "triangulated-low-ground-plane",
                        "normalized_xy": [x, .80]} for x in (.24, .43, .62, .83)]
        rejected = evaluate_local_ground_spill(photo, before, floor_spill, ground_seed)
        self.assertTrue(rejected["rejected"])
        self.assertGreater(rejected["ground_region_added_fraction"], .16)
        body = before.copy()
        body.paste(255, (162, 35, 185, 90))
        accepted = evaluate_local_ground_spill(photo, before, body, ground_seed)
        self.assertFalse(accepted["rejected"])
        self.assertEqual(accepted["ground_region_added_fraction"], 0)

    def test_independent_heldout_ground_repair_can_shrink_mask_without_new_body(self):
        old = np.zeros((100, 100), dtype=np.uint8)
        old[20:80, 20:70] = 255
        fixed = old.copy()
        fixed[73:80, 20:70] = 0
        context = [{"normalized_xy": [x, .4]} for x in (.3, .4, .5, .6)]
        subject_holdouts = [{"normalized_xy": [x, .5]} for x in
                            (.3, .35, .4, .45, .5, .55)]
        negatives = [{"normalized_xy": [x, .75]} for x in (.3, .5)]
        bg_holdouts = [{"normalized_xy": [x, .75]} for x in (.4, .6)]
        repaired = evaluate_track_resegmentation(
            old, fixed, context, subject_holdouts, negatives, bg_holdouts,
            ground_repair=True, foreground_context_count=4)
        self.assertTrue(repaired["accepted"])
        self.assertTrue(repaired["ground_repair_verified"])
        self.assertEqual(repaired["status"], "candidate / heldout-ground-repair")
        # A repair cannot erase a held-out subject point, even if it removes ground.
        damaged = fixed.copy()
        damaged[45:55, 30:60] = 0
        rejected = evaluate_track_resegmentation(
            old, damaged, context, subject_holdouts, negatives, bg_holdouts,
            ground_repair=True, foreground_context_count=4)
        self.assertFalse(rejected["accepted"])

    def test_targeted_resegmentation_produces_new_revision_without_touching_original(self):
        original_hashes = [item["mask_sha256"] for item in self.images]
        cached = self.root / "shared-masks"
        correct = np.zeros((80, 100), dtype=np.uint8)
        correct[10:75, 10:90] = 255
        details = {"components": ["subject"], "missing_parts": [],
                   "background_seed_violations": 0}
        with (patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    return_value=(object(), "mps")) as loader,
              patch("kumiki_blender.reconstruction.masking._sam2_mask",
                    return_value=(correct, details)) as inference):
            first = refine_masks_from_tracks(
                self.sparse, self.mask_revision, self.root / "track-masks-r001",
                sam2_checkpoint=self.weights, cache_dir=cached)
            second = refine_masks_from_tracks(
                self.sparse, self.mask_revision, self.root / "track-masks-r002",
                sam2_checkpoint=self.weights, cache_dir=cached)
        self.assertEqual(first["status"], "candidate / automatic-track-guided-qa")
        self.assertEqual(first["selection_mode"], "automatic")
        self.assertGreaterEqual(first["summary"]["inference_count"], 1)
        self.assertEqual(first["summary"]["accepted"], 1)
        self.assertTrue(next(item for item in first["images"]
                             if item["filename"] == "view-05.png")["track_guided_replaced"])
        self.assertEqual(second["summary"]["inference_count"], 0)
        self.assertEqual(inference.call_count, 1)
        loader.assert_called_once()
        self.assertEqual([digest(self.mask_revision / entry["mask"])
                          for entry in self.images], original_hashes)
        self.assertTrue((self.root / "track-masks-r001/mask-contact-sheet.jpg").is_file())

    def test_verified_background_seed_is_sent_to_sam2_as_negative_click(self):
        evidence = extract_track_guidance(self.sparse, self.mask_revision, self.report)
        negative = {"point3d_id": 9999, "normalized_xy": [.02, .77],
                    "supporting_cameras": 5, "confidence": 1.0,
                    "source_views": [f"view-{i:02d}.png" for i in range(5)],
                    "ground_truth": False}
        evidence["images"]["view-05.png"]["background_seed_points"] = [negative]
        correct = np.zeros((80, 100), dtype=np.uint8)
        correct[10:75, 10:90] = 255
        with (patch("kumiki_blender.reconstruction.mask_track_resegmentation.extract_track_guidance",
                    return_value=evidence),
              patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    return_value=(object(), "mps")),
              patch("kumiki_blender.reconstruction.masking._sam2_mask",
                    return_value=(correct, {"components": ["subject"]})) as inference):
            report = refine_masks_from_tracks(self.sparse, self.mask_revision,
                                              self.root / "negative-revision",
                                              sam2_checkpoint=self.weights)
        self.assertIn([.02, .77], inference.call_args.args[1]["background_points"])
        revised = next(item for item in report["images"] if item["filename"] == "view-05.png")
        self.assertEqual(revised["track_guidance"]["background_seed_count"], 1)

    def test_automatic_ground_repair_retries_without_missing_body_seeds(self):
        evidence = extract_track_guidance(self.sparse, self.mask_revision, self.report)
        image = evidence["images"]["view-05.png"]
        image["seed_points"] = []
        image["foreground_context_points"] = [
            {"normalized_xy": [x, y]} for x, y in ((.22, .35), (.32, .4), (.4, .42))]
        image["holdout_points"] = [
            {"normalized_xy": [x, y]} for x, y in (
                (.2, .25), (.3, .25), (.4, .25), (.25, .45), (.35, .5), (.4, .55))]
        image["background_seed_points"] = [
            {"normalized_xy": [x, .8], "source": "triangulated-low-ground-plane"}
            for x in (.25, .35)]
        image["background_holdout_points"] = [
            {"normalized_xy": [x, .8], "source": "triangulated-low-ground-plane"}
            for x in (.28, .38)]
        selected = self.mask_revision / self.images[-1]["mask"]
        with Image.open(selected) as current:
            fixed = np.asarray(current).copy()
        fixed[57:69, 20:45] = 0  # remove supported ground only
        with (patch("kumiki_blender.reconstruction.mask_track_resegmentation.extract_track_guidance",
                    return_value=evidence),
              patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    return_value=(object(), "mps")),
              patch("kumiki_blender.reconstruction.masking._sam2_mask",
                    return_value=(fixed, {"components": ["subject"]})) as inference):
            report = refine_masks_from_tracks(self.sparse, self.mask_revision,
                                              self.root / "automatic-ground-repair",
                                              sam2_checkpoint=self.weights)
        self.assertEqual(inference.call_count, 1)
        selected_entry = next(item for item in report["images"]
                              if item["filename"] == "view-05.png")
        self.assertTrue(selected_entry["track_guided_replaced"])
        self.assertTrue(selected_entry["track_guidance"]["ground_repair_verified"])
        self.assertEqual(selected_entry["track_guidance"]["status"],
                         "candidate / heldout-ground-repair")

    def test_modified_source_mask_prevents_any_new_revision(self):
        path = self.mask_revision / self.images[0]["mask"]
        with Image.open(path) as old:
            revised = old.copy()
        revised.putpixel((30, 30), 0)
        revised.save(path)
        target = self.root / "no-output"
        with self.assertRaisesRegex(ReferenceError, "modified|changed"):
            refine_masks_from_tracks(self.sparse, self.mask_revision, target,
                                     sam2_checkpoint=self.weights)
        self.assertFalse(target.exists())

    def test_interrupted_revision_resumes_without_rewriting_completed_masks(self):
        target = self.root / "track-masks-interrupted"

        def interrupted(index, total, status):
            if index == 3:
                raise RuntimeError("simulated interruption")

        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            refine_masks_from_tracks(self.sparse, self.mask_revision, target,
                                     sam2_checkpoint=self.weights, progress=interrupted)
        partial = json.loads((target / "mask-report.json").read_text())
        self.assertEqual(partial["status"], "incomplete / track-refinement-review-required")
        self.assertEqual(len(partial["images"]), 3)
        copied_digest = digest(target / "masks/view-00.png.mask.png")
        correct = np.zeros((80, 100), dtype=np.uint8)
        correct[10:75, 10:90] = 255
        with (patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    return_value=(object(), "mps")),
              patch("kumiki_blender.reconstruction.masking._sam2_mask",
                    return_value=(correct, {"components": ["subject"]}))):
            resumed = refine_masks_from_tracks(self.sparse, self.mask_revision, target,
                                               sam2_checkpoint=self.weights, resume=True)
        self.assertEqual(resumed["summary"]["accepted"], 1)
        self.assertEqual(digest(target / "masks/view-00.png.mask.png"), copied_digest)
        with self.assertRaisesRegex(ReferenceError, "identical settings|unfinished"):
            refine_masks_from_tracks(self.sparse, self.mask_revision, target,
                                     sam2_checkpoint=self.weights, resume=True)

    def test_optional_selection_and_polygon_correction_preserves_automatic_revision(self):
        refined_root = self.root / "automatic"
        correct = np.zeros((80, 100), dtype=np.uint8)
        correct[10:75, 10:90] = 255
        with (patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                    return_value=(object(), "mps")),
              patch("kumiki_blender.reconstruction.masking._sam2_mask",
                    return_value=(correct, {"components": ["subject"]}))):
            refine_masks_from_tracks(self.sparse, self.mask_revision, refined_root,
                                     sam2_checkpoint=self.weights)
        bad = self.root / "bad-selection"
        with self.assertRaisesRegex(ReferenceError, "registered cameras"):
            select_track_mask_candidates(self.sparse, refined_root, bad,
                                         use_proposals=["unregistered-photo.png"])
        self.assertFalse(bad.exists())
        edits = self.root / "polygon-edits.json"
        edits.write_text(json.dumps({"images": {"view-05.png": {
            "exclude_polygons": [[[.60, .60], [.73, .60], [.73, .80], [.60, .80]]]
        }}}))
        chosen_root = self.root / "user-corrections"
        report = select_track_mask_candidates(
            self.sparse, refined_root, chosen_root,
            use_proposals=["view-05.png"], edits=edits)
        entry = next(item for item in report["images"] if item["filename"] == "view-05.png")
        self.assertEqual(entry["selection"], "explicit-user-proposal")
        self.assertTrue(entry["user_polygon_edit"])
        self.assertFalse(report["approved_for_reconstruction"])
        self.assertEqual(report["status"], "candidate / optional-user-corrections")
        with Image.open(chosen_root / entry["mask"]) as after, Image.open(
                refined_root / entry["mask"]) as before:
            self.assertEqual(after.getpixel((65, 55)), 0)
            self.assertEqual(before.getpixel((65, 55)), 255)
        self.assertTrue((chosen_root / "mask-contact-sheet.jpg").is_file())
        cli_root = self.root / "cli-selection"
        with redirect_stdout(io.StringIO()):
            result = main(["select-track-masks", "--reconstruction", str(self.sparse),
                           "--mask-revision", str(refined_root),
                           "--output", str(cli_root),
                           "--use-proposal", "view-05.png"])
        self.assertEqual(result, 0)
        self.assertTrue((cli_root / "mask-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
