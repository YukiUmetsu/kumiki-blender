"""Multi-view mask consistency QA with genuine COLMAP binary/text layouts."""

import contextlib
import hashlib
import io
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.mask_comparison import compare_multiview_masks
from kumiki_blender.reconstruction.mask_consistency import audit_multiview_masks
from kumiki_blender.reconstruction.references import ReferenceError
from scripts.validate_foreground_splat_pipeline import validate_foreground_pipeline


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaskConsistencyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sparse = self.root / "sparse"
        photos = self.sparse / "undistorted/images"
        photos.mkdir(parents=True)
        model = self.sparse / "undistorted/sparse"
        model.mkdir(parents=True)
        (model / "cameras.bin").write_bytes(b"first-party camera fixture")
        (model / "points3D.bin").write_bytes(b"first-party 3D fixture")
        self.masks = self.root / "masks"
        self.masks.mkdir()
        images = []
        for index in range(3):
            name = f"view-{index:02d}.jpg"
            path = photos / name
            Image.new("RGB", (100, 80), (118, 112, 105)).save(path)
            images.append({"filename": name, "sha256": _digest(path), "dimensions": [100, 80]})
            mask = Image.new("L", (100, 80), 0)
            mask.paste(255, (20, 10, 80, 75))
            mask.save(self.masks / (name + ".mask.png"))
        (self.sparse / "undistorted-reference-index.json").write_text(
            json.dumps({"images": images}))
        (self.sparse / "reconstruction-report.json").write_text(json.dumps({
            "status": "candidate / visual-review-required", "registered_images": 3,
            "staged_independent_images": 3, "undistorted_images": 3,
        }))
        self.points = [
            [(30, 30, 100), (4, 30, 101), (30, 40, 102), (4, 60, 103), (50, 20, -1)],
            [(30, 30, 100), (4, 30, 101), (30, 40, 102), (30, 50, 103)],
            [(30, 30, 100), (4, 30, 101), (4, 40, 102)],
        ]
        self._write_model()

    def _write_model(self):
        content = bytearray(struct.pack("<Q", 3))
        for index, points in enumerate(self.points):
            name = f"view-{index:02d}.jpg"
            content.extend(struct.pack("<i7di", index+1, 1, 0, 0, 0, 0, 0, 0, 1))
            content.extend(name.encode() + b"\0")
            content.extend(struct.pack("<Q", len(points)))
            for x, y, point_id in points:
                content.extend(struct.pack("<ddq", x, y, point_id))
        (self.sparse / "undistorted/sparse/images.bin").write_bytes(content)

    def _foreground_pipeline_candidates(self):
        mask_revision = self.root / "foreground-mask"
        mask_revision.mkdir()
        shutil.copytree(self.masks, mask_revision / "masks")
        images = json.loads((self.sparse / "undistorted-reference-index.json").read_text())["images"]
        (mask_revision / "mask-report.json").write_text(json.dumps({
            "status": "candidate / inspect-every-mask-before-dense-reconstruction",
            "source_revision": str(self.sparse.resolve()),
            "approved_for_reconstruction": False,
            "images": [{"filename": item["filename"],
                        "source_photo_sha256": item["sha256"],
                        "mask": f'masks/{item["filename"]}.mask.png',
                        "mask_sha256": _digest(mask_revision / "masks" /
                                               (item["filename"] + ".mask.png")),
                        "warnings": []} for item in images],
        }))
        audit_revision = self.root / "foreground-audit"
        audit_multiview_masks(self.sparse, mask_revision / "masks", audit_revision)
        return mask_revision, audit_revision

    def test_foreground_pipeline_accepts_only_matching_review_candidates(self):
        masks, audit = self._foreground_pipeline_candidates()
        target = self.root / "fresh-splat"
        report = validate_foreground_pipeline(self.sparse, masks, audit, target,
                                              require_quality=False)
        self.assertEqual(report["registered_photos"], 3)
        self.assertTrue(report["high_priority_mask_reviews"])
        self.assertEqual(report["status"], "blocked / mask-quality-gates-failed")
        self.assertFalse(report["quality_passed"])
        self.assertTrue(any("part" in issue.lower() for issue in report["quality_failures"]))
        self.assertFalse(target.exists())
        # Replacing a mask after cross-view QA must invalidate the saved audit.
        mask = masks / "masks/view-00.jpg.mask.png"
        with Image.open(mask) as image:
            modified = image.copy()
        modified.putpixel((50, 50), 0)
        modified.save(mask)
        with self.assertRaisesRegex(ReferenceError, "provenance changed|changed since"):
            validate_foreground_pipeline(self.sparse, masks, audit, target)

    def test_foreground_pipeline_rejects_changed_audit_or_camera_model(self):
        masks, audit = self._foreground_pipeline_candidates()
        target = self.root / "fresh-splat"
        report_path = audit / "mask-consistency-report.json"
        report = json.loads(report_path.read_text())
        report["rectified_mask_source"] = str(self.root / "different")
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ReferenceError, "different masks"):
            validate_foreground_pipeline(self.sparse, masks, audit, target)
        report["rectified_mask_source"] = str(masks / "masks")
        report_path.write_text(json.dumps(report))
        model = self.sparse / "undistorted/sparse/images.bin"
        model.write_bytes(model.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ReferenceError, "COLMAP model changed"):
            validate_foreground_pipeline(self.sparse, masks, audit, target)

    def test_foreground_pipeline_accepts_large_cross_view_audit(self):
        masks, audit = self._foreground_pipeline_candidates()
        report_path = audit / "mask-consistency-report.json"
        report = json.loads(report_path.read_text())
        # 26-camera runs can legitimately exceed the standard 1 MiB
        # reconstruction evidence cap with per-feature review markers.
        report["large_review_metadata"] = "x" * (1024 * 1024)
        report_path.write_text(json.dumps(report))
        self.assertGreater(report_path.stat().st_size, 1024 * 1024)
        result = validate_foreground_pipeline(
            self.sparse, masks, audit, self.root / "large-audit-splat",
            require_quality=False)
        self.assertEqual(result["registered_photos"], 3)

    def test_quality_gate_rejects_incomplete_masks_and_cross_view_conflicts(self):
        masks, audit = self._foreground_pipeline_candidates()
        output = self.root / "blocked-training"
        with self.assertRaisesRegex(ReferenceError, "quality gates"):
            validate_foreground_pipeline(self.sparse, masks, audit, output)
        self.assertFalse(output.exists())

    def test_quality_gate_accepts_verified_anatomy_and_consistent_tracks(self):
        # Add actual supporting pixels for conflicting features so every
        # tracked 3D observation now agrees across all three views.
        masks, _ = self._foreground_pipeline_candidates()
        for name, pixel in (("view-00.jpg", (4, 60)),
                            ("view-02.jpg", (4, 40))):
            file = masks / "masks" / (name + ".mask.png")
            with Image.open(file) as image:
                copy = image.copy()
            copy.putpixel(pixel, 255)
            copy.save(file)
        report_path = masks / "mask-report.json"
        report = json.loads(report_path.read_text())
        for entry in report["images"]:
            entry["mask_sha256"] = _digest(masks / entry["mask"])
            entry["guide"] = {"part_points": {"head": [[0.3, 0.375]],
                                                 "torso": [[0.5, 0.65]]},
                              "background_points": [[0.1, 0.1]]}
        report_path.write_text(json.dumps(report))
        audit = self.root / "consistent-audit"
        audit_multiview_masks(self.sparse, masks / "masks", audit)
        validated = validate_foreground_pipeline(
            self.sparse, masks, audit, self.root / "allowed-training",
            min_compared_tracks=3)
        self.assertTrue(validated["quality_passed"])
        self.assertEqual(validated["quality_failures"], [])
        self.assertFalse((self.root / "allowed-training").exists())
        for entry in report["images"]:
            entry["guide"]["component_boxes"] = {"head": [0.2, 0.1, 0.6, 0.6],
                                                   "torso": [0.3, 0.5, 0.7, 0.8]}
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ReferenceError, "outside component boxes"):
            validate_foreground_pipeline(self.sparse, masks, audit,
                                         self.root / "denied-scenery", min_compared_tracks=3)
        for entry in report["images"]:
            del entry["guide"]["component_boxes"]
        # A positive click is a hard anatomical claim, not just a model hint.
        report["images"][0]["guide"]["foreground_points"] = [[0.05, 0.05]]
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ReferenceError, "foreground points"):
            validate_foreground_pipeline(self.sparse, masks, audit,
                                         self.root / "denied-positive-seed", min_compared_tracks=3)

    def test_sam2_requires_full_view_by_view_anatomy_visibility_inventory(self):
        masks, _ = self._foreground_pipeline_candidates()
        for name, pixel in (("view-00.jpg", (4, 60)), ("view-02.jpg", (4, 40))):
            path = masks / "masks" / (name + ".mask.png")
            with Image.open(path) as source:
                adjusted = source.copy()
            adjusted.putpixel(pixel, 255)
            adjusted.save(path)
        report_path = masks / "mask-report.json"
        document = json.loads(report_path.read_text())
        document["requires_anatomy_inventory"] = True
        for item in document["images"]:
            item["mask_sha256"] = _digest(masks / item["mask"])
            item["guide"] = {"part_points": {"head": [[0.3, 0.375]]},
                             "component_boxes": {"head": [0.0, 0.0, 0.99, 0.99]},
                             "background_points": [[0.1, 0.1]]}
        report_path.write_text(json.dumps(document))
        audit = self.root / "anatomy-audit"
        audit_multiview_masks(self.sparse, masks / "masks", audit)

        target = self.root / "anatomy-training"
        initial = validate_foreground_pipeline(self.sparse, masks, audit, target,
                                                min_compared_tracks=3,
                                                require_quality=False)
        self.assertFalse(initial["quality_passed"])
        self.assertTrue(any("visibility inventory" in failure
                            for failure in initial["quality_failures"]))

        visibility = {part: "not_visible" for part in (
            "headdress", "neck", "torso", "paws", "tail")}
        visibility["head"] = "visible"
        for item in document["images"]:
            item["guide"]["part_visibility"] = visibility.copy()
        report_path.write_text(json.dumps(document))
        verified = validate_foreground_pipeline(self.sparse, masks, audit, target,
                                                min_compared_tracks=3)
        self.assertTrue(verified["quality_passed"])
        self.assertEqual(verified["images"][0]["part_visibility"], visibility)

        for item in document["images"]:
            item["guide"]["part_visibility"]["paws"] = "visible"
        report_path.write_text(json.dumps(document))
        missing_paws = validate_foreground_pipeline(self.sparse, masks, audit, target,
                                                    min_compared_tracks=3,
                                                    require_quality=False)
        self.assertTrue(any("paws" in failure for failure in missing_paws["quality_failures"]))

        for item in document["images"]:
            item["guide"]["part_visibility"]["paws"] = "uncertain"
        report_path.write_text(json.dumps(document))
        uncertain = validate_foreground_pipeline(self.sparse, masks, audit, target,
                                                 min_compared_tracks=3,
                                                 require_quality=False)
        self.assertTrue(any("uncertain" in failure for failure in uncertain["quality_failures"]))

        for item in document["images"]:
            item["guide"]["part_visibility"]["paws"] = "not_visible"
            item["guide"]["component_boxes"]["paws"] = [0.1, 0.7, 0.9, 0.99]
        report_path.write_text(json.dumps(document))
        contradictory = validate_foreground_pipeline(self.sparse, masks, audit, target,
                                                    min_compared_tracks=3,
                                                    require_quality=False)
        self.assertTrue(any("not_visible" in failure for failure in contradictory["quality_failures"]))

    def test_flags_one_majority_outlier_and_ambiguous_two_view_track(self):
        target = self.root / "audit"
        result = audit_multiview_masks(self.sparse, self.masks, target)
        self.assertEqual(result["compared_3d_tracks"], 4)
        self.assertEqual(result["mixed_label_tracks"], 2)
        self.assertEqual(result["mixed_label_fraction"], 0.5)
        self.assertEqual(result["ambiguous_label_tracks"], 1)
        self.assertFalse(result["approved_for_reconstruction"])
        self.assertFalse(result["distribution_allowed"])
        self.assertEqual(len(result["images"]), 3)
        self.assertEqual(result["images"][2]["consensus_outliers"], 1)
        self.assertEqual(result["images"][2]["possible_missing_subject"], 1)
        self.assertEqual(result["images"][2]["possible_background_leaks"], 0)
        self.assertEqual(result["images"][2]["review_priority"], "high")
        proposal = result["images"][2]["correction_recommendations"][0]
        self.assertEqual(proposal["kind"], "possible_missing_subject")
        self.assertEqual(proposal["prompt_type"], "foreground_points")
        self.assertTrue(proposal["human_confirmation_required"])
        self.assertTrue((target / "mask-correction-recommendations.json").is_file())
        self.assertEqual(result["images"][0]["review_priority"], "medium")
        self.assertTrue((target / result["contact_sheet"]).is_file())
        self.assertEqual(result["model_sha256"], _digest(self.sparse /
                         "undistorted/sparse/images.bin"))
        for entry in result["images"]:
            self.assertTrue((target / entry["review"]).is_file())
            for detail in entry["detail_reviews"]:
                self.assertTrue((target / detail["review"]).is_file())
                with Image.open(target / detail["review"]) as crop:
                    self.assertEqual(crop.size, (100, 80))
            self.assertEqual(entry["mask_sha256"],
                             _digest(self.masks / (entry["filename"] + ".mask.png")))

    def test_text_model_fallback_and_cli(self):
        (self.sparse / "undistorted/sparse/images.bin").unlink()
        lines = ["# Rectified COLMAP image track model"]
        for index, points in enumerate(self.points):
            lines.append(f"{index+1} 1 0 0 0 0 0 0 1 view-{index:02d}.jpg")
            lines.append(" ".join(f"{x} {y} {point_id}" for x, y, point_id in points))
        (self.sparse / "undistorted/sparse/images.txt").write_text("\n".join(lines) + "\n")
        target = self.root / "cli-audit"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["audit-dense-masks", "--reconstruction", str(self.sparse),
                                   "--rectified-masks", str(self.masks),
                                   "--output", str(target),
                                   "--max-overlay-points", "10"]), 0)
        report = json.loads((target / "mask-consistency-report.json").read_text())
        self.assertTrue(report["model_file"].endswith("images.txt"))
        self.assertEqual(report["mixed_label_tracks"], 2)

    def test_overlay_budget_keeps_cyan_majority_outliers_ahead_of_ambiguous_ties(self):
        # The two-view tie appears before the 3-view disagreement in the model.
        # A one-marker preview must still show the higher-priority cyan marker.
        self.points[0].insert(2, (4, 60, 104))
        self.points[2].insert(2, (30, 65, 104))
        self._write_model()
        target = self.root / "budget-audit"
        result = audit_multiview_masks(self.sparse, self.masks, target,
                                       max_overlay_points=1)
        image = result["images"][2]
        self.assertEqual(image["consensus_outliers"], 1)
        self.assertEqual(image["ambiguous_observations"], 1)
        self.assertEqual(len(image["review_targets"]), 1)
        self.assertEqual(image["review_targets"][0]["kind"], "possible_missing_subject")
        self.assertEqual(image["review_targets"][0]["point3d_id"], 102)
        self.assertEqual(image["review_targets"][0]["foreground_votes"], 2)
        self.assertTrue(any("capped" in warning for warning in image["warnings"]))
        self.assertEqual(image["detail_reviews"][0]["kind"], "possible_missing_subject")
        with Image.open(target / image["review"]) as overlay:
            cyan = overlay.convert("RGB").getpixel((9, 40))
        self.assertGreater(cyan[1], cyan[0] * 1.5)
        self.assertGreater(cyan[2], cyan[0] * 1.5)

    def test_extra_foreground_disagreement_is_red_and_mask_tint_is_visible(self):
        # Two other masks reject this 3D point; the current camera includes it.
        for index, point in enumerate(((4, 70, 105), (4, 70, 105), (30, 70, 105))):
            self.points[index].append(point)
        self._write_model()
        target = self.root / "directional-audit"
        result = audit_multiview_masks(self.sparse, self.masks, target,
                                       max_overlay_points=1)
        image = result["images"][2]
        self.assertEqual(image["possible_background_leaks"], 1)
        self.assertEqual(image["possible_missing_subject"], 1)
        self.assertEqual(len(image["review_targets"]), 1)
        self.assertEqual(image["review_targets"][0]["kind"], "possible_background_leak")
        leak = next(point for point in image["review_targets"]
                    if point["kind"] == "possible_background_leak")
        self.assertEqual(leak["point3d_id"], 105)
        self.assertEqual(leak["background_votes"], 2)
        self.assertAlmostEqual(leak["normalized_xy"][0], 30/99, places=5)
        with Image.open(target / image["review"]) as overlay:
            photo = overlay.convert("RGB")
            red = photo.getpixel((35, 70))
            masked = photo.getpixel((50, 50))
            outside = photo.getpixel((10, 20))
        self.assertGreater(red[0], red[1] * 1.5)
        self.assertGreater(red[0], red[2] * 1.5)
        self.assertGreater(masked[0], outside[0])
        self.assertLess(masked[1], outside[1])

    def test_detail_review_preserves_full_resolution_near_sparse_features(self):
        index_path = self.sparse / "undistorted-reference-index.json"
        index = json.loads(index_path.read_text())
        for entry in index["images"]:
            name = entry["filename"]
            photo_path = self.sparse / "undistorted/images" / name
            Image.new("RGB", (1100, 900), (118, 112, 105)).save(photo_path)
            entry["sha256"] = _digest(photo_path)
            entry["dimensions"] = [1100, 900]
            mask = Image.new("L", (1100, 900), 0)
            mask.paste(255, (20, 10, 80, 75))
            mask.save(self.masks / (name + ".mask.png"))
        index_path.write_text(json.dumps(index))
        target = self.root / "large-photo-audit"
        result = audit_multiview_masks(self.sparse, self.masks, target)
        for entry in result["images"]:
            self.assertEqual(entry["source_photo_sha256"],
                             _digest(self.sparse / "undistorted/images" / entry["filename"]))
            for detail in entry["detail_reviews"]:
                with Image.open(target / detail["review"]) as crop:
                    self.assertEqual(crop.size, (512, 512))

    def test_normalized_review_coordinates_are_clamped_inside_photo(self):
        for points in self.points:
            points.append((99.8, 79.8, 106))
        self._write_model()
        mask_path = self.masks / "view-02.jpg.mask.png"
        with Image.open(mask_path) as existing:
            mask = existing.copy()
        mask.putpixel((99, 79), 255)
        mask.save(mask_path)
        result = audit_multiview_masks(self.sparse, self.masks,
                                       self.root / "boundary-audit")
        candidates = result["images"][2]["review_targets"]
        boundary = next(mark for mark in candidates if mark["point3d_id"] == 106)
        self.assertEqual(boundary["normalized_xy"], [1.0, 1.0])

    def test_repeated_3d_track_in_one_camera_does_not_cast_extra_votes(self):
        # Real COLMAP image models can contain multiple 2D observations of
        # one 3D ID per camera. One identical label counts only once; mixed
        # labels from the same camera must not be chosen arbitrarily.
        self.points[2].append((31, 30, 100))  # same foreground label
        self.points[2].append((30, 40, 102))  # conflicts with image-02 BG
        self._write_model()
        result = audit_multiview_masks(self.sparse, self.masks,
                                       self.root / "duplicate-tracks-audit")
        image = result["images"][2]
        self.assertEqual(image["duplicate_track_observations"], 2)
        self.assertEqual(image["conflicting_duplicate_tracks"], 1)
        self.assertEqual(image["consensus_outliers"], 0)
        self.assertEqual(image["tracked_observations"], 2)
        self.assertTrue(any("deduplicated" in warning for warning in image["warnings"]))

    def _candidate_masks(self):
        candidate = self.root / "candidate-masks"
        candidate.mkdir()
        for mask in self.masks.glob("*.mask.png"):
            shutil.copy2(mask, candidate / mask.name)
        return candidate

    def test_compare_revisions_reports_improvement_without_auto_approval(self):
        candidate = self._candidate_masks()
        mask_path = candidate / "view-02.jpg.mask.png"
        with Image.open(mask_path) as previous:
            changed = previous.copy()
        changed.putpixel((4, 40), 255)  # fixes single camera minority BG
        changed.save(mask_path)
        target = self.root / "comparison"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([
                "compare-dense-masks", "--reconstruction", str(self.sparse),
                "--baseline-masks", str(self.masks),
                "--candidate-masks", str(candidate),
                "--output", str(target), "--max-overlay-points", "5",
            ]), 0)
        report = json.loads((target / "mask-comparison-report.json").read_text())
        self.assertFalse(report["approved_for_reconstruction"])
        self.assertFalse(report["distribution_allowed"])
        self.assertEqual(report["delta"]["mixed_label_tracks_delta"], -1)
        self.assertEqual(report["delta"]["possible_missing_subject_delta"], -1)
        self.assertFalse(report["consistency_regressed"])
        self.assertEqual(len(report["images"]), 3)
        for group in ("baseline", "candidate"):
            self.assertTrue((target / report[group]["audit"]).is_file())
        with Image.open(self.masks / "view-02.jpg.mask.png") as unchanged:
            self.assertEqual(unchanged.getpixel((4, 40)), 0)

    def test_compare_flags_new_background_leak_and_invalid_sources(self):
        candidate = self._candidate_masks()
        path = candidate / "view-00.jpg.mask.png"
        with Image.open(path) as previous:
            changed = previous.copy()
        changed.putpixel((4, 30), 255)  # introduces minority extra FG
        changed.save(path)
        target = self.root / "comparison-leak"
        report = compare_multiview_masks(self.sparse, self.masks, candidate, target)
        self.assertGreater(report["delta"]["mixed_label_tracks_delta"], 0)
        self.assertEqual(report["delta"]["possible_background_leaks_delta"], 1)
        self.assertTrue(report["consistency_regressed"])
        for bad in (self.masks, self.root / "not-a-mask-folder"):
            rejected = self.root / ("rejected-" + bad.name)
            with self.assertRaises(ReferenceError):
                compare_multiview_masks(self.sparse, self.masks, bad, rejected)
            self.assertFalse(rejected.exists())

    def test_tampering_corrupt_tracks_or_no_overlap_rejects_without_output(self):
        target = self.root / "invalid"
        binary = self.sparse / "undistorted/sparse/images.bin"
        original = binary.read_bytes()
        binary.write_bytes(b"truncated")
        with self.assertRaisesRegex(ReferenceError, "registered image count|truncated"):
            audit_multiview_masks(self.sparse, self.masks, target)
        self.assertFalse(target.exists())
        binary.write_bytes(original)
        mask_path = self.masks / "view-00.jpg.mask.png"
        with Image.open(mask_path) as image:
            bad = image.copy()
        bad.putpixel((10, 10), 128)
        bad.save(mask_path)
        with self.assertRaisesRegex(ReferenceError, "binary pixels"):
            audit_multiview_masks(self.sparse, self.masks, target)
        self.assertFalse(target.exists())
        Image.new("L", (100, 80), 0).save(mask_path)
        # Use a valid binary mask again, then replace the tracks with unique IDs.
        with Image.open(self.masks / "view-01.jpg.mask.png") as other:
            other.save(mask_path)
        self.points = [[(30, 30, 100+idx)] for idx in range(3)]
        self._write_model()
        with self.assertRaisesRegex(ReferenceError, "No shared 3D feature tracks"):
            audit_multiview_masks(self.sparse, self.masks, target)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
