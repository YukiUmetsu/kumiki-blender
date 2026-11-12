"""Regression tests for independent shape, placement, color and depth evidence."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from kumiki_blender.character_fidelity.metrics import color, depth, shape, surface_detail
from kumiki_blender.character_fidelity.schema import load_manifest
from kumiki_blender.character_fidelity.evaluate import evaluate_character


def test_translation_is_not_mistaken_for_intrinsic_shape_difference():
    a = np.zeros((100, 100), bool)
    b = np.zeros_like(a)
    a[15:40, 10:40] = True
    b[35:60, 45:75] = True
    report = shape(a, b, registered=True)
    assert report["intrinsic_shape_iou"] == pytest.approx(1)
    assert report["registered_iou"] == 0
    assert report["centroid_offset_xy_px"] == [35, 20]


def test_intrinsic_shape_preserves_aspect_ratio():
    a = np.zeros((100, 100), bool)
    b = np.zeros_like(a)
    a[20:50, 20:60] = True
    b[20:60, 20:50] = True
    result = shape(a, b, registered=True)
    assert result["intrinsic_shape_iou"] < .8
    assert result["reference_aspect_ratio"] != result["sculpt_aspect_ratio"]


def test_unverified_registration_cannot_report_spatial_accuracy():
    a = np.zeros((100, 100), bool)
    a[10:20, 10:20] = True
    result = shape(a, a, registered=False)
    assert result["intrinsic_shape_iou"] == pytest.approx(1)
    assert "registered_iou" not in result
    assert "centroid_offset_xy_px" not in result


def test_missing_part_is_explicit():
    a, b = np.zeros((32, 32), bool), np.zeros((32, 32), bool)
    a[5:12, 5:12] = True
    result = shape(a, b, registered=True)
    assert result["status"] == "missing-part"
    assert result["sculpt_area_px"] == 0


def test_color_and_detail_are_independent():
    m = np.ones((32, 32), dtype=bool)
    red, blue = np.zeros((32, 32, 3), dtype=np.uint8), np.zeros((32, 32, 3), dtype=np.uint8)
    red[:] = [194, 55, 42]
    blue[:] = [35, 74, 205]
    same = color(red, red, m, m)
    different = color(red, blue, m, m)
    assert same["delta_e76"] == pytest.approx(0)
    assert different["delta_e76"] > 60
    assert surface_detail(red, m)["mean_abs_laplacian"] == 0
    # A flat object and a patterned one can have similar median RGB, but
    # the Lab distribution check must still see the missing pattern.
    patterned = np.full_like(red, 128)
    patterned[:, :16, :] = 64
    patterned[:, 16:, :] = 192
    flat = np.full_like(red, 128)
    assert color(patterned, flat, m, m)["lab_histogram_tvd_mean"] > .3
    assert len(surface_detail(patterned, m)["edge_orientation_histogram_8"]) == 8


def test_depth_requires_calibrated_ground_truth():
    mask = np.ones((32, 32), dtype=bool)
    candidate = np.full((32, 32), 5.0)
    truth = np.full((32, 32), 4.75)
    unverified = depth(candidate, mask, reference_depth=truth, reference_mask=mask, calibrated=False)
    assert "depth_mae_m" not in unverified
    validated = depth(candidate, mask, reference_depth=truth, reference_mask=mask, calibrated=True)
    assert validated["depth_mae_m"] == pytest.approx(.25)
    assert validated["matched_pixels"] == 1024


def _make_fixture(root: Path):
    source = Image.new("RGB", (64, 64), (255, 255, 255))
    art = ImageDraw.Draw(source)
    art.ellipse((10, 5, 35, 32), fill=(135, 78, 43))
    source.save(root / "source.png")
    source_mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(source_mask).ellipse((10, 5, 35, 32), fill=255)
    source_mask.save(root / "mask.png")
    manifest = {"schema_version": 1,
        "views": [{"id": "front", "camera": "Camera", "width": 64, "height": 64,
                   "reference_image": "source.png", "alignment": {
                       "verified": True, "source_to_render": [[1, 0, 0], [0, 1, 0]]}}],
        "parts": [{"id": "head", "category": "anatomy", "objects": ["Head*"],
                   "reference_masks": {"front": "mask.png"}}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    capture = root / "capture"
    (capture / "front").mkdir(parents=True)
    labels = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(labels).ellipse((10, 5, 35, 32), fill=(255, 12, 12, 255))
    labels.save(capture / "front" / "labels.png")
    source.save(capture / "front" / "albedo.png")
    np.savez_compressed(capture / "front" / "depth_m.npz", depth_m=np.full((64, 64), 5.0))
    (capture / "capture.json").write_text(json.dumps({"schema_version": 1,
        "palette": {"head": [.95, .0475, .0475]}, "assignments": {"Head": "head"},
        "views": {"front": {"labels": "front/labels.png", "albedo": "front/albedo.png",
                            "depth_npz": "front/depth_m.npz", "part_depth_extent_m": {"head": [4.5, 5.5]}}}}))
    return root / "manifest.json", capture


def test_end_to_end_writes_part_comparison_and_category(tmp_path):
    workspace = tmp_path / "source"
    workspace.mkdir()
    manifest, capture = _make_fixture(workspace)
    output = tmp_path / "analysis"
    report = evaluate_character(manifest, capture, output)
    item = report["parts"][0]
    assert item["category"] == "anatomy"
    assert item["shape"]["registered_iou"] > .9
    assert item["depth"]["status"].startswith("sculpt-only")
    assert (output / "comparisons/front/head.png").is_file()
    with pytest.raises(ValueError, match="new directory"):
        evaluate_character(manifest, capture, output)


def test_manifest_rejects_duplicate_parts_and_path_escape(tmp_path):
    manifest, _ = _make_fixture(tmp_path)
    config = json.loads(manifest.read_text())
    config["parts"].append(config["parts"][0].copy())
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Duplicate part"):
        load_manifest(manifest)
    config["parts"] = config["parts"][:1]
    config["views"][0]["reference_image"] = "../escape.png"
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unsafe asset path"):
        load_manifest(manifest)


def test_pose_mismatch_reports_no_false_registered_score(tmp_path):
    workspace = tmp_path / "source"
    workspace.mkdir()
    manifest, capture = _make_fixture(workspace)
    config = json.loads(manifest.read_text())
    config["parts"][0]["unmatched_pose_views"] = ["front"]
    manifest.write_text(json.dumps(config))
    report = evaluate_character(manifest, capture, tmp_path / "output")
    assert report["parts"][0]["pose_matched"] is False
    assert "registered_iou" not in report["parts"][0]["shape"]
