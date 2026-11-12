"""Offline regression tests for SAM proposal -> Blender capture comparison."""
import hashlib
import json

import pytest
from PIL import Image, ImageDraw

from kumiki_blender.character_fidelity.sam_bridge import compare_sam_to_capture
from tests.test_character_fidelity import _make_fixture


def _fixture(tmp_path, *, verified=True, bad_digest=False, category="anatomy"):
    root = tmp_path / "source"
    root.mkdir()
    manifest, capture = _make_fixture(root)
    if not verified:
        data = json.loads(manifest.read_text())
        data["views"][0]["alignment"]["verified"] = False
        manifest.write_text(json.dumps(data))
    sam = tmp_path / "sam"
    (sam / "masks").mkdir(parents=True)
    Image.open(root / "mask.png").save(sam / "masks" / "head.png")
    digest = hashlib.sha256((root / "source.png").read_bytes()).hexdigest()
    (sam / "report.json").write_text(json.dumps({
        "schema_version": 1, "provider": "sam2.1-local",
        "image_sha256": "0" * 64 if bad_digest else digest,
        "parts": [{"part": "head", "category": category,
                   "proposed_mask": "masks/head.png", "predicted_iou": .99,
                   "approved": False}],
        "semantic_overlap_warnings": [],
    }))
    return manifest, capture, sam


def test_sam_bridge_keeps_proposed_mask_unapproved(tmp_path):
    manifest, capture, sam = _fixture(tmp_path)
    report = compare_sam_to_capture(manifest, capture, sam, tmp_path / "review")
    part = report["parts"][0]
    assert report["status"].startswith("segmentation-review-required")
    assert part["source_mask_provenance"]["source"] == "sam-proposal-unapproved"
    assert part["shape"]["intrinsic_shape_iou"] > .99
    assert part["shape"]["registered_iou"] > .9
    assert (tmp_path / "review/comparisons/front/head.png").is_file()
    assert "unapproved" in (tmp_path / "review/review.md").read_text()


def test_unverified_view_has_no_registered_scores(tmp_path):
    manifest, capture, sam = _fixture(tmp_path, verified=False)
    part = compare_sam_to_capture(manifest, capture, sam, tmp_path / "review")["parts"][0]
    assert "registered_iou" not in part["shape"]
    assert "centroid_offset_xy_px" not in part["shape"]
    assert (tmp_path / "review/comparisons/front/head.png").is_file()
    assert any(row["channel"] == "registration" for row in part["correction_observations"])


def test_different_source_image_hash_refused(tmp_path):
    manifest, capture, sam = _fixture(tmp_path, bad_digest=True)
    with pytest.raises(ValueError, match="SHA-256"):
        compare_sam_to_capture(manifest, capture, sam, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_mislabeled_anatomy_costume_part_refused(tmp_path):
    manifest, capture, sam = _fixture(tmp_path, category="costume")
    with pytest.raises(ValueError, match="category"):
        compare_sam_to_capture(manifest, capture, sam, tmp_path / "review")


def test_overlap_flags_need_for_source_mask_review(tmp_path):
    manifest, capture, sam = _fixture(tmp_path)
    report = json.loads((sam / "report.json").read_text())
    report["semantic_overlap_warnings"] = [{"first": "head", "second": "coat",
        "shared_pixels": 125, "fraction_of_smaller_part": .4}]
    (sam / "report.json").write_text(json.dumps(report))
    result = compare_sam_to_capture(manifest, capture, sam, tmp_path / "review")
    assert result["parts"][0]["source_mask_provenance"]["mask_conflict"]
    assert any(x["channel"] == "segmentation"
               for x in result["parts"][0]["correction_observations"])


def test_output_is_immutable(tmp_path):
    manifest, capture, sam = _fixture(tmp_path)
    compare_sam_to_capture(manifest, capture, sam, tmp_path / "review")
    with pytest.raises(ValueError, match="new directory"):
        compare_sam_to_capture(manifest, capture, sam, tmp_path / "review")


def test_sam_bridge_preserves_existing_measured_depth_reference(tmp_path):
    import numpy as np

    manifest, capture, sam = _fixture(tmp_path)
    np.savez_compressed(manifest.parent / "truth.npz", depth_m=np.full((64, 64), 4.75))
    data = json.loads(manifest.read_text())
    data["views"][0]["reference_depth"] = {"path": "truth.npz",
        "provenance": "measured", "units": "meters"}
    manifest.write_text(json.dumps(data))
    report = compare_sam_to_capture(manifest, capture, sam, tmp_path / "review")
    assert report["parts"][0]["depth"]["status"] == "measured-calibrated-depth"
    assert report["parts"][0]["depth"]["depth_mae_m"] == pytest.approx(.25)
