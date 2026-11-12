"""Regression gates for overlapping SAM masks and unverified camera registration."""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from kumiki_blender.character_fidelity.mask_review import refine_masks
from kumiki_blender.character_fidelity.registration_review import propose_registration
from test_character_fidelity import _make_fixture


def _sam(tmp_path):
    image = tmp_path / "drawing.png"
    Image.new("RGB", (80, 80), "white").save(image)
    sam = tmp_path / "sam"
    (sam / "masks").mkdir(parents=True)
    a = np.zeros((80, 80), np.uint8)
    b = np.zeros_like(a)
    a[10:50, 10:50] = 255
    b[10:50, 30:70] = 255
    Image.fromarray(a).save(sam / "masks" / "wing.png")
    Image.fromarray(b).save(sam / "masks" / "coat.png")
    report = {"schema_version": 1, "provider": "sam2.1-local",
              "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
              "parts": [{"part": "wing", "category": "anatomy", "approved": False,
                         "proposed_mask": "masks/wing.png", "predicted_iou": .97},
                        {"part": "coat", "category": "costume", "approved": False,
                         "proposed_mask": "masks/coat.png", "predicted_iou": .93}]}
    (sam / "report.json").write_text(json.dumps(report))
    return image, sam


def test_ambiguous_pixels_are_not_falsely_assigned(tmp_path):
    im, sam = _sam(tmp_path)
    report = refine_masks(im, sam, tmp_path / "review")
    assert report["mask_review"]["raw_contested_pixels"] == 800
    assert report["mask_review"]["total_ambiguous_pixels"] == 800
    wing = np.array(Image.open(tmp_path / "review/masks/wing.png")) > 0
    coat = np.array(Image.open(tmp_path / "review/masks/coat.png")) > 0
    assert not (wing & coat).any()
    assert wing.sum() == 800 and coat.sum() == 800
    assert np.array(Image.open(tmp_path / "review/raw_masks/wing.png")).sum() // 255 == 1600
    assert report["parts"][0]["mask_review"]["incomplete_shape_evidence"] is True
    assert all(p["approved"] is False for p in report["parts"])


def test_explicit_pixel_ownership_resolves_only_annotated_overlap(tmp_path):
    im, sam = _sam(tmp_path)
    owners = np.zeros((80, 80), np.uint8)
    owners[10:50, 30:40] = 255
    Image.fromarray(owners).save(tmp_path / "wing-owner.png")
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"schema_version": 1,
        "source_image_sha256": hashlib.sha256(im.read_bytes()).hexdigest(),
        "ownership_masks": {"wing": "wing-owner.png"}}))
    report = refine_masks(im, sam, tmp_path / "review", rules=rules)
    assert report["mask_review"]["explicitly_resolved_pixels"] == 400
    assert report["mask_review"]["total_ambiguous_pixels"] == 400
    wing = np.array(Image.open(tmp_path / "review/masks/wing.png")) > 0
    coat = np.array(Image.open(tmp_path / "review/masks/coat.png")) > 0
    assert wing[20, 35] and not coat[20, 35]
    assert not wing[20, 45] and not coat[20, 45]
    assert not np.any(wing & coat)


def test_full_canvas_source_layer_overrides_sam(tmp_path):
    im, sam = _sam(tmp_path)
    source_mask = np.zeros((80, 80), np.uint8)
    source_mask[20:25, 20:25] = 255
    Image.fromarray(source_mask).save(tmp_path / "source-layer.png")
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"schema_version": 1,
        "source_masks": {"wing": "source-layer.png"}}))
    report = refine_masks(im, sam, tmp_path / "refined", rules=rules)
    assert report["mask_review"]["source_overrides"] == ["wing"]
    assert (np.array(Image.open(tmp_path / "refined/masks/wing.png")) > 0).sum() == 25


def test_refine_requires_same_source_and_full_canvas(tmp_path):
    im, sam = _sam(tmp_path)
    data = json.loads((sam / "report.json").read_text())
    data["image_sha256"] = "0" * 64
    (sam / "report.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA-256"):
        refine_masks(im, sam, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_conflicting_ownership_rejected_before_creating_output(tmp_path):
    im, sam = _sam(tmp_path)
    conflicting = np.zeros((80, 80), np.uint8)
    conflicting[15:30, 35:45] = 255
    Image.fromarray(conflicting).save(tmp_path / "a.png")
    Image.fromarray(conflicting).save(tmp_path / "b.png")
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"schema_version": 1,
        "ownership_masks": {"wing": "a.png", "coat": "b.png"}}))
    with pytest.raises(ValueError, match="multiple parts"):
        refine_masks(im, sam, tmp_path / "review", rules=rules)
    assert not (tmp_path / "review").exists()


def _registered_fixture(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    manifest, capture = _make_fixture(root)
    return manifest, capture


def test_registration_proposes_only_without_certifying_camera(tmp_path):
    manifest, capture = _registered_fixture(tmp_path)
    points = [[10, 10], [10, 40], [40, 10], [40, 40], [22, 29]]
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps({"schema_version": 1, "view": "front", "landmarks": [
        {"id": str(i), "source_xy": p, "render_xy": [p[0] + 5, p[1] + 2]}
        for i, p in enumerate(points)]}))
    proposal = propose_registration(manifest, capture, anchors, tmp_path / "fit")
    assert proposal["status"].startswith("reviewable")
    assert proposal["alignment_candidate"]["verified"] is False
    assert proposal["inlier_rmse_px"] < 0.001
    assert proposal["alignment_candidate"]["source_to_render"][0][2] == pytest.approx(5)
    assert (tmp_path / "fit/alignment_overlay.png").is_file()
    assert json.loads(manifest.read_text())["views"][0]["alignment"]["verified"] is True


def test_registration_rejects_unsupported_anchor_evidence(tmp_path):
    manifest, capture = _registered_fixture(tmp_path)
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps({"schema_version": 1, "view": "front",
        "landmarks": [{"id": str(i), "source_xy": [10, 10],
                       "render_xy": [20, 20]} for i in range(4)]}))
    with pytest.raises(ValueError, match="two-dimensional"):
        propose_registration(manifest, capture, anchors, tmp_path / "fit")
    assert not (tmp_path / "fit").exists()


def test_registration_rejects_stale_sha(tmp_path):
    manifest, capture = _registered_fixture(tmp_path)
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps({"schema_version": 1, "view": "front",
        "source_image_sha256": "0"*64, "landmarks": [
           {"id": str(i), "source_xy": [10+i, 10+i*i], "render_xy": [10+i, 10+i*i]}
           for i in range(4)]}))
    with pytest.raises(ValueError, match="SHA-256"):
        propose_registration(manifest, capture, anchors, tmp_path / "fit")


def test_reconciliation_suppresses_sculpt_advice_for_incomplete_mask(tmp_path):
    from kumiki_blender.character_fidelity.sam_bridge import compare_sam_to_capture
    import hashlib, json
    root=tmp_path / "model"
    root.mkdir()
    manifest, capture=_make_fixture(root)
    sam=tmp_path / "sam"
    (sam/"masks").mkdir(parents=True)
    (sam/"masks/head.png").write_bytes((root/"mask.png").read_bytes())
    (sam/"report.json").write_text(json.dumps({"schema_version":1, "provider":"sam2.1-local",
        "image_sha256":hashlib.sha256((root/"source.png").read_bytes()).hexdigest(),
        "parts":[{"part":"head","category":"anatomy","proposed_mask":"masks/head.png",
                  "mask_review":{"incomplete_shape_evidence":True,"unresolved_pixels_excluded":50}}],
        "semantic_overlap_warnings":[] ,"mask_review":{"total_ambiguous_pixels":50}}))
    result=compare_sam_to_capture(manifest,capture,sam,tmp_path/"analysis")
    row=result["parts"][0]
    assert row["source_mask_incomplete"] is True
    assert row["shape"]["source_evidence_status"].startswith("partial")
    assert any(x["channel"]=="mask-evidence" for x in row["correction_observations"])
    assert not any(x["channel"]=="shape" for x in row["correction_observations"])
