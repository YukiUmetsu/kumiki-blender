"""A private, standalone report boundary independent of Blender and modeling."""

import hashlib
import json
import sys

import pytest
from PIL import Image

from kumiki_blender.face_analysis.__main__ import main


def _image(folder, stem="face"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (stem + ".png")
    Image.new("RGB", (100, 120), (145, 140, 130)).save(path)
    return path


def _json(path, data):
    path.write_text(json.dumps(data))
    return path


def _view_entry(image_id, y):
    return {"image_id": image_id, "width": 120, "height": 120, "view": "front",
            "landmarks": [{"landmark_id": "chin-tip", "xy_px": [50, y],
                           "origin": "manual", "visibility": "visible"}],
            "outline_px": [[0, 0], [100, 0], [100, 100], [0, 100]]}


def test_inspect_creates_private_audit_and_overlay_without_changing_source(tmp_path):
    photos = tmp_path / "photos"
    source = _image(photos)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    annotation = _json(tmp_path / "annotations.json", {"schema_version": 1, "images": [
        {"image_id": "face", "view": "front", "landmarks": [
            {"landmark_id": "chin-tip", "xy_px": [50, 100],
             "origin": "manual", "visibility": "visible"}],
         "outline_px": [[8, 8], [92, 8], [92, 110], [8, 110]]}
    ]})
    output = tmp_path / "revision-r01"
    assert main(["inspect", "--images-dir", str(photos), "--output", str(output),
                 "--annotations", str(annotation)]) == 0
    audit = json.loads((output / "photo-audit.json").read_text())
    assert audit["images_total"] == 1
    assert audit["images"][0]["sha256"] == digest
    assert audit["view_counts"] == {"front": 1}
    assert (output / "overlays/face.png").is_file()
    assert (output / "face-evidence.json").is_file()
    assert json.loads((output / "manifest.json").read_text())["likeness_status"] == "unmeasured"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_inspect_without_landmarks_remains_unmeasured_and_detector_missing_is_reported(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    _image(photos)
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    output = tmp_path / "report"
    assert main(["inspect", "--images-dir", str(photos), "--output", str(output),
                 "--detector", "mediapipe"]) == 0
    proposals = json.loads((output / "landmark-proposals.json").read_text())
    assert proposals["images"][0]["status"] == "unavailable"
    assert proposals["images"][0]["reason"] == "not-installed"
    assert json.loads((output / "photo-audit.json").read_text())["view_counts"] == {"unknown": 1}
    assert not (output / "likeness-qa.json").exists()


def test_unannotated_photo_set_has_naturally_ordered_contact_sheet(tmp_path):
    photos = tmp_path / "photos"
    _image(photos, "face10")
    _image(photos, "face2")
    output = tmp_path / "report"
    assert main(["inspect", "--images-dir", str(photos), "--output", str(output)]) == 0
    audit = json.loads((output / "photo-audit.json").read_text())
    assert audit["image_ids"] == ["face2", "face10"]
    assert (output / "contact-sheet.png").is_file()
    assert not (output / "overlays").exists()


def test_inspect_accepts_existing_reference_manifest_and_preserves_its_image_ids(tmp_path):
    _image(tmp_path, "filename-does-not-match-id")
    dossier = _json(tmp_path / "reference-manifest.json", {
        "schema_version": 1, "asset_id": "christ-face", "subject": "Christ the Redeemer",
        "required_views": ["left-profile"], "required_regions": ["face"],
        "images": [{"id": "original-image-id", "file": "filename-does-not-match-id.png",
                    "source_url": "https://example.org/photo", "view": "left-profile",
                    "regions": ["face"]}]
    })
    output = tmp_path / "review"
    assert main(["inspect", "--reference-manifest", str(dossier), "--output", str(output)]) == 0
    audit = json.loads((output / "photo-audit.json").read_text())
    assert audit["view_counts"] == {"profile": 1}
    assert audit["image_ids"] == ["original-image-id"]
    recorded = json.loads((output / "face-evidence.json").read_text())["images"][0]
    assert recorded["view"] == "profile"


def test_cli_rejects_existing_output_without_overwriting_files(tmp_path, capsys):
    photos = tmp_path / "photos"
    _image(photos)
    output = tmp_path / "existing"
    output.mkdir()
    keep = output / "notes.txt"
    keep.write_text("precious review")
    assert main(["inspect", "--images-dir", str(photos), "--output", str(output)]) != 0
    assert keep.read_text() == "precious review"
    assert "already exists" in capsys.readouterr().err


def test_cli_rejects_invalid_annotations_before_creating_revision(tmp_path, capsys):
    photos = tmp_path / "photos"
    _image(photos)
    bad = _json(tmp_path / "bad.json", {"schema_version": 1, "images": [
        {"image_id": "face", "view": "front", "landmarks": [
            {"landmark_id": "chin-tip", "xy_px": [800, 100],
             "origin": "manual", "visibility": "visible"}]}
    ]})
    output = tmp_path / "revision"
    assert main(["inspect", "--images-dir", str(photos), "--output", str(output),
                 "--annotations", str(bad)]) != 0
    assert not output.exists()
    assert "outside" in capsys.readouterr().err


def test_compare_cli_consumes_validated_evidence_bundles_and_reports_region_error(tmp_path):
    reference = _json(tmp_path / "reference.json", {"schema_version": 1, "images": [_view_entry("photo", 80)]})
    candidate = _json(tmp_path / "candidate.json", {"schema_version": 1, "images": [_view_entry("render", 70)]})
    pairs = _json(tmp_path / "pairs.json", {"schema_version": 1, "pairs": [
        {"reference_image_id": "photo", "candidate_image_id": "render",
         "alignment_id": "verified-camera-1", "verified": True}
    ]})
    output = tmp_path / "compare-r01"
    assert main(["compare", "--reference", str(reference), "--candidate", str(candidate),
                 "--pairs", str(pairs), "--output", str(output)]) == 0
    result = json.loads((output / "likeness-qa.json").read_text())
    assert result["regions"]["jaw"]["mean_error_normalized"] == pytest.approx(0.1)
    assert "higher" in result["observations"][0]["observation"]
    assert result["three_d_fidelity_verified"] is False


def test_silhouette_only_rear_pair_is_measured_in_report_manifest(tmp_path):
    rear = {"image_id": "rear", "width": 120, "height": 120, "view": "rear",
            "landmarks": [], "outline_px": [[10, 10], [90, 10], [90, 90], [10, 90]]}
    reference = _json(tmp_path / "reference.json", {"schema_version": 1, "images": [rear]})
    candidate = _json(tmp_path / "candidate.json", {"schema_version": 1, "images": [rear]})
    pairs = _json(tmp_path / "pairs.json", {"schema_version": 1, "pairs": [
        {"reference_image_id": "rear", "candidate_image_id": "rear",
         "alignment_id": "registered", "verified": True}]})
    output = tmp_path / "report"
    assert main(["compare", "--reference", str(reference), "--candidate", str(candidate),
                 "--pairs", str(pairs), "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["likeness_status"] == "projected-2d-measurements-only"


def test_compare_cli_does_not_write_if_pair_claims_invalid_evidence(tmp_path):
    reference = _json(tmp_path / "reference.json", {"schema_version": 1, "images": [_view_entry("photo", 80)]})
    candidate = _json(tmp_path / "candidate.json", {"schema_version": 1, "images": [_view_entry("render", 70)]})
    pairs = _json(tmp_path / "pairs.json", {"schema_version": 1, "pairs": [
        {"reference_image_id": "wrong", "candidate_image_id": "render",
         "alignment_id": "one", "verified": True}
    ]})
    output = tmp_path / "compare-r01"
    assert main(["compare", "--reference", str(reference), "--candidate", str(candidate),
                 "--pairs", str(pairs), "--output", str(output)]) != 0
    assert not output.exists()


def test_reprojection_cli_reuses_existing_camera_math_and_writes_revision(tmp_path, monkeypatch):
    from kumiki_blender.reconstruction import landmarks

    calls = []

    def reprojection(plan, measurements, output):
        calls.append((plan, measurements))
        output.mkdir()
        result = {"schema_version": 1, "status": "review-required",
                  "historical_fidelity_verified": False, "measurements": []}
        (output / "landmark-qa.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(landmarks, "check_landmark_reprojection", reprojection)
    plan, measurements = tmp_path / "camera-plan.json", tmp_path / "3d-points.json"
    output = tmp_path / "report"
    assert main(["reproject", "--camera-plan", str(plan),
                 "--sculpt-landmarks", str(measurements), "--output", str(output)]) == 0
    assert calls == [(plan, measurements)]
    assert json.loads((output / "landmark-qa.json").read_text())["status"] == "review-required"
    assert json.loads((output / "manifest.json").read_text())["three_d_fidelity_verified"] is False


def test_reprojection_failure_cannot_create_partial_report(tmp_path, monkeypatch):
    from kumiki_blender.reconstruction import landmarks

    def reprojection(*_args):
        raise ValueError("invalid camera plan")

    monkeypatch.setattr(landmarks, "check_landmark_reprojection", reprojection)
    output = tmp_path / "report"
    assert main(["reproject", "--camera-plan", "invalid", "--sculpt-landmarks", "invalid",
                 "--output", str(output)]) != 0
    assert not output.exists()
