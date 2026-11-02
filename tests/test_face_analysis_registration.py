"""Unverified image-plane registration suggestions must not masquerade as 3D QA."""

import json
import xml.etree.ElementTree as ET

import pytest

from kumiki_blender.face_analysis.__main__ import main
from kumiki_blender.face_analysis.registration import (RegistrationPair,
                                                       propose_registrations)
from kumiki_blender.face_analysis.reports import write_analysis_revision
from kumiki_blender.face_analysis.schema import Annotation, AnnotationSet, ViewAnnotations


ANCHORS = ("eye-inner-left", "eye-inner-right", "nose-tip")


def _view(image_id, coords, *, view="front", proposed=()):
    records = tuple(Annotation(image_id, key, value,
                               "proposed" if key in proposed else "manual", "visible")
                    for key, value in coords.items())
    return ViewAnnotations(image_id, view, records, ((0, 0), (100, 0), (100, 100), (0, 100)))


def _samples():
    reference = AnnotationSet({"ref": _view("ref", {
        "eye-inner-left": (20, 20), "eye-inner-right": (60, 20),
        "nose-tip": (20, 60), "chin-tip": (40, 80)})})
    candidate = AnnotationSet({"render": _view("render", {
        "eye-inner-left": (10, 25), "eye-inner-right": (50, 25),
        "nose-tip": (10, 65), "chin-tip": (35, 85)})})
    return reference, candidate


def test_proposed_similarity_alignment_reports_heldout_chin_error_not_false_verified_score():
    reference, candidate = _samples()
    report = propose_registrations(reference, candidate,
                                   [RegistrationPair("ref", "render", ANCHORS)])
    pair = report["pairs"][0]
    assert pair["status"] == "proposed-unverified"
    assert pair["transform"]["scale"] == pytest.approx(1)
    assert pair["transform"]["translation_px"] == pytest.approx([10, -5])
    assert pair["anchor_rms_normalized"] == pytest.approx(0, abs=1e-12)
    assert pair["heldout"]["chin-tip"]["delta_normalized"] == pytest.approx([0.05, 0])
    assert pair["heldout"]["chin-tip"]["candidate_aligned_px"] == pytest.approx([45, 80])
    assert pair["anchors_excluded_from_heldout"] is True
    assert report["camera_registration_verified"] is False
    assert report["three_d_fidelity_verified"] is False


def test_similarity_registration_recovers_known_rotation_and_scale():
    reference, _ = _samples()
    # Deliberately generate candidate coordinates from a known 90°/2x
    # candidate-to-reference similarity transform, including a held-out point.
    points = {entry.landmark_id: ((entry.xy_px[1] - 20) / 2,
                                  (100 - entry.xy_px[0]) / 2)
              for entry in reference.images["ref"].landmarks}
    candidate = AnnotationSet({"render": _view("render", points)})
    pair = propose_registrations(reference, candidate,
                                 [RegistrationPair("ref", "render", ANCHORS)])["pairs"][0]
    assert pair["status"] == "proposed-unverified"
    assert pair["transform"]["scale"] == pytest.approx(2)
    assert pair["transform"]["rotation_degrees"] == pytest.approx(90)
    assert pair["transform"]["translation_px"] == pytest.approx([100, 20])
    assert pair["heldout"]["chin-tip"]["error_normalized"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("bad_case", ["collinear", "proposed", "unknown-view", "mismatched-view"])
def test_registration_refuses_unreliable_anchor_evidence(bad_case):
    reference, candidate = _samples()
    if bad_case == "collinear":
        candidate = AnnotationSet({"render": _view("render", {
            "eye-inner-left": (10, 25), "eye-inner-right": (30, 25),
            "nose-tip": (50, 25), "chin-tip": (40, 85)})})
    if bad_case == "proposed":
        candidate = AnnotationSet({"render": _view("render", {
            "eye-inner-left": (10, 25), "eye-inner-right": (50, 25),
            "nose-tip": (10, 65)}, proposed=("nose-tip",))})
    if bad_case in ("unknown-view", "mismatched-view"):
        view = "unknown" if bad_case == "unknown-view" else "profile"
        candidate = AnnotationSet({"render": _view("render", {
            "eye-inner-left": (10, 25), "eye-inner-right": (50, 25),
            "nose-tip": (10, 65)}, view=view)})
    report = propose_registrations(reference, candidate,
                                   [RegistrationPair("ref", "render", ANCHORS)])
    pair = report["pairs"][0]
    assert pair["status"] == "unmeasured"
    assert pair["transform"] is None


def test_registration_rejects_duplicate_anchor_ids_and_unknown_images():
    reference, candidate = _samples()
    with pytest.raises(ValueError, match="duplicate"):
        propose_registrations(reference, candidate,
                              [RegistrationPair("ref", "render", (ANCHORS[0],) * 3)])
    with pytest.raises(ValueError, match="Unknown"):
        propose_registrations(reference, candidate,
                              [RegistrationPair("bad", "render", ANCHORS)])


def test_register_cli_writes_revision_with_proposal_not_likeness_measurement(tmp_path):
    def bundle(path, name, observations):
        data = {"schema_version": 1, "images": [{
            "image_id": name, "width": 120, "height": 120, "view": "front",
            "landmarks": [{"landmark_id": key, "xy_px": list(point),
                           "origin": "manual", "visibility": "visible"}
                          for key, point in observations.items()],
            "outline_px": [[0, 0], [100, 0], [100, 100], [0, 100]]
        }]}
        path.write_text(json.dumps(data))
        return path

    reference, candidate = _samples()
    ref = bundle(tmp_path / "ref.json", "ref", {a.landmark_id: a.xy_px
                                                for a in reference.images["ref"].landmarks})
    ren = bundle(tmp_path / "ren.json", "render", {a.landmark_id: a.xy_px
                                                   for a in candidate.images["render"].landmarks})
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps({"schema_version": 1, "pairs": [{
        "reference_image_id": "ref", "candidate_image_id": "render",
        "anchor_landmark_ids": list(ANCHORS)}]}))
    output = tmp_path / "registration-r01"
    assert main(["register", "--reference", str(ref), "--candidate", str(ren),
                 "--anchors", str(anchors), "--output", str(output)]) == 0
    assert json.loads((output / "registration-proposals.json").read_text())["pairs"][0]["status"] == "proposed-unverified"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["likeness_status"] == "unmeasured"
    assert not (output / "likeness-qa.json").exists()
    preview = output / "overlays" / "registration-ref--render.svg"
    assert preview.exists()
    root = ET.parse(preview).getroot()
    assert root.tag.endswith("svg")
    assert len([element for element in root if element.tag.endswith("polygon")]) == 2
    assert "UNVERIFIED" in preview.read_text()


def test_registration_preview_rejects_untrusted_output_filename(tmp_path):
    report = {"pairs": [{"status": "proposed-unverified", "reference_image_id": "../escape",
                         "candidate_image_id": "render", "reference_outline_px": [[0, 0], [10, 0], [10, 10]]}]}
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="Unsafe registration image ID"):
        write_analysis_revision(output, {}, None, None, registration=report)
    assert not output.exists()
