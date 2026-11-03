"""Optional detectors only propose 2D evidence; no invented 3D accuracy."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from kumiki_blender.face_analysis.detectors import propose_landmarks
from kumiki_blender.face_analysis.inputs import load_photo_set


def _photo(tmp_path):
    path = tmp_path / "front.png"
    Image.new("RGB", (80, 96), (100, 103, 108)).save(path)
    return load_photo_set([path])[0]


def test_detector_disabled_without_optional_imports(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    report = propose_landmarks(_photo(tmp_path))
    assert report == {"status": "unavailable", "detector": "none", "reason": "disabled", "landmarks": []}


def test_detector_missing_dependency_is_not_a_pipeline_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    report = propose_landmarks(_photo(tmp_path), detector="mediapipe")
    assert report["status"] == "unavailable"
    assert report["reason"] == "not-installed"
    assert report["landmarks"] == []


def test_rear_view_is_not_falsely_annotated_as_having_a_face(tmp_path):
    photo = replace(_photo(tmp_path), view="rear")
    result = propose_landmarks(photo, detector="mediapipe")
    assert result["status"] == "no-face"
    assert result["landmarks"] == []


def _fake_mediapipe(points):
    class FaceMesh:
        def __init__(self, **kwargs):
            assert kwargs["max_num_faces"] == 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def process(self, frame):
            assert frame.shape == (96, 80, 3)
            return SimpleNamespace(multi_face_landmarks=(
                [SimpleNamespace(landmark=points)] if points is not None else None))

    return SimpleNamespace(solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=FaceMesh)))


def test_detector_returns_no_face_if_no_landmarks(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", _fake_mediapipe(None))
    report = propose_landmarks(_photo(tmp_path), detector="mediapipe")
    assert report["status"] == "no-face"
    assert report["landmarks"] == []


def test_detector_only_emits_unverified_2d_landmark_proposals(tmp_path, monkeypatch):
    points = [SimpleNamespace(x=0.5, y=0.25, z=0.4) for _ in range(468)]
    monkeypatch.setitem(sys.modules, "mediapipe", _fake_mediapipe(points))
    report = propose_landmarks(_photo(tmp_path), detector="mediapipe")
    nose = next(item for item in report["landmarks"] if item["landmark_id"] == "nose-tip")
    assert report["status"] == "proposed"
    assert nose == {"landmark_id": "nose-tip", "xy_px": [40.0, 24.0],
                    "origin": "proposed", "visibility": "uncertain"}
    assert "z" not in nose and "confidence" not in nose


def test_unknown_detector_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        propose_landmarks(_photo(tmp_path), detector="bfm")
