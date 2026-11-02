"""Behavioral contracts for standalone statue face photo evidence."""

import hashlib
import json

import pytest
from PIL import Image

from kumiki_blender.face_analysis.inputs import (
    inspect_coverage,
    load_annotations,
    load_photo_set,
    photos_from_reference_set,
)
from kumiki_blender.reconstruction.references import ReferenceImage, ReferenceSet


def _image(path, size=(80, 96)):
    Image.new("RGB", size, (111, 104, 98)).save(path)
    return path


def _annotations(path, records):
    path.write_text(json.dumps({"schema_version": 1, "images": records}))
    return path


def test_photo_set_records_original_dimensions_and_hash_across_formats(tmp_path):
    front = _image(tmp_path / "Front.png", (80, 96))
    side = _image(tmp_path / "profile.jpg", (120, 90))
    photos = load_photo_set([front, side])
    assert [(p.image_id, p.width, p.height, p.view) for p in photos] == [
        ("front", 80, 96, "unknown"),
        ("profile", 120, 90, "unknown"),
    ]
    assert photos[0].sha256 == hashlib.sha256(front.read_bytes()).hexdigest()
    assert photos[0].provenance == "user-supplied / rights-unverified"


def test_existing_reference_dossier_can_be_adapted_without_renaming_ids(tmp_path):
    image = _image(tmp_path / "opaque-reference.png")
    source = ReferenceImage("historic-profile", image, "https://example.org/reference",
                            "left-profile", ("nose",), None, None,
                            hashlib.sha256(image.read_bytes()).hexdigest(), (80, 96), 0)
    dossier = ReferenceSet("christ-redeemer", "Monument", ("left-profile",),
                           ("nose",), (source,), "0" * 64)
    photos = photos_from_reference_set(dossier)
    assert [(p.image_id, p.view, p.width, p.height) for p in photos] == [
        ("historic-profile", "profile", 80, 96)
    ]
    assert photos[0].provenance == "validated-reference-manifest / rights-unverified"


def test_photo_set_rejects_colliding_ids_and_symlink(tmp_path):
    png = _image(tmp_path / "Face.png")
    jpg = _image(tmp_path / "face.jpg")
    with pytest.raises(ValueError, match="duplicate image_id"):
        load_photo_set([png, jpg])
    symlink = tmp_path / "shortcut.png"
    symlink.symlink_to(png)
    with pytest.raises(ValueError, match="symlink"):
        load_photo_set([symlink])


def test_photo_set_rejects_too_small_image_and_empty_set(tmp_path):
    with pytest.raises(ValueError, match="1–256"):
        load_photo_set([])
    with pytest.raises(ValueError, match="dimensions"):
        load_photo_set([_image(tmp_path / "tiny.png", (50, 100))])


def test_rear_view_with_no_face_points_is_valid_and_front_default_unknown(tmp_path):
    photos = load_photo_set([_image(tmp_path / "front.png"), _image(tmp_path / "rear.png")])
    path = _annotations(tmp_path / "annotations.json", [
        {"image_id": "front", "landmarks": [
            {"landmark_id": "chin-tip", "xy_px": [42, 80], "origin": "manual", "visibility": "visible"}
        ]},
        {"image_id": "rear", "view": "rear", "landmarks": [],
         "outline_px": [[12, 10], [65, 10], [65, 90], [12, 90]]},
    ])
    annotated = load_annotations(path, photos)
    assert annotated.images["front"].view == "unknown"
    assert annotated.images["rear"].view == "rear"
    assert annotated.images["rear"].landmarks == ()
    coverage = inspect_coverage(photos, annotated)
    assert coverage["images_total"] == 2
    assert coverage["view_counts"] == {"rear": 1, "unknown": 1}
    assert coverage["facial_landmark_views"] == ["front"]


@pytest.mark.parametrize("bad_point", [
    {"landmark_id": "chin-tip", "xy_px": [1000, 5], "origin": "manual", "visibility": "visible"},
    {"landmark_id": "chin-tip", "xy_px": [40, 5], "origin": "automatic", "visibility": "visible"},
    {"landmark_id": "chin-tip", "xy_px": [40, 5], "origin": "manual", "visibility": "missing"},
    {"landmark_id": "invented-organ", "xy_px": [40, 5], "origin": "manual", "visibility": "visible"},
])
def test_annotation_validation_rejects_misleading_coordinates_or_status(tmp_path, bad_point):
    photos = load_photo_set([_image(tmp_path / "front.png")])
    path = _annotations(tmp_path / "bad.json", [{"image_id": "front", "landmarks": [bad_point]}])
    with pytest.raises(ValueError):
        load_annotations(path, photos)


def test_annotation_validation_rejects_duplicate_semantic_point(tmp_path):
    photos = load_photo_set([_image(tmp_path / "front.png")])
    point = {"landmark_id": "chin-tip", "xy_px": [40, 50], "origin": "manual", "visibility": "visible"}
    path = _annotations(tmp_path / "bad.json", [
        {"image_id": "front", "view": "front", "landmarks": [point, point]}
    ])
    with pytest.raises(ValueError, match="duplicate landmark"):
        load_annotations(path, photos)


def test_annotation_rejects_unseen_feature_with_fabricated_position(tmp_path):
    photos = load_photo_set([_image(tmp_path / "rear.png")])
    path = _annotations(tmp_path / "bad.json", [
        {"image_id": "rear", "view": "rear", "landmarks": [
            {"landmark_id": "nose-tip", "xy_px": [40, 30], "origin": "proposed", "visibility": "visible"}
        ]}
    ])
    with pytest.raises(ValueError, match="rear view"):
        load_annotations(path, photos)
