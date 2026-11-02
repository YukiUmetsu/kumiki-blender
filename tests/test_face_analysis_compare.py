"""Calibrated 2D photo/render comparisons must not become fake 3D claims."""

import pytest

from kumiki_blender.face_analysis.compare import (
    ViewPair,
    compare_views,
    summarize_observations,
)
from kumiki_blender.face_analysis.schema import Annotation, AnnotationSet, ViewAnnotations


OUTLINE = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))


def _view(image, view="front", point=(50, 80), *, outline=OUTLINE,
          landmark="chin-tip", origin="manual", visibility="visible"):
    return ViewAnnotations(image, view,
                           (Annotation(image, landmark, point, origin, visibility),), outline)


def _compare(ref, cand, *, alignment="camera-1", verified=True):
    return compare_views(AnnotationSet({ref.image_id: ref}),
                         AnnotationSet({cand.image_id: cand}),
                         [ViewPair(ref.image_id, cand.image_id, alignment, verified)])


def test_matching_aligned_chin_reports_exact_normalized_pixel_displacement():
    result = _compare(_view("photo", point=(50, 80)), _view("render", point=(50, 70)))
    pair = result["pairs"][0]
    assert pair["status"] == "measured"
    assert pair["landmarks"]["chin-tip"]["error_normalized"] == pytest.approx(0.1)
    assert pair["landmarks"]["chin-tip"]["delta_normalized"] == [0.0, -0.1]
    observations = summarize_observations(result)
    assert observations[0]["region"] == "jaw"
    assert "higher" in observations[0]["observation"]
    assert "image coordinates" in observations[0]["observation"]


def test_identical_silhouette_has_iou_one():
    result = _compare(_view("photo"), _view("render"))
    assert result["pairs"][0]["silhouette_iou"] == 1.0


@pytest.mark.parametrize("align,verified,ref_view,candidate_view,ref_outline", [
    ("", True, "front", "front", OUTLINE),
    ("camera-1", False, "front", "front", OUTLINE),
    ("camera-1", True, "front", "profile", OUTLINE),
    ("camera-1", True, "front", "front", None),
    ("camera-1", True, "unknown", "unknown", OUTLINE),
])
def test_unsupported_alignment_or_scale_refuses_numeric_likeness(
        align, verified, ref_view, candidate_view, ref_outline):
    result = _compare(_view("photo", view=ref_view, outline=ref_outline),
                      _view("render", view=candidate_view),
                      alignment=align, verified=verified)
    assert result["pairs"][0]["status"] == "unmeasured"
    assert result["pairs"][0]["landmarks"] == {}
    assert result["pairs"][0]["silhouette_iou"] is None


def test_only_manual_visible_landmarks_are_measured():
    result = _compare(_view("photo"), _view("render", origin="proposed", visibility="uncertain"))
    assert result["pairs"][0]["status"] == "measured"  # Outline-only evidence is valid.
    assert result["pairs"][0]["landmarks"] == {}
    assert result["regions"] == {}  # Do not mislabel silhouette agreement as face likeness.


def test_duplicate_landmark_ids_do_not_get_double_weighted():
    duplicate = Annotation("photo", "chin-tip", (50, 80), "manual", "visible")
    photo = ViewAnnotations("photo", "front", (duplicate, duplicate), OUTLINE)
    with pytest.raises(ValueError, match="duplicate landmark"):
        _compare(photo, _view("render"))


def test_two_aligned_views_aggregate_jaw_evidence_without_claiming_3d_shape():
    reference = AnnotationSet({"p1": _view("p1", "front", (50, 80)),
                               "p2": _view("p2", "profile", (50, 80))})
    candidate = AnnotationSet({"r1": _view("r1", "front", (50, 70)),
                               "r2": _view("r2", "profile", (50, 60))})
    result = compare_views(reference, candidate,
                           [ViewPair("p1", "r1", "camera-front", True),
                            ViewPair("p2", "r2", "camera-side", True)])
    assert result["regions"]["jaw"]["view_count"] == 2
    assert result["regions"]["jaw"]["mean_error_normalized"] == pytest.approx(0.15)
    assert result["three_d_fidelity_verified"] is False
