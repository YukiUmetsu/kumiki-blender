"""Optional 2D landmark suggestions, explicitly unverified for sculpture faces."""

from __future__ import annotations

import math
from typing import Any

from .schema import Photo


# MediaPipe FaceMesh index subset. Not anatomical depth or guaranteed sculpture identity.
_POINTS: tuple[tuple[str, int], ...] = (
    ("head-top", 10), ("forehead-center", 9),
    ("brow-left", 70), ("brow-right", 300),
    ("eye-outer-left", 33), ("eye-inner-left", 133),
    ("eye-inner-right", 362), ("eye-outer-right", 263),
    ("cheek-left", 234), ("cheek-right", 454),
    ("nose-bridge", 168), ("nose-tip", 1), ("nose-base", 4),
    ("mouth-left", 61), ("mouth-right", 291),
    ("upper-lip", 0), ("lower-lip", 17), ("chin-tip", 152),
)


def _result(status: str, detector: str, reason: str, landmarks: list[dict[str, Any]]
            | None = None) -> dict[str, Any]:
    return {"status": status, "detector": detector, "reason": reason,
            "landmarks": landmarks if landmarks is not None else []}


def propose_landmarks(photo: Photo, *, detector: str = "none") -> dict[str, Any]:
    """Return optional *unreviewed* 2D points; never infer unseen anatomy."""
    if detector not in ("none", "mediapipe"):
        raise ValueError(f"Unsupported landmark detector: {detector}")
    if detector == "none":
        return _result("unavailable", detector, "disabled")
    if photo.view == "rear":
        return _result("no-face", detector, "reviewed-rear-view")
    try:
        import mediapipe as mp  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image
    except ImportError:
        return _result("unavailable", detector, "not-installed")
    solutions = getattr(mp, "solutions", None)
    face_mesh = getattr(solutions, "face_mesh", None)
    if face_mesh is None or not hasattr(face_mesh, "FaceMesh"):
        return _result("unavailable", detector, "legacy-api-unavailable")
    try:
        with Image.open(photo.path) as image:
            rgb = np.asarray(image.convert("RGB"))
        with face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=False, min_detection_confidence=0.5) as model:
            prediction = model.process(rgb)
    except Exception as error:
        return _result("unavailable", detector, f"detector-error:{type(error).__name__}")
    faces = getattr(prediction, "multi_face_landmarks", None)
    if not faces:
        return _result("no-face", detector, "no-face-detected")
    points = getattr(faces[0], "landmark", ())
    result = []
    for semantic, index in _POINTS:
        if index >= len(points):
            continue
        p = points[index]
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in (p.x, p.y)):
            continue
        if not (0 <= p.x < 1 and 0 <= p.y < 1):
            continue
        result.append({"landmark_id": semantic,
                       "xy_px": [float(p.x * photo.width), float(p.y * photo.height)],
                       "origin": "proposed", "visibility": "uncertain"})
    if not result:
        return _result("no-face", detector, "no-valid-points")
    return _result("proposed", detector, "manual-review-required", result)
