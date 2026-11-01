"""Compare an original Sphinx sculpt with photographic/survey landmarks.

Only operator-authored point annotations are needed: the photographs and
third-party reference surfaces are never opened, copied, or packaged here.
This is a diagnostic for *image-space proportions*, not photogrammetry or a
claim that either image has an accurately calibrated camera.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LandmarkError(ValueError):
    """An annotation or its alignment is unsuitable for comparison."""


@dataclass(frozen=True)
class LandmarkView:
    view: str
    size: tuple[int, int]
    anchors: tuple[str, ...]
    landmarks: dict[str, tuple[float, float]]


def load_landmark_view(path: str | Path) -> LandmarkView:
    """Load a small, versionable landmark record without reading images."""
    with Path(path).open("r", encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict) or set(data) != {"view", "image_size", "anchors", "landmarks"}:
        raise LandmarkError("Expected view, image_size, anchors and landmarks")
    view = data["view"]
    if view not in ("frontal", "profile-left", "profile-right"):
        raise LandmarkError("View must be frontal, profile-left or profile-right")
    size = data["image_size"]
    if (not isinstance(size, list) or len(size) != 2 or
            any(type(number) is not int or not 16 <= number <= 32768 for number in size)):
        raise LandmarkError("Image size must be two positive pixel dimensions")
    points = data["landmarks"]
    if not isinstance(points, dict) or len(points) < 5 or len(points) > 256:
        raise LandmarkError("Supply between 5 and 256 named landmarks")
    parsed: dict[str, tuple[float, float]] = {}
    for name, point in points.items():
        if (not isinstance(name, str) or not name or len(name) > 80 or
                not isinstance(point, list) or len(point) != 2 or
                any(type(value) not in (float, int) or not math.isfinite(value)
                    for value in point)):
            raise LandmarkError("Landmarks require valid names and finite [x, y] pixel positions")
        if not 0 <= point[0] <= size[0] or not 0 <= point[1] <= size[1]:
            raise LandmarkError(f"Landmark {name!r} is outside the image")
        parsed[name] = (float(point[0]), float(point[1]))
    anchors = data["anchors"]
    if (not isinstance(anchors, list) or not 2 <= len(anchors) <= 8 or
            any(not isinstance(name, str) or name not in parsed for name in anchors) or
            len(set(anchors)) != len(anchors)):
        raise LandmarkError("Use 2-8 distinct shared landmarks as alignment anchors")
    return LandmarkView(view, (size[0], size[1]), tuple(anchors), parsed)


def compare_landmark_views(reference: LandmarkView, candidate: LandmarkView) -> dict[str, Any]:
    """Fit one uniform image scale+translation, then measure facial residuals.

    Avoid anisotropic scaling: it would silently conceal the exact vertical
    compression that motivated this comparison. Avoid a perspective fit as
    well: uncalibrated photographs cannot support metric 3D claims.
    """
    if not isinstance(reference, LandmarkView) or not isinstance(candidate, LandmarkView):
        raise LandmarkError("Comparison needs two loaded landmark views")
    if reference.view != candidate.view or reference.anchors != candidate.anchors:
        raise LandmarkError("View and ordered alignment anchors must match")
    if set(reference.landmarks) != set(candidate.landmarks):
        raise LandmarkError("Reference and candidate must annotate identical features")
    if len(reference.landmarks) <= len(reference.anchors):
        raise LandmarkError("Include at least one independent evaluation landmark")

    anchors = reference.anchors
    rcx = sum(reference.landmarks[name][0] for name in anchors)/len(anchors)
    rcy = sum(reference.landmarks[name][1] for name in anchors)/len(anchors)
    ccx = sum(candidate.landmarks[name][0] for name in anchors)/len(anchors)
    ccy = sum(candidate.landmarks[name][1] for name in anchors)/len(anchors)
    denom = sum((candidate.landmarks[name][0]-ccx)**2 +
                (candidate.landmarks[name][1]-ccy)**2 for name in anchors)
    if denom < 1e-6:
        raise LandmarkError("Candidate anchors are coincident")
    numerator = sum((reference.landmarks[name][0]-rcx)*(candidate.landmarks[name][0]-ccx) +
                    (reference.landmarks[name][1]-rcy)*(candidate.landmarks[name][1]-ccy)
                    for name in anchors)
    scale = numerator/denom
    if scale <= 0 or not math.isfinite(scale):
        raise LandmarkError("Image alignment has a nonpositive uniform scale")
    offset_x, offset_y = rcx-scale*ccx, rcy-scale*ccy
    spread = math.sqrt(sum((reference.landmarks[name][0]-rcx)**2 +
                           (reference.landmarks[name][1]-rcy)**2 for name in anchors)/len(anchors))
    if spread < 1:
        raise LandmarkError("Reference anchors are too close to establish a scale")
    errors = []
    for name in sorted(reference.landmarks):
        if name in anchors:
            continue
        rx, ry = reference.landmarks[name]
        cx, cy = candidate.landmarks[name]
        dx, dy = cx*scale+offset_x-rx, cy*scale+offset_y-ry
        errors.append({
            "feature": name,
            "horizontal_error_fraction": round(dx/spread, 5),
            "vertical_error_fraction": round(dy/spread, 5),
            "distance_fraction": round(math.hypot(dx, dy)/spread, 5),
        })
    errors.sort(key=lambda item: item["distance_fraction"], reverse=True)
    return {
        "view": reference.view,
        "reference_size_px": list(reference.size),
        "candidate_size_px": list(candidate.size),
        "alignment": {"anchors": list(anchors), "uniform_scale": round(scale, 7),
                      "translation_px": [round(offset_x, 4), round(offset_y, 4)]},
        "evaluation_landmarks": len(errors),
        "mean_distance_fraction": round(sum(item["distance_fraction"] for item in errors)/len(errors), 5),
        "largest_mismatches": errors,
        "measurement_type": "uncalibrated image-space proportions only",
        "survey_accuracy_claimed": False,
    }
