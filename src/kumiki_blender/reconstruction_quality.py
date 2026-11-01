"""Evidence-based HOLD/PASS for photo-faithful reconstruction candidates.

This evaluates registered 2D masks and landmark measurements; it never
estimates cameras, infers missing depth, or certifies artistic likeness.
The anatomical reviewer can still reject a numerically passing asset.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


class ReconstructionEvidenceError(ValueError):
    """Evidence is absent, unsafe or not measurable on a shared pixel grid."""


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        return image.size


def _mask(path: Path) -> tuple[tuple[int, int], bytes]:
    from PIL import Image

    with Image.open(path) as image:
        grayscale = image.convert("L")
        pixels = grayscale.tobytes()
        size = grayscale.size
    if not pixels or pixels.count(0)+pixels.count(255) != len(pixels):
        raise ReconstructionEvidenceError(f"Mask must be strictly black/white: {path}")
    if pixels.count(255) == 0:
        raise ReconstructionEvidenceError(f"Mask has no foreground: {path}")
    return size, pixels


def _landmarks(path: Path, size: tuple[int, int]) -> dict[str, tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or len(data) < 3:
        raise ReconstructionEvidenceError("At least three named correspondences are required")
    result = {}
    for name, pair in data.items():
        if not isinstance(name, str) or not name or not isinstance(pair, list) or len(pair) != 2:
            raise ReconstructionEvidenceError(f"Invalid landmark: {name}")
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
               for p in pair):
            raise ReconstructionEvidenceError(f"Nonfinite landmark: {name}")
        if not (0 <= pair[0] < size[0] and 0 <= pair[1] < size[1]):
            raise ReconstructionEvidenceError(f"Landmark outside pixel grid: {name}")
        result[name] = (float(pair[0]), float(pair[1]))
    return result


def compare_registered(*, reference_mask: Path, render_mask: Path,
                       reference_landmarks: Path, render_landmarks: Path,
                       reference_image: Path | None = None,
                       render_image: Path | None = None,
                       registration: str, camera_evidence: str,
                       min_iou: float, max_landmark_error: float,
                       baseline_render_mask: Path | None = None,
                       baseline_render_landmarks: Path | None = None,
                       require_nonregression: bool = False) -> dict[str, Any]:
    """Compute matched-view silhouette IoU and diagonal-normalized point RMS."""
    if registration not in ("calibrated", "survey-orthographic"):
        raise ReconstructionEvidenceError("Unregistered cameras cannot yield match metrics")
    if not isinstance(camera_evidence, str) or len(camera_evidence.strip()) < 10:
        raise ReconstructionEvidenceError("Document the camera registration evidence")
    if not 0 <= min_iou <= 1 or not 0 <= max_landmark_error <= 1:
        raise ReconstructionEvidenceError("Thresholds must be in [0, 1]")
    if require_nonregression and not (baseline_render_mask and baseline_render_landmarks):
        raise ReconstructionEvidenceError("Nonregression requires a complete baseline")
    size, ref_pixels = _mask(reference_mask)
    other_size, render_pixels = _mask(render_mask)
    if size != other_size:
        raise ReconstructionEvidenceError("Masks must have identical pixel grids")
    if reference_image and _image_size(reference_image) != size:
        raise ReconstructionEvidenceError("Reference image and mask grids differ")
    if render_image and _image_size(render_image) != size:
        raise ReconstructionEvidenceError("Render image and mask grids differ")
    ref_points = _landmarks(reference_landmarks, size)
    render_points = _landmarks(render_landmarks, size)
    if ref_points.keys() != render_points.keys():
        raise ReconstructionEvidenceError("Landmark names must match exactly")

    diagonal = math.hypot(*size)

    def scores(pixels: bytes, positions: dict[str, tuple[float, float]]) -> tuple[float, float]:
        intersection = sum(a == 255 and b == 255 for a, b in zip(ref_pixels, pixels))
        union = sum(a == 255 or b == 255 for a, b in zip(ref_pixels, pixels))
        iou = intersection/union
        rms = math.sqrt(sum((ref_points[name][0]-positions[name][0])**2
                            +(ref_points[name][1]-positions[name][1])**2
                            for name in ref_points)/len(ref_points))/diagonal
        return iou, rms

    iou, error = scores(render_pixels, render_points)
    reasons = []
    if iou < min_iou:
        reasons.append(f"silhouette IoU {iou:.4f} < {min_iou:.4f}")
    if error > max_landmark_error:
        reasons.append(f"landmark RMS/diagonal {error:.4f} > {max_landmark_error:.4f}")
    baseline = None
    if baseline_render_mask or baseline_render_landmarks:
        if not baseline_render_mask or not baseline_render_landmarks:
            raise ReconstructionEvidenceError("A baseline needs both mask and landmarks")
        base_size, baseline_pixels = _mask(baseline_render_mask)
        if size != base_size:
            raise ReconstructionEvidenceError("Baseline mask grid does not match")
        baseline_points = _landmarks(baseline_render_landmarks, size)
        if baseline_points.keys() != ref_points.keys():
            raise ReconstructionEvidenceError("Baseline landmark names do not match")
        old_iou, old_error = scores(baseline_pixels, baseline_points)
        baseline = {"iou": old_iou, "landmark_error_ratio": old_error}
        if require_nonregression:
            if iou < old_iou-1e-6:
                reasons.append("silhouette regressed against baseline")
            if error > old_error+1e-6:
                reasons.append("landmarks regressed against baseline")
    return {"status": "HOLD" if reasons else "PASS", "iou": iou,
            "landmark_error_ratio": error, "landmark_count": len(ref_points),
            "pixel_grid": list(size), "baseline": baseline, "reasons": reasons}


def review_reconstruction(path: Path) -> dict[str, Any]:
    """Fail closed on evidence gaps, mismatched views or weak anatomy."""
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReconstructionEvidenceError("Review must be a JSON object")
    failures = []
    foundation = data.get("foundation", {})
    if not isinstance(foundation, dict):
        raise ReconstructionEvidenceError("Foundation must be an object")
    if data.get("goal") != "photo-faithful" or not data.get("subject"):
        failures.append("Missing photo-faithful goal or subject")
    if foundation.get("target_verified") is not True:
        failures.append("Foundation is not authenticated to the real target")
    if foundation.get("method") not in ("authentic-scan", "survey", "photogrammetry", "sculpt"):
        failures.append("No defensible 3D reconstruction method")
    if (not isinstance(foundation.get("sources"), list) or
            not foundation["sources"] or
            not isinstance(foundation.get("constraints"), list) or
            not foundation["constraints"] or
            not isinstance(foundation.get("scale_basis"), str) or
            not foundation["scale_basis"].strip()):
        failures.append("Missing source, scale or 3D constraint evidence")
    if data.get("unresolved_blockers"):
        failures.append("The manifest lists unresolved blockers")

    views = data.get("views", [])
    if not isinstance(views, list):
        raise ReconstructionEvidenceError("Views must be an array")
    by_id = {v.get("id"): v for v in views if isinstance(v, dict)}
    required = {"front", "profile", "three-quarter"}
    if len(by_id) != len(views) or set(by_id) != required:
        failures.append("Exactly one front, profile and three-quarter view is required")
    observed_image_hashes = {"reference": set(), "render": set()}
    measurements = {}

    def contained(relative: str) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ReconstructionEvidenceError("Evidence path must be relative")
        target = (path.parent / relative).resolve()
        if not target.is_relative_to(path.parent) or not target.is_file():
            raise ReconstructionEvidenceError(f"Missing or unsafe evidence path: {relative}")
        return target

    for view_id in sorted(required):
        view = by_id.get(view_id)
        if not view:
            continue
        if view.get("status") != "pass":
            failures.append(f"{view_id}: anatomical reviewer did not approve the view")
            continue
        thresholds = view.get("thresholds", {})
        if not isinstance(thresholds, dict):
            failures.append(f"{view_id}: invalid thresholds object")
            continue
        if len(str(thresholds.get("rationale", "")).strip()) < 10:
            failures.append(f"{view_id}: missing threshold rationale")
            continue
        try:
            photos = {}
            hashes = {}
            for field in ("reference", "render"):
                evidence = contained(view.get(field))
                digest = hashlib.sha256(evidence.read_bytes()).digest()
                if digest in observed_image_hashes[field]:
                    raise ReconstructionEvidenceError(f"Duplicate {field} image across views")
                observed_image_hashes[field].add(digest)
                hashes[field] = digest
                photos[field] = evidence
            if hashes["reference"] == hashes["render"]:
                raise ReconstructionEvidenceError("Reference image is identical to render")
            result = compare_registered(
                reference_mask=contained(view.get("reference_mask")),
                render_mask=contained(view.get("render_mask")),
                reference_landmarks=contained(view.get("reference_landmarks")),
                render_landmarks=contained(view.get("render_landmarks")),
                reference_image=photos["reference"], render_image=photos["render"],
                registration=view.get("registration", ""),
                camera_evidence=view.get("camera_evidence", ""),
                min_iou=float(thresholds["min_iou"]),
                max_landmark_error=float(thresholds["max_landmark_error_ratio"]),
            )
            measurements[view_id] = result
            if result["status"] != "PASS":
                failures.extend(f"{view_id}: {reason}" for reason in result["reasons"])
        except (KeyError, ValueError, OSError, TypeError) as error:
            failures.append(f"{view_id}: {error}")
    features = data.get("critical_features", [])
    if not isinstance(features, list) or not features:
        failures.append("Missing critical anatomical feature review")
    else:
        failures.extend(f"Critical feature on HOLD: {v.get('name', 'unnamed')}"
                        for v in features if isinstance(v, dict)
                        and v.get("status") != "pass")
        if any(not isinstance(v, dict) for v in features):
            failures.append("Malformed critical anatomical feature review")
    return {"status": "HOLD" if failures else "PASS", "subject": data.get("subject"),
            "view_metrics": measurements, "reasons": failures}
