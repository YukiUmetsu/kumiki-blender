"""Anatomical and sculptural landmark reprojection across registered photos.

An artist annotates the *same physical point* in a calibrated 2D photo and
on the 3D sculpt. Camera reprojection exposes mismatches, without claiming
that a statistically small error proves historical likeness.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .blender_reference import load_camera_plan
from .references import ReferenceError


_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


def _vec(values: Any, count: int, label: str) -> tuple[float, ...]:
    if (not isinstance(values, list) or len(values) != count
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e7 for v in values)):
        raise ReferenceError(f"{label} must be {count} finite coordinates")
    return tuple(float(value) for value in values)


def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SLUG.fullmatch(value) or len(value) > 96:
        raise ReferenceError(f"{label} must be a lowercase hyphenated slug")
    return value


def project_sculpt_point(camera: dict[str, Any], point: tuple[float, ...]
                         ) -> tuple[float, float, float] | None:
    """Blender camera local -Z points forward; output COLMAP-style top-left px."""
    position = camera["translation_m"]
    rotation = camera["rotation_3x3"]
    delta = [point[i] - position[i] for i in range(3)]
    local = [sum(rotation[j][i] * delta[j] for j in range(3)) for i in range(3)]
    depth = -local[2]
    if depth <= 1e-8:
        return None
    lens = camera["intrinsics"]
    px = lens["fx_px"]*local[0]/depth + lens["cx_px"]
    py = lens["cy_px"] - lens["fy_px"]*local[1]/depth
    return (px, py, depth)


def _load_measurements(path: Path, asset_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        if path.stat().st_size > 256*1024:
            raise ReferenceError("Landmark annotations exceed 256 KiB")
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        if isinstance(error, ReferenceError):
            raise
        raise ReferenceError(f"Cannot load landmark annotation JSON: {error}") from error
    if (not isinstance(raw, dict) or set(raw) != {
            "schema_version", "asset_id", "required_regions", "measurements"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1
            or raw["asset_id"] != asset_id):
        raise ReferenceError("Landmark annotation schema or asset_id mismatch")
    regions = raw["required_regions"]
    samples = raw["measurements"]
    if (not isinstance(regions, list) or not 1 <= len(regions) <= 32
            or not isinstance(samples, list) or not 1 <= len(samples) <= 2048):
        raise ReferenceError("Landmark annotations need 1–32 regions and 1–2048 measurements")
    regions = [_slug(region, "required_regions") for region in regions]
    if len(set(regions)) != len(regions):
        raise ReferenceError("required_regions contains duplicates")
    unique = set()
    records = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != {
                "image_id", "region", "landmark_id", "sculpt_xyz_m", "image_xy_px"}:
            raise ReferenceError(f"Invalid landmark measurement {index}")
        image_id = _slug(sample["image_id"], "image_id")
        region = _slug(sample["region"], "region")
        landmark_id = _slug(sample["landmark_id"], "landmark_id")
        if region not in regions:
            raise ReferenceError(f"Measurement region {region} not in required_regions")
        key = (image_id, region, landmark_id)
        if key in unique:
            raise ReferenceError(f"Repeated landmark observation for {key}")
        unique.add(key)
        records.append({"image_id": image_id, "region": region, "landmark_id": landmark_id,
                        "sculpt_xyz_m": _vec(sample["sculpt_xyz_m"], 3, "sculpt_xyz_m"),
                        "image_xy_px": _vec(sample["image_xy_px"], 2, "image_xy_px")})
    # Reobserved landmark IDs must represent one physical sculpt vertex/point.
    locations: dict[tuple[str, str], tuple[float, ...]] = {}
    for record in records:
        key = (record["region"], record["landmark_id"])
        previous = locations.setdefault(key, record["sculpt_xyz_m"])
        if any(abs(a-b) > 1e-6 for a, b in zip(previous, record["sculpt_xyz_m"])):
            raise ReferenceError(f"3D sculpt landmark {key} differs between camera observations")
    return regions, records


def check_landmark_reprojection(plan_file: str | Path, annotations: str | Path,
                                output: str | Path, *, max_error_px: float = 12.0) -> dict[str, Any]:
    """Report image/region pixel errors and missing multi-view constraints."""
    if type(max_error_px) not in (int, float) or not math.isfinite(max_error_px) or not 0.1 <= max_error_px <= 500:
        raise ReferenceError("max_error_px must be finite in [0.1, 500]")
    plan = load_camera_plan(plan_file)
    required_regions, observations = _load_measurements(Path(annotations), plan["asset_id"])
    cameras = {camera["image_id"]: camera for camera in plan["cameras"]}
    measurements = []
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in observations:
        name = record["image_id"]
        if name not in cameras:
            raise ReferenceError(f"Landmark uses unknown registered camera {name}")
        camera = cameras[name]
        observed = record["image_xy_px"]
        lens = camera["intrinsics"]
        if not 0 <= observed[0] <= lens["width"] or not 0 <= observed[1] <= lens["height"]:
            raise ReferenceError(f"Landmark pixel outside source image for {name}")
        result = project_sculpt_point(camera, record["sculpt_xyz_m"])
        measurement = {"image_id": name, "region": record["region"],
                       "landmark_id": record["landmark_id"], "observed_px": list(observed)}
        if result is None:
            measurement.update({"status": "behind-camera", "projected_px": None,
                                "error_px": None, "depth_m": None})
        else:
            x, y, depth = result
            error = math.hypot(x - observed[0], y - observed[1])
            measurement.update({"status": "within-tolerance" if error <= max_error_px else "mismatch",
                                "projected_px": [round(x, 4), round(y, 4)],
                                "error_px": round(error, 4), "depth_m": round(depth, 4)})
        measurements.append(measurement)
        by_region[record["region"]].append(measurement)
        by_view[name].append(measurement)

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        errors = [item["error_px"] for item in values if item["error_px"] is not None]
        return {"observations": len(values), "visible_observations": len(errors),
                "median_error_px": round(statistics.median(errors), 4) if errors else None,
                "maximum_error_px": max(errors, default=None),
                "mismatches": sum(item["status"] != "within-tolerance" for item in values)}

    region_review = {}
    for region in required_regions:
        entries = by_region[region]
        distinct_views = sorted({item["image_id"] for item in entries})
        distinct_landmarks = sorted({item["landmark_id"] for item in entries})
        region_review[region] = {**summarize(entries), "view_ids": distinct_views,
                                 "landmark_ids": distinct_landmarks,
                                 "coverage_ready": len(distinct_views) >= 2 and len(distinct_landmarks) >= 3}
    result = {
        "schema_version": 1, "asset_id": plan["asset_id"],
        "status": ("reprojection-threshold-candidate / artistic-review-required"
                   if len(by_view) >= 3 and all(item["coverage_ready"] and item["mismatches"] == 0
                                                for item in region_review.values())
                   else "review-required"),
        "max_error_px": max_error_px,
        "registered_cameras": len(cameras), "annotated_cameras": len(by_view),
        "regions": region_review,
        "views": {name: summarize(values) for name, values in sorted(by_view.items())},
        "measurements": measurements, "historical_fidelity_verified": False,
        "limitations": "Reprojection errors require correct camera calibration, true 2D-3D point identity and independent landmark annotations; sparse points and a few correct landmarks do not prove likeness, texture or unseen form.",
    }
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Landmark QA output already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    (target / "landmark-qa.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
