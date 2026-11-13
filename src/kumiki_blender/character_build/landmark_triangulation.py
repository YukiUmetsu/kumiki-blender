"""Triangulate source drawing landmarks using the *existing* registered cameras.

Every observation must refer to a real 1600x1200 aligned source drawing.
The solver never recenters landmarks, changes cameras, drops inconsistent
views, or invents depth from a single or antipodal projection. No ML needed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from PIL import Image

NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
FREE = re.compile(r"^az(-?\d+(?:\.\d+)?)_el(-?\d+(?:\.\d+)?)$")
AZIMUTH = {
    "front": 0., "front_left": 45., "left": 90., "back_left": 135.,
    "back": 180., "back_right": -135., "right": -90., "front_right": -45.,
}
CANVAS = (1600, 1200)
PIXELS_PER_FIGURE = 1000
BASELINE = 1100
CENTER_X = 800


def _identifier(name, label):
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"{label}: invalid stable identifier")
    return name


def _value(number, label, minimum, maximum):
    if (type(number) not in (float, int) or not math.isfinite(number)
            or not minimum <= number <= maximum):
        raise ValueError(f"{label}: expected a finite value in [{minimum}, {maximum}]")
    return float(number)


def _axes(view):
    if view in AZIMUTH:
        azimuth, elevation = AZIMUTH[view], 0.
    else:
        if not isinstance(view, str):
            raise ValueError("Unknown source camera")
        match = FREE.fullmatch(view)
        if not match or not -89 <= float(match[2]) <= 89:
            raise ValueError(f"Unknown source camera {view!r}")
        azimuth, elevation = float(match[1]), float(match[2])
    a, e = math.radians(azimuth), math.radians(elevation)
    toward = (math.sin(a)*math.cos(e), -math.cos(a)*math.cos(e), math.sin(e))
    right = (math.cos(a), math.sin(a), 0.)
    up = (toward[1]*right[2]-toward[2]*right[1],
          toward[2]*right[0]-toward[0]*right[2],
          toward[0]*right[1]-toward[1]*right[0])
    return right, up


def project(point, view, height_m):
    """Same pixel convention as character_build.views.Camera.project."""
    right, up = _axes(view)
    mpp = height_m / PIXELS_PER_FIGURE
    u = sum(a*b for a, b in zip(point, right))
    elevation = sum(a*b for a, b in zip(point, up)) - (height_m / 2) * up[2]
    return [CENTER_X + u/mpp, BASELINE - PIXELS_PER_FIGURE/2 - elevation/mpp]


def _linear3(matrix, values):
    """Gaussian pivoted 3x3 solve and a conservative observability metric."""
    rows = [[float(x) for x in row] + [float(y)] for row, y in zip(matrix, values)]
    pivot_values = []
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(rows[row][col]))
        magnitude = abs(rows[pivot][col])
        if magnitude < 1e-7:
            raise ValueError("Source cameras cannot triangulate depth: rank-deficient observations")
        pivot_values.append(magnitude)
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for row in range(col+1, 3):
            factor = rows[row][col] / rows[col][col]
            for j in range(col, 4):
                rows[row][j] -= factor * rows[col][j]
    strength = min(pivot_values) / max(pivot_values)
    if strength < .02:
        raise ValueError("Source cameras have an ill-conditioned baseline; add an independent view")
    result = [0., 0., 0.]
    for i in range(2, -1, -1):
        result[i] = (rows[i][3] - sum(rows[i][j] * result[j]
                       for j in range(i+1, 3))) / rows[i][i]
    if any(not math.isfinite(v) or abs(v) > 100 for v in result):
        raise ValueError("Unstable or impossible 3D source landmark coordinates")
    return result, round(strength, 6)


def _one_landmark(item: dict, views: dict, height_m: float) -> dict:
    if not isinstance(item, dict) or set(item) != {
        "id", "observations", "max_reprojection_error_px"
    }:
        raise ValueError("Landmark requires id, observations, and reprojection threshold")
    lid = _identifier(item["id"], "landmark id")
    limit = _value(item["max_reprojection_error_px"],
                   f"{lid}.max_reprojection_error_px", .05, 100)
    observations = item["observations"]
    if not isinstance(observations, list) or not 2 <= len(observations) <= 12:
        raise ValueError(f"{lid}: need 2–12 independently registered camera observations")
    camera_names = set()
    matrix, rhs = [[0.]*3 for _ in range(3)], [0.]*3
    mpp = height_m / PIXELS_PER_FIGURE
    for obs in observations:
        if not isinstance(obs, dict) or set(obs) != {"view", "xy_px", "confidence"}:
            raise ValueError(f"{lid}: invalid 2D source observation record")
        name = obs["view"]
        if name not in views or name in camera_names:
            raise ValueError(f"{lid}: unknown or duplicate registered camera {name}")
        camera_names.add(name)
        xy = obs["xy_px"]
        if not isinstance(xy, list) or len(xy) != 2:
            raise ValueError(f"{lid}: xy_px must be two full-frame pixels")
        x = _value(xy[0], f"{lid}.pixel_x", 0, CANVAS[0])
        y = _value(xy[1], f"{lid}.pixel_y", 0, CANVAS[1])
        weight = _value(obs["confidence"], f"{lid}.confidence", .1, 1.)
        right, up = _axes(name)
        targets = ((x-CENTER_X)*mpp,
                   (BASELINE-PIXELS_PER_FIGURE/2-y)*mpp + height_m/2*up[2])
        for axis, target in ((right, targets[0]), (up, targets[1])):
            for i in range(3):
                rhs[i] += weight * axis[i] * target
                for j in range(3):
                    matrix[i][j] += weight * axis[i] * axis[j]
    point, strength = _linear3(matrix, rhs)
    reprojections = []
    for obs in observations:
        predicted = project(point, obs["view"], height_m)
        error = math.dist(predicted, obs["xy_px"])
        reprojections.append({
            "view": obs["view"], "expected_xy_px": obs["xy_px"],
            "projected_xy_px": [round(v, 6) for v in predicted],
            "reprojection_error_px": round(error, 6),
        })
    worst = max(r["reprojection_error_px"] for r in reprojections)
    if worst > limit:
        raise ValueError(f"{lid}: inconsistent source views, worst reprojection "
                         f"{worst:.3f}px exceeds {limit:.3f}px")
    return {
        "id": lid, "position_m": [round(v, 9) for v in point],
        "views": [r["view"] for r in reprojections],
        "camera_condition_pivot_ratio": strength,
        "max_reprojection_error_px": worst,
        "reprojections": reprojections,
        "status": "triangulated-candidate-source-registration-review-required",
    }


def _safe_image(root: Path, file: str):
    if not isinstance(file, str) or not file or "\\" in file:
        raise ValueError("Source reference path must be local and relative")
    path = Path(file)
    if path.is_absolute() or any(p in (".", "..") for p in path.parts):
        raise ValueError(f"Unsafe source reference path: {file!r}")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source reference escapes spec directory: {file}")
    cur = root
    for token in path.parts:
        cur = cur / token
        if cur.is_symlink():
            raise ValueError(f"Linked source references are not accepted: {file}")
    if not target.is_file() or target.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise FileNotFoundError(f"Missing supported source drawing: {file}")
    with Image.open(target) as image:
        if image.size != CANVAS:
            raise ValueError(f"Reference view {file} is not aligned to 1600x1200")
    return target


def _checksum(file: Path):
    sha = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _solve(spec_file: Path):
    data = json.loads(spec_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {
        "schema_version", "character_id", "height_m", "registered_views", "landmarks"
    }:
        raise ValueError("Invalid source-landmark specification schema")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Source landmark spec schema_version must be 1")
    _identifier(data["character_id"], "character_id")
    height = _value(data["height_m"], "height_m", .01, 100.)
    views = data["registered_views"]
    if not isinstance(views, dict) or not 2 <= len(views) <= 12:
        raise ValueError("Need 2–12 individually registered source drawings")
    provenance = {}
    for name, detail in views.items():
        _axes(name)
        if not isinstance(detail, dict) or set(detail) != {"file", "registration"}:
            raise ValueError(f"View {name} needs file and registration provenance")
        if detail["registration"] not in ("original-aligned", "artist-registered"):
            raise ValueError(f"Unverified camera registration for {name}")
        file = _safe_image(spec_file.parent, detail["file"])
        provenance[name] = {
            "file": detail["file"], "sha256": _checksum(file),
            "registration": detail["registration"],
        }
    landmarks = data["landmarks"]
    if not isinstance(landmarks, list) or not 1 <= len(landmarks) <= 256:
        raise ValueError("Landmarks must list 1–256 distinct anatomical points")
    results, seen = [], set()
    for landmark in landmarks:
        result = _one_landmark(landmark, provenance, height)
        if result["id"] in seen:
            raise ValueError(f"Duplicate source landmark ID {result['id']}")
        seen.add(result["id"])
        results.append(result)
    return {
        "schema_version": 1,
        "character_id": data["character_id"], "height_m": height,
        "source_spec": str(spec_file),
        "source_spec_sha256": _checksum(spec_file),
        "registered_views": provenance,
        "landmarks": results,
        "points_m": {item["id"]: item["position_m"] for item in results},
        "status": "triangulated-candidate-source-registration-review-required",
        "limitations": [
            "Original art cameras and 1600x1200 figure alignment are caller-registered assertions.",
            "2D points are manually or independently annotated; no facial feature recognition is assumed.",
            "Two independent views solve an orthographic point but cannot prove an occluded anatomical surface.",
            "Poor baselines and inconsistent views fail rather than silently altering inputs.",
        ],
    }


def triangulate_landmarks(spec: str | Path, output: str | Path):
    source = Path(spec).resolve()
    report = _solve(source)
    dest = Path(output)
    if dest.exists():
        raise FileExistsError(f"Landmark reconstruction revision exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir()
    (dest / "landmarks.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def load_triangulated_landmarks(path: str | Path):
    source = Path(path)
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Invalid triangulated landmark receipt")
    rebuilt = _solve(Path(record["source_spec"]))
    if rebuilt != record:
        raise ValueError("Registered sources, observations or landmark data changed")
    return record
