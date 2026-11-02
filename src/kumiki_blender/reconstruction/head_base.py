"""Fitting existing anatomical head meshes to measured monument landmarks.

No head mesh is generated here. This module validates a locally supplied GLB,
performs a proper (non-reflecting) similarity registration, and reports the
distance remaining before an artist can claim anything about likeness.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kumiki_blender.asset_import import (
    AssetImportError, AssetProvenance, inspect_source_glb, load_asset_provenance,
)


class HeadFitError(ValueError):
    """Unsafe input or underconstrained head-base registration."""


Vec3 = tuple[float, float, float]
Matrix3 = tuple[Vec3, Vec3, Vec3]
_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


@dataclass(frozen=True)
class HeadLandmark:
    name: str
    region: str
    source_xyz_m: Vec3
    target_xyz_m: Vec3
    radius_m: float


@dataclass(frozen=True)
class HeadFitPlan:
    asset_id: str
    source_glb: Path
    provenance: AssetProvenance
    input_sha256: str
    mesh_name: str
    source_front_axis: str
    fit_mode: str
    landmarks: tuple[HeadLandmark, ...]


@dataclass(frozen=True)
class HeadFitResult:
    asset_id: str
    input_sha256: str
    rotation: Matrix3
    translation_m: Vec3
    scale: float
    rms_error_m: float
    max_error_m: float
    landmark_errors: tuple[dict[str, Any], ...]

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "asset_id": self.asset_id,
            "source_sha256": self.input_sha256,
            "status": "head-fit-candidate / artistic-review-required",
            "scale": self.scale,
            "rotation_3x3": [list(row) for row in self.rotation],
            "translation_m": list(self.translation_m),
            "rms_error_m": round(self.rms_error_m, 8),
            "max_error_m": round(self.max_error_m, 8),
            "landmarks": list(self.landmark_errors),
            "historical_fidelity_verified": False,
            "limitations": "3D landmark registration does not establish likeness, unseen geometry, facial anatomy detail, texture fidelity, or accurate sculptural weathering.",
        }


def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) > 96 or not _SLUG.fullmatch(value):
        raise HeadFitError(f"{label} must be a lowercase hyphenated slug")
    return value


def _vec3(value: Any, label: str) -> Vec3:
    if (not isinstance(value, list) or len(value) != 3
            or any(type(item) not in (int, float) or not math.isfinite(item)
                   or abs(item) > 1e7 for item in value)):
        raise HeadFitError(f"{label} requires three finite meter coordinates")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _relative_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise HeadFitError(f"{label} must be a relative local file")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise HeadFitError(f"{label} cannot escape its manifest directory")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise HeadFitError(f"{label} is unavailable or escapes the manifest directory")
    return candidate


def load_head_fit_plan(path: str | Path) -> HeadFitPlan:
    """Validate source rights, local GLB, correspondence geometry and exact JSON."""
    path = Path(path)
    try:
        if path.stat().st_size > 128 * 1024:
            raise HeadFitError("Head-fit manifest exceeds 128 KiB")
        raw = json.loads(path.read_text(encoding="utf-8"),
                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, ValueError, UnicodeError) as error:
        if isinstance(error, HeadFitError):
            raise
        raise HeadFitError(f"Cannot read head-fit manifest: {error}") from error
    required = {"schema_version", "asset_id", "source_glb", "source_provenance",
                "mesh_name", "fit_mode", "landmarks"}
    if (not isinstance(raw, dict) or set(raw) - {"source_front_axis"} != required
            or raw["schema_version"] != 1 or type(raw["schema_version"]) is not int):
        raise HeadFitError("Invalid head-fit manifest schema")
    asset_id = _slug(raw["asset_id"], "asset_id")
    mesh_name = raw["mesh_name"]
    if (not isinstance(mesh_name, str) or not 1 <= len(mesh_name) <= 128
            or mesh_name != mesh_name.strip() or any(ord(char) < 32 for char in mesh_name)):
        raise HeadFitError("mesh_name must identify one exact imported mesh")
    fit_mode = raw["fit_mode"]
    if fit_mode not in ("rigid", "local-sculpt"):
        raise HeadFitError("fit_mode must be rigid or local-sculpt")
    front_axis = raw.get("source_front_axis", "-Y")
    if front_axis not in ("-X", "+X", "-Y", "+Y"):
        raise HeadFitError("source_front_axis must be one of -X, +X, -Y, +Y")
    rows = raw["landmarks"]
    if not isinstance(rows, list) or not 4 <= len(rows) <= 48:
        raise HeadFitError("Head-fit requires 4–48 manually matched landmarks")
    landmarks = []
    names: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
                "id", "region", "source_xyz_m", "target_xyz_m", "radius_m"}:
            raise HeadFitError(f"Landmark {index} has invalid fields")
        name = _slug(row["id"], "landmark id")
        if name in names:
            raise HeadFitError(f"Landmark has duplicate id: {name}")
        names.add(name)
        radius = row["radius_m"]
        if type(radius) not in (float, int) or not math.isfinite(radius) or not 0.001 <= radius <= 1e5:
            raise HeadFitError(f"Landmark {name} radius_m must be finite and positive")
        landmarks.append(HeadLandmark(name, _slug(row["region"], "region"),
                                      _vec3(row["source_xyz_m"], "source_xyz_m"),
                                      _vec3(row["target_xyz_m"], "target_xyz_m"), float(radius)))
    source = _relative_file(path.parent, raw["source_glb"], "source_glb")
    provenance_path = _relative_file(path.parent, raw["source_provenance"], "source_provenance")
    try:
        provenance = load_asset_provenance(provenance_path)
        report = inspect_source_glb(source, provenance)
    except AssetImportError as error:
        raise HeadFitError(str(error)) from error
    return HeadFitPlan(asset_id, source, provenance, report["input_sha256"],
                       mesh_name, front_axis, fit_mode, tuple(landmarks))


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _centroid(points: list[Vec3]) -> Vec3:
    return tuple(sum(point[i] for point in points) / len(points) for i in range(3))  # type: ignore[return-value]


def _principal_quaternion(matrix: list[list[float]]) -> tuple[float, float, float, float]:
    """Largest eigenvector of symmetric Horn 4x4 matrix, via Jacobi sweeps."""
    a = [row[:] for row in matrix]
    vectors = [[float(i == j) for j in range(4)] for i in range(4)]
    for _ in range(80):
        p, q = max(itertools.combinations(range(4), 2), key=lambda ij: abs(a[ij[0]][ij[1]]))
        if abs(a[p][q]) < 1e-13:
            break
        angle = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c*c*app - 2*c*s*apq + s*s*aqq
        a[q][q] = s*s*app + 2*c*s*apq + c*c*aqq
        a[p][q] = a[q][p] = 0.0
        for k in range(4):
            if k != p and k != q:
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c*akp - s*akq
                a[k][q] = a[q][k] = s*akp + c*akq
            vkp, vkq = vectors[k][p], vectors[k][q]
            vectors[k][p] = c*vkp - s*vkq
            vectors[k][q] = s*vkp + c*vkq
    best = max(range(4), key=lambda index: a[index][index])
    quat = tuple(vectors[i][best] for i in range(4))
    length = math.sqrt(sum(value * value for value in quat))
    return tuple(value / length for value in quat)  # type: ignore[return-value]


def _rotation_from_quaternion(q: tuple[float, float, float, float]) -> Matrix3:
    w, x, y, z = q
    return ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
            (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
            (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))


def _rotated(point: Vec3, rotation: Matrix3) -> Vec3:
    return tuple(_dot(row, point) for row in rotation)  # type: ignore[return-value]


def _check_spread(points: list[Vec3], label: str) -> None:
    max_distance2 = max(_dot(_sub(a, b), _sub(a, b)) for a, b in itertools.combinations(points, 2))
    max_area2 = max(_dot(_cross(_sub(b, a), _sub(c, a)),
                         _cross(_sub(b, a), _sub(c, a)))
                    for a, b, c in itertools.combinations(points, 3))
    if max_distance2 <= 1e-14 or max_area2 / (max_distance2**2) < 1e-7:
        raise HeadFitError(f"{label} landmarks are degenerate or nearly collinear")


def fit_head_landmarks(plan: HeadFitPlan) -> HeadFitResult:
    """Least-squares proper 3D rotation + positive uniform scale + translation."""
    source = [item.source_xyz_m for item in plan.landmarks]
    target = [item.target_xyz_m for item in plan.landmarks]
    _check_spread(source, "Source")
    _check_spread(target, "Target")
    sc, tc = _centroid(source), _centroid(target)
    centered_s = [_sub(v, sc) for v in source]
    centered_t = [_sub(v, tc) for v in target]
    h = [[sum(s[i]*t[j] for s, t in zip(centered_s, centered_t))
          for j in range(3)] for i in range(3)]
    xx, xy, xz = h[0]
    yx, yy, yz = h[1]
    zx, zy, zz = h[2]
    q = _principal_quaternion([
        [xx+yy+zz, yz-zy, zx-xz, xy-yx],
        [yz-zy, xx-yy-zz, xy+yx, zx+xz],
        [zx-xz, xy+yx, -xx+yy-zz, yz+zy],
        [xy-yx, zx+xz, yz+zy, -xx-yy+zz],
    ])
    rotation = _rotation_from_quaternion(q)
    numerator = sum(_dot(t, _rotated(s, rotation)) for s, t in zip(centered_s, centered_t))
    denominator = sum(_dot(s, s) for s in centered_s)
    scale = numerator / denominator
    if not math.isfinite(scale) or not 1e-5 <= scale <= 1e5:
        raise HeadFitError("Landmark registration requires a positive, bounded scale")
    rotated_center = _rotated(sc, rotation)
    translation = tuple(tc[i] - scale*rotated_center[i] for i in range(3))
    # Detect a real mirrored 3D input when both sets contain a stable volume.
    scale_range = max(math.dist(a, b) for a, b in itertools.combinations(source, 2))
    volumes = [(_dot(_cross(_sub(b, a), _sub(c, a)), _sub(d, a)),
                _dot(_cross(_sub(tb, ta), _sub(tcv, ta)), _sub(td, ta)))
               for (a, b, c, d), (ta, tb, tcv, td) in zip(
                   itertools.combinations(source, 4), itertools.combinations(target, 4))]
    if volumes:
        original, fitted = max(volumes, key=lambda pair: abs(pair[0]))
        if abs(original) > 1e-5 * scale_range**3 and original*fitted < 0:
            raise HeadFitError("Target landmark ordering mirrors the anatomical head")
    errors = []
    for item in plan.landmarks:
        point = tuple(scale*v + translation[i] for i, v in enumerate(_rotated(item.source_xyz_m, rotation)))
        distance = math.dist(point, item.target_xyz_m)
        errors.append({"id": item.name, "region": item.region,
                       "aligned_xyz_m": [round(value, 7) for value in point],
                       "target_xyz_m": list(item.target_xyz_m),
                       "error_m": round(distance, 8)})
    distances = [math.dist(tuple(scale*v + translation[i] for i, v in enumerate(_rotated(item.source_xyz_m, rotation))),
                           item.target_xyz_m) for item in plan.landmarks]
    return HeadFitResult(plan.asset_id, plan.input_sha256, rotation, translation,
                         scale, math.sqrt(sum(d*d for d in distances)/len(distances)),
                         max(distances), tuple(errors))


def transform_point(point: Vec3, fit: HeadFitResult) -> Vec3:
    rotated = _rotated(point, fit.rotation)
    return tuple(fit.scale * rotated[i] + fit.translation_m[i] for i in range(3))  # type: ignore[return-value]


def _wendland(distance: float, radius: float) -> float:
    """Compact C2 radial weight: outside radius, source anatomy is untouched."""
    fraction = distance / radius
    if fraction >= 1.0:
        return 0.0
    return (1-fraction)**4 * (4*fraction+1)


def _solve_linear(system: list[list[float]], values: list[Vec3]) -> tuple[Vec3, ...]:
    """Solve all 3 correction coordinates together, rejecting ill-conditioning."""
    n = len(system)
    a = [row[:] + list(rhs) for row, rhs in zip(system, values)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(a[row][column]))
        if abs(a[pivot][column]) < 1e-9:
            raise HeadFitError("Landmark deformation is ill-conditioned; adjust anchor spacing/radii")
        a[column], a[pivot] = a[pivot], a[column]
        divisor = a[column][column]
        for j in range(column, n+3):
            a[column][j] /= divisor
        for row in range(n):
            if row == column:
                continue
            multiplier = a[row][column]
            for j in range(column, n+3):
                a[row][j] -= multiplier * a[column][j]
    result = tuple(tuple(row[n:]) for row in a)
    if any(not math.isfinite(value) for row in result for value in row):
        raise HeadFitError("Landmark deformation produced non-finite offsets")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class HeadDeformation:
    anchors: tuple[Vec3, ...]
    radii: tuple[float, ...]
    coefficients: tuple[Vec3, ...]

    def displace(self, point: Vec3) -> Vec3:
        return tuple(point[axis] + sum(
            _wendland(math.dist(point, anchor), radius) * coefficient[axis]
            for anchor, radius, coefficient in zip(self.anchors, self.radii, self.coefficients)
        ) for axis in range(3))  # type: ignore[return-value]


def head_deformation_field(plan: HeadFitPlan, fit: HeadFitResult) -> HeadDeformation:
    """Non-destructive, localized corrections from aligned anatomy to sculpture.

    The compact kernel interpolates each anchor exactly if the landmark system
    is well-conditioned. It never alters distant vertices. It is intentionally
    not an automatic likeness solver; the resulting key needs sculpt review.
    """
    if plan.fit_mode == "rigid":
        return HeadDeformation((), (), ())
    anchors = tuple(transform_point(item.source_xyz_m, fit) for item in plan.landmarks)
    offsets = tuple(_sub(item.target_xyz_m, aligned)
                    for item, aligned in zip(plan.landmarks, anchors))
    span = statistics.median(math.dist(a, b) for a, b in itertools.combinations(anchors, 2))
    limit = max(0.001, span * 0.45)
    if max(math.sqrt(_dot(offset, offset)) for offset in offsets) > limit:
        raise HeadFitError("Landmark correction is excessive for local sculpting; select a closer anatomical base or use rigid mode")
    radii = tuple(item.radius_m for item in plan.landmarks)
    matrix = [[_wendland(math.dist(position, anchor), radius)
               for anchor, radius in zip(anchors, radii)] for position in anchors]
    coefficients = _solve_linear(matrix, list(offsets))
    if max(math.sqrt(_dot(coefficient, coefficient)) for coefficient in coefficients) > limit * 4:
        raise HeadFitError("Landmark correction is unstable; adjust overlapping radii or anchor positions")
    return HeadDeformation(anchors, radii, coefficients)
