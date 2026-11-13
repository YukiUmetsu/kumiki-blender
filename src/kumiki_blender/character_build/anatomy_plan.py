"""Strict and immutable plans for source-anchored 3D anatomy part authoring.

The schema records whether an artist placed a region, estimated it, or verified
its landmark registration. The Blender worker does not elevate any evidence
level or claim generated geometry matches unpublished views.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_VIEW = frozenset(("front", "front_right", "front_left", "left", "right",
                   "back", "back_left", "back_right"))
_KIND = frozenset(("ocular_globe", "swept_tube", "contour_loft"))
_EVIDENCE = frozenset(("registered-verified", "artist-placed", "inferred"))
_GENERAL = {"id", "kind", "evidence", "attachment"}
_COLOR_KEYS = {"rgba", "roughness", "metallic"}
_FIELDS = {
    "ocular_globe": _GENERAL | {"center_m", "forward_m", "up_m", "radii_m",
                              "iris_fraction", "pupil_fraction", "radial_segments",
                              "latitudes", "materials"},
    "swept_tube": _GENERAL | {"control_points_m", "radii_m", "radial_segments",
                            "subdivisions", "reference_up", "material"},
    "contour_loft": _GENERAL | {"sections_m", "section_subdivisions",
                              "contour_subdivisions", "material"},
}


def _safe_id(value, name):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{name} must be a stable, safe object identifier")
    return value


def _int(value, label, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{label} must be integer between {lo} and {hi}")
    return value


def _scalar(value, label, lo, hi):
    if type(value) not in (float, int) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{label} must be finite between {lo} and {hi}")
    return float(value)


def _vector(value, label, *, n=3, limit=100.):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{label} must be {n} coordinates")
    return [_scalar(v, f"{label}.coordinate", -limit, limit) for v in value]


def _nonzero(value, label):
    v = _vector(value, label)
    if math.sqrt(sum(x*x for x in v)) < 1e-7:
        raise ValueError(f"{label} cannot be a zero vector")
    return v


def _color(mat, label):
    if not isinstance(mat, dict) or set(mat) - _COLOR_KEYS or "rgba" not in mat:
        raise ValueError(f"{label} requires RGBA and optional roughness/metallic")
    if not isinstance(mat["rgba"], list) or len(mat["rgba"]) != 4:
        raise ValueError(f"{label}.rgba must have 4 color channels")
    for color in mat["rgba"]:
        _scalar(color, f"{label}.rgba", 0., 1.)
    _scalar(mat.get("roughness", .55), f"{label}.roughness", .0, 1.)
    _scalar(mat.get("metallic", .0), f"{label}.metallic", .0, 1.)


def _evidence(value, label):
    if (not isinstance(value, dict) or set(value) != {"status", "source_views", "notes"}
            or value["status"] not in _EVIDENCE
            or not isinstance(value["source_views"], list)
            or len(value["source_views"]) > 10
            or any(view not in _VIEW for view in value["source_views"])
            or len(set(value["source_views"])) != len(value["source_views"])
            or not isinstance(value["notes"], str) or len(value["notes"]) > 500):
        raise ValueError(f"{label}: invalid evidence record")
    if value["status"] == "registered-verified" and not value["source_views"]:
        raise ValueError(f"{label}: registered verification needs explicitly named views")
    if value["status"] == "inferred" and not value["notes"].strip():
        raise ValueError(f"{label}: inferred anatomy needs provenance notes")


def _attachment(value, label):
    if not isinstance(value, dict) or value.get("mode") not in ("separate", "bone"):
        raise ValueError(f"{label}: attachment must be separate or bone")
    mode = value["mode"]
    if mode == "separate" and set(value) != {"mode"}:
        raise ValueError(f"{label}: separate attachment has no parent")
    if mode == "bone":
        if set(value) != {"mode", "armature", "bone"}:
            raise ValueError(f"{label}: bone attachment needs armature and bone names")
        _safe_id(value["armature"], f"{label}.armature")
        _safe_id(value["bone"], f"{label}.bone")


def validate_anatomy_spec(data: dict):
    if not isinstance(data, dict) or set(data) != {"schema_version", "character_id", "parts"}:
        raise ValueError("Anatomy spec must contain schema_version, character_id and parts")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Anatomy schema_version must be 1")
    _safe_id(data["character_id"], "character_id")
    parts = data["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 32:
        raise ValueError("Anatomy must specify 1–32 non-destructive parts")
    ids = set()
    estimated_vertices = 0
    for part in parts:
        if not isinstance(part, dict) or part.get("kind") not in _KIND:
            raise ValueError("Unknown anatomy kind")
        pid = _safe_id(part.get("id"), "part.id")
        if pid in ids:
            raise ValueError(f"Duplicate anatomy part {pid}")
        ids.add(pid)
        kind = part["kind"]
        if set(part) != _FIELDS[kind]:
            raise ValueError(f"{pid}: missing or unexpected {kind} properties")
        _evidence(part["evidence"], f"{pid}.evidence")
        _attachment(part["attachment"], f"{pid}.attachment")
        if kind == "ocular_globe":
            _vector(part["center_m"], f"{pid}.center")
            forward = _nonzero(part["forward_m"], f"{pid}.forward")
            up = _nonzero(part["up_m"], f"{pid}.up")
            v = math.sqrt(sum(k*k for k in forward))
            u = math.sqrt(sum(k*k for k in up))
            if abs(sum(a*b for a, b in zip(forward, up)) / (v*u)) > .97:
                raise ValueError(f"{pid}: up and forward eye axes cannot be parallel")
            radii = _vector(part["radii_m"], f"{pid}.radii", limit=2.)
            if min(radii) < .0005:
                raise ValueError(f"{pid}: eye radii must be >=0.5 mm")
            pupil = _scalar(part["pupil_fraction"], f"{pid}.pupil", .025, .8)
            iris = _scalar(part["iris_fraction"], f"{pid}.iris", .05, .94)
            if not pupil < iris:
                raise ValueError(f"{pid}: pupil must be smaller than iris")
            segments = _int(part["radial_segments"], f"{pid}.radial_segments", 16, 96)
            rings = _int(part["latitudes"], f"{pid}.latitudes", 8, 64)
            if not isinstance(part["materials"], dict) or set(part["materials"]) != {
                "sclera", "iris", "pupil"
            }:
                raise ValueError(f"{pid}: eye requires three source-selected material recipes")
            for name, material in part["materials"].items():
                _color(material, f"{pid}.{name}")
            estimated_vertices += segments * (rings + 25)
        elif kind == "swept_tube":
            points, radii = part["control_points_m"], part["radii_m"]
            if not isinstance(points, list) or not 3 <= len(points) <= 64:
                raise ValueError(f"{pid}: tube needs 3–64 source control points")
            if not isinstance(radii, list) or len(points) != len(radii):
                raise ValueError(f"{pid}: tube radii must match control points")
            prev = None
            for i, point in enumerate(points):
                xyz = _vector(point, f"{pid}.point[{i}]")
                radius = _vector(radii[i], f"{pid}.radii[{i}]", n=2, limit=1.)
                if min(radius) < .00005:
                    raise ValueError(f"{pid}: tube width/depth must be >=0.05 mm")
                if prev is not None and math.dist(prev, xyz) < .0001:
                    raise ValueError(f"{pid}: adjacent tube anchors overlap")
                prev = xyz
            _nonzero(part["reference_up"], f"{pid}.reference_up")
            radial = _int(part["radial_segments"], f"{pid}.radial_segments", 12, 64)
            subdivisions = _int(part["subdivisions"], f"{pid}.subdivisions", 2, 16)
            _color(part["material"], f"{pid}.material")
            estimated_vertices += radial * ((len(points)-1)*subdivisions+1)
        else:
            sections = part["sections_m"]
            if not isinstance(sections, list) or not 3 <= len(sections) <= 24:
                raise ValueError(f"{pid}: loft needs 3–24 authored contour stations")
            n = len(sections[0]) if isinstance(sections[0], list) else 0
            if not 6 <= n <= 64 or any(not isinstance(section, list) or len(section) != n
                                       for section in sections):
                raise ValueError(f"{pid}: loft contours must share 6–64 vertices")
            for k, section in enumerate(sections):
                coords = [_vector(point, f"{pid}.contour[{k}]") for point in section]
                for j in range(n):
                    if math.dist(coords[j], coords[(j+1)%n]) < .00002:
                        raise ValueError(f"{pid}: contour has duplicate adjacent landmark")
            axis = [tuple(sum(point[k] for point in section)/n for k in range(3))
                    for section in sections]
            if any(math.dist(axis[i], axis[i+1]) < .0001 for i in range(len(axis)-1)):
                raise ValueError(f"{pid}: loft station centers overlap")
            su = _int(part["section_subdivisions"], f"{pid}.section_subdivisions", 1, 12)
            cu = _int(part["contour_subdivisions"], f"{pid}.contour_subdivisions", 1, 6)
            _color(part["material"], f"{pid}.material")
            estimated_vertices += ((len(sections)-1)*su+1)*n*cu
    if estimated_vertices > 2_000_000:
        raise ValueError("Anatomy plan exceeds 2 million predicted vertices")
    return estimated_vertices


def _hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _assert_source(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.suffix.lower() != ".blend":
        raise ValueError("Anatomy needs a saved local Blender .blend as the source")
    return resolved


def _plan_payload(source: Path, spec_file: Path, spec: dict):
    total = validate_anatomy_spec(spec)
    return {
        "schema_version": 1,
        "source_blend": str(source),
        "source_sha256": _hash(source),
        "spec_file": str(spec_file),
        "spec_sha256": _hash(spec_file),
        "character_id": spec["character_id"],
        "parts": spec["parts"],
        "predicted_vertices_upper_bound": total,
        "status": "geometry-proposal-review-required",
        "limitations": [
            "Positions, shapes and colors are explicitly authored and are NOT inferred from reference artwork.",
            "Registered-verified is the caller's evidence assertion; geometry still needs independent source QA.",
            "Only new separate meshes are created. The source .blend, rig and finished anatomy stay untouched.",
        ],
    }


def plan_anatomy(blend: str | Path, spec_file: str | Path, output: str | Path) -> dict:
    source = _assert_source(blend)
    spec_path = Path(spec_file).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    payload = _plan_payload(source, spec_path, spec)
    folder = Path(output)
    if folder.exists():
        raise FileExistsError(f"Anatomy revision exists: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    (folder/"anatomy-plan.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_anatomy_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or type(plan.get("schema_version")) is not int:
        raise ValueError("Unsupported anatomy plan version")
    spec_file = Path(plan["spec_file"])
    source = _assert_source(plan["source_blend"])
    if _hash(spec_file) != plan["spec_sha256"] or _hash(source) != plan["source_sha256"]:
        raise ValueError("Anatomy source/spec changed: produce a new immutable plan")
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    recomputed = _plan_payload(source, spec_file, spec)
    if plan != recomputed:
        raise ValueError("Anatomy plan contents changed: regenerate from original source")
    return plan
