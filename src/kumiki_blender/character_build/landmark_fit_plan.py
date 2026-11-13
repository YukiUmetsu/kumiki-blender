"""Immutable, source-checked landmark fitting plans for detailed existing meshes."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .landmark_triangulation import load_triangulated_landmarks

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")
_EXPECTED = {"schema_version", "mesh", "expected_vertex_count", "max_displacement_m",
             "max_relative_edge_change", "protected_vertices", "pins"}


def _sha(path: Path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _number(value, label, lo, hi):
    if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{label} must be finite and in [{lo}, {hi}]")
    return float(value)


def validate_fit_spec(data):
    if not isinstance(data, dict) or set(data) != _EXPECTED or data.get("schema_version") != 1 or type(data["schema_version"]) is not int:
        raise ValueError("Landmark fit spec must include only the seven required schema properties")
    if not isinstance(data["mesh"], str) or not _NAME.fullmatch(data["mesh"]):
        raise ValueError("Invalid target mesh object name")
    count = data["expected_vertex_count"]
    if type(count) is not int or not 4 <= count <= 300_000:
        raise ValueError("Expected mesh vertex count must be in [4, 300000]")
    _number(data["max_displacement_m"], "max_displacement_m", .000001, .25)
    _number(data["max_relative_edge_change"], "max_relative_edge_change", 0, .8)
    protected = data["protected_vertices"]
    if (not isinstance(protected, list) or len(protected) > count
            or any(type(v) is not int or v < 0 or v >= count for v in protected)
            or len(protected) != len(set(protected))):
        raise ValueError("Protected vertices must be distinct indices within the source mesh")
    pins = data["pins"]
    if not isinstance(pins, list) or not 1 <= len(pins) <= 24:
        raise ValueError("Select 1–24 landmark constraints")
    names, indices = set(), set()
    for pin in pins:
        if not isinstance(pin, dict) or set(pin) != {"id", "source_vertex", "radius_m"}:
            raise ValueError("Each landmark constraint must specify id, source_vertex and radius_m")
        id, index = pin["id"], pin["source_vertex"]
        if not isinstance(id, str) or not _NAME.fullmatch(id) or id in names:
            raise ValueError("Duplicate or unsafe source landmark ID")
        if type(index) is not int or index < 0 or index >= count or index in indices or index in protected:
            raise ValueError("Landmark vertex index must be distinct, valid and unprotected")
        _number(pin["radius_m"], f"{id}.radius_m", .0001, 2)
        names.add(id)
        indices.add(index)
    return names


def _payload(blend: Path, landmarks: Path, spec_file: Path):
    if not blend.is_file() or blend.suffix.lower() != ".blend":
        raise ValueError("Landmark fitting requires a saved source Blender .blend")
    report = load_triangulated_landmarks(landmarks)
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    names = validate_fit_spec(spec)
    lookup = {row["id"]: row for row in report["landmarks"]}
    if not names <= set(lookup):
        raise ValueError(f"Missing source-triangulated landmarks: {sorted(names-set(lookup))}")
    pin_records = []
    for pin in spec["pins"]:
        landmark = lookup[pin["id"]]
        pin_records.append({
            **pin, "target_world_m": landmark["position_m"],
            "source_views": landmark["views"],
            "source_reprojection_error_px": landmark["max_reprojection_error_px"],
            "camera_condition_pivot_ratio": landmark["camera_condition_pivot_ratio"],
        })
    return {
        "schema_version": 1,
        "source_blend": str(blend), "source_sha256": _sha(blend),
        "landmark_report": str(landmarks), "landmark_report_sha256": _sha(landmarks),
        "fit_spec": str(spec_file), "fit_spec_sha256": _sha(spec_file),
        "character_id": report["character_id"], "mesh": spec["mesh"],
        "expected_vertex_count": spec["expected_vertex_count"],
        "max_displacement_m": spec["max_displacement_m"],
        "max_relative_edge_change": spec["max_relative_edge_change"],
        "protected_vertices": spec["protected_vertices"],
        "pins": pin_records,
        "status": "reviewed-geodesic-candidate-plan",
        "limitations": [
            "Source landmark coordinates rely on manually registered/annotated original views.",
            "Exact source vertex indices are explicit correspondence, not automatic feature matching.",
            "Only pre-rig meshes without shape keys or deforming modifiers may be adjusted.",
            "Mesh and image source fidelity require independent visual review after fitting.",
        ],
    }


def make_landmark_fit_plan(blend: str | Path, landmarks: str | Path,
                           spec: str | Path):
    return _payload(Path(blend).resolve(), Path(landmarks).resolve(), Path(spec).resolve())


def save_landmark_fit_plan(blend: str | Path, landmarks: str | Path,
                           spec: str | Path, output: str | Path):
    payload = make_landmark_fit_plan(blend, landmarks, spec)
    folder = Path(output)
    if folder.exists():
        raise FileExistsError(f"Landmark fit plan revision already exists: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    (folder / "landmark-fit-plan.json").write_text(
        json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    return payload


def load_landmark_fit_plan(path: str | Path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(plan, dict) or type(plan.get("schema_version")) is not int
            or plan["schema_version"] != 1):
        raise ValueError("Invalid landmark fit plan version")
    rebuilt = _payload(Path(plan["source_blend"]), Path(plan["landmark_report"]),
                       Path(plan["fit_spec"]))
    if rebuilt != plan:
        raise ValueError("Landmark fit source, plan or registered view evidence changed: regenerate")
    return plan
