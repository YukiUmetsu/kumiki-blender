"""Joint-local topology review plan generated from reviewed deform skeletons.

Cross-section coverage is a *diagnostic* for joint resolution, not proof of
concentric quad edge loops. A full animation-ready retopo requires visual
review and posed deformation testing in addition to these measurements.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .rig import validate_joints

_CANONICAL = ("hips", "neck", "shoulder.L", "shoulder.R",
              "elbow.L", "elbow.R", "knee.L", "knee.R", "tail0",
              "tail1", "tail2")
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")


def _point(data: dict, name: str) -> list[float]:
    coords = data[name]
    if (not isinstance(coords, list) or len(coords) != 3
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in coords)):
        raise ValueError(f"Joint {name}: expected three finite coordinates")
    return [float(v) for v in coords]


def make_topology_plan(joints_file: str | Path, *, mesh: str = "Low",
                       joints: list[str] | None = None,
                       min_coverage: float = 0.7, min_edges: int = 8,
                       strict: bool = False) -> dict:
    if not isinstance(mesh, str) or not _ID.fullmatch(mesh):
        raise ValueError("mesh must be an exact Blender object name")
    if type(min_coverage) not in (float, int) or not math.isfinite(min_coverage) or not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between zero and one")
    if type(min_edges) is not int or not 3 <= min_edges <= 2000:
        raise ValueError("min_edges must be between 3 and 2000")
    if type(strict) is not bool:
        raise ValueError("strict must be boolean")
    path = Path(joints_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_joints(data)
    points = data["joints"]
    height = data.get("height_m")
    if type(height) not in (int, float) or not math.isfinite(height) or height <= 0:
        raise ValueError("joints.json needs a positive height_m")
    if joints is None:
        requested = [name for name in _CANONICAL if name in points]
    else:
        requested = joints
    if not isinstance(requested, list) or not requested or len(requested) > 64:
        raise ValueError("Select 1–64 explicit joint names")
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate topology zone")
    zones = []
    for name in requested:
        if not isinstance(name, str) or not _ID.fullmatch(name) or name not in points:
            raise ValueError(f"Unknown or unsafe topology joint {name}")
        associated = next(
            (bone for bone in data["bones"] if bone["head"] == name),
            None,
        )
        if associated is None:
            associated = next((bone for bone in data["bones"] if bone["tail"] == name), None)
        if associated is None:
            raise ValueError(f"No bone passes through named joint {name}")
        h, t = _point(points, associated["head"]), _point(points, associated["tail"])
        direction = [t[i] - h[i] for i in range(3)]
        length = math.sqrt(sum(v * v for v in direction))
        if length < 1e-4:
            raise ValueError(f"Bone {associated['name']} is degenerate")
        center = _point(points, name)
        # Three planes through the bone joint, not automatically fitted to the
        # surrounding mesh. 20% of the bone segment defines the sampling span.
        width = max(.003 * height, min(.2 * length, .06 * height))
        zones.append({
            "id": name,
            "source_bone": associated["name"],
            "center_m": center,
            "axis": [round(v / length, 9) for v in direction],
            "half_width_m": round(width, 9),
            "search_radius_m": round(max(.1 * height, .75 * length), 9),
        })
    plan = {
        "schema_version": 1,
        "mesh": mesh,
        "joints_file": str(path.resolve()),
        "joints_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "zones": zones,
        "min_cross_section_coverage": min_coverage,
        "min_crossing_edges": min_edges,
        "strict": strict,
        "status": "joint-topology-review-plan",
    }
    validate_topology_plan(plan)
    return plan


def validate_topology_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("Invalid topology plan version")
    if not isinstance(plan.get("mesh"), str) or not _ID.fullmatch(plan["mesh"]):
        raise ValueError("Invalid mesh object name")
    if not re.fullmatch(r"[0-9a-f]{64}", plan.get("joints_sha256", "")):
        raise ValueError("Invalid joint source hash")
    zones = plan.get("zones")
    if not isinstance(zones, list) or not 1 <= len(zones) <= 64:
        raise ValueError("Topology needs 1–64 registered joint zones")
    names = set()
    for zone in zones:
        if not isinstance(zone, dict) or set(zone) != {
            "id", "source_bone", "center_m", "axis", "half_width_m", "search_radius_m"
        }:
            raise ValueError("Invalid zone contract")
        name = zone["id"]
        if not isinstance(name, str) or not _ID.fullmatch(name) or name in names:
            raise ValueError("Duplicate or unsafe topology zone")
        names.add(name)
        if not isinstance(zone["source_bone"], str) or not _ID.fullmatch(zone["source_bone"]):
            raise ValueError("Invalid topology source bone")
        _point({"center": zone["center_m"], "axis": zone["axis"]}, "center")
        axis = _point({"center": zone["center_m"], "axis": zone["axis"]}, "axis")
        if abs(math.sqrt(sum(c * c for c in axis)) - 1) > 1e-5:
            raise ValueError("Topology zone axis must be unit length")
        for key in ("half_width_m", "search_radius_m"):
            value = zone[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 1e-5 <= value <= 20:
                raise ValueError(f"Invalid {key}")
    coverage = plan.get("min_cross_section_coverage")
    if type(coverage) not in (int, float) or not math.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError("Invalid coverage threshold")
    edges = plan.get("min_crossing_edges")
    if type(edges) is not int or not 3 <= edges <= 2000:
        raise ValueError("Invalid crossing-edge threshold")
    if type(plan.get("strict")) is not bool:
        raise ValueError("strict must be boolean")


def save_topology_plan(joints_file: str | Path, output: str | Path, **settings) -> dict:
    plan = make_topology_plan(joints_file, **settings)
    folder = Path(output)
    if folder.exists():
        raise FileExistsError(f"Output revision exists: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    (folder / "topology-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def load_topology_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_topology_plan(plan)
    joints = Path(plan["joints_file"])
    if not joints.is_file() or hashlib.sha256(joints.read_bytes()).hexdigest() != plan["joints_sha256"]:
        raise ValueError("Source joints changed; regenerate topology guide plan")
    return plan
