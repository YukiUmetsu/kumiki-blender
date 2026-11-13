"""Validated and source-locked pre-skin local joint-loop refinement plan.

This is a *topology density aid* for an approved quad retopo cage, not a
replacement for artist-directed edge-flow sculpting or an auto-rigger.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .topology_zones import load_topology_plan


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_joint_refine_plan(plan: dict) -> None:
    keys = {"schema_version", "source_blend", "source_sha256", "topology_plan",
            "topology_plan_sha256", "mesh", "zones", "cuts",
            "max_added_vertices", "axial_alignment_min"}
    if not isinstance(plan, dict) or set(plan) != keys:
        raise ValueError("Joint refine plan has missing or unexpected fields")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("Joint refine schema_version must be 1")
    for key in ("source_blend", "topology_plan"):
        if not isinstance(plan[key], str) or not Path(plan[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute existing file path")
    if not plan["source_blend"].endswith(".blend"):
        raise ValueError("Source file must be Blender .blend")
    for key in ("source_sha256", "topology_plan_sha256"):
        val = plan[key]
        if (not isinstance(val, str) or len(val) != 64 or
                any(char not in "0123456789abcdef" for char in val)):
            raise ValueError(f"{key} must be a SHA-256 checksum")
    if not isinstance(plan["mesh"], str) or not plan["mesh"]:
        raise ValueError("Mesh object name must be non-empty")
    zones = plan["zones"]
    if not isinstance(zones, list) or not 1 <= len(zones) <= 16 or any(
            not isinstance(z, str) or not z for z in zones
    ) or len(set(zones)) != len(zones):
        raise ValueError("Select 1–16 unique reviewed joint names")
    if type(plan["cuts"]) is not int or not 1 <= plan["cuts"] <= 4:
        raise ValueError("Joint edge loop cuts must be from 1 to 4")
    if type(plan["max_added_vertices"]) is not int or not 16 <= plan["max_added_vertices"] <= 100_000:
        raise ValueError("max_added_vertices must be between 16 and 100000")
    x = plan["axial_alignment_min"]
    if type(x) not in (int, float) or not .3 <= x <= .99:
        raise ValueError("axial_alignment_min must be in [0.3, 0.99]")


def make_joint_refine_plan(source_blend: str | Path, topology_plan: str | Path,
                           zones: list[str], *, cuts: int = 2,
                           max_added_vertices: int = 25_000,
                           axial_alignment_min: float = .6) -> dict:
    source = Path(source_blend).resolve()
    topo = Path(topology_plan).resolve()
    if not source.is_file() or source.suffix.lower() != ".blend":
        raise ValueError("Source must be an existing editable .blend")
    registered = load_topology_plan(topo)
    lookup = {z["id"] for z in registered["zones"]}
    if not isinstance(zones, list) or not zones or any(z not in lookup for z in zones):
        raise ValueError("Every selected joint must appear in the reviewed topology plan")
    plan = {
        "schema_version": 1,
        "source_blend": str(source), "source_sha256": _sha(source),
        "topology_plan": str(topo), "topology_plan_sha256": _sha(topo),
        "mesh": registered["mesh"], "zones": zones, "cuts": cuts,
        "max_added_vertices": max_added_vertices,
        "axial_alignment_min": axial_alignment_min,
    }
    validate_joint_refine_plan(plan)
    return plan


def save_joint_refine_plan(source_blend: str | Path, topology_plan: str | Path,
                           zones: list[str], output: str | Path, **kwargs) -> dict:
    plan = make_joint_refine_plan(source_blend, topology_plan, zones, **kwargs)
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"Joint refine revision already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    (root / "joint-refine-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def load_joint_refine_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_joint_refine_plan(plan)
    # This is cheap enough for local AI; stale registered joints require
    # an entirely new geometry candidate, never silent acceptance.
    rebuilt = make_joint_refine_plan(
        plan["source_blend"], plan["topology_plan"], plan["zones"],
        cuts=plan["cuts"], max_added_vertices=plan["max_added_vertices"],
        axial_alignment_min=plan["axial_alignment_min"])
    if rebuilt != plan:
        raise ValueError("Source Blender/joints/topology plan changed; create a new revision")
    return plan
