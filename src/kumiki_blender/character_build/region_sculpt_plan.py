"""Safe, bounded, data-only plans for region-local Blender sculpt corrections.

A sculpt plan moves existing vertices; it does not pretend to create anatomical
detail, reconstruct occluded regions, or automatically approve artistic quality.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

_OBJECT = re.compile(r"^[a-zA-Z0-9_.:-]{1,96}$")


def _vector(value, label: str) -> None:
    if (not isinstance(value, list) or len(value) != 3
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)):
        raise ValueError(f"{label} must contain 3 finite numeric coordinates")


def load_region_sculpt_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or set(plan) - {
        "schema_version", "object", "operations", "max_displacement_m",
        "expected_vertex_count", "expected_mesh_sha256", "protected_vertex_indices"
    }:
        raise ValueError("Sculpt plan contains unknown fields")
    if type(plan.get("schema_version")) is not int or plan["schema_version"] != 1:
        raise ValueError("Sculpt plan must use schema_version 1")
    if not isinstance(plan.get("object"), str) or not _OBJECT.fullmatch(plan["object"]):
        raise ValueError("Sculpt plan requires an exact Blender mesh object name")
    operations = plan.get("operations")
    if not isinstance(operations, list) or not 1 <= len(operations) <= 24:
        raise ValueError("Sculpt plan requires 1 to 24 operations")
    bound = plan.get("max_displacement_m", .05)
    if type(bound) not in (float, int) or not math.isfinite(bound) or not 0 < bound <= 1:
        raise ValueError("max_displacement_m must be in (0, 1]")
    expected = plan.get("expected_vertex_count")
    if expected is not None and (type(expected) is not int or not 3 <= expected <= 10_000_000):
        raise ValueError("expected_vertex_count must be between 3 and 10 million")
    mesh_sha = plan.get("expected_mesh_sha256")
    if mesh_sha is not None and (
            not isinstance(mesh_sha, str) or
            not re.fullmatch(r"[0-9a-f]{64}", mesh_sha)):
        raise ValueError("expected_mesh_sha256 must be a lowercase SHA-256 hex digest")
    protected = plan.get("protected_vertex_indices", [])
    if (not isinstance(protected, list) or len(protected) > 50000 or
            any(type(v) is not int or v < 0 or
                (expected is not None and v >= expected) for v in protected) or
            len(protected) != len(set(protected))):
        raise ValueError("protected_vertex_indices must be unique in-range vertex IDs")
    if protected and (mesh_sha is None or expected is None):
        raise ValueError("Protected vertex IDs require expected_mesh_sha256 and expected_vertex_count")
    total = 0.0
    for i, op in enumerate(operations):
        required = {"center_m", "radius_m", "move_m", "falloff"}
        allowed = required | {"distance_mode", "seed_vertex", "max_seed_distance_m"}
        if not isinstance(op, dict) or not required.issubset(op) or set(op) - allowed:
            raise ValueError(f"Operation {i} requires center_m, radius_m, move_m, falloff")
        _vector(op["center_m"], f"operation {i}.center_m")
        _vector(op["move_m"], f"operation {i}.move_m")
        radius = op["radius_m"]
        if type(radius) not in (float, int) or not math.isfinite(radius) or not 1e-5 <= radius <= 10:
            raise ValueError(f"Operation {i} radius_m must be in [0.00001, 10]")
        if op["falloff"] not in {"smoothstep", "linear"}:
            raise ValueError(f"Operation {i} must use smoothstep or linear falloff")
        mode = op.get("distance_mode", "euclidean")
        if mode not in {"euclidean", "geodesic"}:
            raise ValueError(f"Operation {i} has unsupported distance_mode")
        if mode == "geodesic":
            if expected is None or mesh_sha is None:
                raise ValueError("Geodesic seed IDs require expected_vertex_count and expected_mesh_sha256")
            seed = op.get("seed_vertex")
            if type(seed) is not int or not 0 <= seed < expected:
                raise ValueError(f"Operation {i} requires an in-range seed_vertex")
            tolerance = op.get("max_seed_distance_m")
            if (type(tolerance) not in (int, float) or
                    not math.isfinite(tolerance) or not 0 < tolerance <= radius):
                raise ValueError(f"Operation {i} requires max_seed_distance_m within radius")
            if seed in protected:
                raise ValueError("A geodesic seed cannot be protected")
        elif "seed_vertex" in op or "max_seed_distance_m" in op:
            raise ValueError("Geodesic seed options require distance_mode=geodesic")
        total += math.sqrt(sum(v * v for v in op["move_m"]))
    if total > bound + 1e-10:
        raise ValueError("Sculpt operation movements exceed the conservative displacement budget")
    return plan
