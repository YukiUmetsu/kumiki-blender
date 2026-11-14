"""Reviewed, immutable pose-driven sculpt-corrective plans for character rigs.

Corrective *geometry* is authored as a separate rest-space mesh with precisely
the same vertex order/topology as its bound source mesh. No automation
synthesizes new character anatomy. Blender can use the source drawing/sculpt to
supply that correction; we bind it to actual pose rotations non-destructively.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")
_DIRECTIONS = {"positive", "negative"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_corrective_config(config: dict) -> dict:
    """Only explicit authored target parts, specific joints, and safe budgets."""
    if not isinstance(config, dict) or set(config) != {"armature", "correctives"}:
        raise ValueError("Correctives config needs armature and correctives only")
    if not isinstance(config["armature"], str) or not _NAME.fullmatch(config["armature"]):
        raise ValueError("Invalid corrective armature name")
    rows = config["correctives"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError("Specify 1-16 reviewed joint sculpt correctives")
    assigned = set()
    for item in rows:
        if not isinstance(item, dict) or set(item) != {
                "mesh", "target_object", "shape_key", "bone", "axis",
                "direction", "start_degrees", "full_degrees",
                "max_displacement_m", "min_changed_vertices", "max_changed_vertices"}:
            raise ValueError("Corrective entries must provide exact mesh, sculpt, joint, range and budgets")
        for field in ("mesh", "target_object", "shape_key", "bone"):
            if not isinstance(item[field], str) or not _NAME.fullmatch(item[field]):
                raise ValueError(f"Invalid corrective {field}")
        if item["mesh"] == item["target_object"]:
            raise ValueError("Rest-space sculpt target must be a separate mesh object")
        identity = (item["mesh"], item["shape_key"])
        if identity in assigned:
            raise ValueError(f"Duplicate corrective shape key {identity}")
        assigned.add(identity)
        if item["axis"] not in ("X", "Y", "Z"):
            raise ValueError("Corrective axis must be X, Y or Z")
        if item["direction"] not in _DIRECTIONS:
            raise ValueError("Corrective direction must be positive or negative")
        start, full, bound = (item[name] for name in (
            "start_degrees", "full_degrees", "max_displacement_m"))
        if (any(type(value) not in (int, float) or not math.isfinite(value)
                for value in (start, full, bound)) or
                not 0 <= start < full <= 170 or full - start < 1):
            raise ValueError("Corrective activation angles must be finite, 0 <= start < full <= 170, ramp >= 1 degree")
        if not 0 < bound <= .25:
            raise ValueError("Corrective max_displacement_m must be within (0, 0.25]")
        minimum, maximum = item["min_changed_vertices"], item["max_changed_vertices"]
        if (type(minimum) is not int or type(maximum) is not int or
                not 1 <= minimum <= maximum <= 100_000):
            raise ValueError("Corrective vertex budgets require 1 <= min <= max <= 100000")
    return config


def validate_corrective_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or set(plan) != {
            "schema_version", "source_blend", "source_sha256", "armature", "correctives"}:
        raise ValueError("Corrective plan must contain locked source, armature and reviewed sculpts")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("Unsupported joint corrective plan version")
    if (not isinstance(plan["source_blend"], str) or
            not Path(plan["source_blend"]).is_absolute() or
            Path(plan["source_blend"]).suffix.lower() != ".blend"):
        raise ValueError("Corrective source must be an absolute .blend path")
    if (not isinstance(plan["source_sha256"], str) or
            not re.fullmatch(r"[0-9a-f]{64}", plan["source_sha256"])):
        raise ValueError("Corrective source SHA-256 must be lowercase 64-char hex")
    validate_corrective_config({k: plan[k] for k in ("armature", "correctives")})
    return plan


def make_corrective_plan(source_blend: str | Path, config: dict) -> dict:
    source = Path(source_blend).resolve()
    if not source.is_file() or source.suffix.lower() != ".blend":
        raise ValueError("Correctives need an existing saved source .blend")
    validate_corrective_config(config)
    plan = {
        "schema_version": 1, "source_blend": str(source),
        "source_sha256": _sha256(source),
        "armature": config["armature"], "correctives": config["correctives"],
    }
    return validate_corrective_plan(plan)


def save_corrective_plan(source_blend: str | Path, config_path: str | Path,
                         output: str | Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    plan = make_corrective_plan(source_blend, config)
    dest = Path(output)
    if dest.exists():
        raise FileExistsError(f"Immutable corrective plan revision already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir()
    (dest / "joint-corrective-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def load_corrective_plan(plan_path: str | Path) -> dict:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    validate_corrective_plan(plan)
    source = Path(plan["source_blend"])
    if not source.is_file() or _sha256(source) != plan["source_sha256"]:
        raise ValueError("Original .blend changed after the corrective was planned; relock the source")
    return plan


def activation_expression(direction: str, start_degrees: float, full_degrees: float,
                          *, variable: str = "bend") -> str:
    """Blender expression, source angles converted to radians, clamped to 0..1.

    Transform-channel rotation variables are in radians. No Python driver
    globals or user-controlled expression fragments are accepted.
    """
    if direction not in _DIRECTIONS or variable != "bend":
        raise ValueError("Unsupported rotation direction or driver variable")
    if (any(type(v) not in (int, float) or not math.isfinite(v)
            for v in (start_degrees, full_degrees)) or
            not 0 <= start_degrees < full_degrees <= 170 or
            full_degrees - start_degrees < 1):
        raise ValueError("Invalid corrective activation interval")
    sign = "1" if direction == "positive" else "-1"
    start = math.radians(start_degrees)
    width = math.radians(full_degrees - start_degrees)
    return f"min(1.0,max(0.0,(({sign})*bend-{start:.12g})/{width:.12g}))"
