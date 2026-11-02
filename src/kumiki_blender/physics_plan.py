"""Small declarative rigid-body shot plan, safe to validate without Blender.

Every shot is a separate immutable artifact. No general-purpose expressions,
paths, code evaluation, or implicit object ownership in the agent input.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


class PhysicsPlanError(ValueError):
    """Invalid agent-authored physical shot plan."""


_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,100}\Z")
_SHAPES = {"CONVEX_HULL", "BOX", "SPHERE", "CAPSULE", "CYLINDER", "MESH"}


@dataclass(frozen=True)
class PhysicsBody:
    target: str
    kind: str
    shape: str
    mass_kg: float
    friction: float
    restitution: float
    margin_m: float
    offset_m: tuple[float, float, float]


@dataclass(frozen=True)
class GroundCollider:
    height_m: float
    half_extent_m: float
    thickness_m: float
    friction: float
    restitution: float


@dataclass(frozen=True)
class PhysicsShot:
    asset_id: str
    action_id: str
    duration_frames: int
    gravity_mps2: tuple[float, float, float]
    substeps: int
    solver_iterations: int
    bodies: tuple[PhysicsBody, ...]
    ground: GroundCollider | None = None


def _object(value: Any, label: str, required: set[str], optional: set[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - (required | optional):
        raise PhysicsPlanError(f"{label} requires {sorted(required)} and only permits {sorted(optional)} extras")
    return value


def _int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PhysicsPlanError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _float(value: Any, label: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise PhysicsPlanError(f"{label} must be finite in [{minimum}, {maximum}]")
    return float(value)


def _vector(value: Any, label: str, minimum: float, maximum: float) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise PhysicsPlanError(f"{label} must be a 3-vector")
    return tuple(_float(v, f"{label}[{i}]", minimum, maximum) for i, v in enumerate(value))  # type: ignore[return-value]


def parse_physics_plan(value: Any) -> PhysicsShot:
    raw = _object(value, "plan", {"schema_version", "asset_id", "action_id", "duration_frames", "bodies"},
                  {"gravity_mps2", "substeps", "solver_iterations", "ground"})
    if raw["schema_version"] != 1 or type(raw["schema_version"]) is not int:
        raise PhysicsPlanError("Only physics schema_version 1 is supported")
    asset = raw["asset_id"]
    action = raw["action_id"]
    if not isinstance(asset, str) or not _SLUG.fullmatch(asset):
        raise PhysicsPlanError("asset_id must be a lowercase hyphenated slug")
    if not isinstance(action, str) or not _NAME.fullmatch(action):
        raise PhysicsPlanError("action_id must be an alphanumeric-safe identifier")
    duration = _int(raw["duration_frames"], "duration_frames", 1, 3600)
    gravity = _vector(raw.get("gravity_mps2", [0, 0, -9.81]), "gravity_mps2", -100, 100)
    substeps = _int(raw.get("substeps", 16), "substeps", 1, 1000)
    iterations = _int(raw.get("solver_iterations", 20), "solver_iterations", 1, 1000)
    ground = None
    if "ground" in raw:
        plane = _object(raw["ground"], "ground", set(),
                        {"height_m", "half_extent_m", "thickness_m", "friction", "restitution"})
        ground = GroundCollider(
            height_m=_float(plane.get("height_m", 0), "ground.height_m", -100, 100),
            half_extent_m=_float(plane.get("half_extent_m", 8), "ground.half_extent_m", .25, 100),
            thickness_m=_float(plane.get("thickness_m", .2), "ground.thickness_m", .01, 5),
            friction=_float(plane.get("friction", .75), "ground.friction", 0, 1),
            restitution=_float(plane.get("restitution", .1), "ground.restitution", 0, 1),
        )
    originals = raw["bodies"]
    if not isinstance(originals, list) or not 1 <= len(originals) <= 128:
        raise PhysicsPlanError("bodies must be an array of 1–128 rules")
    bodies = []
    targets = set()
    for index, raw_body in enumerate(originals):
        label = f"bodies[{index}]"
        body = _object(raw_body, label, {"target", "kind"},
                       {"shape", "mass_kg", "friction", "restitution", "margin_m", "offset_m"})
        target = body["target"]
        if (not isinstance(target, str) or not (target.startswith("name:") or target.startswith("role:"))
                or not _NAME.fullmatch(target.split(":", 1)[1])):
            raise PhysicsPlanError(f"{label}.target must be name:<object> or role:<kumiki_part>")
        if target in targets:
            raise PhysicsPlanError(f"Duplicate physics target {target!r}")
        targets.add(target)
        kind = body["kind"]
        if kind not in ("active", "passive"):
            raise PhysicsPlanError(f"{label}.kind must be active or passive")
        shape = body.get("shape", "CONVEX_HULL" if kind == "active" else "MESH")
        if shape not in _SHAPES:
            raise PhysicsPlanError(f"{label}.shape must be a Blender collision shape")
        offset = _vector(body.get("offset_m", [0, 0, 0]), f"{label}.offset_m", -20, 20)
        if kind == "passive" and any(abs(x) > 1e-9 for x in offset):
            raise PhysicsPlanError(f"{label}.offset_m on a passive collider would move the static reference")
        bodies.append(PhysicsBody(
            target=target, kind=kind, shape=shape,
            mass_kg=_float(body.get("mass_kg", 1), f"{label}.mass_kg", .001, 50_000),
            friction=_float(body.get("friction", .5), f"{label}.friction", 0, 1),
            restitution=_float(body.get("restitution", .2), f"{label}.restitution", 0, 1),
            margin_m=_float(body.get("margin_m", .001), f"{label}.margin_m", 0, .2),
            offset_m=offset,
        ))
    if not any(body.kind == "active" for body in bodies):
        raise PhysicsPlanError("A shot needs at least one active rigid body")
    return PhysicsShot(asset, action, duration, gravity, substeps, iterations, tuple(bodies), ground)


def load_physics_plan(source: str | Path) -> PhysicsShot:
    path = Path(source)
    try:
        if path.stat().st_size > 128 * 1024:
            raise PhysicsPlanError("Physics plan exceeds 128 KiB")
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, ValueError, UnicodeError) as error:
        if isinstance(error, PhysicsPlanError):
            raise
        raise PhysicsPlanError(f"Invalid physics JSON: {error}") from error
    return parse_physics_plan(value)
