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
    impact_material: str | None = None


@dataclass(frozen=True)
class GroundCollider:
    height_m: float
    half_extent_m: float
    thickness_m: float
    friction: float
    restitution: float


@dataclass(frozen=True)
class PhysicsConstraint:
    id: str
    kind: str
    body_a: str
    body_b: str
    pivot_m: tuple[float, float, float]
    axis: tuple[float, float, float]
    limits: tuple[float, float] | None
    disable_collisions: bool


@dataclass(frozen=True)
class PhysicsContactPair:
    """Explicitly authorized body-body proximity cue, not a Bullet manifold."""

    body_a: str
    body_b: str
    material: str
    tolerance_m: float


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
    constraints: tuple[PhysicsConstraint, ...] = ()
    contact_pairs: tuple[PhysicsContactPair, ...] = ()


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
                  {"gravity_mps2", "substeps", "solver_iterations", "ground", "constraints", "contact_pairs"})
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
                       {"shape", "mass_kg", "friction", "restitution", "margin_m", "offset_m", "impact_material"})
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
        material = body.get("impact_material")
        if material is not None and (not isinstance(material, str) or not _NAME.fullmatch(material)):
            raise PhysicsPlanError(f"{label}.impact_material must be a safe semantic label")
        bodies.append(PhysicsBody(
            target=target, kind=kind, shape=shape,
            mass_kg=_float(body.get("mass_kg", 1), f"{label}.mass_kg", .001, 50_000),
            friction=_float(body.get("friction", .5), f"{label}.friction", 0, 1),
            restitution=_float(body.get("restitution", .2), f"{label}.restitution", 0, 1),
            margin_m=_float(body.get("margin_m", .001), f"{label}.margin_m", 0, .2),
            offset_m=offset,
            impact_material=material,
        ))
    if not any(body.kind == "active" for body in bodies):
        raise PhysicsPlanError("A shot needs at least one active rigid body")
    raw_constraints = raw.get("constraints", [])
    if not isinstance(raw_constraints, list) or len(raw_constraints) > 64:
        raise PhysicsPlanError("constraints must be an array of at most 64 entries")
    constraints = []
    seen_ids = set()
    for index, raw_constraint in enumerate(raw_constraints):
        label = f"constraints[{index}]"
        item = _object(raw_constraint, label,
                       {"id", "kind", "body_a", "body_b", "pivot_m"},
                       {"axis", "limits_deg", "limits_m", "disable_collisions"})
        ident = item["id"]
        if not isinstance(ident, str) or not _NAME.fullmatch(ident) or ident in seen_ids:
            raise PhysicsPlanError(f"{label}.id must be a unique safe identifier")
        seen_ids.add(ident)
        kind = item["kind"]
        if kind not in ("fixed", "hinge", "slider"):
            raise PhysicsPlanError(f"{label}.kind must be fixed, hinge or slider")
        refs = (item["body_a"], item["body_b"])
        for ref in refs:
            if (not isinstance(ref, str) or not (ref.startswith("name:") or ref.startswith("role:"))
                    or not _NAME.fullmatch(ref.split(":", 1)[1])):
                raise PhysicsPlanError(f"{label} must reference name:<object> or role:<part>")
        if refs[0] == refs[1]:
            raise PhysicsPlanError(f"{label} cannot constrain a body to itself")
        pivot = _vector(item["pivot_m"], f"{label}.pivot_m", -100, 100)
        axis = _vector(item.get("axis", [0, 0, 1]), f"{label}.axis", -1, 1)
        axis_length = math.sqrt(sum(x*x for x in axis))
        if not .99 <= axis_length <= 1.01:
            raise PhysicsPlanError(f"{label}.axis must be a unit vector")
        if "limits_deg" in item and kind != "hinge" or "limits_m" in item and kind != "slider":
            raise PhysicsPlanError(f"{label} limits must match the hinge or slider kind")
        limit_name = "limits_deg" if kind == "hinge" else "limits_m"
        limits = None
        if limit_name in item:
            raw_limits = item[limit_name]
            if not isinstance(raw_limits, list) or len(raw_limits) != 2:
                raise PhysicsPlanError(f"{label}.{limit_name} must contain [lower, upper]")
            bound = 180 if kind == "hinge" else 20
            limits = tuple(_float(v, f"{label}.{limit_name}", -bound, bound) for v in raw_limits)
            if limits[0] > limits[1]:
                raise PhysicsPlanError(f"{label}.{limit_name} must be ordered")
        disable = item.get("disable_collisions", True)
        if type(disable) is not bool:
            raise PhysicsPlanError(f"{label}.disable_collisions must be boolean")
        constraints.append(PhysicsConstraint(ident, kind, refs[0], refs[1], pivot,
                                             tuple(v/axis_length for v in axis), limits, disable))
    raw_pairs = raw.get("contact_pairs", [])
    if not isinstance(raw_pairs, list) or len(raw_pairs) > 64:
        raise PhysicsPlanError("contact_pairs must contain at most 64 explicit pairs")
    pairs = []
    seen_pairs: set[frozenset[str]] = set()
    for index, raw_pair in enumerate(raw_pairs):
        label = f"contact_pairs[{index}]"
        item = _object(raw_pair, label, {"body_a", "body_b", "material"}, {"tolerance_m"})
        refs = (item["body_a"], item["body_b"])
        for ref in refs:
            if (not isinstance(ref, str) or not (ref.startswith("name:") or ref.startswith("role:"))
                    or not _NAME.fullmatch(ref.split(":", 1)[1])):
                raise PhysicsPlanError(f"{label} requires name:<object> or role:<part> selectors")
        key = frozenset(refs)
        if len(key) != 2 or key in seen_pairs:
            raise PhysicsPlanError(f"{label} must be a unique pair of different bodies")
        seen_pairs.add(key)
        material = item["material"]
        if not isinstance(material, str) or not _NAME.fullmatch(material):
            raise PhysicsPlanError(f"{label}.material must be a safe SFX label")
        pairs.append(PhysicsContactPair(refs[0], refs[1], material,
                                        _float(item.get("tolerance_m", .014), f"{label}.tolerance_m", .001, .04)))
    return PhysicsShot(asset, action, duration, gravity, substeps, iterations,
                       tuple(bodies), ground, tuple(constraints), tuple(pairs))


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
