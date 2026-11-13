"""Blender-native animator control plans derived from existing reviewed joints.json.

No automatic skeleton inference and no new ML dependency. The generated plan
keeps edit bones and IK/FK targets separate from the exported deform skeleton.
Existing rig bones remain the source of truth for source geometry.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .rig import validate_joints

PROFILE_KINDS = {
    "stylized-biped-tail": "arms",
    "avian-biped-wings": "wings",
    "generic-biped": "arms",
}
_ID = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.:-]{0,95}$")


def _vector(data, label):
    if (not isinstance(data, list) or len(data) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in data)):
        raise ValueError(f"{label} must be three finite coordinates")
    return [float(v) for v in data]


def _point(joints, name):
    if name not in joints:
        raise ValueError(f"Missing joint {name}")
    return _vector(joints[name], name)


def _bone_map(data):
    validate_joints(data)
    bones = {}
    for bone in data["bones"]:
        name = bone.get("name")
        if not isinstance(name, str) or not _ID.fullmatch(name) or name in bones:
            raise ValueError(f"Duplicate or unsafe bone name: {name}")
        bones[name] = bone
    for b in bones.values():
        for key in ("head", "tail"):
            _point(data["joints"], b[key])
    return bones


def build_control_plan(joints_file: str | Path, profile: str, *,
                       armature: str = "CharacterRig", pole_angle_deg: float = 0) -> dict:
    """Build an FK wing/arm/tail + two-bone leg IK control specification."""
    if profile not in PROFILE_KINDS:
        raise ValueError(f"Unknown control profile {profile}. Available: {sorted(PROFILE_KINDS)}")
    if not _ID.fullmatch(armature):
        raise ValueError("Unsafe armature object name")
    if (type(pole_angle_deg) not in (int, float) or not math.isfinite(pole_angle_deg)
            or abs(pole_angle_deg) > 180):
        raise ValueError("pole_angle_deg must be in +/-180 degrees")
    path = Path(joints_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    bones = _bone_map(data)
    joints = data["joints"]
    height = data.get("height_m")
    if type(height) not in (int, float) or not math.isfinite(height) or height <= 0:
        raise ValueError("joints.json must include a positive height_m")
    if "root" not in bones:
        raise ValueError("The original deform rig must have a root bone")
    controls: list[dict] = []
    constraints: list[dict] = []
    control_map: dict[str, str] = {}
    for side in ("L", "R"):
        thigh = f"thigh.{side}"
        shin = f"shin.{side}"
        if thigh not in bones or shin not in bones or bones[shin].get("parent") != thigh:
            raise ValueError(f"Two-bone leg IK needs {thigh} -> {shin}")
        ankle = _point(joints, bones[shin]["tail"])
        knee = _point(joints, bones[shin]["head"])
        target = f"CTRL_foot_IK.{side}"
        pole = f"CTRL_knee_pole.{side}"
        size = max(.008, min(.05, height * .09))
        controls.extend([
            {"name": target, "kind": "ik-target", "parent": "root",
             "head": ankle, "tail": [ankle[0], ankle[1] - size, ankle[2]],
             "source_bone": shin},
            {"name": pole, "kind": "pole", "parent": "root",
             "head": [knee[0], knee[1] - max(height * .35, .05), knee[2]],
             "tail": [knee[0], knee[1] - max(height * .35, .05), knee[2] + size],
             "source_bone": shin},
        ])
        constraints.append({"type": "IK", "owner": shin, "target": target,
                            "pole": pole, "chain_count": 2, "pole_angle_deg": pole_angle_deg})
    limb = "wing" if PROFILE_KINDS[profile] == "wings" else "arm"
    for side in ("L", "R"):
        upper = f"upper_{limb}.{side}"
        fore = f"fore_{limb}.{side}"
        if upper not in bones or fore not in bones or bones[fore].get("parent") != upper:
            raise ValueError(f"{profile} requires reviewed {upper} -> {fore} chain")
        for name in (upper, fore):
            control_map[name] = f"CTRL_{name}"
    if profile == "stylized-biped-tail":
        names = sorted((n for n in bones if re.fullmatch(r"tail[1-9]\d*", n)),
                       key=lambda n: int(n.removeprefix("tail")))
        if not names or names[0] != "tail1":
            raise ValueError("stylized-biped-tail needs reviewed tail1, tail2 ... bones")
        for name in names:
            control_map[name] = f"CTRL_{name}"
    for name, cname in control_map.items():
        bone = bones[name]
        head, tail = _point(joints, bone["head"]), _point(joints, bone["tail"])
        parent = bone.get("parent")
        if parent is None:
            raise ValueError(f"FK controller {name} requires an upstream bone")
        controls.append({
            "name": cname, "kind": "fk", "parent": control_map.get(parent, parent),
            "source_bone": name, "head": head, "tail": tail,
        })
        constraints.append({"type": "COPY_ROTATION", "owner": name, "target": cname})
    plan = {
        "schema_version": 1,
        "profile": profile,
        "armature": armature,
        "source_joints_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_joints_file": str(path.resolve()),
        "controls": controls,
        "constraints": constraints,
        "status": "animator-controls-candidate",
        "limitations": [
            "Existing reviewed skeleton positions are mandatory; no inferred anatomy.",
            "Knee IK pole placement and pole angle are defaults requiring posed review.",
            "Control bones must be baked to deform bones for portable glTF animation.",
            "This does not add facial blendshapes, physically correct wing folding, or a Rigify rig.",
        ],
    }
    validate_control_plan(plan)
    return plan


def validate_control_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("Invalid control plan schema")
    if plan.get("profile") not in PROFILE_KINDS or not isinstance(plan.get("armature"), str):
        raise ValueError("Unknown control rig profile")
    if not _ID.fullmatch(plan["armature"]):
        raise ValueError("Unsafe armature name")
    if not isinstance(plan.get("source_joints_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", plan["source_joints_sha256"]
    ):
        raise ValueError("Invalid joints source hash")
    controls = plan.get("controls")
    constraints = plan.get("constraints")
    if not isinstance(controls, list) or not 2 <= len(controls) <= 64:
        raise ValueError("Control rig needs 2–64 controls")
    if not isinstance(constraints, list) or not 2 <= len(constraints) <= 64:
        raise ValueError("Control rig needs 2–64 constraints")
    names = set()
    for control in controls:
        if not isinstance(control, dict) or set(control) != {
            "name", "kind", "parent", "source_bone", "head", "tail"
        }:
            raise ValueError("Malformed control bone")
        name = control["name"]
        if not isinstance(name, str) or not _ID.fullmatch(name) or name in names:
            raise ValueError("Control bone names must be unique and safe")
        names.add(name)
        if control["kind"] not in ("ik-target", "pole", "fk"):
            raise ValueError("Unknown control kind")
        for key in ("parent", "source_bone"):
            if not isinstance(control[key], str) or not _ID.fullmatch(control[key]):
                raise ValueError("Invalid control bone parent/owner")
        h, t = _vector(control["head"], "head"), _vector(control["tail"], "tail")
        if math.dist(h, t) < 1e-4:
            raise ValueError(f"Degenerate control {name}")
    if any(c["parent"] == c["name"] for c in controls):
        raise ValueError("Self-parenting controller")
    graph = {c["name"]: c["parent"] for c in controls}
    for name in graph:
        seen, current = set(), name
        while current in graph:
            if current in seen:
                raise ValueError("Control parent cycle")
            seen.add(current)
            current = graph[current]
    active_owners = set()
    for row in constraints:
        if not isinstance(row, dict) or row.get("type") not in ("IK", "COPY_ROTATION"):
            raise ValueError("Invalid control constraint")
        owner, target = row.get("owner"), row.get("target")
        if not isinstance(owner, str) or not _ID.fullmatch(owner) or target not in names:
            raise ValueError("Invalid owner/target control")
        if owner in active_owners:
            raise ValueError("A deform bone can have only one new control constraint")
        active_owners.add(owner)
        if row["type"] == "IK":
            if (set(row) != {"type", "owner", "target", "pole", "chain_count", "pole_angle_deg"}
                    or row["pole"] not in names or row["chain_count"] != 2):
                raise ValueError("Invalid 2-bone IK constraint")
            angle = row["pole_angle_deg"]
            if type(angle) not in (int, float) or not math.isfinite(angle) or abs(angle) > 180:
                raise ValueError("Invalid pole angle")
        elif set(row) != {"type", "owner", "target"}:
            raise ValueError("Invalid FK copy rotation constraint")


def save_control_plan(joints_file: str | Path, profile: str, output: str | Path, *,
                      armature: str = "CharacterRig", pole_angle_deg: float = 0) -> dict:
    plan = build_control_plan(joints_file, profile, armature=armature,
                              pole_angle_deg=pole_angle_deg)
    folder = Path(output)
    if folder.exists():
        raise FileExistsError(f"Output revision exists: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    (folder / "controls.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def load_control_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_control_plan(plan)
    if not Path(plan["source_joints_file"]).is_file() or hashlib.sha256(
        Path(plan["source_joints_file"]).read_bytes()
    ).hexdigest() != plan["source_joints_sha256"]:
        raise ValueError("Reviewed joints source changed; regenerate controls plan")
    return plan
