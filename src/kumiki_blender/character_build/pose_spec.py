"""Data-only contract for deterministic native-Blender rig pose QA.

Validation runs in ordinary Python. Blender evaluation is a separate worker.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

_SAFE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_AXIS = {"X", "Y", "Z"}


def load_pose_plan(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - {
        "schema_version", "armature", "poses", "collision_pairs",
        "max_p95_edge_stretch", "render"
    }:
        raise ValueError("Pose QA plan has unknown properties")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Only pose QA schema_version 1 is supported")
    if not isinstance(raw.get("armature"), str) or not _SAFE.fullmatch(raw["armature"]):
        raise ValueError("armature must be a named Blender armature object")
    poses = raw.get("poses")
    if not isinstance(poses, list) or not 1 <= len(poses) <= 64:
        raise ValueError("Pose QA requires 1 to 64 named poses")
    names = set()
    for pose in poses:
        if not isinstance(pose, dict) or set(pose) != {"name", "bones"}:
            raise ValueError("Each pose needs only name and bones")
        name = pose["name"]
        if not isinstance(name, str) or not _SAFE.fullmatch(name) or name in names:
            raise ValueError("Pose names must be unique and filename safe")
        names.add(name)
        bones = pose["bones"]
        if not isinstance(bones, dict) or not 1 <= len(bones) <= 32:
            raise ValueError("Every pose must rotate 1 to 32 bones")
        for bone, rotation in bones.items():
            if not isinstance(bone, str) or not _SAFE.fullmatch(bone):
                raise ValueError(f"{name}: invalid bone name")
            if not isinstance(rotation, dict) or set(rotation) != {"axis", "degrees"}:
                raise ValueError(f"{name}:{bone}: requires axis and degrees")
            if rotation["axis"] not in _AXIS:
                raise ValueError(f"{name}:{bone}: axis must be X/Y/Z")
            angle = rotation["degrees"]
            if type(angle) not in (int, float) or not math.isfinite(angle) or abs(angle) > 180:
                raise ValueError(f"{name}:{bone}: rotation outside +/-180 degrees")
    pairs = raw.get("collision_pairs", [])
    if not isinstance(pairs, list) or len(pairs) > 64:
        raise ValueError("collision_pairs must be at most 64 pairs")
    unique_pairs = set()
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(name, str) or not _SAFE.fullmatch(name) for name in pair)
                or pair[0] == pair[1]):
            raise ValueError("Collision pairs require two distinct mesh object names")
        key = tuple(sorted(pair))
        if key in unique_pairs:
            raise ValueError("Duplicate collision pair")
        unique_pairs.add(key)
    threshold = raw.get("max_p95_edge_stretch")
    if threshold is not None and (type(threshold) not in (float, int)
                                  or not math.isfinite(threshold) or not 0 <= threshold <= 10):
        raise ValueError("max_p95_edge_stretch must be a finite nonnegative ratio")
    if type(raw.get("render", False)) is not bool:
        raise ValueError("render must be boolean")
    return raw
