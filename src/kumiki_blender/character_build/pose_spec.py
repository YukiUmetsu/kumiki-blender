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
        "max_p95_edge_stretch", "min_pose_displacement_m",
        "max_new_collision_pairs", "weight_zones", "render"
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
    minimum_motion = raw.get("min_pose_displacement_m")
    if minimum_motion is not None and (
            type(minimum_motion) not in (int, float) or
            not math.isfinite(minimum_motion) or not 0 < minimum_motion <= 1):
        raise ValueError("min_pose_displacement_m must be finite and in (0, 1] meter")
    maximum_new = raw.get("max_new_collision_pairs")
    if maximum_new is not None:
        if type(maximum_new) is not int or not 0 <= maximum_new <= 100000:
            raise ValueError("max_new_collision_pairs must be an integer in [0, 100000]")
        if not pairs:
            raise ValueError("max_new_collision_pairs requires explicit collision_pairs")
    zones = raw.get("weight_zones", [])
    if not isinstance(zones, list) or len(zones) > 64:
        raise ValueError("weight_zones must be at most 64 explicit semantic regions")
    region_keys = set()
    for zone in zones:
        if not isinstance(zone, dict) or set(zone) != {
                "mesh", "region_group", "allowed_bones",
                "max_outside_weight", "min_vertices"}:
            raise ValueError("weight_zones entries need mesh, region_group, "
                             "allowed_bones, max_outside_weight and min_vertices")
        for key in ("mesh", "region_group"):
            if not isinstance(zone[key], str) or not _SAFE.fullmatch(zone[key]):
                raise ValueError(f"weight_zones.{key} must be a safe named Blender object/group")
        identity = (zone["mesh"], zone["region_group"])
        if identity in region_keys:
            raise ValueError(f"Duplicate weight zone {identity}")
        region_keys.add(identity)
        bones = zone["allowed_bones"]
        if (not isinstance(bones, list) or not 1 <= len(bones) <= 32 or
                any(not isinstance(v, str) or not _SAFE.fullmatch(v) for v in bones) or
                len(set(bones)) != len(bones)):
            raise ValueError("weight_zones.allowed_bones must contain 1-32 unique bone names")
        outside = zone["max_outside_weight"]
        if (type(outside) not in (float, int) or not math.isfinite(outside)
                or not 0 <= outside <= 1):
            raise ValueError("weight_zones.max_outside_weight must be a finite fraction in [0,1]")
        minimum = zone["min_vertices"]
        if type(minimum) is not int or not 1 <= minimum <= 10_000_000:
            raise ValueError("weight_zones.min_vertices must be a positive integer")
    if type(raw.get("render", False)) is not bool:
        raise ValueError("render must be boolean")
    return raw
