"""Small reviewed-joints contract without NumPy, SciPy or Blender imports.

Native Blender's embedded Python may not contain SciPy. Planning FK/IK and
topology sections only needs the coordinates and bone graph, not mesh IO.
"""
from __future__ import annotations

import math
import re

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")


def validate_joints(data: dict) -> None:
    """Reject malformed, unsafe or disconnected skeletons before generating workers."""
    if not isinstance(data, dict):
        raise ValueError("Joint specification must be an object")
    joints, bones = data.get("joints"), data.get("bones")
    if not isinstance(joints, dict) or not 1 <= len(joints) <= 1024:
        raise ValueError("joints.json must contain 1–1024 joint coordinates")
    if not isinstance(bones, list) or not 1 <= len(bones) <= 512:
        raise ValueError("joints.json must contain 1–512 deform bones")
    for name, coordinates in joints.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"Invalid joint name {name!r}")
        if (not isinstance(coordinates, list) or len(coordinates) != 3 or
                any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1000
                    for v in coordinates)):
            raise ValueError(f"Joint {name}: expected finite Blender-space XYZ coordinates")
    seen = set()
    for entry in bones:
        if not isinstance(entry, dict) or set(entry) - {"name", "head", "tail", "parent"}:
            raise ValueError("Invalid bone object")
        name = entry.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in seen:
            raise ValueError(f"Invalid or duplicate bone name {name!r}")
        head, tail = entry.get("head"), entry.get("tail")
        if head not in joints or tail not in joints:
            raise ValueError(f"Bone {name} references missing joints")
        if math.dist(joints[head], joints[tail]) < 1e-4:
            raise ValueError(f"Bone {name} has zero length")
        if entry.get("parent") not in seen | {None}:
            raise ValueError(f"Bone {name} needs its parent listed before it")
        seen.add(name)
