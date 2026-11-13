"""Immutable source-locked plans for simultaneous body-and-face performance clips.

No Blender, ML, external libraries or audio service needed in this module.
Both animations must already be authored and keyed in one saved .blend.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

_SAFE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_performance_plan(plan: dict) -> None:
    required = {
        "schema_version", "source_blend", "source_sha256", "armature",
        "mouth_object", "clip", "start_frame", "end_frame", "fps",
        "time_alignment_tolerance_s",
    }
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("Performance plan requires only the documented fields")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("Unsupported performance plan version")
    for key in ("armature", "mouth_object", "clip"):
        if not isinstance(plan[key], str) or not _SAFE.fullmatch(plan[key]):
            raise ValueError(f"Invalid performance {key}")
    if plan["armature"] == plan["mouth_object"]:
        raise ValueError("Face mesh and armature must have distinct object names")
    if not isinstance(plan["source_blend"], str) or not plan["source_blend"].endswith(".blend"):
        raise ValueError("Source must be a saved Blender file")
    if not isinstance(plan["source_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", plan["source_sha256"]):
        raise ValueError("Expected source SHA-256")
    for key in ("start_frame", "end_frame", "fps"):
        if type(plan[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    start, end, fps = plan["start_frame"], plan["end_frame"], plan["fps"]
    if not 0 <= start < end <= 10000 or end - start > 1200:
        raise ValueError("Performance clip must contain 2–1201 bounded frames")
    if not 8 <= fps <= 120:
        raise ValueError("fps must be between 8 and 120")
    tolerance = plan["time_alignment_tolerance_s"]
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or not 0 < tolerance <= .2:
        raise ValueError("time_alignment_tolerance_s must be in (0, 0.2]")


def make_performance_plan(blend: str | Path, clip: str, start: int, end: int,
                          *, armature: str = "CharacterRig", mouth_object: str = "Mouth",
                          fps: int = 24, time_alignment_tolerance_s: float = .05) -> dict:
    path = Path(blend).resolve()
    if not path.is_file() or path.suffix.lower() != ".blend":
        raise ValueError("A saved .blend file containing body and face Actions is required")
    plan = {
        "schema_version": 1, "source_blend": str(path),
        "source_sha256": _hash_file(path), "armature": armature,
        "mouth_object": mouth_object, "clip": clip,
        "start_frame": start, "end_frame": end, "fps": fps,
        "time_alignment_tolerance_s": time_alignment_tolerance_s,
    }
    validate_performance_plan(plan)
    return plan


def save_performance_plan(blend: str | Path, clip: str, start: int, end: int,
                          output: str | Path, **options) -> dict:
    plan = make_performance_plan(blend, clip, start, end, **options)
    folder = Path(output)
    if folder.exists():
        raise FileExistsError(f"Performance revision exists: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    (folder / "performance-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def load_performance_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_performance_plan(plan)
    source = Path(plan["source_blend"])
    if not source.is_file() or _hash_file(source) != plan["source_sha256"]:
        raise ValueError("Source Blender file changed: create a new performance plan")
    return plan
