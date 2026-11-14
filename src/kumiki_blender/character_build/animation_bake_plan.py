"""Strict, source-locked animation bake contracts for Blender control rigs."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def validate_bake_plan(data: dict) -> None:
    base_fields = {
        "schema_version", "source_blend", "source_sha256", "armature",
        "clip", "start_frame", "end_frame", "step", "fps",
        "max_position_error_m", "max_rotation_error_deg",
    }
    if (not isinstance(data, dict) or
            set(data) not in (base_fields, base_fields | {"bake_correctives"})):
        raise ValueError("Animation bake plan needs explicit source and frame bounds")
    if type(data.get("bake_correctives", False)) is not bool:
        raise ValueError("bake_correctives must be boolean")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported animation bake plan version")
    for label in ("armature", "clip"):
        if not isinstance(data[label], str) or not _NAME.fullmatch(data[label]):
            raise ValueError(f"Invalid animation {label}")
    if not isinstance(data["source_blend"], str) or not data["source_blend"].endswith(".blend"):
        raise ValueError("source_blend must be a .blend path")
    if not isinstance(data["source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", data["source_sha256"]):
        raise ValueError("Invalid source SHA-256")
    for label in ("start_frame", "end_frame", "step", "fps"):
        if type(data[label]) is not int:
            raise ValueError(f"{label} must be an integer")
    start, end, step, fps = (data[k] for k in ("start_frame", "end_frame", "step", "fps"))
    if not 0 <= start < end <= 10000 or end - start > 1200:
        raise ValueError("Animation must be 2–1201 frames long within 0–10000")
    if not 1 <= step <= 8 or not 1 <= fps <= 120:
        raise ValueError("step or fps outside supported range")
    for name, maximum in (("max_position_error_m", .1), ("max_rotation_error_deg", 90)):
        value = data[name]
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"{name} must be finite and between zero and {maximum}")


def make_bake_plan(blend: str | Path, clip: str, start: int, end: int, *,
                   armature: str = "CharacterRig", step: int = 1, fps: int = 24,
                   max_position_error_m: float = .002,
                   max_rotation_error_deg: float = 2.0,
                   bake_correctives: bool = False) -> dict:
    path = Path(blend).resolve()
    if not path.is_file() or path.suffix.lower() != ".blend":
        raise ValueError("A saved .blend with the animated control rig is required")
    data = {
        "schema_version": 1, "source_blend": str(path),
        "source_sha256": _digest(path), "armature": armature,
        "clip": clip, "start_frame": start, "end_frame": end, "step": step,
        "fps": fps, "max_position_error_m": max_position_error_m,
        "max_rotation_error_deg": max_rotation_error_deg,
        "bake_correctives": bake_correctives,
    }
    validate_bake_plan(data)
    return data


def save_bake_plan(blend: str | Path, clip: str, start: int, end: int,
                   output: str | Path, **settings) -> dict:
    plan = make_bake_plan(blend, clip, start, end, **settings)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"Animation plan revision exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target / "bake-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def load_bake_plan(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_bake_plan(data)
    source = Path(data["source_blend"])
    if not source.is_file() or _digest(source) != data["source_sha256"]:
        raise ValueError("Animated Blender source changed; generate a new bake plan")
    return data
