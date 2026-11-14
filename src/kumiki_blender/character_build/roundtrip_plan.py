"""Conservative immutable source and GLB locks for evaluated surface roundtrips."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_roundtrip_plan(
    blend: str | Path,
    glb: str | Path,
    *,
    armature: str = "CharacterRig",
    first_frame: int = 1,
    last_frame: int = 24,
    fps: int = 24,
    samples: int = 5,
    max_vertices_per_mesh: int = 150_000,
    max_p95_error_m: float = .002,
    max_max_error_m: float = .01,
) -> dict:
    """Lock the exact baked .blend and its exported GLB before comparing poses."""
    source, portable = Path(blend).resolve(), Path(glb).resolve()
    if not source.is_file() or source.suffix.lower() != ".blend":
        raise ValueError("Roundtrip needs an existing baked Blender .blend")
    if not portable.is_file() or portable.suffix.lower() != ".glb":
        raise ValueError("Roundtrip needs an existing exported .glb")
    result = {
        "schema_version": 1,
        "source_blend": str(source), "source_sha256": file_sha256(source),
        "exported_glb": str(portable), "glb_sha256": file_sha256(portable),
        "armature": armature,
        "first_frame": first_frame, "last_frame": last_frame, "fps": fps,
        "samples": samples, "max_vertices_per_mesh": max_vertices_per_mesh,
        "max_p95_error_m": max_p95_error_m, "max_max_error_m": max_max_error_m,
    }
    validate_roundtrip_plan(result)
    return result


def frame_samples(first: int, last: int, count: int) -> list[int]:
    if type(first) is not int or type(last) is not int or type(count) is not int:
        raise ValueError("Frame samples must be integers")
    if not 0 <= first < last <= 10000 or not 3 <= count <= 11 or last - first < count - 1:
        raise ValueError("Frame range must accommodate 3-11 distinct sampled frames")
    frames = [first + (last - first) * i // (count - 1) for i in range(count)]
    if len(set(frames)) != count or frames[0] != first or frames[-1] != last:
        raise ValueError("Frame samples are not strictly increasing")
    return frames


def validate_roundtrip_plan(plan: dict) -> None:
    required = {
        "schema_version", "source_blend", "source_sha256",
        "exported_glb", "glb_sha256", "armature",
        "first_frame", "last_frame", "fps", "samples",
        "max_vertices_per_mesh", "max_p95_error_m", "max_max_error_m",
    }
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("Roundtrip plan has missing or unexpected fields")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("Only roundtrip plan version 1 is supported")
    if not isinstance(plan["armature"], str) or not _NAME.fullmatch(plan["armature"]):
        raise ValueError("Roundtrip requires a safe exact armature name")
    for field, suffix in (("source_blend", ".blend"), ("exported_glb", ".glb")):
        value = plan[field]
        if (not isinstance(value, str) or not Path(value).is_absolute()
                or Path(value).suffix.lower() != suffix):
            raise ValueError(f"{field} must be an absolute {suffix} path")
    for field in ("source_sha256", "glb_sha256"):
        value = plan[field]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{field} must be lowercase SHA-256")
    if (type(plan["fps"]) is not int or not 1 <= plan["fps"] <= 120 or
            type(plan["max_vertices_per_mesh"]) is not int or
            not 10 <= plan["max_vertices_per_mesh"] <= 500_000):
        raise ValueError("FPS or maximum mesh vertex count outside supported limits")
    frame_samples(plan["first_frame"], plan["last_frame"], plan["samples"])
    for field, upper in (("max_p95_error_m", .1), ("max_max_error_m", .25)):
        value = plan[field]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f"{field} must be finite and within (0,{upper}]")
    if plan["max_p95_error_m"] > plan["max_max_error_m"]:
        raise ValueError("P95 positional error must be no greater than maximum error")
