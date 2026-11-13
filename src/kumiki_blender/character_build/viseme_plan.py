"""Offline, source-locked Rhubarb-style viseme cue plans for Blender shape keys.

No speech recognition runs here. The caller supplies timestamps from a local
lip-sync tool or authored cue list. Existing mouth anatomy and keys remain the
artist's source of truth; this pipeline creates only animation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

_CODES = frozenset("ABCDEFGHX")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_cues(path: str | Path) -> list[dict]:
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("mouthCues"), list):
        raise ValueError("Cue JSON needs Rhubarb-style mouthCues array")
    cues = source["mouthCues"]
    if not 1 <= len(cues) <= 5000:
        raise ValueError("Expected 1–5000 mouth cues")
    normalized = []
    last_end = 0.0
    for i, cue in enumerate(cues):
        if not isinstance(cue, dict) or set(cue) != {"start", "end", "value"}:
            raise ValueError(f"Cue {i} must contain only start/end/value")
        start, end, value = cue["start"], cue["end"], cue["value"]
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start or end > 300):
            raise ValueError(f"Cue {i}: invalid start/end timestamps")
        if start < last_end - 1e-5:
            raise ValueError(f"Cue {i}: cue intervals must not overlap or be out of order")
        if value not in _CODES or not isinstance(value, str) or len(value) != 1:
            raise ValueError(f"Cue {i}: invalid Rhubarb mouth code")
        normalized.append({"start": round(float(start), 6),
                           "end": round(float(end), 6), "value": value})
        last_end = float(end)
    return normalized


def _map_keys(keys: dict | None, values: set[str]) -> dict:
    mapping = {v: f"viseme_{v}" for v in sorted(values) if v != "X"} if keys is None else keys
    if not isinstance(mapping, dict):
        raise ValueError("shape_keys must be a cue-to-shape-key object")
    if not values.difference({"X"}) <= set(mapping):
        raise ValueError("Every non-silence cue needs a shape-key mapping")
    if any(code not in _CODES - {"X"} or not isinstance(name, str) or not _NAME.fullmatch(name)
           for code, name in mapping.items()):
        raise ValueError("Invalid shape-key mapping")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Distinct cues must not share one shape key")
    return mapping


def make_viseme_plan(blend: str | Path, cues_file: str | Path, mouth_object: str,
                     clip: str, *, fps: int = 24, first_frame: int = 1,
                     ramp_frames: float = 1.5, shape_keys: dict | None = None) -> dict:
    blend, cues_file = Path(blend).resolve(), Path(cues_file).resolve()
    if not blend.is_file() or blend.suffix.lower() != ".blend":
        raise ValueError("Source must be a saved Blender .blend file")
    if not cues_file.is_file():
        raise FileNotFoundError(f"No local cue file: {cues_file}")
    if not isinstance(mouth_object, str) or not _NAME.fullmatch(mouth_object):
        raise ValueError("mouth_object must be an exact Blender object name")
    if not isinstance(clip, str) or not _NAME.fullmatch(clip):
        raise ValueError("clip must be a filename-safe name")
    if type(fps) is not int or not 8 <= fps <= 120:
        raise ValueError("fps must be an integer between 8 and 120")
    if type(first_frame) is not int or not 0 <= first_frame <= 10000:
        raise ValueError("first_frame outside supported range")
    if (type(ramp_frames) not in (float, int) or not math.isfinite(ramp_frames)
            or not 0.1 <= ramp_frames <= 8):
        raise ValueError("ramp_frames must be in [0.1, 8]")
    cues = load_cues(cues_file)
    if first_frame + cues[-1]["end"] * fps > 12000:
        raise ValueError("Cue timeline would exceed 12000 Blender frames")
    if not any(cue["value"] != "X" for cue in cues):
        raise ValueError("Viseme track needs at least one non-silence cue")
    mapping = _map_keys(shape_keys, {c["value"] for c in cues})
    return {
        "schema_version": 1,
        "source_blend": str(blend), "source_blend_sha256": _digest(blend),
        "source_cues": str(cues_file), "source_cues_sha256": _digest(cues_file),
        "mouth_object": mouth_object, "clip": clip, "fps": fps,
        "first_frame": first_frame, "ramp_frames": ramp_frames,
        "shape_keys": mapping, "cues": cues,
        "status": "facial-animation-candidate",
    }


def save_viseme_plan(blend: str | Path, cues_file: str | Path, mouth_object: str,
                     clip: str, output: str | Path, **options) -> dict:
    plan = make_viseme_plan(blend, cues_file, mouth_object, clip, **options)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"Facial plan revision exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target / "viseme-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def load_viseme_plan(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Invalid viseme plan version")
    required = {"schema_version", "source_blend", "source_blend_sha256", "source_cues",
                "source_cues_sha256", "mouth_object", "clip", "fps", "first_frame",
                "ramp_frames", "shape_keys", "cues", "status"}
    if set(data) != required:
        raise ValueError("Unexpected fields in the viseme plan")
    source, cues = Path(data["source_blend"]), Path(data["source_cues"])
    if not source.is_file() or _digest(source) != data["source_blend_sha256"]:
        raise ValueError("Source Blender file changed; regenerate viseme plan")
    if not cues.is_file() or _digest(cues) != data["source_cues_sha256"]:
        raise ValueError("Source audio cues changed; regenerate viseme plan")
    live = make_viseme_plan(source, cues, data["mouth_object"], data["clip"],
                            fps=data["fps"], first_frame=data["first_frame"],
                            ramp_frames=data["ramp_frames"],
                            shape_keys=data["shape_keys"])
    if data != live:
        raise ValueError("Cached viseme plan differs from its source cues and settings")
    return data
