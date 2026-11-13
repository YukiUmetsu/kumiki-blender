"""Compose a SHA-locked body Action and offline facial cues into one local plan.

The source .blend needs a baked skeleton Action and authored neutral mouth
shape keys, but does not need any facial keyframes. Blender applies them in a
new output revision, then exports the synchronized performance GLB.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .performance_plan import make_performance_plan
from .viseme_plan import make_viseme_plan


def make_talking_plan(blend: str | Path, cues: str | Path, clip: str,
                      start: int, end: int, *, armature: str = "CharacterRig",
                      mouth_object: str = "Mouth", fps: int = 24,
                      ramp_frames: float = 1.5,
                      time_alignment_tolerance_s: float = .05,
                      shape_keys: dict | None = None) -> dict:
    body = make_performance_plan(
        blend, clip, start, end, armature=armature, mouth_object=mouth_object,
        fps=fps, time_alignment_tolerance_s=time_alignment_tolerance_s)
    face = make_viseme_plan(
        blend, cues, mouth_object, clip, fps=fps, first_frame=start,
        ramp_frames=ramp_frames, shape_keys=shape_keys)
    last_cue_end_frame = start + face["cues"][-1]["end"] * fps
    if last_cue_end_frame > end + 1e-5:
        raise ValueError("Viseme cue duration extends beyond the baked body Action range")
    if body["source_sha256"] != face["source_blend_sha256"]:
        raise ValueError("Face and body were not planned from the same Blender source")
    payload = {
        "schema_version": 1,
        "performance": body,
        "visemes": face,
        "status": "talking-performance-plan",
        "limitations": [
            "Source Blender needs an active baked deform skeleton Action and authored neutral mouth shapes.",
            "Mouth cues are applied locally; no new facial anatomy, audio recognition or visual approval.",
            "Body and face must share the same start/end frame timeline.",
        ],
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "plan_sha256": digest}


def save_talking_plan(blend: str | Path, cues: str | Path, clip: str,
                      start: int, end: int, output: str | Path, **options) -> dict:
    plan = make_talking_plan(blend, cues, clip, start, end, **options)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"Talking performance revision exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target / "talking-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def load_talking_plan(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {
        "schema_version", "performance", "visemes", "status", "limitations", "plan_sha256"
    } or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Invalid talking performance plan")
    checksum = hashlib.sha256(json.dumps(
        {k: v for k, v in data.items() if k != "plan_sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if checksum != data["plan_sha256"]:
        raise ValueError("Talking performance plan settings changed; regenerate the plan")
    body, face = data["performance"], data["visemes"]
    if not isinstance(body, dict) or not isinstance(face, dict):
        raise ValueError("Talking performance plan needs both input contracts")
    try:
        live = make_talking_plan(
            body["source_blend"], face["source_cues"], body["clip"],
            body["start_frame"], body["end_frame"],
            armature=body["armature"], mouth_object=body["mouth_object"],
            fps=body["fps"], ramp_frames=face["ramp_frames"],
            time_alignment_tolerance_s=body["time_alignment_tolerance_s"],
            shape_keys=face["shape_keys"])
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed or missing talking performance settings") from error
    if live != data:
        raise ValueError("Talking performance source, cues or settings changed; regenerate the plan")
    return data
