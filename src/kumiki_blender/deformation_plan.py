"""Bounded declarative bake plan for topology-stable Blender vertex deformation.

No Blender dependency: agents can validate a plan before launching the heavy
headless worker. The exporter uses native glTF morph targets and animation,
not a proprietary stream or an invented rigid-body/cloth solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


class DeformationPlanError(ValueError):
    """Invalid, unsafe or unreasonably expensive deformation bake request."""


_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,100}\Z")


@dataclass(frozen=True)
class DeformationShot:
    asset_id: str
    action_id: str
    target: str
    duration_frames: int
    sample_stride: int
    max_vertices: int

    @property
    def frames(self) -> tuple[int, ...]:
        frames = list(range(0, self.duration_frames + 1, self.sample_stride))
        if frames[-1] != self.duration_frames:
            frames.append(self.duration_frames)
        return tuple(frames)


def parse_deformation_plan(raw: Any) -> DeformationShot:
    if not isinstance(raw, dict):
        raise DeformationPlanError("Deformation plan must be an object")
    required = {"schema_version", "asset_id", "action_id", "target", "duration_frames"}
    optional = {"sample_stride", "max_vertices"}
    if not required <= raw.keys() or raw.keys() - (required | optional):
        raise DeformationPlanError("Deformation plan has missing required or unknown fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise DeformationPlanError("Only deformation schema_version 1 is supported")
    asset_id, action_id, target = raw["asset_id"], raw["action_id"], raw["target"]
    if not isinstance(asset_id, str) or not _SLUG.fullmatch(asset_id):
        raise DeformationPlanError("asset_id must be a lowercase hyphenated slug")
    if not isinstance(action_id, str) or not _SAFE.fullmatch(action_id):
        raise DeformationPlanError("action_id must be an alphanumeric-safe identifier")
    if (not isinstance(target, str) or not (target.startswith("name:") or target.startswith("role:"))
            or not _SAFE.fullmatch(target.split(":", 1)[1])):
        raise DeformationPlanError("target must select one name:<mesh> or role:<kumiki_part>")

    def bounded(value: Any, name: str, lower: int, upper: int) -> int:
        if type(value) is not int or not lower <= value <= upper:
            raise DeformationPlanError(f"{name} must be an integer in [{lower}, {upper}]")
        return value

    duration = bounded(raw["duration_frames"], "duration_frames", 2, 240)
    stride = bounded(raw.get("sample_stride", 1), "sample_stride", 1, 16)
    maximum = bounded(raw.get("max_vertices", 8000), "max_vertices", 3, 12_000)
    shot = DeformationShot(asset_id, action_id, target, duration, stride, maximum)
    if len(shot.frames) - 1 > 64:
        raise DeformationPlanError("Morph target budget: at most 64 sampled deformation poses")
    if (len(shot.frames) - 1) * maximum > 450_000:
        raise DeformationPlanError("Morph vertex budget: max_vertices × sampled target count exceeds 450000")
    return shot


def load_deformation_plan(source: str | Path) -> DeformationShot:
    try:
        path = Path(source)
        if path.stat().st_size > 32 * 1024:
            raise DeformationPlanError("Deformation plan exceeds 32 KiB")
        raw = json.loads(path.read_text(encoding="utf-8"),
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        return parse_deformation_plan(raw)
    except (OSError, UnicodeError, ValueError) as error:
        if isinstance(error, DeformationPlanError):
            raise
        raise DeformationPlanError(f"Invalid deformation JSON: {error}") from error
