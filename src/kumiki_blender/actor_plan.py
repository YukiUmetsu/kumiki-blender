"""Validated actor plan: a Character Studio pack plus a staging profile.

No Blender dependency. The pack (character.json + layers/) is the same format
Kumiki's CharacterPresenter plays, so one saved rig drives both 2D and 3D output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


class ActorPlanError(ValueError):
    """Invalid or unsafe actor pack or profile."""


@dataclass(frozen=True)
class ActorPart:
    id: str
    parent: str | None
    asset: str
    variants: dict[str, str]
    pivot: tuple[float, float]
    rest: dict[str, float]
    z: float


@dataclass(frozen=True)
class PartStage:
    """How one flat layer sits in depth. Metres, toward the camera is negative."""
    layer: float | None = None
    inflate: float | None = None
    back: float = .55


@dataclass(frozen=True)
class Effector:
    id: str
    part: str
    variant: str | None
    joints: int
    start: tuple[float, float] | None
    tip: tuple[float, float] | None


@dataclass(frozen=True)
class ActorPlan:
    actor_id: str
    revision: str
    pack: Path
    height_m: float
    parts: tuple[ActorPart, ...]
    assets: dict[str, Path]
    stage: dict[str, PartStage] = field(default_factory=dict)
    effectors: tuple[Effector, ...] = ()
    grid_step: int = 4
    roughness: float = .62
    logical_sizes: dict[str, tuple[float, float]] = field(default_factory=dict)
    ground_parts: tuple[str, ...] = ()

    @property
    def by_id(self) -> dict[str, ActorPart]:
        return {part.id: part for part in self.parts}

    def asset_of(self, part: ActorPart, variant: str | None) -> str:
        """Same lookup as Kumiki partAssetKey: an unknown variant keeps the default drawing."""
        return part.variants.get(variant, part.asset) if variant else part.asset


def _point(raw: Any, name: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2 \
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in raw):
        raise ActorPlanError(f"{name} must be [x, y] pixels")
    return float(raw[0]), float(raw[1])


def _contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ActorPlanError("Pack paths must be relative")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ActorPlanError(f"Pack path leaves the pack: {relative}")
    return path


def parse_actor(pack: Path, character: Any, profile: Any) -> ActorPlan:
    if not isinstance(character, dict) or not isinstance(character.get("parts"), list) or not character["parts"]:
        raise ActorPlanError("character.json has no parts")
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ActorPlanError("Only actor profile schema_version 1 is supported")
    unknown = profile.keys() - {"schema_version", "height_m", "parts", "effectors", "grid_step", "roughness", "ground_parts"}
    if unknown:
        raise ActorPlanError(f"Actor profile has unknown fields: {sorted(unknown)}")
    height = profile.get("height_m")
    if not isinstance(height, (int, float)) or isinstance(height, bool) or not .02 <= height <= 3:
        raise ActorPlanError("height_m must be between 0.02 and 3 metres")
    assets_raw = character.get("assets") or {}
    assets: dict[str, Path] = {}
    logical: dict[str, tuple[float, float]] = {}
    for key, asset in assets_raw.items():
        if isinstance(asset, dict) and asset.get("kind") == "image":
            assets[key] = _contained(pack, asset.get("uri", ""))
            logical[key] = (float(asset["width"]), float(asset["height"]))
    parts = []
    for raw in character["parts"]:
        pivot = raw.get("pivot") or {}
        parts.append(ActorPart(str(raw["id"]), raw.get("parent"), str(raw["asset"]), dict(raw.get("variants") or {}),
                               (float(pivot.get("x", 0)), float(pivot.get("y", 0))),
                               dict(raw.get("rest") or {}), float(raw.get("z", 0))))
    ids = {part.id for part in parts}
    for part in parts:
        if part.parent is not None and part.parent not in ids:
            raise ActorPlanError(f"Part {part.id} has an unknown parent {part.parent}")
        for key in (part.asset, *part.variants.values()):
            if key not in assets:
                raise ActorPlanError(f"Part {part.id} needs an image asset, missing {key}")
    stage = {}
    for part_id, raw in (profile.get("parts") or {}).items():
        if part_id not in ids:
            raise ActorPlanError(f"Profile staged an unknown part {part_id}")
        extra = raw.keys() - {"layer", "inflate", "back"}
        if extra:
            raise ActorPlanError(f"Part stage has unknown fields: {sorted(extra)}")
        stage[part_id] = PartStage(raw.get("layer"), raw.get("inflate"), raw.get("back", .55))
    effectors = []
    for effector_id, raw in (profile.get("effectors") or {}).items():
        if raw.get("part") not in ids:
            raise ActorPlanError(f"Effector {effector_id} targets unknown part {raw.get('part')}")
        joints = raw.get("joints", 3)
        if type(joints) is not int or not 2 <= joints <= 6:
            raise ActorPlanError(f"Effector {effector_id} needs 2 to 6 joints")
        effectors.append(Effector(effector_id, raw["part"], raw.get("variant"), joints,
                                  _point(raw["start"], "start") if "start" in raw else None,
                                  _point(raw["tip"], "tip") if "tip" in raw else None))
    ground = tuple(profile.get("ground_parts") or ())
    for part_id in ground:
        if part_id not in ids:
            raise ActorPlanError(f"ground_parts names an unknown part {part_id}")
    step = profile.get("grid_step", 4)
    if type(step) is not int or not 1 <= step <= 16:
        raise ActorPlanError("grid_step must be 1 to 16")
    return ActorPlan(str(character.get("characterId")), str(character.get("revision")), pack, float(height),
                     tuple(parts), assets, stage, tuple(effectors), step, float(profile.get("roughness", .62)), logical, ground)


def load_actor(pack: Path, profile_path: Path) -> ActorPlan:
    return parse_actor(pack, json.loads((pack / "character.json").read_text()),
                       json.loads(profile_path.read_text()))
