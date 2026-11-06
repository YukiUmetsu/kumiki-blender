"""Validated plan for one rendered shot. No Blender dependency.

A shot joins a set, a cast and a per-frame bake. Kumiki resolves every frame and
this package only renders them. Bakes use Kumiki's right-handed, Y-up metres;
`to_blender_position` and `to_blender_quaternion` are the only axis conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

RENDER_PROFILES = {
    "draft": {"samples": 16, "resolution_scale": "2"},
    "review": {"samples": 64, "resolution_scale": "1"},
    "final": {"samples": 128, "resolution_scale": "1"},
}
_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}\Z", re.IGNORECASE)
_SPEC_FIELDS = {"schema_version", "shot_id", "set", "bake", "actors", "render", "look"}
_LOOK_FIELDS = {"view_transform", "look", "exposure", "aperture_fstop", "motion_blur_shutter"}


class ShotPlanError(ValueError):
    """Invalid, unsafe or inconsistent shot request."""


def to_blender_position(position: Any) -> tuple[float, float, float]:
    x, y, z = position
    return (x, -z, y)


def to_blender_quaternion(orientation: Any) -> tuple[float, float, float, float]:
    """XYZW Y-up quaternion to WXYZ Z-up, by a +90 degree turn about X."""
    x, y, z, w = orientation
    half = math.sqrt(.5)
    return (half * (w - x), half * (x + w), half * (y - z), half * (z + y))


@dataclass(frozen=True)
class Bake:
    fps_num: int
    fps_den: int
    duration_frames: int
    camera: dict[str, list[Any]]
    objects: dict[str, dict[str, list[Any]]]
    actors: dict[str, dict[str, Any]]
    lights: dict[str, list[float]]
    contacts: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ShotActor:
    id: str
    pack: Path
    profile: Path


@dataclass(frozen=True)
class ShotPlan:
    shot_id: str
    root: Path
    set_path: Path
    bake: Bake
    actors: tuple[ShotActor, ...]
    profile: str
    width: int
    height: int
    look: dict[str, Any]


def _length(values: Any, name: str, frames: int) -> list[Any]:
    if not isinstance(values, list):
        raise ShotPlanError(f"{name} must be a per-frame list")
    if len(values) != frames:
        raise ShotPlanError(f"{name} has {len(values)} frames, expected {frames}")
    return values


def _quaternions(values: list[Any], name: str) -> None:
    for frame, value in enumerate(values):
        if not math.isclose(sum(v * v for v in value), 1, abs_tol=1e-3):
            raise ShotPlanError(f"{name} frame {frame} is not a unit quaternion")


def parse_bake(raw: Any) -> Bake:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ShotPlanError("Only bake schema_version 1 is supported")
    fps = raw.get("fps") or {}
    num, den = fps.get("num"), fps.get("den")
    if type(num) is not int or type(den) is not int or num < 1 or den < 1:
        raise ShotPlanError("fps must be a positive rational")
    frames = raw.get("duration_frames")
    if type(frames) is not int or not 1 <= frames <= 18000:
        raise ShotPlanError("duration_frames must be 1 to 18000")
    camera = raw.get("camera") or {}
    for key in ("position", "orientation", "fov", "focus_distance"):
        _length(camera.get(key), f"camera.{key}", frames)
    _quaternions(camera["orientation"], "camera.orientation")
    objects = raw.get("objects") or {}
    for name, channels in objects.items():
        for key in ("position", "orientation"):
            _length(channels.get(key), f"{name}.{key}", frames)
        _quaternions(channels["orientation"], f"{name}.orientation")
        if "bend" in channels:
            _length(channels["bend"], f"{name}.bend", frames)
    actors = raw.get("actors") or {}
    for name, channels in actors.items():
        if not channels:
            continue
        _length(channels.get("parts"), f"{name}.parts", frames)
        root = channels.get("root") or {}
        _length(root.get("position"), f"{name}.root.position", frames)
        _length(root.get("yaw"), f"{name}.root.yaw", frames)
        for effector, track in (channels.get("effectors") or {}).items():
            _length(track.get("position"), f"{name}.{effector}.position", frames)
            _length(track.get("influence"), f"{name}.{effector}.influence", frames)
            _length(track.get("holding"), f"{name}.{effector}.holding", frames)
    lights = raw.get("lights") or {}
    for name, values in lights.items():
        _length(values, f"lights.{name}", frames)
    contacts = tuple(raw.get("contacts") or ())
    for contact in contacts:
        frame = contact.get("frame")
        if type(frame) is not int or not 0 <= frame < frames:
            raise ShotPlanError(f"contact frame {frame} is outside the shot")
    return Bake(num, den, frames, camera, objects, actors, lights, contacts)


def _contained(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ShotPlanError("Shot paths must be relative")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ShotPlanError(f"Path leaves the shot root: {relative}")
    return path


def load_shot_plan(path: Path) -> ShotPlan:
    root = path.resolve().parent
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ShotPlanError("Only shot schema_version 1 is supported")
    unknown = raw.keys() - _SPEC_FIELDS
    if unknown:
        raise ShotPlanError(f"Shot has unknown fields: {sorted(unknown)}")
    shot_id = raw.get("shot_id")
    if not isinstance(shot_id, str) or not _SLUG.fullmatch(shot_id):
        raise ShotPlanError("shot_id must be a safe identifier")
    render = raw.get("render") or {}
    profile = render.get("profile")
    if profile not in RENDER_PROFILES:
        raise ShotPlanError(f"Unknown render profile {profile}")
    width, height = render.get("width"), render.get("height")
    if type(width) is not int or type(height) is not int or not 16 <= width <= 3840 or not 16 <= height <= 3840 \
            or width % 2 or height % 2:
        raise ShotPlanError("width and height must be even integers from 16 to 3840")
    look = raw.get("look") or {}
    if look.keys() - _LOOK_FIELDS:
        raise ShotPlanError(f"Look has unknown fields: {sorted(look.keys() - _LOOK_FIELDS)}")
    set_path = _contained(root, raw.get("set"))
    bake = parse_bake(json.loads(_contained(root, raw.get("bake")).read_text()))
    actors = []
    for actor in raw.get("actors") or []:
        if not isinstance(actor.get("id"), str) or not _SLUG.fullmatch(actor["id"]):
            raise ShotPlanError("Actor id must be a safe identifier")
        actors.append(ShotActor(actor["id"], _contained(root, actor.get("pack")), _contained(root, actor.get("profile"))))
    cast = {actor.id for actor in actors}
    missing = bake.actors.keys() - cast
    if missing:
        raise ShotPlanError(f"Bake drives actors that are not in the cast: {sorted(missing)}")
    return ShotPlan(shot_id, root, set_path, bake, tuple(actors), profile, width, height, look)


def to_blender_object_quaternion(orientation: Any) -> tuple[float, float, float, float]:
    """XYZW Y-up object rotation to WXYZ Z-up. Local axes convert with the world."""
    x, y, z, w = orientation
    return (w, x, -z, y)
