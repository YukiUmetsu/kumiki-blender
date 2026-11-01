"""Typed, declarative AssetSpec v2 for agent-authored component assemblies.

This is an optional authoring dependency. Legacy generators and Blender builds
retain their standard-library-only import path unless AssetSpec is requested.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class AssetSpecError(ValueError):
    """Invalid, unsafe or unsupported declarative asset description."""


Slug = Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=64)]
Measure = Annotated[float, Field(ge=0.001, le=1_000, allow_inf_nan=False)]
Position = Annotated[float, Field(ge=-10_000, le=10_000, allow_inf_nan=False)]
Point3 = tuple[Position, Position, Position]
RadiusHeight = tuple[Measure, Position]
SurfaceAxis = Literal["+x", "-x", "+y", "-y", "+z", "-z"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True)


class MaterialSpec(StrictModel):
    preset: Literal[
        "painted-steel", "galvanized-steel", "cast-concrete", "natural-stone",
        "weathered-stone", "timber", "optical-glass",
    ]
    color: Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")] | None = None
    roughness: Annotated[float, Field(ge=0.02, le=1.0, allow_inf_nan=False)] | None = None


class BoxComponent(StrictModel):
    type: Literal["solid.box"]
    id: Slug
    material: Slug
    size_m: tuple[Measure, Measure, Measure]
    origin_m: Point3 = (0.0, 0.0, 0.0)


class RoundedBoxComponent(BoxComponent):
    type: Literal["solid.rounded-box"]
    corner_radius_m: Measure
    corner_segments: Annotated[int, Field(ge=2, le=12)] = 5

    @model_validator(mode="after")
    def check_radius(self) -> "RoundedBoxComponent":
        if self.corner_radius_m >= min(self.size_m) / 2:
            raise ValueError("corner_radius_m must be smaller than half the shortest box dimension")
        return self


class LatheComponent(StrictModel):
    type: Literal["solid.lathe"]
    id: Slug
    material: Slug
    profile_rz_m: Annotated[tuple[RadiusHeight, ...], Field(min_length=2, max_length=128)]
    origin_m: Point3 = (0.0, 0.0, 0.0)
    radial_segments: Annotated[int, Field(ge=8, le=256)] = 48
    normal_axis: SurfaceAxis = "+z"


class HollowLatheComponent(LatheComponent):
    type: Literal["solid.hollow-lathe"]
    wall_thickness_m: Measure

    @model_validator(mode="after")
    def check_wall(self) -> "HollowLatheComponent":
        if self.wall_thickness_m >= min(radius for radius, _ in self.profile_rz_m):
            raise ValueError("wall_thickness_m must be smaller than every profile radius")
        if any(b[1] - a[1] < 1e-5 for a, b in zip(self.profile_rz_m, self.profile_rz_m[1:])):
            raise ValueError("hollow lathe profile heights must strictly increase")
        return self


class RodComponent(StrictModel):
    type: Literal["mechanical.rod"]
    id: Slug
    material: Slug
    start_m: Point3
    end_m: Point3
    radius_m: Measure
    radial_segments: Annotated[int, Field(ge=8, le=128)] = 24


class StudRingComponent(StrictModel):
    type: Literal["detail.radial-studs"]
    id: Slug
    material: Slug
    center_m: Point3
    ring_radius_m: Measure
    stud_radius_m: Measure
    stud_height_m: Measure
    count: Annotated[int, Field(ge=3, le=128)]
    radial_segments: Annotated[int, Field(ge=8, le=48)] = 16
    normal_axis: SurfaceAxis = "+z"


class LinearFastenersComponent(StrictModel):
    type: Literal["detail.linear-fasteners"]
    id: Slug
    material: Slug
    start_m: Point3
    end_m: Point3
    count: Annotated[int, Field(ge=2, le=128)]
    head_radius_m: Measure
    head_height_m: Measure
    normal_axis: SurfaceAxis = "+z"
    head_style: Literal["button", "hex"] = "button"

    @model_validator(mode="after")
    def check_spacing(self) -> "LinearFastenersComponent":
        normal_index = {"x": 0, "y": 1, "z": 2}[self.normal_axis[1]]
        if abs(self.end_m[normal_index] - self.start_m[normal_index]) > 1e-6:
            raise ValueError("linear fastener row must lie in a plane perpendicular to normal_axis")
        spacing = math.dist(self.start_m, self.end_m) / (self.count - 1)
        if spacing <= 2.4 * self.head_radius_m:
            raise ValueError("linear fastener heads overlap or have insufficient clearance")
        return self


class PanelFrameComponent(StrictModel):
    type: Literal["detail.panel-frame"]
    id: Slug
    material: Slug
    size_m: tuple[Measure, Measure]
    trim_width_m: Measure
    depth_m: Measure
    origin_m: Point3 = (0.0, 0.0, 0.0)
    plane: Literal["xy", "xz", "yz"] = "xy"

    @model_validator(mode="after")
    def check_trim_width(self) -> "PanelFrameComponent":
        if self.trim_width_m >= min(self.size_m) / 2:
            raise ValueError("trim_width_m must be smaller than half the shortest panel dimension")
        return self


Component = Annotated[
    Union[BoxComponent, RoundedBoxComponent, LatheComponent, HollowLatheComponent,
          RodComponent, StudRingComponent, LinearFastenersComponent, PanelFrameComponent],
    Field(discriminator="type"),
]


class QualitySpec(StrictModel):
    preset: Literal["smoke", "production", "cinematic-hero"] = "production"
    views: Annotated[
        tuple[Literal["hero", "front", "side", "detail"], ...],
        Field(min_length=1, max_length=4),
    ] = ("hero", "front", "side", "detail")

    @model_validator(mode="after")
    def check_views(self) -> "QualitySpec":
        if self.views[0] != "hero" or len(self.views) != len(set(self.views)):
            raise ValueError("views must start with hero and must be unique")
        return self


class AssetSpec(StrictModel):
    schema_version: Literal[2]
    asset_id: Slug
    generator: Literal["component-assembly"]
    materials: Annotated[dict[Slug, MaterialSpec], Field(min_length=1, max_length=32)]
    components: Annotated[tuple[Component, ...], Field(min_length=1, max_length=256)]
    quality: QualitySpec = Field(default_factory=QualitySpec)

    @model_validator(mode="after")
    def check_references(self) -> "AssetSpec":
        ids = [component.id for component in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("component ids must be unique")
        for component in self.components:
            if component.material not in self.materials:
                raise ValueError(f"{component.id}: unknown material role {component.material!r}")
            if isinstance(component, RodComponent) and component.start_m == component.end_m:
                raise ValueError(f"{component.id}: rod endpoints must differ")
            if isinstance(component, StudRingComponent) and component.stud_radius_m >= component.ring_radius_m:
                raise ValueError(f"{component.id}: stud radius must be smaller than ring radius")
            if (isinstance(component, StudRingComponent) and
                    2*component.stud_radius_m >=
                    2*component.ring_radius_m*math.sin(math.pi/component.count)):
                raise ValueError(f"{component.id}: neighboring radial studs must not overlap")
        return self


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AssetSpecError(f"Duplicate JSON property: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise AssetSpecError(f"Nonfinite number is not allowed: {value}")


def load_asset_spec(source: str | Path) -> AssetSpec:
    """Read a bounded JSON recipe without paths, code execution or silent keys."""
    path = Path(source)
    try:
        if path.stat().st_size > 262_144:
            raise AssetSpecError("AssetSpec exceeds the 256 KiB input limit")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                         parse_constant=_reject_nonfinite)
        # JSON arrays map to immutable tuple fields in Pydantic's JSON mode,
        # even with strict numeric/string validation enabled.
        return AssetSpec.model_validate_json(json.dumps(raw, allow_nan=False))
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise AssetSpecError(f"Invalid AssetSpec {path}: {error}") from error


def asset_json_schema() -> dict[str, object]:
    """Publish the same contract used for runtime validation to agent clients."""
    return AssetSpec.model_json_schema()
