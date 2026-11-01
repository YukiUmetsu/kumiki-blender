"""Typed, declarative AssetSpec v2 for agent-authored component assemblies.

This is an optional authoring dependency. Legacy generators and Blender builds
retain their standard-library-only import path unless AssetSpec is requested.
"""

from __future__ import annotations

import json
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


class LatheComponent(StrictModel):
    type: Literal["solid.lathe"]
    id: Slug
    material: Slug
    profile_rz_m: Annotated[tuple[RadiusHeight, ...], Field(min_length=2, max_length=128)]
    origin_m: Point3 = (0.0, 0.0, 0.0)
    radial_segments: Annotated[int, Field(ge=8, le=256)] = 48


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


Component = Annotated[
    Union[BoxComponent, LatheComponent, RodComponent, StudRingComponent],
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
