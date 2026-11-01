"""Blender-independent compiler from declarative components to a scene plan."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .architecture_details import box_mesh
from .architecture_geometry import (Mesh, MeshInstance, Vec3, audit_mesh, place_mesh,
                                    revolve_profile, world_vertices)
from .craft_details import (detail_frame, fastener_head_mesh, mitered_panel_rails,
                            rounded_box_mesh)

if TYPE_CHECKING:
    from .asset_spec import (AssetSpec, BoxComponent, HollowLatheComponent, LatheComponent,
                             LinearFastenersComponent, PanelFrameComponent,
                             RodComponent, RoundedBoxComponent, StudRingComponent)


class ComposerError(ValueError):
    """A validated component could not be safely assembled."""


@dataclass(frozen=True)
class ComponentDefinition:
    kind: str
    description: str
    builder: Callable[..., tuple[MeshInstance, ...]]


@dataclass(frozen=True)
class AssemblyPlan:
    asset_id: str
    instances: tuple[MeshInstance, ...]
    bounds_min: Vec3
    bounds_max: Vec3
    named_parts: tuple[tuple[str, int], ...]
    recipe_sha256: str


def _box(component: BoxComponent) -> tuple[MeshInstance, ...]:
    return (place_mesh(box_mesh(*component.size_m), name=component.id,
                       origin=component.origin_m, material=component.material),)


def _rounded_box(component: RoundedBoxComponent) -> tuple[MeshInstance, ...]:
    mesh = rounded_box_mesh(component.size_m, component.corner_radius_m,
                            corner_segments=component.corner_segments)
    return (place_mesh(mesh, name=component.id, origin=component.origin_m,
                       material=component.material),)


def _lathe(component: LatheComponent) -> tuple[MeshInstance, ...]:
    mesh = revolve_profile(component.profile_rz_m, radial_segments=component.radial_segments)
    x, y, z = detail_frame(component.normal_axis)
    return (place_mesh(mesh, name=component.id, origin=component.origin_m,
                       axis_x=x, axis_y=y, axis_z=z,
                       material=component.material),)


def _hollow_lathe(component: HollowLatheComponent) -> tuple[MeshInstance, ...]:
    """Physically hollow shell with closed annular lips and a real inner wall."""
    profile = component.profile_rz_m
    n = component.radial_segments
    section_count = len(profile)
    vertices = tuple(
        ((radius-thickness)*math.cos(math.tau*j/n),
         (radius-thickness)*math.sin(math.tau*j/n), z)
        for thickness in (0.0, component.wall_thickness_m)
        for radius, z in profile
        for j in range(n)
    )
    inner_offset = section_count*n
    faces: list[tuple[int, ...]] = []
    for ring in range(section_count-1):
        for i in range(n):
            j = (i+1) % n
            a, b = ring*n+i, (ring+1)*n+i
            faces.append((a, ring*n+j, (ring+1)*n+j, b))
            faces.append((inner_offset+a, inner_offset+b,
                          inner_offset+(ring+1)*n+j, inner_offset+ring*n+j))
    last = (section_count-1)*n
    for i in range(n):
        j = (i+1) % n
        faces.append((i, inner_offset+i, inner_offset+j, j))
        faces.append((last+i, last+j, inner_offset+last+j, inner_offset+last+i))
    x, y, z = detail_frame(component.normal_axis)
    return (place_mesh(Mesh(vertices, tuple(faces)), name=component.id,
                       origin=component.origin_m, axis_x=x, axis_y=y,
                       axis_z=z, material=component.material),)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _rod(component: RodComponent) -> tuple[MeshInstance, ...]:
    length = math.dist(component.start_m, component.end_m)
    if length < 1e-4:
        raise ComposerError(f"{component.id}: rod is too short")
    z: Vec3 = tuple((b-a)/length for a, b in zip(component.start_m, component.end_m))
    helper: Vec3 = (0, 0, 1) if abs(z[2]) < 0.9 else (1, 0, 0)
    x = _cross(helper, z)
    norm = math.sqrt(sum(v*v for v in x))
    x = tuple(v/norm for v in x)
    y = _cross(z, x)
    mesh = revolve_profile(((component.radius_m, 0), (component.radius_m, length)),
                           radial_segments=component.radial_segments)
    return (place_mesh(mesh, name=component.id, origin=component.start_m,
                       axis_x=x, axis_y=y, axis_z=z, material=component.material),)


def _stud_ring(component: StudRingComponent) -> tuple[MeshInstance, ...]:
    radius, height = component.stud_radius_m, component.stud_height_m
    mesh = revolve_profile(((radius, 0), (radius, height)),
                           radial_segments=component.radial_segments)
    cx, cy, cz = component.center_m
    x, y, z = detail_frame(component.normal_axis)
    return tuple(place_mesh(
        mesh, name=f"{component.id}-{index:03d}",
        origin=tuple(center + component.ring_radius_m*(
            x[axis]*math.cos(math.tau*index/component.count) +
            y[axis]*math.sin(math.tau*index/component.count))
            for axis, center in enumerate((cx, cy, cz))),
        axis_x=x, axis_y=y, axis_z=z, material=component.material,
    ) for index in range(component.count))


def _linear_fasteners(component: LinearFastenersComponent) -> tuple[MeshInstance, ...]:
    mesh = fastener_head_mesh(component.head_style, component.head_radius_m,
                              component.head_height_m)
    x, y, z = detail_frame(component.normal_axis)
    return tuple(place_mesh(
        mesh, name=f"{component.id}-{index:03d}",
        origin=tuple(start + (end-start)*index/(component.count-1)
                     for start, end in zip(component.start_m, component.end_m)),
        axis_x=x, axis_y=y, axis_z=z, material=component.material,
    ) for index in range(component.count))


def _panel_frame(component: PanelFrameComponent) -> tuple[MeshInstance, ...]:
    x, y, z = detail_frame({"xy": "+z", "xz": "-y", "yz": "+x"}[component.plane])
    return tuple(place_mesh(
        mesh, name=f"{component.id}-{label}", origin=component.origin_m,
        axis_x=x, axis_y=y, axis_z=z, material=component.material,
    ) for label, mesh in mitered_panel_rails(
        component.size_m, component.trim_width_m, component.depth_m))


_INSTANCE_SUFFIX = {
    "detail.radial-studs": r"-[0-9]{3}",
    "detail.linear-fasteners": r"-[0-9]{3}",
    "detail.panel-frame": r"-(?:top|right|bottom|left)",
}


def matches_component_instance(kind: str, component_id: str, name: str) -> bool:
    """Share expansion-name rules across the compiler and Blender handoff."""
    suffix = _INSTANCE_SUFFIX.get(kind)
    if suffix is None:
        return name == component_id
    return re.fullmatch(re.escape(component_id) + suffix, name) is not None


def origin_component_id(name: str, components: tuple) -> str | None:
    """Resolve the typed declaration that emitted a named mesh instance."""
    for component in components:
        if matches_component_instance(component.type, component.id, name):
            return component.id
    return None


COMPONENT_REGISTRY: dict[str, ComponentDefinition] = {
    item.kind: item for item in (
        ComponentDefinition("solid.box", "Centered closed rectangular solid", _box),
        ComponentDefinition("solid.rounded-box", "Real filleted manufactured solid", _rounded_box),
        ComponentDefinition("solid.lathe", "Capped radial profile for machined parts", _lathe),
        ComponentDefinition("solid.hollow-lathe", "Open shade/shell with physical inner and outer walls", _hollow_lathe),
        ComponentDefinition("mechanical.rod", "Oriented cylindrical connector", _rod),
        ComponentDefinition("detail.radial-studs", "Instanced circular fastener detail", _stud_ring),
        ComponentDefinition("detail.linear-fasteners", "Instanced surface-oriented machine fasteners", _linear_fasteners),
        ComponentDefinition("detail.panel-frame", "Four individually mitered raised frame rails", _panel_frame),
    )
}


def plan_assembly(spec: AssetSpec) -> AssemblyPlan:
    """Compile a validated spec into named, audited, metrically bounded meshes."""
    result: list[MeshInstance] = []
    groups: dict[str, int] = {}
    for component in spec.components:
        definition = COMPONENT_REGISTRY[component.type]
        parts = definition.builder(component)
        if len(result) + len(parts) > 4096:
            raise ComposerError("AssetSpec exceeds the 4096-instance safety limit")
        groups[component.type] = groups.get(component.type, 0) + len(parts)
        for part in parts:
            report = audit_mesh(part.mesh)
            if report.boundary_edges or report.nonmanifold_edges or report.volume_m3 <= 0:
                raise ComposerError(f"{part.name}: component mesh is not a positive closed solid")
        result.extend(parts)
    names = [part.name for part in result]
    if len(set(names)) != len(names):
        raise ComposerError("Expanded component instance names collide")
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for part in result:
        for vertex in world_vertices(part):
            for axis in range(3):
                mins[axis] = min(mins[axis], vertex[axis])
                maxs[axis] = max(maxs[axis], vertex[axis])
    canonical = json.dumps(spec.model_dump(mode="json", exclude_none=True),
                           sort_keys=True, separators=(",", ":"), allow_nan=False)
    return AssemblyPlan(spec.asset_id, tuple(result), tuple(mins), tuple(maxs),
                        tuple(sorted(groups.items())),
                        hashlib.sha256(canonical.encode("utf-8")).hexdigest())
