"""Blender-independent compiler from declarative components to a scene plan."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .architecture_details import box_mesh
from .architecture_geometry import (Mesh, MeshInstance, Vec3, audit_mesh, place_mesh,
                                    revolve_profile, world_vertices)

if TYPE_CHECKING:
    from .asset_spec import (AssetSpec, BoxComponent, HollowLatheComponent, LatheComponent,
                             RodComponent, StudRingComponent)


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


def _lathe(component: LatheComponent) -> tuple[MeshInstance, ...]:
    mesh = revolve_profile(component.profile_rz_m, radial_segments=component.radial_segments)
    return (place_mesh(mesh, name=component.id, origin=component.origin_m,
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
    return (place_mesh(Mesh(vertices, tuple(faces)), name=component.id,
                       origin=component.origin_m, material=component.material),)


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
    return tuple(place_mesh(
        mesh, name=f"{component.id}-{index:03d}",
        origin=(cx + component.ring_radius_m*math.cos(math.tau*index/component.count),
                cy + component.ring_radius_m*math.sin(math.tau*index/component.count), cz),
        material=component.material,
    ) for index in range(component.count))


COMPONENT_REGISTRY: dict[str, ComponentDefinition] = {
    item.kind: item for item in (
        ComponentDefinition("solid.box", "Centered closed rectangular solid", _box),
        ComponentDefinition("solid.lathe", "Capped radial profile for machined parts", _lathe),
        ComponentDefinition("solid.hollow-lathe", "Open shade/shell with physical inner and outer walls", _hollow_lathe),
        ComponentDefinition("mechanical.rod", "Oriented cylindrical connector", _rod),
        ComponentDefinition("detail.radial-studs", "Instanced circular fastener detail", _stud_ring),
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
    canonical = spec.model_dump_json(exclude_none=True)
    return AssemblyPlan(spec.asset_id, tuple(result), tuple(mins), tuple(maxs),
                        tuple(sorted(groups.items())),
                        hashlib.sha256(canonical.encode("utf-8")).hexdigest())
