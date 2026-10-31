"""Original, configurable building-detail study assembled from reusable parts.

This is a stone townhouse/facade demonstration, not a verified reconstruction
of a historic landmark. Doors, glazing, masonry, moldings and tiles all remain
individually addressable MeshInstances for Blender, GLB and future animation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .architecture_details import (
    Opening, box_mesh, cornice_extrusion, coursed_wall, dentil_row,
    door_assembly, gable_roof, window_assembly,
)
from .architecture_geometry import MeshInstance, Vec3, place_mesh, world_vertices


class FacadeError(ValueError):
    """An architectural facade study cannot be assembled as specified."""


@dataclass(frozen=True)
class FacadeStudy:
    asset_id: str
    bays: int
    instances: tuple[MeshInstance, ...]
    part_counts: tuple[tuple[str, int], ...]
    bounds_min: Vec3
    bounds_max: Vec3


def _translated(instance: MeshInstance, *, offset: Vec3) -> MeshInstance:
    return replace(instance, origin=tuple(instance.origin[i] + offset[i] for i in range(3)))


def _oriented(instance: MeshInstance, *, offset: Vec3, axes: tuple[Vec3, Vec3, Vec3]) -> MeshInstance:
    """Compose a rotation/translation with an existing rigid instance frame."""
    def rotate(v: Vec3) -> Vec3:
        return tuple(sum(v[i] * axes[i][axis] for i in range(3)) for axis in range(3))
    xyz = rotate(instance.origin)
    return replace(instance, origin=tuple(xyz[i] + offset[i] for i in range(3)),
                   axis_x=rotate(instance.axis_x), axis_y=rotate(instance.axis_y),
                   axis_z=rotate(instance.axis_z))


def make_facade_study(*, bays: int = 3) -> FacadeStudy:
    """Compose an enclosed, decorated stone building with configurable bay count.

    Alternating physical masonry courses respect all true openings. Every bay
    gets an upper glazed window, all but the center get a lower window, and the
    central bay receives a panelled door. The pitch is 3.8 m; larger odd bay
    counts stretch the facade without stretching individual detail geometry.
    """
    if type(bays) is not int or bays not in (3, 5, 7):
        raise FacadeError("bays must be 3, 5 or 7 for a symmetric central entrance")
    pitch, depth, wall_height = 3.8, 7.2, 8.5
    width = pitch * bays + 1.0
    center_bay = bays // 2
    front_y = -depth / 2
    openings: list[Opening] = []
    window_centers: list[tuple[int, float, str, float]] = []
    for bay in range(bays):
        x = 0.5 + (bay + 0.5) * pitch
        if bay != center_bay:
            openings.append(Opening(x - 1.02, x + 1.02, 1.22, 3.77))
            window_centers.append((bay, x, "lower", (1.22 + 3.77) / 2))
        else:
            openings.append(Opening(x - 1.15, x + 1.15, 0.29, 3.94))
        openings.append(Opening(x - 1.02, x + 1.02, 4.69, 7.24))
        window_centers.append((bay, x, "upper", (4.69 + 7.24) / 2))

    items: list[MeshInstance] = []
    front = coursed_wall(width=width, height=wall_height, thickness=0.49,
                         openings=tuple(openings), prefix="masonry-front")
    for i, stone in enumerate(front):
        if i % 9 == 0:
            stone = replace(stone, material="wall-stone-light")
        items.append(_translated(stone, offset=(0, front_y, 0)))

    # Enclose the volume with real side and back masonry, not an open backdrop.
    rotations = (
        ("left", depth, (0, front_y, 0), ((0, 1, 0), (-1, 0, 0), (0, 0, 1))),
        ("right", depth, (width, -front_y, 0), ((0, -1, 0), (1, 0, 0), (0, 0, 1))),
        ("back", width, (width, -front_y, 0), ((-1, 0, 0), (0, -1, 0), (0, 0, 1))),
    )
    for side, span, offset, axes in rotations:
        wall = coursed_wall(width=span, height=wall_height, thickness=0.49,
                           prefix=f"masonry-{side}")
        for i, stone in enumerate(wall):
            if i % 11 == 0:
                stone = replace(stone, material="wall-stone-light")
            items.append(_oriented(stone, offset=offset, axes=axes))

    for bay, x, floor, z in window_centers:
        parts = window_assembly(prefix=f"window-{bay:02d}-{floor}", width=2.04,
                                height=2.55, rows=2, columns=2)
        items.extend(_translated(part, offset=(x, front_y, z)) for part in parts)
    entrance_x = 0.5 + (center_bay + 0.5) * pitch
    parts = door_assembly(prefix="door-main", width=2.30, height=3.65)
    items.extend(_translated(part, offset=(entrance_x, front_y, (0.29 + 3.94) / 2))
                 for part in parts)

    # A heavy foundation, a projecting string course and an ornate eave band.
    items.append(place_mesh(box_mesh(width + 0.7, depth + 0.7, 0.33),
                            name="foundation-perimeter", origin=(width/2, 0, -0.18),
                            material="foundation"))
    profile_axes = {"axis_x": (0, 1, 0), "axis_y": (0, 0, 1), "axis_z": (1, 0, 0)}
    items.append(place_mesh(cornice_extrusion(width + 0.70, 0.28, 0.23),
                            name="cornice-string-course", origin=(-0.35, front_y - 0.08, 4.07),
                            material="trim-stone", **profile_axes))
    items.append(place_mesh(cornice_extrusion(width + 0.90, 0.51, 0.46),
                            name="cornice-eave-band", origin=(-0.45, front_y - 0.12, 8.16),
                            material="trim-stone", **profile_axes))
    teeth_count = round(width / 0.38)
    items.extend(_translated(tooth, offset=(-0.15, front_y - 0.45, 8.01))
                 for tooth in dentil_row(prefix="dentil-eave", length=width + 0.3,
                                         count=teeth_count, tooth_width=0.22))

    items.extend(gable_roof(width=width, depth=depth, eave_z=8.63, rise=2.6,
                            overhang=0.55, tile_rows=8,
                            tile_columns=round((width+1.1) / 0.52)))

    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    counts: dict[str, int] = {}
    for obj in items:
        group = obj.name.split("-")[0]
        counts[group] = counts.get(group, 0) + 1
        for vertex in world_vertices(obj):
            for axis in range(3):
                mins[axis] = min(mins[axis], vertex[axis])
                maxs[axis] = max(maxs[axis], vertex[axis])
    if len({obj.name for obj in items}) != len(items):
        raise FacadeError("Facade instance names must be unique")
    return FacadeStudy(asset_id=f"stone-facade-{bays}-bay-study", bays=bays,
                       instances=tuple(items), part_counts=tuple(sorted(counts.items())),
                       bounds_min=tuple(mins), bounds_max=tuple(maxs))
