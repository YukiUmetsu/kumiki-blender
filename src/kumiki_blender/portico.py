"""Modular architectural arcade study assembled from reusable solids.

This is an original detail demonstrator, not a recreation of a specific
historic building. Its columns, voussoirs and cornice remain separate,
named instances with physically closed geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .architecture_geometry import (
    MeshInstance,
    Vec3,
    arch_voussoirs,
    extrude_profile,
    place_mesh,
    revolve_profile,
    world_vertices,
)


class PorticoError(ValueError):
    """Invalid arcade layout request."""


@dataclass(frozen=True)
class PorticoStudy:
    asset_id: str
    bays: int
    instances: tuple[MeshInstance, ...]
    part_counts: tuple[tuple[str, int], ...]
    bounds_min: Vec3
    bounds_max: Vec3


def make_portico_study(*, bays: int = 3) -> PorticoStudy:
    """Three-dimensional, 4.2 m-pitch limestone arcade with fluted columns.

    Each column and flute shares a mesh datablock, while every stone has
    real planar mortar spacing for readable highlights and silhouettes.
    """
    if type(bays) is not int or not 1 <= bays <= 20:
        raise PorticoError("bays must be an integer from 1 to 20")
    pitch = 4.2
    overall_span = pitch * bays
    columns = revolve_profile((
        (0.48, 0), (0.48, 0.13), (0.42, 0.23), (0.36, 0.35),
        (0.32, 0.48), (0.32, 4.08), (0.36, 4.2),
        (0.44, 4.34), (0.50, 4.43), (0.50, 4.55),
    ), radial_segments=48)
    flute = revolve_profile(((0.018, 0), (0.018, 3.52)), radial_segments=10)
    pedestal = extrude_profile(((-0.68, -0.65), (0.68, -0.65),
                                (0.68, 0.65), (-0.68, 0.65)), 0.26)
    # Arcade spring line is level with the capital. The keystone projects
    # forward an additional 150 mm, as an architectural shadow detail.
    arch = arch_voussoirs(1.64, 1.96, 0.60, stone_count=11, joint_gap_m=0.006)
    keystone = arch_voussoirs(1.64, 1.96, 0.75, stone_count=11, joint_gap_m=0.006)[5]

    instances: list[MeshInstance] = []
    for col in range(bays + 1):
        x = -overall_span / 2 + col * pitch
        instances.append(place_mesh(pedestal, name=f"pedestal-{col:02d}",
                                    origin=(x, 0.15, -0.10), material="stone-foundation"))
        instances.append(place_mesh(columns, name=f"column-{col:02d}",
                                    origin=(x, 0.15, 0), material="limestone"))
        for index in range(16):
            angle = math.tau * index / 16
            instances.append(place_mesh(flute, name=f"column-{col:02d}-flute-{index:02d}",
                                        origin=(x + 0.324 * math.cos(angle),
                                                0.15 + 0.324 * math.sin(angle), 0.51),
                                        material="limestone-light"))

    # Local mesh axes: X=horizontal, Y=height, Z=thickness. Rotate into
    # world X/Z, thickness pointing toward the camera along world -Y.
    arch_axes = {"axis_x": (1, 0, 0), "axis_y": (0, 0, 1), "axis_z": (0, -1, 0)}
    for bay in range(bays):
        center_x = -overall_span / 2 + (bay + 0.5) * pitch
        for index, block in enumerate(arch):
            instances.append(place_mesh(keystone if index == 5 else block,
                                        name=f"arch-{bay:02d}-stone-{index:02d}",
                                        origin=(center_x, 0.35, 4.55),
                                        material="keystone" if index == 5 else
                                                 ("limestone" if index % 3 else "limestone-warm"),
                                        **arch_axes))

    # A concave molding profile, extruded the full length of the arcade.
    cornice_profile = ((0, 0), (0.78, 0), (0.78, 0.14), (0.91, 0.17),
                       (0.91, 0.27), (0.78, 0.31), (0.84, 0.44),
                       (0.71, 0.53), (0.71, 0.64), (0, 0.64))
    cornice = extrude_profile(cornice_profile, overall_span + 1.6)
    instances.append(place_mesh(cornice, name="cornice-main", origin=(-overall_span / 2 - 0.8, -0.65, 6.52),
                                axis_x=(0, 1, 0), axis_y=(0, 0, 1), axis_z=(1, 0, 0),
                                material="limestone-light"))
    floor_span = overall_span + 2.4
    plinth = extrude_profile(((-floor_span / 2, -1.15), (floor_span / 2, -1.15),
                              (floor_span / 2, 1.15), (-floor_span / 2, 1.15)), 0.28)
    instances.append(place_mesh(plinth, name="plinth-main", origin=(0, 0, -0.28), material="stone-foundation"))

    # Exact world bounds include the stone courses, flutes and cornice.
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for instance in instances:
        for vertex in world_vertices(instance):
            for axis in range(3):
                mins[axis] = min(mins[axis], vertex[axis])
                maxs[axis] = max(maxs[axis], vertex[axis])

    return PorticoStudy(
        asset_id="portico-detail-study",
        bays=bays,
        instances=tuple(instances),
        part_counts=(("arch_stones", bays * 11), ("columns", bays + 1),
                     ("flutes", (bays + 1) * 16), ("pedestals", bays + 1),
                     ("cornices", 1), ("plinths", 1)),
        bounds_min=tuple(mins), bounds_max=tuple(maxs),
    )
