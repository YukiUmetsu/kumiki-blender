"""Modular architectural arcade study assembled from reusable solids.

This is an original detail demonstrator, not a recreation of a specific
historic building. Its columns, voussoirs and cornice remain separate,
named instances with physically closed geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .architecture_details import classical_column

from .architecture_geometry import (
    MeshInstance,
    Vec3,
    arch_voussoirs,
    extrude_profile,
    place_mesh,
    world_vertices,
)


class PorticoError(ValueError):
    """Invalid arcade layout request."""


@dataclass(frozen=True)
class PorticoStudy:
    asset_id: str
    bays: int
    order: str
    instances: tuple[MeshInstance, ...]
    part_counts: tuple[tuple[str, int], ...]
    bounds_min: Vec3
    bounds_max: Vec3


def make_portico_study(*, bays: int = 3, order: str = "ionic") -> PorticoStudy:
    """Three-dimensional, 4.2 m-pitch limestone arcade with fluted columns.

    Each shaft has real concave flutes, distinct moulded base/capital solids
    and carved Ionic-style volutes. Arch stones have actual mortar spacing.
    """
    if type(bays) is not int or not 1 <= bays <= 20:
        raise PorticoError("bays must be an integer from 1 to 20")
    if order not in ("tuscan", "doric", "ionic"):
        raise PorticoError("order must be tuscan, doric or ionic")
    pitch = 4.2
    overall_span = pitch * bays
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
        for component in classical_column(prefix=f"column-{col:02d}", height=4.55,
                                          base_radius=0.36, order=order):
            instances.append(replace(component, origin=(component.origin[0] + x,
                                                        component.origin[1] + 0.15,
                                                        component.origin[2])))

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

    # Exact world bounds include grooved shafts, volutes and cornice.
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for instance in instances:
        for vertex in world_vertices(instance):
            for axis in range(3):
                mins[axis] = min(mins[axis], vertex[axis])
                maxs[axis] = max(maxs[axis], vertex[axis])

    return PorticoStudy(
        asset_id="portico-detail-study" if order == "ionic" else f"portico-{order}-study",
        bays=bays,
        order=order,
        instances=tuple(instances),
        part_counts=(("arch_stones", bays * 11), ("columns", bays + 1),
                     ("column_bases", (bays + 1) * (0 if order == "doric" else 2)),
                     ("column_capitals", (bays + 1) * (4 if order == "ionic" else 3)),
                     ("volutes", (bays + 1) * (8 if order == "ionic" else 0)),
                     ("pedestals", bays + 1),
                     ("cornices", 1), ("plinths", 1)),
        bounds_min=tuple(mins), bounds_max=tuple(maxs),
    )
