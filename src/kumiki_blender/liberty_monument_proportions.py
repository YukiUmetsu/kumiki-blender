"""Conservative NPS-size study for the assembled Liberty model.

Unlike the existing illustrative default, this optional transformation uses
published *physical dimensions* for the monument's principal heights and
some recognizable components. It is a dimensionally constrained blockout,
not a registered 3D scan or a claim of architectural/face likeness.

NPS: https://www.nps.gov/stli/learn/historyculture/statue-statistics.htm
The pedestal's ~16.8 m upper breadth is a coarse **inference** from the
public 2017 NYC topobathymetric LiDAR tile 970190.las, not an NPS dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

from .architecture_geometry import Mesh, MeshInstance, place_mesh


PEDESTAL_HEIGHT_M = 46.94
STATUE_HEIGHT_M = 46.05
TOTAL_HEIGHT_M = PEDESTAL_HEIGHT_M + STATUE_HEIGHT_M
HEAD_HEIGHT_M = 5.26
HEAD_WIDTH_M = 3.05
HEEL_TO_CRANIUM_M = 33.86
WAIST_WIDTH_M = 10.67
TABLET_HEIGHT_M = 7.19
TABLET_WIDTH_M = 4.14
TABLET_THICKNESS_M = .61
UPPER_PEDESTAL_BREADTH_EST_M = 16.8  # *inferred*, coarse airborne LAS section


def _extent(part: MeshInstance, axis: int) -> tuple[float, float]:
    vals = [p[axis] for p in part.mesh.vertices]
    return min(vals), max(vals)


@dataclass(frozen=True)
class MonumentProportionMap:
    """Source-model anchors and NPS target sections, without camera claims."""

    pedestal_top: float
    statue_bottom: float
    torch_top: float
    head_low: float
    head_high: float
    neck_low: float
    neck_high: float
    torso_xy: float
    head_xy: float
    pedestal_xy: float
    tablet_x: float
    tablet_y: float
    tablet_z: float
    tablet_x_width: float
    tablet_y_width: float
    tablet_z_height: float

    @classmethod
    def from_instances(cls, instances: list[MeshInstance]) -> "MonumentProportionMap":
        named = {part.name: part for part in instances}
        pedestal = [part for part in instances if part.name.startswith("liberty-pedestal-")]
        head = [part for part in instances if part.name.startswith("liberty-head-")]
        face = named["liberty-head-face"]
        gown = named["liberty-gown-underrobe"]
        slab = named["liberty-tablet-slab"]
        footplate = named["liberty-statue-footplate"]
        neck = named["liberty-neck"]
        if not pedestal:
            # Even with --no-pedestal the original construction has the same
            # 9.12m upper support elevation; all statue parts stay in frame.
            pedestal_top = 9.12
            pedestal_span = 9.65
        else:
            pedestal_top = max(_extent(part, 2)[1] for part in pedestal)
            pedestal_span = (max(_extent(part, 0)[1] for part in pedestal)
                             - min(_extent(part, 0)[0] for part in pedestal))
        head_span = (max(_extent(part, 0)[1] for part in head)
                     - min(_extent(part, 0)[0] for part in head))
        gw = _extent(gown, 0)
        tx, ty, tz = (_extent(slab, i) for i in range(3))
        return cls(
            pedestal_top=pedestal_top,
            statue_bottom=_extent(footplate, 2)[0],
            torch_top=max(_extent(part, 2)[1] for part in instances),
            head_low=_extent(face, 2)[0],
            head_high=_extent(face, 2)[1],
            neck_low=_extent(neck, 2)[0],
            neck_high=_extent(neck, 2)[1],
            torso_xy=WAIST_WIDTH_M/(gw[1]-gw[0]),
            head_xy=HEAD_WIDTH_M/head_span,
            pedestal_xy=UPPER_PEDESTAL_BREADTH_EST_M/pedestal_span,
            tablet_x=(tx[0]+tx[1])/2,
            tablet_y=(ty[0]+ty[1])/2,
            tablet_z=(tz[0]+tz[1])/2,
            tablet_x_width=TABLET_WIDTH_M/(tx[1]-tx[0]),
            tablet_y_width=TABLET_THICKNESS_M/(ty[1]-ty[0]),
            tablet_z_height=TABLET_HEIGHT_M/(tz[1]-tz[0]),
        )

    def statue_z(self, z: float) -> float:
        return (PEDESTAL_HEIGHT_M
                +(z-self.statue_bottom)*STATUE_HEIGHT_M/(self.torch_top-self.statue_bottom))

    def head_z(self, z: float) -> float:
        head_top = PEDESTAL_HEIGHT_M+HEEL_TO_CRANIUM_M
        return head_top+(z-self.head_high)*HEAD_HEIGHT_M/(self.head_high-self.head_low)

    def map_part(self, part: MeshInstance) -> MeshInstance:
        name = part.name
        pedestal = name.startswith("liberty-pedestal-")
        head = name.startswith(("liberty-head-", "liberty-hair-", "liberty-crown-"))
        neck = name == "liberty-neck"
        tablet = name.startswith("liberty-tablet-")
        transformed = []
        for x, y, z in part.mesh.vertices:
            if pedestal:
                xyz = (x*self.pedestal_xy, y*self.pedestal_xy,
                       z*PEDESTAL_HEIGHT_M/self.pedestal_top)
            elif head:
                xyz = (x*self.head_xy, y*self.head_xy, self.head_z(z))
            elif neck:
                alpha = (z-self.neck_low)/(self.neck_high-self.neck_low)
                neck_top = self.head_z(self.neck_high)
                neck_base = self.statue_z(self.neck_low)
                xyz = (x*self.head_xy, y*self.head_xy,
                       neck_base+alpha*(neck_top-neck_base))
            elif tablet:
                xyz = (
                    self.tablet_x*self.torso_xy+(x-self.tablet_x)*self.tablet_x_width,
                    self.tablet_y*self.torso_xy+(y-self.tablet_y)*self.tablet_y_width,
                    self.statue_z(self.tablet_z)+(z-self.tablet_z)*self.tablet_z_height,
                )
            else:
                xyz = (x*self.torso_xy, y*self.torso_xy, self.statue_z(z))
            transformed.append(xyz)
        return place_mesh(Mesh(tuple(transformed), part.mesh.faces), name=name,
                          material=part.material)


def apply_monument_proportions(instances: list[MeshInstance]) -> list[MeshInstance]:
    mapper = MonumentProportionMap.from_instances(instances)
    return [mapper.map_part(part) for part in instances]
