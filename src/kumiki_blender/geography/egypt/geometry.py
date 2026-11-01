"""Small, deterministic mesh primitives for zoomable Egypt-map miniatures.

All dimensions are in the diorama's world units. This kernel is deliberately
independent of Blender so landmark proportions and topology are unit-testable.
These are interpretive architectural miniatures, not surveyed reconstructions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DistrictBuilding:
    x: float
    y: float
    width: float
    depth: float
    height: float
    palette: int
    rooftop: bool


def _hash2(a: int, b: int, seed: int) -> float:
    value = (a*0x9E3779B1 ^ b*0x85EBCA77 ^ seed*0xC2B2AE3D) & 0xFFFFFFFF
    value = ((value ^ (value >> 16))*0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15))*0x846CA68B) & 0xFFFFFFFF
    return ((value ^ (value >> 16)) & 0xFFFFFF)/0xFFFFFF


def district_layout(columns: int, rows: int, spacing: float,
                    seed: int) -> tuple[DistrictBuilding, ...]:
    """Walkable streets, two roof vocabularies, no random global state."""
    if not 2 <= columns <= 24 or not 2 <= rows <= 24 or not .03 <= spacing <= .18:
        raise ValueError("City district grid is out of its supported miniature range")
    buildings = []
    for row in range(rows):
        for col in range(columns):
            # Two deliberately wider streets reveal the block structure in
            # a macro shot instead of making a single solid box collection.
            if (columns >= 6 and col == columns//2 and row % 3 != 0):
                continue
            if rows >= 6 and row == rows//2 and col % 4 != 0:
                continue
            variance = _hash2(col, row, seed)
            other = _hash2(row+17, col+11, seed)
            buildings.append(DistrictBuilding(
                (col-(columns-1)/2)*spacing + (other-.5)*spacing*.12,
                (row-(rows-1)/2)*spacing + (variance-.5)*spacing*.12,
                spacing*(.57+.23*variance), spacing*(.55+.24*other),
                .055+.105*(.35*variance+.65*other),
                min(2, int(_hash2(row+6, col+5, seed+7)*3)),
                _hash2(col+7, row+21, seed+3) > .53,
            ))
    return tuple(buildings)


@dataclass
class MeshSketch:
    """A batched watertight hard-surface mesh, with optional thin foliage."""

    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)

    def _append(self, vertices: list[tuple[float, float, float]],
                faces: list[tuple[int, ...]]) -> None:
        offset = len(self.vertices)
        self.vertices.extend(vertices)
        self.faces.extend(tuple(index+offset for index in face) for face in faces)

    def box(self, x: float, y: float, base: float, width: float, depth: float,
            height: float) -> None:
        self.tapered_prism(x, y, base, width, depth, height, taper=1.0)

    def tapered_prism(self, x: float, y: float, base: float, width: float,
                      depth: float, height: float, *, taper: float = .80) -> None:
        if min(width, depth, height, taper) <= 0:
            raise ValueError("A miniature masonry volume requires positive dimensions")
        vertices = []
        for z, factor in ((base, 1.0), (base+height, taper)):
            vertices.extend((x+sx*width*.5*factor, y+sy*depth*.5*factor, z)
                            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)))
        self._append(vertices, [(3, 2, 1, 0), (4, 5, 6, 7),
                                (0, 1, 5, 4), (1, 2, 6, 5),
                                (2, 3, 7, 6), (3, 0, 4, 7)])

    def lathe(self, x: float, y: float, base: float,
              profile: tuple[tuple[float, float], ...], *, sides: int = 12) -> None:
        """Profile as (height above base, radius); ideal for columns and domes."""
        if sides < 3 or len(profile) < 2 or any(radius <= 0 for _, radius in profile):
            raise ValueError("Lathe requires at least two positive-radius rings")
        verts = [(x+radius*math.cos(2*math.pi*i/sides),
                  y+radius*math.sin(2*math.pi*i/sides), base+h)
                 for h, radius in profile for i in range(sides)]
        faces = [tuple(reversed(range(sides)))]
        for ring in range(len(profile)-1):
            a, b = ring*sides, (ring+1)*sides
            faces.extend((a+i, a+(i+1)%sides, b+(i+1)%sides, b+i)
                         for i in range(sides))
        top = (len(profile)-1)*sides
        faces.append(tuple(top+i for i in range(sides)))
        self._append(verts, faces)

    def hemisphere(self, x: float, y: float, base: float, radius: float,
                   *, sides: int = 16) -> None:
        profile = tuple((radius*math.sin(k*math.pi/12),
                         max(.001, radius*math.cos(k*math.pi/12)))
                        for k in range(7))
        self.lathe(x, y, base, profile, sides=sides)

    def obelisk(self, x: float, y: float, base: float, width: float,
                height: float) -> None:
        self.tapered_prism(x, y, base, width, width, height*.83, taper=.65)
        self.tapered_prism(x, y, base+height*.83, width*.65,
                           width*.65, height*.17, taper=.012)

    def palm(self, x: float, y: float, base: float, height: float,
             *, leaf_reach: float = .095, trunk_only: bool = False,
             leaf_only: bool = False) -> None:
        """Tapered, slightly crooked stem and individually arched palm leaves."""
        if not leaf_only:
            self.lathe(x, y, base, ((0, .013), (height*.35, .011),
                                    (height*.72, .010), (height, .006)), sides=7)
        if trunk_only:
            return
        for leaf in range(9):
            angle = leaf*2*math.pi/9 + .14*math.sin(leaf*2.3)
            ux, uy = math.cos(angle), math.sin(angle)
            vx, vy = -uy, ux
            verts = []
            for section in range(5):
                t = section/4
                reach = leaf_reach*t
                z = base+height + .045*math.sin(math.pi*t) - .067*t*t
                half = .017*(1-t)**.85+.0004
                verts.extend(((x+ux*reach+vx*half, y+uy*reach+vy*half, z),
                              (x+ux*reach-vx*half, y+uy*reach-vy*half, z)))
            self._append(verts, [(i*2, i*2+1, i*2+3, i*2+2) for i in range(4)])
            # Side leaflets make a pinnate date-palm silhouette in city zooms
            # instead of a nine-pointed geometric rosette seen from above.
            for section in (1, 2, 3):
                t = (section+.22)/4
                reach = leaf_reach*t
                cx, cy = x+ux*reach, y+uy*reach
                cz = base+height+.045*math.sin(math.pi*t)-.067*t*t
                for side in (-1, 1):
                    tip_reach = leaf_reach*(.32-.12*t)
                    vertices = [
                        (cx-ux*.006, cy-uy*.006, cz),
                        (cx+ux*.007, cy+uy*.007, cz),
                        (cx+ux*.014+vx*side*tip_reach,
                         cy+uy*.014+vy*side*tip_reach, cz-.017),
                    ]
                    self._append(vertices, [(0, 1, 2)])
