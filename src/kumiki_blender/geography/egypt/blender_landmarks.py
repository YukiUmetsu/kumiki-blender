"""Detailed, batched, editable architectural miniatures on the Egypt atlas."""

from __future__ import annotations

import math
from typing import Any

from ...blender_scene import _create_box
from .geometry import MeshSketch, district_layout
from .terrain import RIVER_CHANNELS, geo_to_xy, sample_terrain, xy_to_geo
from .blender_support import _attach_sketch, _catmull, _xy_ground


def _pyramid(bpy: Any, collection: Any, name: str,
             lon: float, lat: float, width: float, height: float, material: Any) -> Any:
    x, y = geo_to_xy(lon, lat)
    z = sample_terrain(lon, lat).height_m+.012
    stone = MeshSketch()
    # Twelve physically separate tapering courses with thin recessed joints.
    # A single primitive pyramid had no readable masonry in Giza close-ups.
    for course in range(12):
        base_fraction = course/12
        end_fraction = (course+1)/12
        lower = width*(1-base_fraction)
        upper = max(.001, width*(1-end_fraction))
        stone.tapered_prism(x, y, z+height*base_fraction, lower, lower,
                            height/12-.0023, taper=max(.008, upper/lower))
    return _attach_sketch(bpy, collection, name, stone, material, "giza-pyramid-miniature")


def _temple(bpy: Any, collection: Any, sandstone: Any,
            shadow_stone: Any, limestone: Any) -> list[Any]:
    """Miniature Luxor: battered pylons, capitals, twin obelisks, sphinx avenue."""
    x, y = geo_to_xy(32.73, 25.7)
    z = sample_terrain(32.73, 25.7).height_m+.025
    objects = []
    plinth = _create_box(bpy, collection, "Luxor / raised temple plinth", (.66, .48, .05),
                         (x, y, z+.015), shadow_stone)
    plinth["kumiki_part"] = "luxor-temple-miniature"
    objects.append(plinth)
    architecture, carved, avenue = MeshSketch(), MeshSketch(), MeshSketch()
    for side in (-1, 1):
        architecture.tapered_prism(x+side*.185, y-.155, z+.04,
                                   .215, .12, .205, taper=.75)
        # Real raised door jambs and lintel; the middle remains an open gate.
        architecture.box(x+side*.066, y-.155, z+.04, .024, .12, .16)
        carved.obelisk(x+side*.108, y-.297, z+.04, .036, .295)
    architecture.box(x, y+.205, z+.04, .56, .022, .12)
    for side in (-1, 1):
        architecture.box(x+side*.295, y+.075, z+.04, .023, .26, .115)
    for iy in range(4):
        for ix in (-2, -1, 1, 2):
            cx = x+ix*.09
            cy = y-.035+iy*.069
            architecture.lathe(cx, cy, z+.04,
                               ((0, .024), (.012, .025), (.026, .017),
                                (.149, .015), (.157, .024), (.172, .027)), sides=12)
            carved.box(cx, cy, z+.21, .059, .056, .016)
    # An avenue of ram-headed guardian glyphs gives the foreground scale.
    for j in range(4):
        for side in (-1, 1):
            ax, ay = x+side*.10, y-.405-j*.06
            gz = _xy_ground(ax, ay, .016)
            avenue.box(ax, ay, gz, .038, .048, .028)
            avenue.lathe(ax, ay-.025, gz+.025, ((0, .017), (.030, .015)), sides=8)
    objects.extend((
        _attach_sketch(bpy, collection, "Luxor / monumental enclosure and colonnade",
                       architecture, sandstone, "luxor-temple-miniature"),
        _attach_sketch(bpy, collection, "Luxor / obelisks and stone capitals",
                       carved, limestone, "luxor-temple-miniature"),
        _attach_sketch(bpy, collection, "Luxor / sphinx avenue",
                       avenue, sandstone, "luxor-temple-miniature"),
    ))
    return objects


def _add_city_district(bpy: Any, collection: Any, name: str, lon: float, lat: float,
                       offset: tuple[float, float], grid: tuple[int, int, float],
                       seed: int, walls: tuple[Any, ...], roof: Any,
                       windows: Any) -> tuple[list[Any], int]:
    """Street-spaced houses with parapets, stair hatches and dark window insets."""
    origin = geo_to_xy(lon, lat)
    sketches = [MeshSketch() for _ in walls]
    roofline, glazing = MeshSketch(), MeshSketch()
    count = 0
    for block in district_layout(*grid, seed):
        x, y = origin[0]+offset[0]+block.x, origin[1]+offset[1]+block.y
        landmark = ({"Cairo": (31.58, 30.13, .205, .155),
                     "Alexandria": (29.98, 31.14, .115, .085)}).get(name)
        if landmark:
            lx, ly = geo_to_xy(landmark[0], landmark[1])
            if abs(x-lx) < landmark[2] and abs(y-ly) < landmark[3]:
                continue
        if sample_terrain(*xy_to_geo(x, y)).kind in ("red-sea", "mediterranean", "lake-nasser"):
            continue
        floor = _xy_ground(x, y, .014)
        sketches[block.palette].box(x, y, floor, block.width, block.depth, block.height)
        top = floor+block.height
        # Roof parapets must form an actual perimeter in a close orbit.
        for side in (-1, 1):
            roofline.box(x+side*(block.width*.5-.003), y, top,
                         .006, block.depth, .012)
            roofline.box(x, y+side*(block.depth*.5-.003), top,
                         block.width, .006, .012)
        if block.rooftop:
            roofline.box(x+.008, y+.004, top+.003,
                         block.width*.32, block.depth*.40, .023)
        for side in (-1, 1):
            glazing.box(x+side*block.width*.22, y-block.depth*.5-.0013,
                        floor+block.height*.49,
                        block.width*.145, .0027, block.height*.20)
        count += 1
    objects = [_attach_sketch(bpy, collection, f"{name} / city masonry shade {shade}",
                              sketch, material, "city-architecture")
               for shade, (sketch, material) in enumerate(zip(sketches, walls))
               if sketch.faces]
    objects.extend((_attach_sketch(bpy, collection, f"{name} / parapet rooftops",
                                   roofline, roof, "city-roof-detail"),
                    _attach_sketch(bpy, collection, f"{name} / recessed window accents",
                                   glazing, windows, "city-facade-detail")))
    return objects, count


def _add_cairo_mosque(bpy: Any, collection: Any, limestone: Any,
                      patina: Any) -> list[Any]:
    """Citadel-inspired mosque silhouette: cascading domes and two minarets."""
    x, y = geo_to_xy(31.58, 30.13)
    z = _xy_ground(x, y, .018)
    masonry, domes = MeshSketch(), MeshSketch()
    masonry.box(x, y, z, .33, .27, .043)
    masonry.box(x, y, z+.043, .27, .225, .118)
    masonry.lathe(x, y, z+.161, ((0, .112), (.045, .104)), sides=12)
    domes.hemisphere(x, y, z+.206, .117)
    for side in (-1, 1):
        domes.hemisphere(x+side*.097, y-.059, z+.163, .053, sides=12)
        masonry.lathe(x+side*.155, y+.12, z+.04,
                      ((0, .026), (.14, .024), (.155, .034),
                       (.172, .019), (.28, .017), (.292, .024)), sides=12)
        domes.obelisk(x+side*.155, y+.12, z+.332, .036, .08)
    return [
        _attach_sketch(bpy, collection, "Cairo / citadel mosque walls and minarets",
                       masonry, limestone, "cairo-mosque-miniature"),
        _attach_sketch(bpy, collection, "Cairo / copper domes and spires",
                       domes, patina, "cairo-mosque-miniature"),
    ]


def _add_alexandria_harbor(bpy: Any, collection: Any, sandstone: Any,
                           patina: Any) -> list[Any]:
    """Tiered Pharos-inspired beacon, stone mole and harbor bollards."""
    x, y = geo_to_xy(29.98, 31.14)
    z = _xy_ground(x, y, .023)
    stone, crown = MeshSketch(), MeshSketch()
    stone.box(x, y, z, .30, .16, .034)
    stone.tapered_prism(x, y, z+.034, .155, .145, .18, taper=.79)
    stone.lathe(x, y, z+.214, ((0, .064), (.092, .048)), sides=8)
    stone.lathe(x, y, z+.306, ((0, .034), (.071, .033)), sides=10)
    crown.hemisphere(x, y, z+.377, .044, sides=12)
    crown.obelisk(x, y, z+.419, .022, .06)
    for side in (-1, 1):
        for j in range(3):
            stone.box(x+side*.12, y-.047+j*.043, z+.035,
                      .016, .016, .014)
    return [
        _attach_sketch(bpy, collection, "Alexandria / Pharos beacon and harbor stonework",
                       stone, sandstone, "alexandria-harbor-miniature"),
        _attach_sketch(bpy, collection, "Alexandria / beacon copper crown",
                       crown, patina, "alexandria-harbor-miniature"),
    ]


def _add_aswan_dam(bpy: Any, collection: Any, concrete: Any,
                   shadow_stone: Any) -> list[Any]:
    """A legible reservoir barrage, with spillway bays and downstream wall."""
    x, y = geo_to_xy(32.90, 24.13)
    base = .089  # structure crosses submerged surface at Lake Nasser's head
    wall, gates = MeshSketch(), MeshSketch()
    wall.tapered_prism(x, y, base, .49, .086, .115, taper=.76)
    wall.box(x, y, base+.112, .50, .067, .012)
    for i in range(6):
        gx = x+(i-2.5)*.073
        gates.box(gx, y-.046, base+.009, .013, .011, .077)
        gates.box(gx, y+.041, base+.025, .013, .011, .066)
    return [
        _attach_sketch(bpy, collection, "Aswan / high dam embankment and road",
                       wall, concrete, "aswan-dam-miniature"),
        _attach_sketch(bpy, collection, "Aswan / spillway bays and buttresses",
                       gates, shadow_stone, "aswan-dam-miniature"),
    ]


def _add_sharm_marina(bpy: Any, collection: Any, sandstone: Any,
                      patina: Any) -> list[Any]:
    """Small waterfront resort silhouettes and jetties on Sinai's southern tip."""
    x, y = geo_to_xy(34.29, 27.98)
    stone, blue = MeshSketch(), MeshSketch()
    for j in range(4):
        sx, sy = x+(j-1.5)*.087, y+.04+(j%2)*.047
        base = _xy_ground(sx, sy, .015)
        stone.box(sx, sy, base, .065, .047, .074+(j%2)*.029)
        blue.hemisphere(sx, sy, base+.074+(j%2)*.029, .029, sides=10)
    return [
        _attach_sketch(bpy, collection, "Sharm / waterfront resort miniatures",
                       stone, sandstone, "sharm-marina-miniature"),
        _attach_sketch(bpy, collection, "Sharm / turquoise roof domes",
                       blue, patina, "sharm-marina-miniature"),
    ]


def _add_nile_fields(bpy: Any, collection: Any,
                     crops: tuple[Any, ...]) -> tuple[list[Any], int]:
    """Hundreds of actual irrigated field surfaces with visible parcel seams."""
    sketches = [MeshSketch() for _ in crops]
    route = _catmull(RIVER_CHANNELS["main-nile"], steps=9)
    count = 0
    for index in range(4, len(route)-4, 3):
        # No cultivated parcels inside the northern dense Cairo urban center.
        lon, lat = route[index]
        if lat > 29.55 or lat < 24.3:
            continue
        before, after = geo_to_xy(*route[index-2]), geo_to_xy(*route[index+2])
        dx, dy = after[0]-before[0], after[1]-before[1]
        mag = max(1e-6, math.hypot(dx, dy))
        tx, ty = dx/mag, dy/mag
        nx, ny = -ty, tx
        cx, cy = geo_to_xy(lon, lat)
        for side in (-1, 1):
            for band in range(2):
                offset = side*(.095+.064*band)
                px, py = cx+nx*offset, cy+ny*offset
                width, length = .051, .079
                corners = [(px+sign1*tx*length*.5+sign2*nx*width*.5,
                            py+sign1*ty*length*.5+sign2*ny*width*.5)
                           for sign1, sign2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                vertices = [(vx, vy, _xy_ground(vx, vy, .017)) for vx, vy in corners]
                # Delta-style geometry, not a green decal: every field is a
                # separate four-corner piece draped over the real terrain.
                palette = (index//3+side+band*3) % len(crops)
                sketches[palette]._append(vertices, [(0, 1, 2, 3)])
                count += 1
    objects = [_attach_sketch(bpy, collection, f"Nile / cultivated parcel type {i}",
                              sketch, material, "agricultural-parcel")
               for i, (sketch, material) in enumerate(zip(sketches, crops)) if sketch.faces]
    return objects, count


def _add_palm_groves(bpy: Any, collection: Any, bark: Any,
                     fronds: Any) -> tuple[list[Any], int]:
    """Palm oases at Siwa, Faiyum and the Nile settlements, not uniform poles."""
    trunks, leaves = MeshSketch(), MeshSketch()
    count = 0
    centers = ((25.62, 29.15, 15), (30.76, 29.34, 13),
               (31.20, 30.54, 18), (32.24, 25.77, 18), (32.74, 24.44, 13))
    for patch, (lon, lat, amount) in enumerate(centers):
        px, py = geo_to_xy(lon, lat)
        for i in range(amount):
            theta = i*2.3999632297
            radius = .035 + .205*math.sqrt((i+.5)/amount)
            x = px+math.cos(theta)*radius
            y = py+math.sin(theta)*radius*.70
            sample = sample_terrain(*xy_to_geo(x, y))
            if sample.kind in ("mediterranean", "red-sea", "lake-nasser", "lake-qarun"):
                continue
            base = sample.height_m+.009
            height = .105+.065*((i+patch*3)%7)/6
            trunks.palm(x, y, base, height, trunk_only=True)
            leaves.palm(x, y, base, height, leaf_only=True,
                        leaf_reach=.073+.035*(i%5)/4)
            count += 1
    return [
        _attach_sketch(bpy, collection, "Egypt / individual palm trunks",
                       trunks, bark, "palm-trunk"),
        _attach_sketch(bpy, collection, "Egypt / curved palm crowns",
                       leaves, fronds, "palm-frond"),
    ], count
