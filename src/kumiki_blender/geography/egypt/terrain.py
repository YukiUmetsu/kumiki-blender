"""Deterministic, source-independent relief atlas inspired by an Egypt map diorama.

This is illustrated geography, not a DEM or a claim of survey-grade borders.
Coordinates are deliberately curated and simplified for cinematic readability.
No Blender import or third-party package is needed for planning and tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


WEST, EAST, SOUTH, NORTH = 24.0, 38.0, 21.7, 32.6
WIDTH, DEPTH = 16.0, 11.4


@dataclass(frozen=True)
class EgyptMapSpec:
    # Finer base mesh reduces staircase coastlines in city-level macro shots;
    # the high preset approaches the bounded 300k-cell geometry budget.
    columns: int = 480
    rows: int = 352

    def __post_init__(self) -> None:
        if (type(self.columns) is not int or type(self.rows) is not int
                or not 12 <= self.columns <= 700 or not 12 <= self.rows <= 700
                or self.columns * self.rows > 300_000):
            raise ValueError("Egypt terrain mesh must use 12–700 divisions/axis and <=300k cells")


@dataclass(frozen=True)
class TerrainSample:
    height_m: float
    kind: str
    rgb: tuple[float, float, float]
    vegetation: float


@dataclass(frozen=True)
class TerrainMesh:
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int, int], ...]
    face_kinds: tuple[str, ...]


@dataclass(frozen=True)
class MapLabel:
    text: str
    lon: float
    lat: float
    size: float
    angle: float = 0.0
    style: str = "ink"


# Paths are longitude, latitude. The main Nile and delta arms follow an
# intentionally illustrative northbound course, not a navigable waterway file.
RIVER_CHANNELS: dict[str, tuple[tuple[float, float], ...]] = {
    "main-nile": (
        (32.90, 24.09), (32.88, 24.34), (32.80, 24.57), (32.74, 24.78),
        (32.76, 25.06), (32.75, 25.31), (32.68, 25.54), (32.65, 25.70),
        (32.58, 25.91), (32.70, 26.13), (32.72, 26.34), (32.57, 26.57),
        (32.38, 26.71), (32.19, 26.94), (31.91, 27.16), (31.72, 27.48),
        (31.43, 27.72), (31.22, 28.09), (31.23, 28.45), (31.08, 28.73),
        (31.07, 29.04), (31.19, 29.33), (31.24, 29.62), (31.25, 30.05),
    ),
    "rosetta": (
        (31.25, 30.05), (31.09, 30.28), (30.87, 30.51), (30.69, 30.75),
        (30.56, 31.04), (30.39, 31.20), (30.37, 31.47),
    ),
    "damietta": (
        (31.25, 30.05), (31.46, 30.30), (31.58, 30.58), (31.70, 30.83),
        (31.75, 31.06), (31.85, 31.24), (31.85, 31.49),
    ),
    "delta-distributary-west": ((31.02, 30.24), (30.73, 30.42), (30.43, 30.61), (30.22, 31.02)),
    "delta-distributary-east": ((31.48, 30.36), (31.92, 30.53), (32.16, 30.80), (32.34, 31.10)),
    "suez-canal": ((32.35, 31.26), (32.31, 31.10), (32.32, 30.84),
                   (32.35, 30.62), (32.44, 30.39), (32.55, 30.08)),
}


BORDER_LINES: dict[str, tuple[tuple[float, float], ...]] = {
    "libya-egypt": ((25.0, 22.05), (25.0, 26.0), (25.0, 29.4), (25.0, 31.3)),
    "sudan-egypt": ((25.0, 22.0), (28.0, 22.0), (31.6, 22.0)),
    "sinai-east": ((34.25, 31.24), (34.3, 30.9), (34.5, 30.30),
                   (34.65, 29.95), (34.87, 29.52)),
}


LAKE_NASSER = (
    (32.91, 24.06), (32.91, 23.78), (32.87, 23.60), (32.84, 23.38),
    (32.95, 23.20), (32.87, 23.02), (32.94, 22.75), (32.81, 22.56),
    (32.75, 22.33), (32.62, 22.06),
)

LAKE_NASSER_ARMS = (
    ((31.90, 23.09), (32.50, 22.97), (32.90, 23.13)),
    ((33.50, 22.56), (33.05, 22.66), (32.80, 22.65)),
)


CITY_MARKERS: dict[str, tuple[float, float]] = {
    "Alexandria": (29.92, 31.20),
    "Cairo": (31.24, 30.04),
    "Luxor": (32.65, 25.70),
    "Aswan": (32.90, 24.09),
    "Sharm El-Sheikh": (34.33, 27.92),
}


LABELS: tuple[MapLabel, ...] = (
    MapLabel("Mediterranean Sea", 29.8, 32.03, .57, style="sea"),
    MapLabel("Delta", 31.05, 30.99, .37, style="sage"),
    MapLabel("Western Desert", 28.30, 26.65, .66, angle=24),
    # Offset this caption to the north-west so oblique close-ups reveal the
    # depression's escarpment rather than covering its lowest saltpan.
    MapLabel("Qattara Depression", 27.88, 30.08, .32, angle=13),
    MapLabel("Eastern Desert", 33.57, 26.45, .52, angle=-35),
    MapLabel("Sinai", 33.70, 29.67, .47, angle=-7),
    MapLabel("Red Sea", 36.50, 25.95, .65, angle=-57, style="sea"),
    MapLabel("Nile River", 31.25, 27.45, .32, angle=-51, style="river"),
    MapLabel("Lake Nasser", 31.91, 22.68, .29, angle=0, style="river"),
    MapLabel("Suez Canal", 32.65, 30.89, .22, style="ink"),
    MapLabel("LIBYA", 24.67, 26.85, .47, angle=89, style="border"),
)


# In geographic space: visible eastern branch sea and the two gulfs.
RED_SEA_POLYGONS = (
    (
        (34.42, 27.96), (34.55, 27.72), (34.67, 27.36), (34.85, 26.94),
        (34.94, 26.34), (35.03, 25.70), (35.21, 24.85), (35.43, 24.17),
        (35.65, 23.38), (35.98, 22.35), (36.22, 21.65), (38.2, 21.65),
        (38.2, 22.65), (38.2, 23.42), (38.2, 24.17), (38.12, 24.94),
        (38.02, 25.75), (37.85, 26.55), (37.15, 27.39), (34.69, 28.05),
    ),
    # Gulf of Suez, western finger with its long, tapered head.
    (
        (34.52, 27.97), (34.08, 28.23), (33.60, 28.50), (33.24, 28.90),
        (32.82, 29.31), (32.68, 29.69), (32.56, 30.03), (32.60, 30.16),
        (32.78, 29.98), (33.13, 29.53), (33.42, 29.04), (33.84, 28.68),
        (34.29, 28.28), (34.67, 28.07),
    ),
    # Gulf of Aqaba, eastern, narrow arm enclosing the Sinai triangle.
    (
        (34.47, 27.94), (34.65, 28.18), (34.78, 28.50), (34.89, 28.85),
        (34.95, 29.27), (35.02, 29.60), (35.18, 29.64), (35.15, 29.23),
        (35.07, 28.83), (34.98, 28.45), (34.83, 28.07), (34.65, 27.87),
    ),
)


COASTLINE = (
    (24.0, 31.55), (25.1, 31.49), (26.2, 31.39), (27.4, 31.29),
    (28.45, 31.20), (29.32, 31.22), (29.90, 31.22), (30.33, 31.38),
    (30.76, 31.51), (31.22, 31.61), (31.75, 31.46), (32.24, 31.25),
    (32.73, 31.16), (33.4, 31.12), (34.18, 31.12), (34.65, 31.29),
    (35.05, 31.54), (35.36, 31.95), (35.6, 32.36), (38.0, 33.05),
)


def _validate_geo(lon: float, lat: float) -> None:
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in (lon, lat)):
        raise ValueError("Longitude and latitude must be finite numbers")
    if not WEST <= lon <= EAST or not SOUTH <= lat <= NORTH:
        raise ValueError("Geographic point must be inside the illustration extent")


def geo_to_xy(lon: float, lat: float) -> tuple[float, float]:
    _validate_geo(lon, lat)
    return ((lon-WEST)/(EAST-WEST)-.5)*WIDTH, ((lat-SOUTH)/(NORTH-SOUTH)-.5)*DEPTH


def xy_to_geo(x: float, y: float) -> tuple[float, float]:
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y))
            or not -WIDTH/2 <= x <= WIDTH/2 or not -DEPTH/2 <= y <= DEPTH/2):
        raise ValueError("Map XY must be finite and inside the map footprint")
    return (WEST+(x/WIDTH+.5)*(EAST-WEST), SOUTH+(y/DEPTH+.5)*(NORTH-SOUTH))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b-a)*t


def _smooth(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t*t*(3-2*t)


def _gaussian(x: float, center: float, radius: float) -> float:
    return math.exp(-((x-center)/radius)**2)


def _hashed(ix: int, iy: int) -> float:
    n = (ix*374761393 + iy*668265263 + 0x9E3779B9) & 0xFFFFFFFF
    n = ((n ^ (n >> 13))*1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFF) / 16777215


def _noise(x: float, y: float) -> float:
    ix, iy = math.floor(x), math.floor(y)
    u, v = _smooth(x-ix), _smooth(y-iy)
    return _lerp(_lerp(_hashed(ix, iy), _hashed(ix+1, iy), u),
                 _lerp(_hashed(ix, iy+1), _hashed(ix+1, iy+1), u), v)


def _fractal(x: float, y: float) -> float:
    return (.52*_noise(x, y) + .28*_noise(x*2.13+9, y*2.13-7)
            + .14*_noise(x*4.39-3, y*4.39+13)
            + .06*_noise(x*8.74+19, y*8.74-17))


def _distance_to_path(lon: float, lat: float, path: tuple[tuple[float, float], ...]) -> float:
    best = float("inf")
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        ax, bx = ax*.86, bx*.86
        px = lon*.86
        dx, dy = bx-ax, by-ay
        t = max(0.0, min(1.0, ((px-ax)*dx+(lat-ay)*dy)/(dx*dx+dy*dy)))
        best = min(best, math.hypot(px-(ax+t*dx), lat-(ay+t*dy)))
    return best


def _inside(lon: float, lat: float, polygon: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    j = len(polygon)-1
    for i, (x, y) in enumerate(polygon):
        oldx, oldy = polygon[j]
        if (y > lat) != (oldy > lat) and lon < (oldx-x)*(lat-y)/(oldy-y)+x:
            inside = not inside
        j = i
    return inside


def _coast_latitude(lon: float) -> float:
    for a, b in zip(COASTLINE, COASTLINE[1:]):
        if a[0] <= lon <= b[0]:
            return _lerp(a[1], b[1], (lon-a[0])/(b[0]-a[0]))
    return COASTLINE[-1][1]


def _lake_width(lat: float) -> float:
    return .14 + .22*_gaussian(lat, 23.18, .58) + .12*_gaussian(lat, 22.51, .28)


def _water_kind(lon: float, lat: float) -> str | None:
    if lat > _coast_latitude(lon):
        return "mediterranean"
    if any(_inside(lon, lat, shape) for shape in RED_SEA_POLYGONS):
        return "red-sea"
    if ((lon-30.64)/.23)**2 + ((lat-29.49)/.085)**2 < 1.0:
        return "lake-qarun"
    irregular_bank = 1+.17*math.sin(lon*38+lat*24)+.10*math.sin(lon*59-lat*43)
    if (21.95 < lat < 24.15 and
            _distance_to_path(lon, lat, LAKE_NASSER) < _lake_width(lat)*irregular_bank):
        return "lake-nasser"
    # Distinct arms and drowned side wadis give Nasser its long, scalloped banks.
    if ((22.85 < lat < 23.25 and
         _distance_to_path(lon, lat, LAKE_NASSER_ARMS[0]) < .085)
            or (22.43 < lat < 22.80 and
                _distance_to_path(lon, lat, LAKE_NASSER_ARMS[1]) < .08)):
        return "lake-nasser"
    return None


def sample_terrain(lon: float, lat: float) -> TerrainSample:
    """Geographic masks, real relief deformation and texture color at a point."""
    _validate_geo(lon, lat)
    water = _water_kind(lon, lat)
    if water:
        d = _fractal(lon*5.1, lat*5.1)
        if water == "mediterranean":
            rgb = (.035+d*.028, .150+d*.048, .255+d*.060)
        elif water == "red-sea":
            rgb = (.035+d*.036, .176+d*.066, .280+d*.085)
        elif water == "lake-nasser":
            rgb = (.045+d*.050, .245+d*.070, .340+d*.080)
        else:
            rgb = (.075+d*.040, .29+d*.065, .30+d*.060)
        return TerrainSample(.075 if water.startswith("lake") else .060, water, rgb, 0)

    main_distance = _distance_to_path(lon, lat, RIVER_CHANNELS["main-nile"])
    delta_distance = min(_distance_to_path(lon, lat, RIVER_CHANNELS[arm])
                         for arm in ("rosetta", "damietta"))
    # Nile valley is a narrow, irregular, green ribbon; delta widens toward sea.
    river_green = math.exp(-(main_distance/.18)**2)
    delta_width = .16 + 1.22*_smooth((lat-30.0)/1.5)
    delta_green = (_gaussian(lon, 31.22, max(.15, delta_width))
                   * _smooth((lat-29.79)/.35)) if lat > 29.79 else 0.0
    faiyum = .68*_gaussian(lon, 30.63, .39)*_gaussian(lat, 29.52, .25)
    vegetation = max(river_green, delta_green*.96,
                     .5*math.exp(-(delta_distance/.18)**2), faiyum)
    # The Sinai highlands and Red Sea Hills are creased by oblique ridges,
    # while broad western hammadas and dune seas remain much lower.
    sinai = (_gaussian(lon, 33.78, 1.06)*_gaussian(lat, 28.78, 1.45))
    east_hills = (_gaussian(lon, 34.3+.18*math.sin(lat*2), .96)
                  * _gaussian(lat, 25.50, 3.20))
    east_foothills = (_gaussian(lon, 33.35, 1.22)*_gaussian(lat, 25.8, 2.7))
    southwest = (_gaussian(lon, 26.0, 1.2)*_gaussian(lat, 23.2, 2.4))
    western_escarpments = (
        _gaussian(lon, 28.5, 2.1)*_gaussian(lat, 27.4, .54)
        + .65*_gaussian(lon, 29.3, 1.55)*_gaussian(lat, 25.1, .65))
    western_scarps = (
        .25*_gaussian(lon, 28.4, 2.8)
        * _gaussian(lat, 27.8+.25*math.sin((lon-28.5)*1.15), .32)
        + .21*_gaussian(lon, 29.0, 2.15)
        * _gaussian(lat, 25.6+.25*math.sin((lon-29)*1.30), .36)
        + .12*_gaussian(lon, 26.3, 1.45)
        * _gaussian(lat, 23.5+.18*math.sin((lon-26)*1.3), .36))
    qattara = (_gaussian(lon, 28.7, 1.35)*_gaussian(lat, 29.43, .72))
    rim = (_gaussian(lon, 28.8, 1.68)*_gaussian(lat, 29.45, 1.02)-qattara)
    wavy = _fractal(lon*.9, lat*1.2)
    grain = _fractal(lon*7.6, lat*8.0)
    folded = 1.0-abs(2*_fractal(lon*5.1+lat*.8, lat*3.7-lon*.48)-1.0)
    # Keep the Sinai massif standing proud *between* its two narrow gulfs.
    # In the first atlas the broad coastal-shelf fade erased most of Sinai's
    # relief, even though the Eastern Desert further south remained rugged.
    ridge = (sinai*.91 + east_hills*.86 + east_foothills*.28)*(.55+.49*folded)
    sinai_spine = .18*sinai*_gaussian(lon, 33.76, .68)
    plateau = (.10*(wavy-.36) + .11*southwest + .34*max(0, rim)
               + .18*western_escarpments*(.40+.60*folded)
               + western_scarps*(.75+.28*folded))
    # Break up dune alignments with multiscale phase warping; avoid the
    # equally spaced, manufactured corrugations of a single sine wave.
    warped = _fractal(lon*2.6+8, lat*2.9-3)
    dunes = (.013*math.sin(lon*12.3+lat*2.7+10*warped)
             * (1-vegetation)**2 * (1-.6*western_escarpments))
    # Deliberate vertical exaggeration is essential to an oblique desktop
    # diorama: the map is ~16 units wide, so unscaled 0.2-unit plateaus read
    # as a flat printed sheet in a 3/4 camera view. Amplify geological forms
    # without raising the fertile Nile floor or open-water elevations.
    height = max(.085, .246 + 1.38*plateau + 1.27*(ridge+sinai_spine)
                 + .125*(grain-.5) + 1.45*dunes - .245*qattara)
    # Coastal alluvial shelves soften mountainous-to-sea transitions. A
    # pixel-sharp abrupt switch created artificial needle cliffs in oblique
    # inspection shots; the erosion transition must exist in mesh geometry.
    if lon > 31.5 and lat < 30.5:
        coast_dist = min(_distance_to_path(lon, lat, polygon+(polygon[0],))
                         for polygon in RED_SEA_POLYGONS)
        # Narrow Sinai needs a tighter shelf than the broad Red Sea hills or
        # its central peaks disappear completely between the two coastlines.
        shelf_width = .30 if lat > 27.85 else .55
        shelf = _smooth(coast_dist/shelf_width)
        height = _lerp(.095, height, shelf)
    if 31.5 < lon < 34.0 and lat < 24.35:
        bank_noise = 1+.17*math.sin(lon*38+lat*24)+.10*math.sin(lon*59-lat*43)
        bank_edge = _distance_to_path(lon, lat, LAKE_NASSER)-_lake_width(lat)*bank_noise
        # The narrow drowned side wadis once formed sawtooth banks because
        # only the reservoir's main centerline received an erosion shelf.
        if 22.75 < lat < 23.34:
            bank_edge = min(bank_edge, _distance_to_path(lon, lat, LAKE_NASSER_ARMS[0])-.085)
        if 22.34 < lat < 22.91:
            bank_edge = min(bank_edge, _distance_to_path(lon, lat, LAKE_NASSER_ARMS[1])-.080)
        height = _lerp(.09, height, _smooth(max(0, bank_edge)/.56))
    height = _lerp(height, .15+.016*(grain-.5), min(.93, vegetation*.88))
    height = _lerp(height, .13, .65*_smooth((lat-30.8)/.9))
    # Compress only extreme summits continuously: without this the folded
    # noise creates needle-like 2.5-unit spikes above a 16-unit-wide map.
    # This preserves escarpment and Nile-valley height differences while
    # keeping the silhouette geological instead of resembling stalagmites.
    if height > .32:
        height = .32 + 1.34*math.tanh((height-.32)/1.34)

    if vegetation > .45:
        green = (.19+.095*grain, .29+.18*grain, .13+.07*grain)
        sand = (.66+.15*grain, .52+.16*grain, .35+.13*grain)
        rgb = tuple(_lerp(a, b, min(1.0, vegetation*.95)) for a, b in zip(sand, green))
        kind = "delta" if lat >= 30.15 else "nile"
    else:
        warm = min(1.0, .77*sinai+.83*east_hills+.33*east_foothills
                   +.50*southwest+1.0*western_scarps+.65*max(0, rim))
        dune_bright = .11*_fractal(lon*3.2, lat*4.1)
        sand = (.82+dune_bright, .69+dune_bright*.8, .49+dune_bright*.65)
        rock = (.62+.08*grain, .35+.09*grain, .26+.07*grain)
        rgb = tuple(_lerp(a, b, warm*.9) for a, b in zip(sand, rock))
        rgb = tuple(_lerp(a, b, .45*qattara)
                    for a, b in zip(rgb, (.91, .85, .72)))
        kind = "highlands" if max(sinai, east_hills) > .45 else "desert"
    return TerrainSample(height, kind, tuple(min(1.0, max(0.0, c)) for c in rgb), vegetation)


def terrain_mesh(spec: EgyptMapSpec = EgyptMapSpec()) -> TerrainMesh:
    """Shared shore vertices avoid both cracks and water-shaded mountain spikes.

    Classify cells before their four corners. A cell that is painted water
    must actually be low water geometry, even when a corner's land sampling
    falls just outside an irregular coastline polygon. Adjacent land slopes
    smoothly down into the same shoreline vertex instead of opening a crack.
    """
    vertices = []
    faces = []
    kinds = []
    water_kinds = {"mediterranean", "red-sea", "lake-nasser", "lake-qarun"}
    for row in range(spec.rows):
        lat = _lerp(SOUTH, NORTH, (row+.5)/spec.rows)
        for col in range(spec.columns):
            lon = _lerp(WEST, EAST, (col+.5)/spec.columns)
            kinds.append(sample_terrain(lon, lat).kind)
    for row in range(spec.rows+1):
        lat = _lerp(SOUTH, NORTH, row/spec.rows)
        for col in range(spec.columns+1):
            lon = _lerp(WEST, EAST, col/spec.columns)
            x, y = geo_to_xy(lon, lat)
            height = sample_terrain(lon, lat).height_m
            if any(kinds[near_row*spec.columns+near_col] in water_kinds
                   for near_row in (row-1, row) if 0 <= near_row < spec.rows
                   for near_col in (col-1, col) if 0 <= near_col < spec.columns):
                height = min(height, .084)
            vertices.append((x, y, height))
    stride = spec.columns+1
    for row in range(spec.rows):
        for col in range(spec.columns):
            a = row*stride+col
            faces.append((a, a+1, a+stride+1, a+stride))
    return TerrainMesh(tuple(vertices), tuple(faces), tuple(kinds))
