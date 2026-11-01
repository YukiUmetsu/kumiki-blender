"""Download GSI elevation tiles and cache a self-contained Fuji heightmap.

Run with the host Python + Pillow. No network access is required afterward by
the Blender build step. Data: Geospatial Information Authority of Japan (GSI).
"""

from __future__ import annotations

from array import array
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import math
from pathlib import Path
import sys
import urllib.error
import urllib.request

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from kumiki_blender.geography.fuji.dem import FujiDem, FujiSpec, load_dem, save_dem  # noqa: E402
from kumiki_blender.geography.fuji.patches import (  # noqa: E402
    FujiDetailPatch, FujiPatchSpec, _base_sample, dem_fingerprint, save_patch,
)


BASE_ZOOM = 13
DETAIL_ZOOM = 15
TILE = 256
HOST = "https://cyberjapandata.gsi.go.jp/xyz"
SOURCES = ("dem5a_png", "dem5b_png", "dem5c_png", "dem_png")
NO_DATA = 128 << 16


def mercator_xy(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    return ((lon+180)/360*(1 << zoom)*TILE,
            (1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*(1 << zoom)*TILE)


def unpack_height(rgb: tuple[int, int, int]) -> float | None:
    value = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
    if value == NO_DATA:
        return None
    if value >= 1 << 23:
        value -= 1 << 24
    return value * .01


def fetch_tile(zoom: int, x: int, y: int) -> list[float | None]:
    """Select the highest-quality available cell, not merely the first file."""
    pixels: list[float | None] = [None] * (TILE*TILE)
    for source in SOURCES:
        url = f"{HOST}/{source}/{zoom}/{x}/{y}.png"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "kumiki-blender-fuji-terrain/1.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
        with Image.open(BytesIO(payload)) as image:
            rgb = image.convert("RGB")
            if rgb.size != (TILE, TILE):
                raise ValueError(f"Unexpected DEM tile dimensions at {url}")
            data = rgb.tobytes()
            for i, color in enumerate(zip(data[::3], data[1::3], data[2::3])):
                if pixels[i] is None:
                    pixels[i] = unpack_height(color)
        if None not in pixels:
            break
    return pixels


def _tile_bounds(lon_min: float, lon_max: float, lat_min: float, lat_max: float,
                 zoom: int) -> list[tuple[int, int]]:
    west, north = mercator_xy(lon_min, lat_max, zoom)
    east, south = mercator_xy(lon_max, lat_min, zoom)
    tx0, tx1 = int(west//TILE), int(east//TILE)
    ty0, ty1 = int(north//TILE), int(south//TILE)
    # Bilinear sampling can require one more pixel to the east or south.
    return [(x, y) for y in range(ty0, ty1+2) for x in range(tx0, tx1+2)]


def _load_tiles(coords: list[tuple[int,int]], zoom: int) -> dict[tuple[int,int], list[float | None]]:
    print(f"Retrieving {len(coords)} GSI elevation tiles at zoom {zoom}", flush=True)
    # Four concurrent requests keep the GSI public infrastructure load modest.
    with ThreadPoolExecutor(max_workers=4) as pool:
        tiles = dict(zip(coords, pool.map(lambda xy: fetch_tile(zoom,*xy), coords)))
    return tiles

def sample_tiles(tiles: dict[tuple[int,int], list[float | None]], zoom: int,
                 lon: float, lat: float) -> float | None:
    def sample_pixel(px: int, py: int) -> float | None:
        tile = tiles.get((px//TILE, py//TILE))
        return None if tile is None else tile[(py%TILE)*TILE+(px%TILE)]

    px, py = mercator_xy(lon, lat, zoom)
    x, y = int(math.floor(px)), int(math.floor(py))
    fx, fy = px-x, py-y
    corners = ((sample_pixel(x,y), (1-fx)*(1-fy)),
               (sample_pixel(x+1,y), fx*(1-fy)),
               (sample_pixel(x,y+1), (1-fx)*fy),
               (sample_pixel(x+1,y+1), fx*fy))
    valid = [(h, w) for h, w in corners if h is not None]
    return sum(h*w for h, w in valid)/sum(w for _,w in valid) if valid else None


def prepare(path: Path, *, spec: FujiSpec = FujiSpec()) -> FujiDem:
    base = _load_tiles(_tile_bounds(spec.longitude_min, spec.longitude_max,
                                    spec.latitude_min, spec.latitude_max,
                                    BASE_ZOOM), BASE_ZOOM)
    # Native GSI detail within key inspection areas, blended to avoid seams.
    # The first Hōei crater floor is at approximately 138.7451E, 35.34098N.
    patches = ((138.7274, 35.3606, 1.65), (138.7451, 35.34098, 1.35))
    high_coords: set[tuple[int,int]] = set()
    for lon,lat,radius in patches:
        dx = radius/(111.320*math.cos(math.radians(lat)))
        dy = radius/111.132
        high_coords.update(_tile_bounds(lon-dx, lon+dx, lat-dy, lat+dy, DETAIL_ZOOM))
    high = _load_tiles(sorted(high_coords), DETAIL_ZOOM)

    values = array("f")
    for row in range(spec.samples):
        lat = spec.latitude_max - row/(spec.samples-1)*(spec.latitude_max-spec.latitude_min)
        for col in range(spec.samples):
            lon = spec.longitude_min + col/(spec.samples-1)*(spec.longitude_max-spec.longitude_min)
            height = sample_tiles(base,BASE_ZOOM,lon,lat)
            if height is None:
                raise ValueError(f"No GSI elevation for ({lon:.6f}, {lat:.6f}); refusing flat fill")
            for cx,cy,radius in patches:
                distance = math.hypot((lon-cx)*111.320*math.cos(math.radians(cy)),
                                      (lat-cy)*111.132)
                if distance < radius:
                    detailed = sample_tiles(high,DETAIL_ZOOM,lon,lat)
                    if detailed is not None:
                        weight = max(0.,min(1., (radius-distance)/.32))
                        weight = weight*weight*(3-2*weight)
                        height = height*(1-weight)+detailed*weight
            values.append(height)
    dem = FujiDem(spec, values,
                  "GSI DEM5A/5B/5C/10B tiles, z13 base + z15 summit/Hoei refinement; © GSI Japan")
    save_dem(path, dem)
    print(f"Saved {path} — {dem.statistics()}", flush=True)
    return dem


def prepare_crater_patches(dem_path: Path, output_dir: Path,
                           *, factor: int = 4, span_cells: int = 80) -> list[Path]:
    """Keep original z15 GSI detail around the two distinctive craters.

    Separate patch meshes replace coarse faces, using a stitched subdivision
    of the 1025-cell global DEM. Edges match exactly; only the central region
    retains near-native (~6 m) zoom-15 survey relief. No random displacement.
    """
    if factor not in (2,4,8) or span_cells<16 or span_cells>120 or span_cells%2:
        raise ValueError("Invalid crater patch refinement or span")
    dem=load_dem(dem_path)
    spec=dem.spec
    if spec.samples!=1025:
        raise ValueError("Crater patches currently target the 1025-sample parent GSI raster")
    centres=(("summit",138.7274,35.3606), ("hoei",138.7451,35.34098))
    patches=[]
    for name,lon,lat in centres:
        row=round((spec.latitude_max-lat)/(spec.latitude_max-spec.latitude_min)*(spec.samples-1))
        col=round((lon-spec.longitude_min)/(spec.longitude_max-spec.longitude_min)*(spec.samples-1))
        s=FujiPatchSpec(name,row-span_cells//2,row+span_cells//2,
                        col-span_cells//2,col+span_cells//2,factor,spec.samples)
        if any(s.overlaps(p) for p in patches):
            raise ValueError("Crater refinement extents overlap")
        patches.append(s)

    # Download only the native tiles intersecting these two ~2 km patches.
    coords=set()
    for s in patches:
        lon0=spec.longitude_min+s.col_start/(spec.samples-1)*(spec.longitude_max-spec.longitude_min)
        lon1=spec.longitude_min+s.col_end/(spec.samples-1)*(spec.longitude_max-spec.longitude_min)
        lat0=spec.latitude_max-s.row_end/(spec.samples-1)*(spec.latitude_max-spec.latitude_min)
        lat1=spec.latitude_max-s.row_start/(spec.samples-1)*(spec.latitude_max-spec.latitude_min)
        coords.update(_tile_bounds(lon0,lon1,lat0,lat1,DETAIL_ZOOM))
    tiles=_load_tiles(sorted(coords),DETAIL_ZOOM)

    output_dir=Path(output_dir)
    fingerprint=dem_fingerprint(dem)
    results=[]
    for s in patches:
        values=array("f")
        available=0
        for r in range(s.rows):
            rr=s.row_start+r/s.factor
            lat=spec.latitude_max-rr/(spec.samples-1)*(spec.latitude_max-spec.latitude_min)
            for c in range(s.cols):
                cc=s.col_start+c/s.factor
                lon=spec.longitude_min+cc/(spec.samples-1)*(spec.longitude_max-spec.longitude_min)
                baseline=_base_sample(dem,s,r,c)
                # Shared grid edges are bit-for-bit the underlying sampled relief.
                if r in (0,s.rows-1) or c in (0,s.cols-1):
                    values.append(baseline)
                    continue
                measured=sample_tiles(tiles,DETAIL_ZOOM,lon,lat)
                if measured is not None:
                    available+=1
                    # 180 m shoulder connects the original 25 m GSI surface
                    # smoothly to the 5 m-source detail in the patch centre.
                    nearest=min(r,s.rows-1-r)*spec.depth_km/(spec.samples-1)/s.factor
                    nearest=min(nearest,min(c,s.cols-1-c)*spec.width_km/(spec.samples-1)/s.factor)
                    t=max(0.,min(1.,nearest/.18))
                    t=t*t*(3-2*t)
                    values.append(baseline*(1-t)+measured*t)
                else:
                    values.append(baseline)
        fraction=available/((s.rows-2)*(s.cols-2))
        if fraction < .90:
            raise ValueError(f"{s.name} lacks GSI z15 data ({fraction:.1%}); refusing fake refinement")
        patch=FujiDetailPatch(s,values,"GSI DEM5A/5B/5C zoom-15 elevations",fingerprint,fraction)
        target=output_dir/f"fuji-{s.name}-z15-patch.bin"
        if target.exists():
            raise FileExistsError(f"Won't overwrite crater survey patch: {target}")
        save_patch(target,patch)
        print(f"Saved {target} — {s.rows}×{s.cols}, native z15 coverage {fraction:.1%}",flush=True)
        results.append(target)
    return results


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--crater-patches":
        source=Path(sys.argv[2]) if len(sys.argv)>2 else Path(".review/fuji-dem/fuji-heightmap-v2.bin")
        target_dir=Path(sys.argv[3]) if len(sys.argv)>3 else source.parent
        prepare_crater_patches(source,target_dir)
    else:
        target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".review/fuji-dem/fuji-heightmap.bin")
        if target.exists():
            raise SystemExit(f"Won't overwrite existing DEM: {target}")
        prepare(target)
