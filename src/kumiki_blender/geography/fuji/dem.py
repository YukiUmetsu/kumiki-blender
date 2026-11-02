"""Portable, dependency-free reader and metric sampling of Mount Fuji GSI elevations.

The source heightmap is a cached derivative of GSI's DEM5A / DEM10B tiles,
not a hand-sculpted cone. X/Y positions use a local equirectangular metric
projection; elevation remains true metres above sea level.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import sys
import zlib


MAGIC = b"FUJI-DEM-1\n"
SUMMIT = (138.7274, 35.3606)


@dataclass(frozen=True)
class FujiSpec:
    longitude_min: float = 138.5844
    longitude_max: float = 138.8704
    latitude_min: float = 35.2435
    latitude_max: float = 35.4777
    samples: int = 1025

    def __post_init__(self) -> None:
        if not (self.longitude_min < SUMMIT[0] < self.longitude_max and
                self.latitude_min < SUMMIT[1] < self.latitude_max and
                self.samples >= 33 and self.samples <= 4097):
            raise ValueError("Invalid Mount Fuji geographic envelope or grid size")

    @property
    def width_km(self) -> float:
        return (self.longitude_max-self.longitude_min)*111.320*math.cos(math.radians(SUMMIT[1]))

    @property
    def depth_km(self) -> float:
        return (self.latitude_max-self.latitude_min)*111.132

    def xy(self, longitude: float, latitude: float) -> tuple[float, float]:
        return ((longitude-SUMMIT[0])*111.320*math.cos(math.radians(SUMMIT[1])),
                (latitude-SUMMIT[1])*111.132)


@dataclass(frozen=True)
class FujiDem:
    spec: FujiSpec
    elevations: array
    source: str

    def __post_init__(self) -> None:
        if len(self.elevations) != self.spec.samples ** 2:
            raise ValueError("DEM cell count does not match its declared size")
        if any(not math.isfinite(h) or h < -500 or h > 9000 for h in self.elevations):
            raise ValueError("DEM includes invalid, missing or out-of-range heights")

    def height(self, longitude: float, latitude: float) -> float:
        spec = self.spec
        u = (longitude-spec.longitude_min)/(spec.longitude_max-spec.longitude_min)*(spec.samples-1)
        # Row zero is the NORTH edge (raster convention).
        v = (spec.latitude_max-latitude)/(spec.latitude_max-spec.latitude_min)*(spec.samples-1)
        if not (0 <= u <= spec.samples-1 and 0 <= v <= spec.samples-1):
            raise ValueError("Geographic sample outside Fuji terrain envelope")
        x, y = min(int(u), spec.samples-2), min(int(v), spec.samples-2)
        fx, fy = u-x, v-y
        i = y*spec.samples+x
        a, b = self.elevations[i], self.elevations[i+1]
        c, d = self.elevations[i+spec.samples], self.elevations[i+spec.samples+1]
        return (a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy

    def statistics(self) -> dict[str, float | int | list[float]]:
        heights = self.elevations
        peak = max(range(len(heights)), key=heights.__getitem__)
        x, y = peak % self.spec.samples, peak // self.spec.samples
        lon = self.spec.longitude_min + x/(self.spec.samples-1)*(self.spec.longitude_max-self.spec.longitude_min)
        lat = self.spec.latitude_max - y/(self.spec.samples-1)*(self.spec.latitude_max-self.spec.latitude_min)
        return {"minimum_m": round(min(heights), 2), "maximum_m": round(heights[peak], 2),
                "peak_lon_lat": [round(lon, 6), round(lat, 6)],
                "samples": self.spec.samples, "width_km": round(self.spec.width_km, 2),
                "depth_km": round(self.spec.depth_km, 2)}


def save_dem(path: Path, dem: FujiDem) -> None:
    """Cache tile-derived height samples in a checksummed cross-platform file."""
    from dataclasses import asdict

    values = array("f", dem.elevations)
    if sys.byteorder != "little":
        values.byteswap()
    data = values.tobytes()
    header = json.dumps({"bounds": asdict(dem.spec), "source": dem.source,
                         "crc32": zlib.crc32(data), "dtype": "float32-le"},
                        separators=(",", ":")).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MAGIC + struct.pack("<I", len(header)) + header + zlib.compress(data, 9))


def load_dem(path: Path) -> FujiDem:
    raw = Path(path).read_bytes()
    if not raw.startswith(MAGIC) or len(raw) < len(MAGIC)+4:
        raise ValueError("Invalid Fuji DEM signature")
    start = len(MAGIC)
    header_len, = struct.unpack_from("<I", raw, start)
    if header_len > 8192 or start+4+header_len >= len(raw):
        raise ValueError("Invalid Fuji DEM header")
    meta = json.loads(raw[start+4:start+4+header_len])
    if meta.get("dtype") != "float32-le":
        raise ValueError("Unsupported DEM cell encoding")
    spec = FujiSpec(**meta["bounds"])
    # Refuse oversized/corrupt streams rather than blindly decompressing.
    inflater = zlib.decompressobj()
    data = inflater.decompress(raw[start+4+header_len:], spec.samples**2*4+1)
    if len(data) != spec.samples**2*4 or not inflater.eof or inflater.unconsumed_tail:
        raise ValueError("Invalid DEM payload size")
    if zlib.crc32(data) != meta["crc32"]:
        raise ValueError("Invalid DEM checksum")
    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return FujiDem(spec, values, meta["source"])
