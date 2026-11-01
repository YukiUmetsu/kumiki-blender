"""Checksummed, geographically aligned GSI detail meshes for Fuji's craters.

Patch edges use the parent DEM verbatim. Interior samples come from the GSI
zoom-15 raster and are feathered into the parent heightmap at their boundary.
The format never stores an artist-invented height displacement.
"""

from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import struct
import sys
import zlib

from .dem import FujiDem


MAGIC = b"FUJI-PATCH-1\n"


@dataclass(frozen=True)
class FujiPatchSpec:
    name: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    factor: int
    parent_samples: int

    def __post_init__(self) -> None:
        if (not self.name or not 2 <= self.factor <= 8 or
                not 0 <= self.row_start < self.row_end < self.parent_samples or
                not 0 <= self.col_start < self.col_end < self.parent_samples or
                (self.rows * self.cols) > 1_000_000):
            raise ValueError("Invalid Fuji detail-patch dimensions")

    @property
    def rows(self) -> int:
        return (self.row_end-self.row_start)*self.factor+1

    @property
    def cols(self) -> int:
        return (self.col_end-self.col_start)*self.factor+1

    def overlaps(self, other: "FujiPatchSpec") -> bool:
        return (max(self.row_start,other.row_start) < min(self.row_end,other.row_end) and
                max(self.col_start,other.col_start) < min(self.col_end,other.col_end))


@dataclass(frozen=True)
class FujiDetailPatch:
    spec: FujiPatchSpec
    elevations: array
    source: str
    parent_crc32: int
    surveyed_fraction: float

    def __post_init__(self) -> None:
        if (len(self.elevations) != self.spec.rows*self.spec.cols or
                any(not math.isfinite(h) or h < -500 or h > 9000 for h in self.elevations) or
                not 0 <= self.surveyed_fraction <= 1):
            raise ValueError("Invalid Fuji detail-patch heights")


def _little_bytes(values: array) -> bytes:
    result = array("f", values)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def dem_fingerprint(dem: FujiDem) -> int:
    return zlib.crc32(_little_bytes(dem.elevations))


def validate_patches(patches: tuple[FujiDetailPatch, ...], dem: FujiDem,
                     divisions: int) -> None:
    if not patches:
        return
    if (dem.spec.samples-1) % divisions:
        raise ValueError("Coarse DEM grid cannot align detail patches")
    stride = (dem.spec.samples-1)//divisions
    for i, patch in enumerate(patches):
        s = patch.spec
        if (s.parent_samples != dem.spec.samples or patch.parent_crc32 != dem_fingerprint(dem) or
                any(value % stride for value in (s.row_start,s.row_end,s.col_start,s.col_end))):
            raise ValueError(f"Detail patch {s.name} does not match the parent terrain grid")
        if any(s.name == previous.spec.name for previous in patches[:i]):
            raise ValueError(f"Duplicate detail patch name: {s.name}")
        if any(s.overlaps(previous.spec) for previous in patches[:i]):
            raise ValueError(f"Overlapping detail patch: {s.name}")
        # The complete rim must coincide with the parent DEM, to sub-millimetre
        # precision, before its coarse faces are replaced by fine triangles.
        for r in range(s.rows):
            for c in (0,s.cols-1):
                if abs(patch.elevations[r*s.cols+c] - _base_sample(dem,s,r,c)) > .001:
                    raise ValueError(f"Seam height mismatch: {s.name}")
        for c in range(s.cols):
            for r in (0,s.rows-1):
                if abs(patch.elevations[r*s.cols+c] - _base_sample(dem,s,r,c)) > .001:
                    raise ValueError(f"Seam height mismatch: {s.name}")


def _base_sample(dem: FujiDem, spec: FujiPatchSpec, r: int, c: int) -> float:
    """Sample the parent *grid* in row/column space, avoiding geographic roundoff."""
    rr=spec.row_start+r/spec.factor
    cc=spec.col_start+c/spec.factor
    r0,c0=int(rr),int(cc)
    fr,fc=rr-r0,cc-c0
    if r0 == dem.spec.samples-1:
        r0-=1; fr=1.
    if c0 == dem.spec.samples-1:
        c0-=1; fc=1.
    w=dem.spec.samples
    heights=dem.elevations
    a=heights[r0*w+c0]*(1-fc)+heights[r0*w+c0+1]*fc
    b=heights[(r0+1)*w+c0]*(1-fc)+heights[(r0+1)*w+c0+1]*fc
    return a*(1-fr)+b*fr


def save_patch(path: Path, patch: FujiDetailPatch) -> None:
    values=_little_bytes(patch.elevations)
    meta={"spec":asdict(patch.spec),"source":patch.source,"parent_crc32":patch.parent_crc32,
          "surveyed_fraction":patch.surveyed_fraction,"crc32":zlib.crc32(values),
          "dtype":"float32-le"}
    header=json.dumps(meta,separators=(",",":")).encode()
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(MAGIC+struct.pack("<I",len(header))+header+zlib.compress(values,9))


def load_patch(path: Path) -> FujiDetailPatch:
    raw=Path(path).read_bytes()
    if not raw.startswith(MAGIC) or len(raw)<len(MAGIC)+4:
        raise ValueError("Invalid Fuji detail-patch signature")
    pos=len(MAGIC)
    size,=struct.unpack_from("<I",raw,pos)
    if not 0 < size < 8192 or len(raw) <= pos+4+size:
        raise ValueError("Invalid detail-patch header")
    meta=json.loads(raw[pos+4:pos+4+size])
    if meta.get("dtype")!="float32-le":
        raise ValueError("Unsupported detail-patch encoding")
    spec=FujiPatchSpec(**meta["spec"])
    inflater=zlib.decompressobj()
    data=inflater.decompress(raw[pos+4+size:],spec.rows*spec.cols*4+1)
    if (len(data)!=spec.rows*spec.cols*4 or not inflater.eof or
            inflater.unconsumed_tail or zlib.crc32(data)!=meta["crc32"]):
        raise ValueError("Invalid detail-patch payload or checksum")
    values=array("f")
    values.frombytes(data)
    if sys.byteorder!="little":
        values.byteswap()
    return FujiDetailPatch(spec,values,meta["source"],meta["parent_crc32"],
                           meta["surveyed_fraction"])
