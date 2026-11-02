"""Inspect authentic *coarse* airborne LiDAR of the installed monument.

Input: 2017 NYC Topobathymetric LAZ/LAS tile 970190.las (uncompressed LAS
only here), publicly indexed through NYS ITS LAS_Indexes FeatureServer/11.
The point cloud is not redistributed. This intentionally produces sectional
*evidence*, not a watertight mesh or fictional precise statue measurements.

Usage:
    python3 scripts/liberty_lidar_sections.py \
      --source .review/liberty-r64-survey/970190.las \
      --output .review/liberty-r64-survey/lidar-sections.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path


SOURCE_URL = "https://gisdata.ny.gov/elevation/LIDAR/NYC_TopoBathymetric2017/970190.las"


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    at = fraction*(len(ordered)-1)
    index = int(at)
    nxt = min(index+1, len(ordered)-1)
    return ordered[index]+(ordered[nxt]-ordered[index])*(at-index)


def analyze_las(source: Path, *, easting: float = 971895., northing: float = 190390.,
                radius_ft: float = 80., start_ft: float = 140., end_ft: float = 305.,
                band_ft: float = 10.) -> dict:
    if not source.is_file():
        raise ValueError(f"LAS file does not exist: {source}")
    if not (0 < radius_ft <= 1000 and 0 <= start_ft < end_ft <= 1000 and
            0 < band_ft <= end_ft-start_ft):
        raise ValueError("Invalid crop / height band")
    with source.open("rb") as handle:
        header = handle.read(375)
        if len(header) < 227 or header[:4] != b"LASF":
            raise ValueError("Not an uncompressed LAS point cloud")
        major, minor = header[24], header[25]
        point_offset = struct.unpack_from("<I", header, 96)[0]
        point_format = header[104] & 0x3f
        record_size = struct.unpack_from("<H", header, 105)[0]
        legacy_count = struct.unpack_from("<I", header, 107)[0]
        count = (struct.unpack_from("<Q", header, 247)[0]
                 if (major, minor) >= (1, 4) and not legacy_count else legacy_count)
        scales = struct.unpack_from("<3d", header, 131)
        offsets = struct.unpack_from("<3d", header, 155)
        if not (point_format <= 10 and record_size >= 20 and
                all(math.isfinite(v) and v > 0 for v in scales) and
                all(math.isfinite(v) for v in offsets)):
            raise ValueError("Unsupported LAS point-format or coordinate scale")
        if source.stat().st_size < point_offset+count*record_size:
            raise ValueError("LAS point count exceeds file size (or compressed LAZ)")
        handle.seek(point_offset)
        buckets: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
        remaining = count
        digest = hashlib.sha256()
        # Hash the source in a separate streaming pass, preserving the simple
        # binary reader's exact byte counts and avoiding a NumPy dependency.
        with source.open("rb") as checksum_source:
            for chunk in iter(lambda: checksum_source.read(2**20), b""):
                digest.update(chunk)
        while remaining:
            batch_count = min(remaining, 4096)
            raw = handle.read(batch_count*record_size)
            for index in range(batch_count):
                ix, iy, iz = struct.unpack_from("<iii", raw, index*record_size)
                x, y, z = (ix*scales[0]+offsets[0],
                           iy*scales[1]+offsets[1], iz*scales[2]+offsets[2])
                if not start_ft <= z < end_ft or (x-easting)**2+(y-northing)**2 > radius_ft**2:
                    continue
                band = int((z-start_ft)//band_ft)
                buckets[band][0].append(x)
                buckets[band][1].append(y)
            remaining -= batch_count
    bands = []
    for index, (xs, ys) in sorted(buckets.items()):
        bands.append({"z_ft": [round(start_ft+index*band_ft, 2),
                                 round(start_ft+(index+1)*band_ft, 2)],
                      "point_count": len(xs),
                      "easting_p05_p50_p95_ft": [round(_quantile(xs, q), 2)
                                                  for q in (.05, .5, .95)],
                      "northing_p05_p50_p95_ft": [round(_quantile(ys, q), 2)
                                                   for q in (.05, .5, .95)]})
    return {
        "source_url": SOURCE_URL,
        "source_sha256": digest.hexdigest(),
        "survey": "NYC Topobathymetric LiDAR (2017), tile 970190",
        "capture": "Airborne, mostly top/upper-facing surfaces; NOT a complete exterior scan",
        "header": {"las_version": f"{major}.{minor}", "point_format": point_format,
                   "point_count": count, "record_size": record_size},
        "coordinate_frame": "NAD83 / New York Long Island State Plane (US survey feet); vertical NAVD88",
        "crop": {"center_easting_ft": easting, "center_northing_ft": northing,
                 "radius_ft": radius_ft, "z_min_ft": start_ft, "z_max_ft": end_ft,
                 "band_ft": band_ft,
                 "center_status": "manually inferred from high-elevation cluster; not surveyed"},
        "sections": bands,
        "limitations": [
            "5th–95th percentiles are sampled points, not the full copper surface silhouette",
            "No image intrinsics, side-wall coverage, body detail or face anatomy can be recovered",
            "Do not register a mesh or claim photo fidelity using these extents alone",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--easting", type=float, default=971895.)
    parser.add_argument("--northing", type=float, default=190390.)
    parser.add_argument("--radius-ft", type=float, default=80.)
    args = parser.parse_args()
    record = analyze_las(args.source, easting=args.easting, northing=args.northing,
                         radius_ft=args.radius_ft)
    if args.output.exists():
        raise SystemExit("Output exists; choose a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(f"Analyzed {record['header']['point_count']} LAS points; "
          f"{len(record['sections'])} non-empty height bands")


if __name__ == "__main__":
    main()
