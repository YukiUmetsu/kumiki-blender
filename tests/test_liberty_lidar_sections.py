"""Small synthetic LAS fixture for the authentic-monument survey inspector."""

from __future__ import annotations

import runpy
import struct
import tempfile
import unittest
from pathlib import Path


analyze_las = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                  "scripts" / "liberty_lidar_sections.py"))["analyze_las"]


class LibertyLidarSectionsTests(unittest.TestCase):
    def test_las_14_sections_and_near_field_filtering(self):
        header = bytearray(375)
        header[:4] = b"LASF"
        header[24:26] = bytes((1, 4))
        struct.pack_into("<I", header, 96, 375)
        header[104] = 6
        struct.pack_into("<H", header, 105, 30)
        struct.pack_into("<Q", header, 247, 3)
        struct.pack_into("<3d", header, 131, .01, .01, .01)
        points = b"".join(struct.pack("<iii18x", x, y, z) for x, y, z in (
            (97189500, 19039000, 15000),
            (97191500, 19039000, 16500),
            (97209500, 19039000, 17500),
        ))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.las"
            source.write_bytes(header+points)
            report = analyze_las(source)
            self.assertEqual(report["header"]["point_count"], 3)
            self.assertEqual([row["point_count"] for row in report["sections"]], [1, 1])
            self.assertEqual([row["z_ft"] for row in report["sections"]],
                             [[150.0, 160.0], [160.0, 170.0]])
            self.assertEqual(report["header"]["point_format"], 6)
            source.write_bytes(b"not a point cloud")
            with self.assertRaises(ValueError):
                analyze_las(source)
            with self.assertRaises(ValueError):
                analyze_las(source, radius_ft=0)


if __name__ == "__main__":
    unittest.main()
