"""Deterministic, tileable PBR texture maps without asset-site dependencies.

PNG output is generated using only the Python standard library. The maps can
be embedded in both a Blender scene and its portable GLB; unlike procedural
shader nodes, normal/roughness/color image links survive glTF export.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class TextureMaps:
    resolution: int
    base_color_png: bytes
    roughness_png: bytes
    normal_png: bytes


def _noise_corner(x: int, y: int, seed: int) -> float:
    n = (x * 0x9E3779B1 + y * 0x85EBCA77 + seed * 0xC2B2AE3D) & 0xFFFFFFFF
    n ^= n >> 16
    n = (n * 0x7FEB352D) & 0xFFFFFFFF
    n ^= n >> 15
    n = (n * 0x846CA68B) & 0xFFFFFFFF
    n ^= n >> 16
    return n / 0xFFFFFFFF


def _tile_noise(u: float, v: float, frequency: int, seed: int) -> float:
    x, y = u * frequency, v * frequency
    ix, iy = math.floor(x), math.floor(y)
    tx, ty = x - ix, y - iy
    sx, sy = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    a = _noise_corner(ix % frequency, iy % frequency, seed)
    b = _noise_corner((ix + 1) % frequency, iy % frequency, seed)
    c = _noise_corner(ix % frequency, (iy + 1) % frequency, seed)
    d = _noise_corner((ix + 1) % frequency, (iy + 1) % frequency, seed)
    return ((a * (1 - sx) + b * sx) * (1 - sy)
            + (c * (1 - sx) + d * sx) * sy)


def _surface_height(u: float, v: float, seed: int) -> float:
    return (0.48 * _tile_noise(u, v, 3, seed)
            + 0.27 * _tile_noise(u, v, 11, seed + 13)
            + 0.17 * _tile_noise(u, v, 29, seed + 37)
            + 0.08 * _tile_noise(u, v, 67, seed + 71))


def _stone_weathering(u: float, v: float, seed: int) -> float:
    """Tileable mineral bloom and narrow vertical rain tracks in UV space."""
    warped_u = u + 0.018 * math.sin(math.tau * v)
    streak = _tile_noise(warped_u, 0.5 + 0.055 * math.sin(math.tau * v),
                         17, seed + 101)
    blotch = _tile_noise(u, v, 4, seed + 137)
    return max(0.0, (streak - 0.55) * 2.1) + max(0.0, (blotch - 0.61) * 0.65)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _encode_rgb_png(size: int, pixels: bytes) -> bytes:
    if len(pixels) != size * size * 3:
        raise ValueError("Invalid RGB texture buffer")
    scanlines = b"".join(b"\x00" + pixels[row * size * 3:(row + 1) * size * 3]
                         for row in range(size))
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
            + _png_chunk(b"IEND", b""))


def _clamp(value: float) -> int:
    return max(0, min(255, round(value)))


@lru_cache(maxsize=32)
def make_pbr_maps(kind: str, color: str, roughness: float, *, resolution: int = 256,
                  weathering_strength: float = 0.5) -> TextureMaps:
    """Generate subtle, seam-safe albedo, roughness and tangent-normal images.

    The maps are intentionally restrained. They add stone/concrete grain and
    manufactured-surface microvariation without replacing measured reference
    textures when a specific building requires historically correct materials.
    """
    if kind not in {"natural-stone", "weathered-stone", "cast-concrete", "painted-steel", "galvanized-steel", "verdigris-copper", "timber"}:
        raise ValueError("Unsupported procedural PBR material kind")
    if not isinstance(color, str) or len(color) != 7 or color[0] != "#":
        raise ValueError("Expected six-digit hexadecimal sRGB")
    try:
        base = bytes.fromhex(color[1:])
    except ValueError as error:
        raise ValueError("Expected six-digit hexadecimal sRGB") from error
    if not isinstance(resolution, int) or type(resolution) is bool or not 16 <= resolution <= 512 or resolution & (resolution - 1):
        raise ValueError("Resolution must be a power of two between 16 and 512")
    if isinstance(roughness, bool) or not isinstance(roughness, (int, float)) or not math.isfinite(roughness) or not 0.02 <= roughness <= 1:
        raise ValueError("Roughness must be in [0.02, 1]")
    if (isinstance(weathering_strength, bool) or not isinstance(weathering_strength, (int, float))
            or not math.isfinite(weathering_strength) or not 0 <= weathering_strength <= 1):
        raise ValueError("Weathering strength must be in [0, 1]")
    seed = int.from_bytes(hashlib.sha256(f"kumiki-pbr-v1:{kind}:{color}".encode()).digest()[:4], "big")
    height = [_surface_height((i % resolution) / resolution, (i // resolution) / resolution, seed)
              for i in range(resolution * resolution)]
    albedo, rough, normal = bytearray(), bytearray(), bytearray()
    color_variance = {"natural-stone": 0.30, "weathered-stone": 0.30, "cast-concrete": 0.17,
                      "painted-steel": 0.055, "galvanized-steel": 0.09,
                      "verdigris-copper": 0.21, "timber": 0.24}[kind]
    bump_strength = {"natural-stone": 3.0, "weathered-stone": 3.4, "cast-concrete": 2.4,
                     "painted-steel": 0.42, "galvanized-steel": 1.1,
                     "verdigris-copper": 1.5, "timber": 1.7}[kind]
    for index, h in enumerate(height):
        x, y = index % resolution, index // resolution
        variation = (h - 0.5) * color_variance
        weathering = 0.0
        if kind == "weathered-stone":
            weathering = _stone_weathering(x / resolution, y / resolution, seed)
            variation -= weathering * 0.20 * weathering_strength
        if kind == "verdigris-copper":
            u, v = x / resolution, y / resolution
            # Ancient oxidized copper varies between mint malachite, dark
            # turquoise and shaded grey. Avoid brown rust or green wallpaper.
            oxide = (0.52 * _tile_noise(u, v, 4, seed + 131)
                     + 0.30 * _tile_noise(u, v, 15, seed + 179)
                     + 0.18 * _tile_noise(u, v, 39, seed + 221))
            weathering = _stone_weathering(u, v, seed + 257)
            age = (oxide - 0.5) * 0.42 * weathering_strength
            mineral = weathering * 0.13 * weathering_strength
            albedo.extend((
                _clamp(base[0] * (1 + variation - age - mineral)),
                _clamp(base[1] * (1 + variation*.65 + age*.25 - mineral*.45)),
                _clamp(base[2] * (1 + variation*.65 + age*.65 - mineral*.60)),
            ))
        else:
            for channel in base:
                albedo.append(_clamp(channel * (1 + variation)))
        local_roughness = max(0.02, min(1, roughness + (h - 0.5) * color_variance * 0.75
                                       + weathering * 0.14 * weathering_strength))
        rough.extend([_clamp(local_roughness * 255)] * 3)
        left, right = height[y * resolution + (x - 1) % resolution], height[y * resolution + (x + 1) % resolution]
        below, above = height[((y - 1) % resolution) * resolution + x], height[((y + 1) % resolution) * resolution + x]
        nx, ny, nz = -(right - left) * bump_strength, -(above - below) * bump_strength, 1.0
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        normal.extend((_clamp((component / length * 0.5 + 0.5) * 255) for component in (nx, ny, nz)))
    return TextureMaps(resolution, _encode_rgb_png(resolution, albedo),
                       _encode_rgb_png(resolution, rough), _encode_rgb_png(resolution, normal))
