"""Bounded OBJ/MTL UV-raster coverage for private texture-atlas QA.

Only UV regions actually referenced by polygon faces are used to estimate
color and missing texels. No OBJ, MTL, bitmap, or mask source is modified.
This is an optional diagnostic, not a claim that the mesh is complete.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .references import ReferenceError


_MAX_UV_COORDS = 3_000_000
_MAX_UV_FACES = 10_000_000


def _material_textures(path: Path, atlas_names: set[str]) -> dict[str, str]:
    materials: dict[str, str] = {}
    seen: set[str] = set()
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("newmtl "):
            current = stripped[7:].strip()
            if not current or current in seen:
                raise ReferenceError("Repeated or empty OBJ material name")
            seen.add(current)
        elif stripped.startswith("map_Kd "):
            atlas = stripped[7:].strip()
            if not current or atlas not in atlas_names:
                raise ReferenceError("UV material is missing or has an unpinned atlas")
            if materials.get(current) is not None:
                raise ReferenceError("UV material has more than one diffuse atlas")
            materials[current] = atlas
    if not materials:
        raise ReferenceError("OBJ has no usable diffuse atlas materials")
    return materials


def rasterize_uv_coverage(obj: Path, mtl: Path,
                          sample_sizes: dict[str, tuple[int, int]]) -> tuple[dict[str, Any], dict[str, int]]:
    """Rasterize material-tagged OBJ faces into per-atlas Pillow L masks.

    ``sample_sizes`` must correspond exactly to the texture thumbnail sizes
    used by color/exposure QA. Unsupported tiling or missing UVs are rejected
    rather than estimating exposure on unrelated atlas padding.
    """
    from PIL import Image, ImageDraw

    materials = _material_textures(mtl, set(sample_sizes))
    masks = {name: Image.new("L", dimensions, 0)
             for name, dimensions in sample_sizes.items()}
    painters = {name: ImageDraw.Draw(mask) for name, mask in masks.items()}
    uv: list[tuple[float, float]] = []
    faces = {name: 0 for name in masks}
    current: str | None = None
    face_count = 0
    with obj.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("vt "):
                values = stripped.split()
                if len(values) not in (3, 4):
                    raise ReferenceError("OBJ UV texture coordinate must have two or three values")
                try:
                    u, v = float(values[1]), float(values[2])
                except ValueError as error:
                    raise ReferenceError("OBJ has an invalid UV coordinate") from error
                if (not math.isfinite(u) or not math.isfinite(v)
                        or not -0.0001 <= u <= 1.0001
                        or not -0.0001 <= v <= 1.0001):
                    raise ReferenceError("UV coverage cannot analyze tiled/out-of-range coordinates")
                uv.append((min(1, max(0, u)), min(1, max(0, v))))
                if len(uv) > _MAX_UV_COORDS:
                    raise ReferenceError("UV atlas coverage exceeds safe texture-coordinate budget")
            elif stripped.startswith("usemtl "):
                current = materials.get(stripped[7:].strip())
            elif stripped.startswith("f "):
                face_count += 1
                if face_count > _MAX_UV_FACES:
                    raise ReferenceError("UV atlas coverage exceeds safe face budget")
                if current is None:
                    raise ReferenceError("OBJ face has no pinned diffuse UV material")
                tokens = stripped.split()[1:]
                if not 3 <= len(tokens) <= 32:
                    raise ReferenceError("UV face must have 3–32 vertices")
                points = []
                width, height = sample_sizes[current]
                for token in tokens:
                    parts = token.split("/")
                    if len(parts) < 2 or not parts[1]:
                        raise ReferenceError("Textured OBJ face is missing a UV index")
                    try:
                        index = int(parts[1])
                    except ValueError as error:
                        raise ReferenceError("Textured OBJ UV index is not an integer") from error
                    # OBJ negative indices are relative to the vt count at
                    # this face; 0 is never a valid OBJ index.
                    offset = index-1 if index > 0 else len(uv)+index
                    if index == 0 or not 0 <= offset < len(uv):
                        raise ReferenceError("Textured OBJ face references an unknown UV index")
                    u, v = uv[offset]
                    points.append((round(u*(width-1)), round((1-v)*(height-1))))
                painters[current].polygon(points, fill=255)
                faces[current] += 1
    if not face_count or any(not mask.getbbox() for mask in masks.values()):
        raise ReferenceError("Atlas coverage needs at least one mapped UV face per texture page")
    return masks, faces
