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


def _draw_intersected_edge(painter: Any, first: tuple[float, float],
                           second: tuple[float, float]) -> None:
    """Mark every texel cell crossed by an exact (unrounded) UV edge.

    Pixel centers are at integral coordinates; cell boundaries are at +/−0.5.
    This supercover traversal is linear in edge length, not its bounding area.
    """
    x0, y0 = first[0] + 0.5, first[1] + 0.5
    x1, y1 = second[0] + 0.5, second[1] + 0.5
    ix, iy = math.floor(x0), math.floor(y0)
    ex, ey = math.floor(x1), math.floor(y1)
    dx, dy = x1 - x0, y1 - y0
    step_x = (dx > 0) - (dx < 0)
    step_y = (dy > 0) - (dy < 0)
    step_time_x = 1.0 / abs(dx) if dx else math.inf
    step_time_y = 1.0 / abs(dy) if dy else math.inf
    next_x = ix + 1 if dx > 0 else ix
    next_y = iy + 1 if dy > 0 else iy
    hit_x = (next_x - x0) / dx if dx else math.inf
    hit_y = (next_y - y0) / dy if dy else math.inf
    painter.point((ix, iy), fill=255)
    while (ix, iy) != (ex, ey):
        if hit_x < hit_y:
            ix += step_x
            hit_x += step_time_x
        elif hit_y < hit_x:
            iy += step_y
            hit_y += step_time_y
        else:
            # A diagonal edge touches both neighbors at a pixel-cell corner.
            painter.point((ix + step_x, iy), fill=255)
            painter.point((ix, iy + step_y), fill=255)
            ix += step_x
            iy += step_y
            hit_x += step_time_x
            hit_y += step_time_y
        painter.point((ix, iy), fill=255)


def rasterize_uv_coverage(obj: Path, mtl: Path,
                          sample_sizes: dict[str, tuple[int, int]], *,
                          atlas_names: set[str] | None = None,
                          conservative: bool = False) -> tuple[dict[str, Any], dict[str, int]]:
    """Rasterize material-tagged OBJ faces into per-atlas Pillow L masks.

    ``sample_sizes`` selects the pages and dimensions to rasterize. An optional
    ``atlas_names`` can list *all* pinned atlas names while only one selected
    page gets a native-resolution mask, avoiding simultaneously allocating
    enormous masks for a multi-page scan. All OBJ faces remain UV-validated.
    With conservative=True, also mark texels intersected by unrounded UV
    polygon edges. This protects edge texels from destructive gutter writes.
    Unsupported tiling or missing UVs are rejected rather than guessing.
    """
    from PIL import Image, ImageDraw

    pinned_names = set(sample_sizes) if atlas_names is None else set(atlas_names)
    if not sample_sizes or not set(sample_sizes) <= pinned_names:
        raise ReferenceError("UV coverage page selection must use pinned atlas names")
    for size in sample_sizes.values():
        if (len(size) != 2 or any(type(value) is not int or not 1 <= value <= 32768
                                  for value in size)):
            raise ReferenceError("UV coverage mask dimensions must be positive and bounded")
    if sum(width * height for width, height in sample_sizes.values()) > 100_000_000:
        raise ReferenceError("UV coverage masks exceed the 100-megapixel safety budget")
    materials = _material_textures(mtl, pinned_names)
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
                exact_points: list[tuple[float, float]] = []
                selected = current in sample_sizes
                if selected:
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
                    if selected:
                        u, v = uv[offset]
                        exact = (u*(width-1), (1-v)*(height-1))
                        points.append((round(exact[0]), round(exact[1])))
                        if conservative:
                            exact_points.append(exact)
                if selected:
                    painters[current].polygon(points, fill=255)
                    if conservative:
                        for index, first in enumerate(exact_points):
                            _draw_intersected_edge(
                                painters[current], first,
                                exact_points[(index + 1) % len(exact_points)])
                    faces[current] += 1
    if not face_count or any(not mask.getbbox() for mask in masks.values()):
        raise ReferenceError("Atlas coverage needs at least one mapped UV face per texture page")
    return masks, faces
