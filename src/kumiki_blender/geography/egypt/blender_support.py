"""Blender-only geographic materials, water meshes, map type and terrain surfaces."""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Any

from ...blender_scene import _color
from ...blender_textures import ensure_metric_uvs
from .geometry import MeshSketch
from .terrain import (DEPTH, EAST, NORTH, SOUTH, WEST, WIDTH, EgyptMapSpec,
                      geo_to_xy, sample_terrain, terrain_mesh, xy_to_geo)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", zlib.crc32(kind+payload) & 0xFFFFFFFF))


def _color_atlas_png(width: int = 1024, height: int = 800) -> bytes:
    """Embedded geographic albedo; fine atlas uses fast bilinear supersampling.

    Sampling 1.8 million full geographic masks and multifractals was adding
    several minutes to every camera/art review. Sampling at half each axis,
    then reconstructing pixels with small deterministic mineral grain, keeps
    smooth coastlines and a high-resolution embedded texture at ~1/4 the cost.
    """
    if (type(width) is not int or type(height) is not int
            or not 16 <= width <= 2048 or not 16 <= height <= 2048
            or width*height > 2_500_000):
        raise ValueError("Atlas dimensions outside safe range")
    stride = 2 if width >= 512 and height >= 384 else 1
    coarse = []
    if stride > 1:
        cw = (width-1)//stride+2
        ch = (height-1)//stride+2
        for row in range(ch):
            pixel_y = min(height-1, row*stride)
            lat = NORTH - (pixel_y+.5)*(NORTH-SOUTH)/height
            pixels = bytearray()
            for col in range(cw):
                pixel_x = min(width-1, col*stride)
                lon = WEST+(pixel_x+.5)*(EAST-WEST)/width
                pixels.extend(round(255*channel) for channel in sample_terrain(lon, lat).rgb)
            coarse.append(pixels)
    compressor = zlib.compressobj(6)
    compressed = []
    for row in range(height):
        scan = bytearray((0,))  # PNG filter 0
        if stride == 1:
            lat = NORTH - (row+.5)*(NORTH-SOUTH)/height
            for col in range(width):
                lon = WEST + (col+.5)*(EAST-WEST)/width
                scan.extend(round(255*channel)
                            for channel in sample_terrain(lon, lat).rgb)
        else:
            iy, ry = divmod(row, stride)
            upper, lower = coarse[iy], coarse[iy+1]
            wy = ry/stride
            for col in range(width):
                ix, rx = divmod(col, stride)
                wx = rx/stride
                a, b = 3*ix, 3*(ix+1)
                mineral = ((col*13 ^ row*71 ^ (col//7)*17) & 7)-3
                for channel in range(3):
                    north = upper[a+channel]*(1-wx)+upper[b+channel]*wx
                    south = lower[a+channel]*(1-wx)+lower[b+channel]*wx
                    value = round(north*(1-wy)+south*wy + mineral*.33)
                    scan.append(min(255, max(0, value)))
        compressed.append(compressor.compress(scan))
    compressed.append(compressor.flush())
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", b"".join(compressed))
            + _png_chunk(b"IEND", b""))


def _catmull(points: tuple[tuple[float, float], ...], steps: int = 14) -> list[tuple[float, float]]:
    """Smooth centerline through control points; endpoints are not extrapolated."""
    output = []
    for index in range(len(points)-1):
        a = points[max(0, index-1)]
        b = points[index]
        c = points[index+1]
        d = points[min(len(points)-1, index+2)]
        for step in range(steps):
            t = step/steps
            output.append(tuple(.5*((2*b[j]) + (-a[j]+c[j])*t +
                                      (2*a[j]-5*b[j]+4*c[j]-d[j])*t*t +
                                      (-a[j]+3*b[j]-3*c[j]+d[j])*t*t*t)
                                for j in range(2)))
    output.append(points[-1])
    return output


def _river_ribbon(name: str, route: tuple[tuple[float, float], ...]
                  ) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    """True surface ribbon with a smooth, changing heading and terrain-following Z."""
    if len(route) < 2:
        raise ValueError("River ribbon requires two geographic points")
    smooth_route = _catmull(route)
    coordinates = [geo_to_xy(*point) for point in smooth_route]
    width = (.009 if name == "border" else .047 if name == "main-nile"
             else .026 if name == "suez-canal" else .032)
    vertices = []
    for index, ((x, y), (lon, lat)) in enumerate(zip(coordinates, smooth_route)):
        prev = coordinates[max(0, index-1)]
        after = coordinates[min(len(coordinates)-1, index+1)]
        dx, dy = after[0]-prev[0], after[1]-prev[1]
        length = max(math.hypot(dx, dy), 1e-8)
        ox, oy = -dy/length*width, dx/length*width
        z = sample_terrain(lon, lat).height_m + (.035 if name != "suez-canal" else .048)
        vertices.extend(((x+ox, y+oy, z), (x-ox, y-oy, z)))
    faces = [(2*i, 2*i+1, 2*i+3, 2*i+2) for i in range(len(vertices)//2-1)]
    return vertices, faces


def _make_plain_material(bpy: Any, name: str, color: str, *, roughness: float,
                         metallic: float = 0.0) -> Any:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = _color(color)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return material


def _terrain_contact_z(x: float, y: float, clearance: float) -> float:
    lon, lat = xy_to_geo(x, y)
    return sample_terrain(lon, lat).height_m + clearance


def _label_hover_height(points: Any, clearance: float) -> float:
    """One stable Z plane for a complete word, clearing its highest ridge."""
    heights = [_terrain_contact_z(x, y, clearance) for x, y in points]
    if not heights:
        raise ValueError("A place label must have visible glyph vertices")
    return max(heights)


def _vertex_islands(vertex_count: int, edges: Any) -> tuple[tuple[int, ...], ...]:
    """Connected glyph solids: lift each complete letter, not its individual vertices."""
    parent = list(range(vertex_count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in edges:
        a, b = root(first), root(second)
        if a != b:
            parent[max(a, b)] = min(a, b)
    islands: dict[int, list[int]] = {}
    for index in range(vertex_count):
        islands.setdefault(root(index), []).append(index)
    return tuple(tuple(indices) for indices in islands.values())


def _attach(bpy: Any, collection: Any, name: str, vertices: Any, faces: Any,
            material: Any, role: str, *, uv: bool = True) -> Any:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    if uv:
        ensure_metric_uvs(mesh, tile_size_m=.35)
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj["kumiki_part"] = role
    return obj

def _attach_sketch(bpy: Any, collection: Any, name: str, sketch: MeshSketch,
                   material: Any, role: str) -> Any:
    if not sketch.faces:
        raise ValueError(f"Empty Egypt architecture mesh: {name}")
    return _attach(bpy, collection, name, sketch.vertices, sketch.faces, material, role)


def _xy_ground(x: float, y: float, clearance: float = 0.0) -> float:
    return sample_terrain(*xy_to_geo(x, y)).height_m + clearance


def _add_terrain(bpy: Any, collection: Any, spec: EgyptMapSpec, atlas: Any,
                 land: Any, sea: Any) -> Any:
    kernel = terrain_mesh(spec)
    mesh = bpy.data.meshes.new("Egypt / true relief / geographic grid")
    mesh.from_pydata(kernel.vertices, [], kernel.faces)
    mesh.update()
    mesh.materials.append(land)
    mesh.materials.append(sea)
    for polygon, kind in zip(mesh.polygons, kernel.face_kinds):
        polygon.material_index = int(kind in ("mediterranean", "red-sea", "lake-nasser", "lake-qarun"))
        polygon.use_smooth = True
    uv = mesh.uv_layers.new(name="Egypt / lat-long atlas")
    for loop in mesh.loops:
        point = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = ((point.x/WIDTH)+.5, (point.y/DEPTH)+.5)
    obj = bpy.data.objects.new("Egypt relief / editable sculpted topography", mesh)
    collection.objects.link(obj)
    obj["kumiki_part"] = "geographic-relief"
    obj["kumiki_geography"] = "illustrative / not surveyed DEM"
    return obj


def _text_mesh(bpy: Any, collection: Any, text: str, name: str,
               location: tuple[float, float, float], size: float, material: Any,
               *, angle: float = 0.0, upright: bool = False,
               font: Any = None, depth: float = .004,
               clear_relief: bool = False) -> Any:
    letters = bpy.data.curves.new(name, "FONT")
    letters.body = text
    letters.size = size
    letters.align_x = "CENTER"
    letters.align_y = "CENTER"
    letters.extrude = depth
    letters.bevel_depth = min(.008, depth*.45)
    letters.bevel_resolution = 2
    if font is not None:
        letters.font = font
    obj = bpy.data.objects.new(name, letters)
    collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (math.pi/2 if upright else 0, 0, math.radians(angle))
    obj.data.materials.append(material)
    # GLB only exports meshes. Conversion happens before source save, so
    # glyphs are self-contained on macOS/Windows even without these fonts.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")
    converted = bpy.context.view_layer.objects.active
    if clear_relief:
        # Keep the entire phrase on a single rigid plane. Per-vertex draping
        # mangled glyph outlines, while per-letter heights made adjacent
        # letters overlap in oblique projection over the Eastern Desert.
        # All contours now remain legible above the highest touched summit.
        bpy.context.view_layer.update()
        matrix = converted.matrix_world.copy()
        parts = _vertex_islands(len(converted.data.vertices),
                                (edge.vertices for edge in converted.data.edges))
        coordinates = ((world.x, world.y)
                       for indices in parts for i in indices
                       for world in (matrix @ converted.data.vertices[i].co,))
        converted.location.z = _label_hover_height(coordinates, .085)
    converted["kumiki_part"] = "map-type"
    converted["kumiki_text"] = text
    return converted


def _serif_font(bpy: Any) -> Any:
    """Use an available local serif for conversion; never depend on it at playback."""
    paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
    )
    for path in paths:
        if Path(path).is_file():
            return bpy.data.fonts.load(path)
    return None


def _marker(bpy: Any, collection: Any, name: str, lon: float, lat: float,
            material: Any) -> Any:
    x, y = geo_to_xy(lon, lat)
    z = sample_terrain(lon, lat).height_m + .056
    # The sphere's radial silhouette holds up under a high oblique camera.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=.043,
                                         location=(x, y, z))
    obj = bpy.context.object
    obj.name = f"City marker / {name}"
    for previous in tuple(obj.users_collection):
        previous.objects.unlink(obj)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    obj["kumiki_part"] = "city-marker"
    obj["kumiki_place"] = name
    return obj
