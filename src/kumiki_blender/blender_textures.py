"""Create glTF-compatible packed PBR textures and simple metric UV maps."""

from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path
from typing import Any

from .procedural_pbr import make_pbr_maps
from .surfaces import PBRSurface


def ensure_metric_uvs(mesh: Any, *, tile_size_m: float = 1.5) -> None:
    """Create per-face planar UVs for generated meshes, tiling in meter space.

    This maps arbitrary extruded profiles, stone voussoirs, tapered supports and
    boxes without relying on destructive operator-based UV unwrapping. The
    texture itself is seamlessly tileable; sharp projection changes land on
    physical polygon boundaries, not through the middle of a face.
    """
    if mesh.uv_layers:
        return
    uv = mesh.uv_layers.new(name="Kumiki / metric tile")
    for polygon in mesh.polygons:
        normal = polygon.normal
        axis = max(range(3), key=lambda i: abs(normal[i]))
        for index in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[index].vertex_index].co
            coords = ((vertex[1], vertex[2]) if axis == 0 else
                      (vertex[0], vertex[2]) if axis == 1 else (vertex[0], vertex[1]))
            uv.data[index].uv = (coords[0] / tile_size_m, coords[1] / tile_size_m)


def cylindrical_uv_ring(points: tuple[tuple[float, float, float], ...], *,
                        radius: float, tile_size_m: float = 1.5) -> tuple[tuple[float, float], ...]:
    """Seam-aware, meter-scaled UVs for the side of a fluted/lathed column.

    The number of turns is an integer, preventing a visible albedo/normal seam
    at the back of a cylindrical object even when circumference is not an exact
    multiple of the requested texture scale.
    """
    if not all(math.isfinite(n) and n > 0 for n in (radius, tile_size_m)):
        raise ValueError("Cylindrical UV radius and tile size must be positive and finite")
    angles = [(math.atan2(y, x) % math.tau) for x, y, _ in points]
    if angles and max(angles) - min(angles) > math.pi:
        angles = [angle + math.tau if angle < math.pi else angle for angle in angles]
    repeats = max(1, round(math.tau * radius / tile_size_m))
    return tuple((angle * repeats / math.tau, z / tile_size_m)
                 for angle, (_, _, z) in zip(angles, points))


def ensure_cylindrical_uvs(mesh: Any, *, tile_size_m: float = 1.5) -> None:
    """Assign seamless cylindrical side UVs, with planar UVs on end caps."""
    if mesh.uv_layers:
        return
    outer_radius = max(math.hypot(v.co.x, v.co.y) for v in mesh.vertices)
    uv = mesh.uv_layers.new(name="Kumiki / wrapped column")
    for polygon in mesh.polygons:
        indices = tuple(polygon.loop_indices)
        points = tuple(tuple(mesh.vertices[mesh.loops[i].vertex_index].co) for i in indices)
        if abs(polygon.normal.z) > 0.78:
            coords = tuple((x / tile_size_m, y / tile_size_m) for x, y, _ in points)
        else:
            coords = cylindrical_uv_ring(points, radius=outer_radius, tile_size_m=tile_size_m)
        for index, coord in zip(indices, coords):
            uv.data[index].uv = coord


def _packed_png(bpy: Any, label: str, payload: bytes, *, non_color: bool) -> Any:
    name = f"Kumiki / {label} / {hashlib.sha256(payload).hexdigest()[:16]}"
    cached = bpy.data.images.get(name)
    if cached is not None:
        return cached
    # Blender's glTF exporter expects Image Texture datablocks backed by PNG
    # images. Pack into the .blend so no temporary path is needed downstream.
    with tempfile.TemporaryDirectory(prefix="kumiki-pbr-") as temporary:
        path = Path(temporary) / "tile.png"
        path.write_bytes(payload)
        image = bpy.data.images.load(str(path), check_existing=False)
        image.name = name
        if non_color:
            image.colorspace_settings.name = "Non-Color"
        image.pack()
    if image.packed_file is None:
        raise RuntimeError(f"Failed to pack generated texture {name}")
    return image


def attach_packed_pbr(bpy: Any, material: Any, spec: PBRSurface) -> None:
    """Wire color, roughness and tangent-space normal images into Principled."""
    if not spec.texture_kind:
        return
    maps = make_pbr_maps(spec.texture_kind, spec.color, spec.roughness,
                         weathering_strength=spec.weathering_strength)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")

    albedo = nodes.new("ShaderNodeTexImage")
    albedo.label = "Kumiki / embedded albedo"
    albedo.image = _packed_png(bpy, "base-color", maps.base_color_png, non_color=False)
    links.new(albedo.outputs["Color"], shader.inputs["Base Color"])

    rough = nodes.new("ShaderNodeTexImage")
    rough.label = "Kumiki / embedded roughness"
    rough.image = _packed_png(bpy, "roughness", maps.roughness_png, non_color=True)
    links.new(rough.outputs["Color"], shader.inputs["Roughness"])

    normal_image = nodes.new("ShaderNodeTexImage")
    normal_image.label = "Kumiki / embedded micro-normal"
    normal_image.image = _packed_png(bpy, "normal", maps.normal_png, non_color=True)
    normal = nodes.new("ShaderNodeNormalMap")
    normal.label = "Kumiki / subtle surface relief"
    normal.inputs["Strength"].default_value = 0.42
    links.new(normal_image.outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], shader.inputs["Normal"])
