"""Local-only photographic QA of a REAL Sphinx mesh before source selection.

OBJ/FBX/GLB meshes can be purchased or independently licensed. Their geometry
and embedded textures must never be copied into the open-source repository
without explicit redistribution rights. This module imports those files in
Blender, renders four inspection PNGs, and writes metadata only: no GLB, OBJ,
FBX, blend file, geometry coordinates, or source texture is exported.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_import import AssetImportError
from .render_profiles import select_render_profile


ALLOWED_FORMATS = frozenset({".glb", ".obj", ".fbx"})
FRONT_DIRECTIONS = {
    "-X": (-1.0, 0.0),
    "+X": (1.0, 0.0),
    "-Y": (0.0, -1.0),
    "+Y": (0.0, 1.0),
}
MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class SphinxReviewInput:
    source: Path
    output: Path
    front_axis: str


def prepare_sphinx_review(source: str | Path, output: str | Path, *,
                          front_axis: str = "-X") -> SphinxReviewInput:
    """Preflight read-only asset intake without executing Blender or exporting it."""
    if front_axis not in FRONT_DIRECTIONS:
        raise AssetImportError("Sphinx front axis must be one of -X, +X, -Y, +Y")
    path = Path(source).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in ALLOWED_FORMATS:
        raise AssetImportError("Sphinx review source must be a local GLB, OBJ, or FBX mesh")
    if path.stat().st_size == 0 or path.stat().st_size > MAX_SOURCE_BYTES:
        raise AssetImportError("Sphinx review source must be nonempty and at most 2 GiB")
    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise AssetImportError("Review directory already exists; choose a new revision")
    return SphinxReviewInput(path, destination, front_axis)


def review_camera_layout(
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    *, front_axis: str,
) -> dict[str, dict[str, Any]]:
    """Compute head-focused cameras from the actual scan, never a guessed face.

    Camera placement is deliberately separate from geometry editing. For the
    Sphinx, the head sits at the forward end of its recumbent body, not at the
    overall object's origin. The front axis describes the *imported* mesh.
    """
    if front_axis not in FRONT_DIRECTIONS:
        raise AssetImportError("Unknown Sphinx front axis")
    values = (*bounds_min, *bounds_max)
    if any(type(value) not in (float, int) or not math.isfinite(value)
           for value in values):
        raise AssetImportError("Sphinx review bounds must be finite")
    sizes = [b-a for a, b in zip(bounds_min, bounds_max)]
    if any(size <= 1e-5 for size in sizes):
        raise AssetImportError("Sphinx review needs nondegenerate mesh bounds")
    dx, dy = FRONT_DIRECTIONS[front_axis]
    sx, sy = -dy, dx
    center_x = (bounds_min[0]+bounds_max[0])/2
    center_y = (bounds_min[1]+bounds_max[1])/2
    mid_z = (bounds_min[2]+bounds_max[2])/2
    extent = abs(dx)*sizes[0]+abs(dy)*sizes[1]
    cross = abs(sx)*sizes[0]+abs(sy)*sizes[1]
    # 35% toward the nose end; above body height and below the headdress tip.
    face = (center_x+dx*extent*.35, center_y+dy*extent*.35,
            bounds_min[2]+sizes[2]*.81)
    whole = (center_x, center_y, mid_z)

    def camera(target: tuple[float, float, float], direction: tuple[float, float],
               *, scale: float) -> dict[str, Any]:
        distance = max(sizes)*1.7
        position = (target[0]+direction[0]*distance,
                    target[1]+direction[1]*distance,
                    target[2]+sizes[2]*.09)
        return {"camera": position, "target": target, "ortho_scale": scale}

    return {
        "frontal": camera(face, (dx, dy), scale=max(cross*.68, sizes[2]*.70)),
        "face-three-quarter": camera(
            face, (dx*.83+sx*.55, dy*.83+sy*.55),
            scale=max(cross*.64, sizes[2]*.68)),
        "side-profile": camera(whole, (sx, sy),
                               scale=max(extent*1.25, sizes[2]*1.55)),
        "whole-frontal": camera(whole, (dx, dy),
                                scale=max(cross*1.30, sizes[2]*1.55)),
    }


def render_sphinx_source_review(review: SphinxReviewInput, *,
                                bpy_module: Any) -> dict[str, Any]:
    """Render-only review: no scene archive or mesh/texture redistribution."""
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError("Sphinx source review requires Blender 4.2 or newer")
    # Command is intended for --factory-startup; remove the default cube,
    # lights and camera if Blender was launched with a saved startup scene.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    if review.source.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(review.source))
    elif review.source.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(review.source))
    else:
        bpy.ops.wm.obj_import(filepath=str(review.source))

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("Sphinx source contains no mesh objects")
    from mathutils import Vector

    points = [obj.matrix_world @ Vector(corner)
              for obj in mesh_objects for corner in obj.bound_box]
    low = tuple(float(min(p[axis] for p in points)) for axis in range(3))
    high = tuple(float(max(p[axis] for p in points)) for axis in range(3))
    views = review_camera_layout(low, high, front_axis=review.front_axis)
    scene = bpy.context.scene
    profile = select_render_profile(960, 720, 32)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile.samples
    scene.render.resolution_x = profile.width
    scene.render.resolution_y = profile.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.new("Scan inspection / soft neutral environment")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (.31, .31, .31, 1)
    background.inputs["Strength"].default_value = .65
    scene.world = world
    center = tuple((a+b)/2 for a, b in zip(low, high))
    size = max(high[a]-low[a] for a in range(3))
    for name, position, energy in (
        ("Sculpture soft key", (center[0]-size, center[1]-size, center[2]+size), 1700),
        ("Sculpture rim", (center[0]+size, center[1]+size, center[2]+size), 1050),
    ):
        lamp = bpy.data.lights.new(name, type="AREA")
        lamp.energy = energy
        lamp.shape = "DISK"
        lamp.size = size*.85
        obj = bpy.data.objects.new(name, lamp)
        scene.collection.objects.link(obj)
        obj.location = position
        direction = Vector(center)-obj.location
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data = bpy.data.cameras.new("Sphinx source inspection")
    camera_data.type = "ORTHO"
    camera_obj = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    review.output.mkdir(parents=True, exist_ok=False)
    screenshots = []
    for name, view in views.items():
        camera_obj.location = view["camera"]
        direction = Vector(view["target"])-camera_obj.location
        camera_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        camera_data.ortho_scale = view["ortho_scale"]
        output = review.output / f"{name}.png"
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        screenshots.append(output.name)

    digest = hashlib.sha256()
    with review.source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    report = {
        "review_kind": "local-only-sphinx-geometry",
        "source_filename": review.source.name,
        "source_sha256": digest.hexdigest(),
        "source_format": review.source.suffix.lower(),
        "front_axis": review.front_axis,
        "mesh_objects": len(mesh_objects),
        "triangles": sum(sum(len(poly.vertices)-2 for poly in obj.data.polygons)
                         for obj in mesh_objects),
        "bounds": [low, high],
        "previews": screenshots,
        "no_source_geometry_exported": True,
        "no_source_textures_exported": True,
        "source_authenticity_verified": False,
        "source_license_verified": False,
        "note": ("Compare the frontal/profile photographs and check provenance "
                 "before using this source commercially; PNGs are local QA only."),
    }
    (review.output/"review.json").write_text(json.dumps(report, indent=2)+"\n",
                                              encoding="utf-8")
    return report
