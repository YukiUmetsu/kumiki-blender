"""Blender background capture: visible semantic IDs, unlit color and metric Z depth.

Usage: blender -b scene.blend --python /path/to/character_fidelity/blender_capture.py --
       --manifest path/config.json --output path/new-capture

Does not save or overwrite the source .blend. Scene changes are in-memory only.
"""

from __future__ import annotations

import argparse
import colorsys
import fnmatch
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from kumiki_blender.character_fidelity.schema import load_manifest  # noqa: E402


def material_as_emission(original, cache):
    if original is None:
        return None
    if original.name in cache:
        return cache[original.name]
    clone = original.copy()
    clone.name = "CF_QA_ALBEDO_" + original.name
    clone.use_nodes = True
    nodes, links = clone.node_tree.nodes, clone.node_tree.links
    output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")
    shader = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    emit = nodes.new("ShaderNodeEmission")
    if shader is not None:
        base = shader.inputs.get("Base Color")
        if base is not None:
            if base.is_linked:
                links.new(base.links[0].from_socket, emit.inputs["Color"])
            else:
                emit.inputs["Color"].default_value = base.default_value
    else:
        emit.inputs["Color"].default_value = original.diffuse_color
    links.new(emit.outputs[0], output.inputs["Surface"])
    cache[original.name] = clone
    return clone


def camera_depth_extent(objects, camera):
    inverse = camera.matrix_world.inverted()
    depths = []
    for obj in objects:
        if not hasattr(obj, "bound_box"):
            continue
        for corner in obj.bound_box:
            # Camera looks down local negative Z.
            v = inverse @ (obj.matrix_world @ Vector(corner))
            if -v.z > 0:
                depths.append(-v.z)
    return [round(min(depths), 5), round(max(depths), 5)] if depths else None


def build_visible_bvh(groups):
    """Single world-space BVH of evaluated, assigned geometry; includes modifiers."""
    from mathutils.bvhtree import BVHTree
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices, faces = [], []
    for objs in groups.values():
        for obj in objs:
            evaluated = obj.evaluated_get(depsgraph)
            try:
                mesh = evaluated.to_mesh()
            except RuntimeError:
                continue
            if mesh is None:
                continue
            offset = len(vertices)
            matrix = evaluated.matrix_world
            vertices.extend(matrix @ vertex.co for vertex in mesh.vertices)
            faces.extend(tuple(offset + index for index in face.vertices)
                         for face in mesh.polygons if len(face.vertices) >= 3)
            evaluated.to_mesh_clear()
    if not faces:
        raise ValueError("No renderable assigned mesh geometry for depth capture")
    return BVHTree.FromPolygons(vertices, faces, all_triangles=False)


def render_depth(camera, scene, bvh, width, height):
    """Metric optical-axis Z at camera pixel centers; nan for ray misses."""
    frame = camera.data.view_frame(scene=scene)
    left, right = min(v.x for v in frame), max(v.x for v in frame)
    bottom, top = min(v.y for v in frame), max(v.y for v in frame)
    negative_z = sum(v.z for v in frame) / len(frame)
    matrix = camera.matrix_world
    rotation = matrix.to_3x3()
    back_transform = matrix.inverted()
    orthographic = camera.data.type == "ORTHO"
    out = np.full((height, width), np.nan, dtype=np.float32)
    if camera.data.type not in ("ORTHO", "PERSP"):
        raise ValueError("Only orthographic and perspective reference cameras are supported")
    forward = rotation @ Vector((0, 0, -1))
    for row in range(height):
        cy = top - (row + .5) / height * (top - bottom)
        for col in range(width):
            cx = left + (col + .5) / width * (right - left)
            if orthographic:
                ray_origin = matrix @ Vector((cx, cy, 0))
                ray_direction = forward
            else:
                ray_origin = matrix.translation
                ray_direction = (rotation @ Vector((cx, cy, negative_z))).normalized()
            hit, _, _, _ = bvh.ray_cast(ray_origin, ray_direction)
            if hit is not None:
                depth_m = -(back_transform @ hit).z
                if depth_m > 0:
                    out[row, col] = depth_m
    return out


def capture(manifest_path: Path, output: Path) -> None:
    config = load_manifest(manifest_path)
    if output.exists() or output.is_symlink():
        raise ValueError("Capture output must not already exist")
    scene = bpy.context.scene
    objects = [obj for obj in scene.objects if obj.type in ("MESH", "CURVE", "SURFACE", "FONT")]
    groups, assignments, matched = {}, {}, {}
    for part in config["parts"]:
        names = [obj for obj in objects if any(fnmatch.fnmatchcase(obj.name, glob) for glob in part["objects"])]
        if not names:
            raise ValueError(f"No Blender objects matched semantic part {part['id']}")
        for obj in names:
            if obj.name in assignments:
                raise ValueError(f"Object {obj.name} belongs to both {assignments[obj.name]} and {part['id']}")
            assignments[obj.name] = part["id"]
        groups[part["id"]] = names
        matched[part["id"]] = [o.name for o in names]
    palette = {part["id"]: [round(v, 6) for v in colorsys.hsv_to_rgb(i / len(config["parts"]), 0.95, 0.95)]
               for i, part in enumerate(config["parts"])}
    camera_by_view = {}
    for view in config["views"]:
        camera = bpy.data.objects.get(view["camera"])
        if camera is None or camera.type != "CAMERA":
            raise ValueError(f"Missing Blender camera: {view['camera']}")
        camera_by_view[view["id"]] = camera
    original_hide = {obj.name: obj.hide_render for obj in objects}
    original_color = {obj.name: tuple(obj.color) for obj in objects}
    original_materials = {obj.name: [slot.material for slot in obj.material_slots] for obj in objects}
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "assignments": assignments, "palette": palette, "views": {},
              "source_scene_saved": False, "occlusion": "all assigned parts rendered together"}
    try:
        for obj in objects:
            obj.hide_render = obj.name not in assignments
            if obj.name in assignments:
                obj.color = (*palette[assignments[obj.name]], 1.0)
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = True
        scene.view_settings.view_transform = "Standard"
        
        # Workbench flat object color provides a single visibility-aware ID pass.
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "OBJECT"
        scene.display.shading.show_shadows = False
        scene.display.shading.show_cavity = False
        for view in config["views"]:
            vid = view["id"]
            view_dir = output / vid
            view_dir.mkdir()
            scene.camera = camera_by_view[vid]
            scene.render.resolution_x, scene.render.resolution_y = view["width"], view["height"]
            scene.render.filepath = str(view_dir / "labels.png")
            bpy.ops.render.render(write_still=True)
            record["views"][vid] = {"labels": f"{vid}/labels.png", "albedo": f"{vid}/albedo.png",
                "depth_npz": f"{vid}/depth_m.npz", "part_depth_extent_m": {
                    pid: camera_depth_extent(items, scene.camera) for pid, items in groups.items()}}
        # Replace material shaders *in memory* with unlit Base Color; preserve texture connections.
        material_cache = {}
        for obj in objects:
            if obj.name not in assignments:
                continue
            for index, slot in enumerate(obj.material_slots):
                slot.material = material_as_emission(original_materials[obj.name][index], material_cache)
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 1
        scene.render.image_settings.color_mode = "RGBA"
        bvh = build_visible_bvh(groups)
        for view in config["views"]:
            vid, folder = view["id"], output / view["id"]
            scene.camera = camera_by_view[vid]
            scene.render.resolution_x, scene.render.resolution_y = view["width"], view["height"]
            scene.render.filepath = str(folder / "albedo.png")
            bpy.ops.render.render(write_still=True)
            depth_map = render_depth(scene.camera, scene, bvh, view["width"], view["height"])
            np.savez_compressed(folder / "depth_m.npz", depth_m=depth_map)
        (output / "capture.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    finally:
        for obj in objects:
            obj.hide_render = original_hide[obj.name]
            obj.color = original_color[obj.name]
            if obj.name in original_materials:
                for slot, material in zip(obj.material_slots, original_materials[obj.name]):
                    slot.material = material


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    capture(args.manifest, args.output)


if __name__ == "__main__":
    main()
