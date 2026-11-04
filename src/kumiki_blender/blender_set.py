"""Build a set from data: materials, world, lights and props.

Positions use Kumiki Y-up metres. Mesh shapes use Blender axes with +Z as the
object's up, so a lathe profile is (radius, height) in either convention.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

from .shot_plan import ShotPlanError, to_blender_position


def _input(shader: Any, name: str, value: Any) -> None:
    if name in shader.inputs:
        shader.inputs[name].default_value = value


def _texture(nodes: Any, shader_input: Any, path: Path, coordinates: Any, colorspace: str) -> Any:
    image = nodes.nodes.new("ShaderNodeTexImage")
    image.image = _load_image(path)
    image.image.colorspace_settings.name = colorspace
    nodes.links.new(coordinates, image.inputs["Vector"])
    if shader_input is not None:
        nodes.links.new(image.outputs["Color"], shader_input)
    return image


_BPY: list[Any] = []


def _load_image(path: Path) -> Any:
    return _BPY[0].data.images.load(str(path), check_existing=True)


def build_material(bpy: Any, name: str, spec: dict[str, Any], root: Path) -> Any:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree
    shader = nodes.nodes["Principled BSDF"]
    _input(shader, "Base Color", (*spec.get("color", (.8, .8, .8)), 1))
    _input(shader, "Roughness", spec.get("roughness", .5))
    _input(shader, "Metallic", spec.get("metallic", 0))
    _input(shader, "Coat Weight", spec.get("coat", 0))
    _input(shader, "Coat Roughness", spec.get("coat_roughness", .1))
    _input(shader, "Sheen Weight", spec.get("sheen", 0))
    if "emission" in spec:
        _input(shader, "Emission Color", (*spec["emission"], 1))
        _input(shader, "Emission Strength", spec.get("emission_strength", 1))
    maps = spec.get("maps")
    if maps:
        coordinates = nodes.nodes.new("ShaderNodeTexCoord")
        mapping = nodes.nodes.new("ShaderNodeMapping")
        tile = maps.get("tile", (1, 1))
        mapping.inputs["Scale"].default_value = (tile[0], tile[1], 1)
        nodes.links.new(coordinates.outputs["UV"], mapping.inputs["Vector"])
        out = mapping.outputs["Vector"]
        if "base_color" in maps:
            _texture(nodes, shader.inputs["Base Color"], root / maps["base_color"], out, "sRGB")
        if "roughness" in maps:
            _texture(nodes, shader.inputs["Roughness"], root / maps["roughness"], out, "Non-Color")
        if "normal" in maps:
            image = _texture(nodes, None, root / maps["normal"], out, "Non-Color")
            normal = nodes.nodes.new("ShaderNodeNormalMap")
            normal.inputs["Strength"].default_value = maps.get("normal_strength", .7)
            nodes.links.new(image.outputs["Color"], normal.inputs["Color"])
            nodes.links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    return material


def _link(bpy: Any, obj: Any) -> Any:
    bpy.context.collection.objects.link(obj)
    return obj


def _mesh_object(bpy: Any, name: str, vertices: list[Any], faces: list[Any], material: Any, smooth: bool) -> Any:
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    if smooth:
        data.polygons.foreach_set("use_smooth", np.ones(len(data.polygons), bool))
    if material is not None:
        data.materials.append(material)
    return _link(bpy, bpy.data.objects.new(name, data))


def _box(bpy: Any, name: str, spec: dict[str, Any], material: Any) -> Any:
    x, y, z = (value / 2 for value in spec["size"])
    vertices = [(sx * x, sy * y, sz * z) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    obj = _mesh_object(bpy, name, vertices, faces, material, False)
    data = obj.data
    layer = data.uv_layers.new()
    span = max(spec["size"][0], spec["size"][1])
    for loop in data.loops:
        vertex = data.vertices[loop.vertex_index].co
        layer.data[loop.index].uv = (vertex.x / span + .5, vertex.y / span + .5)
    bevel = spec.get("bevel", .002)
    if bevel:
        modifier = obj.modifiers.new("bevel", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    return obj


def _lathe(bpy: Any, name: str, spec: dict[str, Any], material: Any) -> Any:
    profile = spec["profile"]
    segments = spec.get("segments", 96)
    if len(profile) < 2:
        raise ShotPlanError(f"Lathe {name} needs a profile")
    vertices = []
    for radius, height in profile:
        vertices += [(radius * math.cos(step / segments * math.tau), radius * math.sin(step / segments * math.tau), height)
                     for step in range(segments)]
    faces = []
    for ring in range(len(profile) - 1):
        for step in range(segments):
            a = ring * segments + step
            b = ring * segments + (step + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    obj = _mesh_object(bpy, name, vertices, faces, material, True)
    return obj


def _document(bpy: Any, name: str, spec: dict[str, Any], material: Any) -> Any:
    width, height = spec["width"], spec["height"]
    rng = random.Random(spec.get("seed", 1))
    cell = spec.get("cell", .003)
    columns, rows = max(2, int(width / cell)), max(2, int(height / cell))
    curl = spec.get("curl", .003)
    phases = [rng.uniform(0, 6) for _ in range(3)]
    roll = spec.get("roll")
    vertices, uvs = [], []
    for row in range(rows + 1):
        for column in range(columns + 1):
            u, v = column / columns, row / rows
            x, y = (u - .5) * width, (v - .5) * height
            z = curl * 4 * (u - .5) ** 2 + .25 * curl * math.sin(v * 5 + phases[0]) * (u - .5) * 2 \
                + .15 * curl * math.sin(u * 7 + phases[1]) * math.sin(v * 4 + phases[2])
            if roll and v > roll[0]:
                angle = (v - roll[0]) / (1 - roll[0]) * roll[1]
                radius = height * (1 - roll[0]) / max(roll[1], 1e-3)
                y = (roll[0] - .5) * height + radius * math.sin(angle)
                z += radius * (1 - math.cos(angle))
            vertices.append((x, y, z + .0005))
            uvs.append((u, v))
    faces = [(row * (columns + 1) + column, row * (columns + 1) + column + 1,
              (row + 1) * (columns + 1) + column + 1, (row + 1) * (columns + 1) + column)
             for row in range(rows) for column in range(columns)]
    obj = _mesh_object(bpy, name, vertices, faces, material, True)
    layer = obj.data.uv_layers.new()
    for loop in obj.data.loops:
        layer.data[loop.index].uv = uvs[loop.vertex_index]
    obj.modifiers.new("thickness", "SOLIDIFY").thickness = spec.get("thickness", .00025)
    bend = obj.modifiers.new("bend", "SIMPLE_DEFORM")
    bend.deform_method = "BEND"
    bend.deform_axis = "X"
    bend.angle = 0
    return obj


BUILDERS = {"box": _box, "lathe": _lathe, "document": _document}


def _place(obj: Any, spec: dict[str, Any]) -> None:
    obj.rotation_mode = "XYZ"
    obj.location = to_blender_position(spec.get("location", (0, 0, 0)))
    x, y, z = (math.radians(value) for value in spec.get("rotation_deg", (0, 0, 0)))
    obj.rotation_euler = (x, -z, y)
    obj.scale = spec.get("scale", (1, 1, 1))


def _world(bpy: Any, spec: dict[str, Any] | None, root: Path) -> None:
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("set-world")
    scene.world.use_nodes = True
    tree = scene.world.node_tree
    background = tree.nodes["Background"]
    if not spec:
        background.inputs["Strength"].default_value = 0
        return
    background.inputs["Strength"].default_value = spec.get("strength", 1)
    if "hdri" in spec:
        environment = tree.nodes.new("ShaderNodeTexEnvironment")
        environment.image = _load_image(root / spec["hdri"])
        mapping = tree.nodes.new("ShaderNodeMapping")
        coordinates = tree.nodes.new("ShaderNodeTexCoord")
        mapping.inputs["Rotation"].default_value = (0, 0, math.radians(spec.get("rotation_deg", 0)))
        tree.links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
        tree.links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
        tree.links.new(environment.outputs["Color"], background.inputs["Color"])


def _light(bpy: Any, spec: dict[str, Any]) -> Any:
    data = bpy.data.lights.new(spec["id"], spec.get("type", "AREA"))
    data.energy = spec.get("energy", 1)
    data.color = spec.get("color", (1, 1, 1))
    if data.type == "AREA":
        data.shape = "RECTANGLE"
        size = spec.get("size", (.5, .5))
        data.size, data.size_y = size[0], size[1]
    obj = _link(bpy, bpy.data.objects.new(spec["id"], data))
    _place(obj, spec)
    return obj


def build_set(bpy: Any, path: Path) -> dict[str, Any]:
    """Create the set. Returns objects by id so a bake can address them."""
    _BPY[:] = [bpy]
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1:
        raise ShotPlanError("Only set schema_version 1 is supported")
    root = path.parent
    materials = {name: build_material(bpy, name, spec, root) for name, spec in (raw.get("materials") or {}).items()}
    _world(bpy, raw.get("world"), root)
    objects: dict[str, Any] = {}
    for spec in raw.get("objects") or []:
        kind, name = spec.get("kind"), spec.get("id")
        if kind not in BUILDERS:
            raise ShotPlanError(f"Unknown set object kind {kind}")
        if name in objects:
            raise ShotPlanError(f"Duplicate set object id {name}")
        material = None
        if spec.get("material") is not None:
            if spec["material"] not in materials:
                raise ShotPlanError(f"Set object {name} uses unknown material {spec['material']}")
            material = materials[spec["material"]]
        if kind == "document" and spec.get("face"):
            material = build_material(bpy, f"{name}-paper", {
                "roughness": spec.get("roughness", .62), "sheen": .25,
                "maps": {"base_color": spec["face"]}}, root)
        objects[name] = BUILDERS[kind](bpy, name, spec, material)
        _place(objects[name], spec)
    for spec in raw.get("lights") or []:
        objects[spec["id"]] = _light(bpy, spec)
    return objects
