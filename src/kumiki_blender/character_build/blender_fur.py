"""Blender worker: strand fur on a textured low-poly mesh with Blender's procedural hair assets.

Arguments (JSON): mesh (with a UV map), basecolor (PNG on that UV map), output, render (bool),
weights (optional PLY: red channel = fur weight, see ``scatter.fur_weights``),
systems: [{name, mask (PNG on the UV map, white = fur), density_per_m2, length_m, radius_m,
clump, frizz_m, seed, box_min, box_max (optional, metres), use_fur_attribute, comb (direction
vector), comb_strength}].

Each system is a geometry-nodes modifier on the mesh: Generate Hair Curves (density and
mask texture on the UV map) -> Clump -> Frizz -> Set Hair Curve Profile. The mesh stays in
the output. Each strand takes the base colour at its root (the surface UV it stores), so
the fur keeps the character's colours.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import (activate, arguments, clear_startup_scene,  # noqa: E402
                                                           import_mesh, write_report)

ASSETS = Path(bpy.utils.resource_path("LOCAL")) / "datafiles" / "assets" / "nodes" / "procedural_hair_node_assets.blend"
NEEDED = ("Generate Hair Curves", "Clump Hair Curves", "Frizz Hair Curves", "Set Hair Curve Profile")


def hair_material(basecolor: Path):
    material = bpy.data.materials.new("FurStrands")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    uv = nodes.new("ShaderNodeAttribute")
    uv.attribute_name = "surface_uv_coordinate"
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(basecolor), check_existing=True)
    tip = nodes.new("ShaderNodeHairInfo")
    shade = nodes.new("ShaderNodeMix")
    shade.data_type, shade.blend_type = "RGBA", "MULTIPLY"
    shade.inputs["B"].default_value = (0.75, 0.72, 0.7, 1.0)  # roots a little darker than tips
    ramp = nodes.new("ShaderNodeMath")
    ramp.operation = "SUBTRACT"
    ramp.inputs[0].default_value = 1.0
    links.new(uv.outputs["Vector"], image.inputs["Vector"])
    links.new(tip.outputs["Intercept"], ramp.inputs[1])
    links.new(ramp.outputs[0], shade.inputs["Factor"])
    links.new(image.outputs["Color"], shade.inputs["A"])
    bsdf = nodes["Principled BSDF"]
    links.new(shade.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    return material


def fur_tree(name: str, groups: dict, system: dict, material, mask_image):
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    group_in, group_out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    uv = nodes.new("GeometryNodeInputNamedAttribute")
    uv.data_type = "FLOAT_VECTOR"
    uv.inputs["Name"].default_value = "UVMap"

    def group(key: str):
        node = nodes.new("GeometryNodeGroup")
        node.node_tree = groups[key]
        return node

    generate = group("Generate Hair Curves")
    generate.inputs["Surface Source"].default_value = "Input"  # the "Hair Surface" geometry input
    generate.inputs["Attach to Surface"].default_value = True
    generate.inputs["Hair Length"].default_value = system["length_m"]
    generate.inputs["Density"].default_value = system["density_per_m2"]
    generate.inputs["Control Points"].default_value = 6
    generate.inputs["Seed"].default_value = system.get("seed", 0)
    generate.inputs["Hair Material"].default_value = material
    if mask_image is not None:
        generate.inputs["Mask Texture"].default_value = mask_image
    if system.get("use_fur_attribute"):  # per-vertex weight transferred from the drawings' colours
        weight = nodes.new("GeometryNodeInputNamedAttribute")
        weight.data_type = "FLOAT"
        weight.inputs["Name"].default_value = "fur"
        links.new(weight.outputs["Attribute"], generate.inputs["Density Mask"])
    if system.get("box_min") and system.get("box_max") and system.get("use_fur_attribute"):
        raise ValueError(f"System {name}: use a box or the fur attribute, not both")
    if system.get("box_min") and system.get("box_max"):
        # Grow this system only inside a box (metres), e.g. longer fur on the tail.
        position = nodes.new("GeometryNodeInputPosition")
        above = nodes.new("FunctionNodeCompare")
        above.data_type, above.mode, above.operation = "VECTOR", "ELEMENT", "GREATER_EQUAL"
        below = nodes.new("FunctionNodeCompare")
        below.data_type, below.mode, below.operation = "VECTOR", "ELEMENT", "LESS_EQUAL"
        above.inputs["B"].default_value = system["box_min"]
        below.inputs["B"].default_value = system["box_max"]
        inside = nodes.new("FunctionNodeBooleanMath")
        inside.operation = "AND"
        links.new(position.outputs["Position"], above.inputs["A"])
        links.new(position.outputs["Position"], below.inputs["A"])
        links.new(above.outputs["Result"], inside.inputs[0])
        links.new(below.outputs["Result"], inside.inputs[1])
        links.new(inside.outputs[0], generate.inputs["Density Mask"])
    links.new(group_in.outputs[0], generate.inputs["Hair Surface"])
    links.new(uv.outputs["Attribute"], generate.inputs["Surface UV Map"])
    clump = group("Clump Hair Curves")
    clump.inputs["Factor"].default_value = system.get("clump", 0.5)
    clump.inputs["Guide Distance"].default_value = 4 * system["length_m"]
    frizz = group("Frizz Hair Curves")
    frizz.inputs["Distance"].default_value = system.get("frizz_m", 0.15 * system["length_m"])
    profile = group("Set Hair Curve Profile")
    profile.inputs["Radius"].default_value = system["radius_m"]
    join = nodes.new("GeometryNodeJoinGeometry")
    comb_target = generate.outputs["Curves"]
    if system.get("comb"):
        # Brush the strands: each point moves along the comb direction by length * strength * t^1.5,
        # with t the position along the strand (0 at the root, which stays on the surface).
        comb = nodes.new("GeometryNodeSetPosition")
        parameter = nodes.new("GeometryNodeSplineParameter")
        power = nodes.new("ShaderNodeMath")
        power.operation, power.inputs[1].default_value = "POWER", 1.5
        scale = nodes.new("ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.inputs[0].default_value = [c * system["length_m"] * system.get("comb_strength", 0.8)
                                         for c in system["comb"]]
        links.new(parameter.outputs["Factor"], power.inputs[0])
        links.new(power.outputs[0], scale.inputs["Scale"])
        links.new(generate.outputs["Curves"], comb.inputs["Geometry"])
        links.new(scale.outputs["Vector"], comb.inputs["Offset"])
        comb_target = comb.outputs["Geometry"]
    links.new(comb_target, clump.inputs["Geometry"])
    links.new(clump.outputs["Geometry"], frizz.inputs["Geometry"])
    links.new(frizz.outputs["Geometry"], profile.inputs["Geometry"])
    links.new(group_in.outputs[0], join.inputs["Geometry"])
    links.new(profile.outputs["Geometry"], join.inputs["Geometry"])
    links.new(join.outputs["Geometry"], group_out.inputs[0])
    return tree


def main() -> None:
    args = arguments()
    output = Path(args["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    clear_startup_scene()
    body = import_mesh(args["mesh"], "FurBody")
    with bpy.data.libraries.load(str(ASSETS)) as (src, dst):
        dst.node_groups = list(NEEDED)
    groups = {g.name: g for g in dst.node_groups}
    base_material = bpy.data.materials.new("Body")
    base_material.use_nodes = True
    texture = base_material.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(args["basecolor"], check_existing=True)
    base_material.node_tree.links.new(texture.outputs["Color"],
                                      base_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    body.data.materials.clear()
    body.data.materials.append(base_material)
    if args.get("weights"):  # PLY whose red channel is the fur weight; nearest point to each vertex
        from mathutils.kdtree import KDTree

        donor = import_mesh(args["weights"], "FurWeights")
        colours = donor.data.color_attributes[0].data
        tree = KDTree(len(donor.data.vertices))
        for vertex in donor.data.vertices:
            tree.insert(donor.matrix_world @ vertex.co, vertex.index)
        tree.balance()
        attribute = body.data.attributes.new("fur", "FLOAT", "POINT")
        for vertex in body.data.vertices:
            _co, index, _distance = tree.find(body.matrix_world @ vertex.co)
            attribute.data[vertex.index].value = colours[index].color[0]
        bpy.data.objects.remove(donor)
        activate(body)
    strands = hair_material(Path(args["basecolor"]))
    body.data.materials.append(strands)  # the curves use the second material slot
    counts = {}
    for system in args["systems"]:
        mask = bpy.data.images.load(system["mask"], check_existing=True) if system.get("mask") else None
        if mask is not None:
            mask.colorspace_settings.name = "Non-Color"
        modifier = body.modifiers.new(system["name"], "NODES")
        modifier.node_group = fur_tree(f"Fur_{system['name']}", groups, system, strands, mask)
    evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    geometry = evaluated.evaluated_geometry()  # keep a reference: the curves live inside it
    curves = geometry.curves
    counts["strands"] = len(curves.curves) if curves is not None else 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "fur.blend"))
    renders = []
    if args.get("render"):
        scene = bpy.context.scene
        if args.get("engine", "CYCLES") == "CYCLES":  # hair needs ray tracing to read as fur
            scene.render.engine = "CYCLES"
            scene.cycles.samples = int(args.get("samples", 48))
            scene.cycles.use_denoising = True
            try:
                bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "METAL"
                bpy.context.preferences.addons["cycles"].preferences.get_devices()
                scene.cycles.device = "GPU"
            except Exception:  # no Metal device: CPU
                scene.cycles.device = "CPU"
        else:
            scene.render.engine = "BLENDER_EEVEE"
            scene.render.hair_type = "STRIP"  # "STRAND" draws one-pixel lines and ignores the radius
        scene.render.resolution_x = scene.render.resolution_y = int(args.get("render_px", 900))
        world = bpy.data.worlds.new("Studio")
        scene.world, world.use_nodes = world, True
        world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
        sun = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "SUN"))
        scene.collection.objects.link(sun)
        sun.rotation_euler, sun.data.energy = (0.8, 0.0, -0.6), 3.0
        cam = bpy.data.objects.new("FurCam", bpy.data.cameras.new("FurCam"))
        scene.collection.objects.link(cam)
        scene.camera = cam
        low, high = (Vector([f(v[i] for v in [body.matrix_world @ Vector(c) for c in body.bound_box]) for i in range(3)])
                     for f in (min, max))
        centre, size = (low + high) / 2, max(high - low)
        cam.data.type, cam.data.ortho_scale = "ORTHO", 1.15 * size
        for azimuth in args.get("render_azimuths", [0, 45, 135]):
            a = math.radians(azimuth)
            d = Vector((math.sin(a), -math.cos(a), 0.15)).normalized()
            cam.location = centre + d * 3 * size
            cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
            path = output / f"fur_{azimuth}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            renders.append(path.name)
    write_report(output, {"systems": args["systems"], **counts, "renders": renders, "files": ["fur.blend"] + renders})


main()
