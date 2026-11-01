"""High-detail torch study for Liberty. Blender 4.2+; independent of staging.

This is a geometric study, not a surveyed reconstruction. All details are
real mesh geometry and remain separately editable.
"""
from __future__ import annotations
import math
import bpy
from mathutils import Vector


def _material(name, color, metallic=0.7, roughness=0.38):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return m


def _lathe(name, profile, material, segments=96, flutes=0, flute_depth=0.0):
    """Turn a closed radial profile, optionally cutting true longitudinal flutes."""
    vertices, faces = [], []
    for z, radius in profile:
        for j in range(segments):
            a = 2 * math.pi * j / segments
            r = radius - flute_depth * (0.5 + 0.5 * math.cos(a * flutes)) ** 8 if flutes else radius
            vertices.append((r * math.cos(a), r * math.sin(a), z))
    for i in range(len(profile)-1):
        for j in range(segments):
            k = (j+1) % segments
            faces.append((i*segments+j, i*segments+k, (i+1)*segments+k, (i+1)*segments+j))
    faces.append(tuple(reversed(tuple(range(segments)))))
    faces.append(tuple((len(profile)-1)*segments+j for j in range(segments)))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    for p in mesh.polygons:
        p.use_smooth = len(p.vertices) == 4
    bevel = obj.modifiers.new("Soft cast edges", "BEVEL")
    bevel.width = .004
    bevel.segments = 2
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return obj


def _rod(name, start, end, radius, material, vertices=12):
    direction = Vector(end) - Vector(start)
    midpoint = (Vector(start) + Vector(end)) / 2
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius,
                                       depth=direction.length, location=midpoint)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    bevel = obj.modifiers.new("Cast-edge rounding", "BEVEL")
    bevel.width = min(.006, radius * .3)
    bevel.segments = 2
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return obj


def build_torch_detail(origin=(0, 0, 0), scale=1.0):
    """Construct separately selectable shaft, gallery, brackets and flame.

    Local Z is the torch axis. The object can be parented to a hand control;
    transforms are applied only to a common empty, leaving parts editable.
    """
    bronze = _material("Liberty / torch oxidized copper", (.12, .37, .30))
    dark = _material("Liberty / torch recessed bronze", (.075, .17, .15))
    gold = _material("Liberty / torch gilded flame", (.78, .47, .11), .82, .25)
    root = bpy.data.objects.new("Liberty / torch detailed assembly", None)
    bpy.context.collection.objects.link(root)
    parts = []
    parts.append(_lathe("Torch / fluted grip and neck", [
        (0, .115), (.04, .13), (.09, .125), (.15, .105), (.28, .092),
        (.47, .09), (.57, .106), (.65, .12), (.70, .15), (.74, .18),
        (.79, .20), (.84, .195), (.89, .23), (.92, .23)], bronze, flutes=16,
        flute_depth=.008))
    parts.append(_lathe("Torch / gallery lower cornice", [
        (.89, .21), (.91, .27), (.935, .29), (.96, .285), (.98, .26)], bronze))
    parts.append(_lathe("Torch / gallery upper coping", [
        (1.18, .27), (1.205, .29), (1.235, .30), (1.26, .28)], bronze))
    parts.append(_lathe("Torch / dark recessed gallery", [
        (.98, .22), (1.00, .235), (1.18, .235)], dark))
    parts.append(_lathe("Torch / burner cup", [
        (1.25, .21), (1.29, .22), (1.33, .19), (1.38, .15), (1.42, .12)], bronze))
    for j in range(24):
        a = j * math.tau / 24
        c, s = math.cos(a), math.sin(a)
        parts.append(_rod(f"Torch / gallery baluster {j:02d}",
                          (.251*c, .251*s, .987),
                          (.251*c, .251*s, 1.195), .012, bronze))
    for j in range(12):
        a = j * math.tau / 12
        c, s = math.cos(a), math.sin(a)
        parts.append(_rod(f"Torch / diagonal gallery brace {j:02d}",
                          (.19*c, .19*s, .79),
                          (.255*c, .255*s, .93), .014, bronze))
    # Individual asymmetric flame sheets, with volumetric thickness, rather
    # than a single cone. Stylized until calibrated to monument photography.
    for j in range(9):
        a = math.tau*j/9
        height = 0.38 + 0.12*math.sin(j*2.4)
        verts = []
        for k in range(9):
            t = k/8
            z = 1.36 + height*t
            angle = a + .42*t + .1*math.sin(t*math.pi)
            radius = (.15*(1-t) + .018) * (1+.17*math.sin(j))
            for side in (-1, 1):
                aa = angle + side * (.21*(1-t)+.015)
                verts.append((radius*math.cos(aa), radius*math.sin(aa), z))
        faces = []
        for k in range(8):
            n = 2*k
            faces.append((n,n+1,n+3,n+2))
        mesh = bpy.data.meshes.new(f"Flame / sheet {j:02d}")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
        solid = obj.modifiers.new("Cast flame thickness", "SOLIDIFY")
        solid.thickness = .018
        obj.data.materials.append(gold)
        parts.append(obj)
    for part in parts:
        part.parent = root
    root.location = origin
    root.scale = (scale,)*3
    root["modeling_status"] = "reference-guided detail study; pending hand contact and photo comparison"
    return root
