"""Christ the Redeemer: original procedural, dimensioned Art Deco sculpture.

30 m figure / 29.4 m surveyed arm span / 8 m pedestal.
Original geometry only; no external scan or mesh is incorporated.
Run: blender -b --python-exit-code 1 --python scripts/landmarks/christ_redeemer.py --
     --output .review/christ.glb --views .review/christ --verify
"""
import argparse
from array import array
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

TAU = math.tau
STONE = None
CARVING = None
GRANITE = None
HEAD_TILT_DEG = 22.0  # Measured against authentic side-view scan previews.
MOSAIC_PITCH = .72 / 24  # 3 cm squares split into ~4.2 cm triangular tesserae.


def soapstone_mosaic():
    """Three original, tileable, metric albedo/roughness/normal soapstone maps.

    The 3 cm tesserae are substantially larger than the subtle height variation:
    the seams are depressions of less than a millimetre, not heavy black gridwork.
    """
    size = 768
    cells = 24
    tile_size = size // cells
    albedo = array("f")
    roughness = array("f")
    heights = array("f")

    def hash01(a, b, c):
        n = math.sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5453
        return n - math.floor(n)

    for y in range(size):
        cy, fy = divmod(y, tile_size)
        for x in range(size):
            cx, fx = divmod(x, tile_size)
            upper = int(fx + fy >= tile_size)
            # The original stone is distinctly assembled from hand-set
            # tesserae. At normal viewing distances the prior .046 modulation
            # vanished into Eevee noise and read as featureless white plastic.
            variation = (hash01(cx, cy, upper) - .5) * .088
            fine = (hash01(x // 3, y // 3, 7) - .5) * .006
            # Fine joints between the triangles, without hard dark grid lines.
            joint = min(fx, fy, tile_size - 1 - fx,
                        tile_size - 1 - fy, abs(fx + fy - tile_size) / 1.414)
            grout = -.032 * max(0, 1 - joint / 1.3)
            shade = variation + fine + grout
            albedo.extend((.665 + shade, .673 + shade, .650 + shade, 1))
            rough = min(.985, .895 + .075*hash01(cx, cy, upper) +
                        .022*(hash01(x//4, y//4, 19)-.5) - 1.6*grout)
            roughness.extend((rough, rough, rough, 1))
            heights.append(-.00040*max(0,1-joint/1.5) +
                           .000035*(hash01(x//3,y//3,23)-.5))

    # Finite differences use the physical UV repeat distance, rather than the
    # image pixel count, to keep grout depth realistic at any render size.
    normals = array("f")
    pixel_metres = .72 / size
    for y in range(size):
        for x in range(size):
            i=y*size+x
            left=heights[y*size+(x-1)%size]
            right=heights[y*size+(x+1)%size]
            below=heights[((y-1)%size)*size+x]
            above=heights[((y+1)%size)*size+x]
            dx=(right-left)/(2*pixel_metres)
            dy=(above-below)/(2*pixel_metres)
            inv=1/math.sqrt(dx*dx+dy*dy+1)
            normals.extend((.5-.5*dx*inv,.5-.5*dy*inv,.5+.5*inv,1))

    maps=[]
    for name,pixels in (("Soapstone | diffuse triangular tesserae",albedo),
                        ("Soapstone | high-roughness grain",roughness),
                        ("Soapstone | micron relief and grout",normals)):
        image=bpy.data.images.new(name,width=size,height=size,alpha=False)
        image.file_format="PNG"
        if name != "Soapstone | diffuse triangular tesserae":
            image.colorspace_settings.name="Non-Color"
        image.pixels.foreach_set(pixels)
        image.pack()
        maps.append(image)
    return tuple(maps)


def granite_blocks():
    """Original weathered ashlar granite with 1.44 m staggered blocks."""
    size=512
    block=size//4
    pixels=array("f")

    def hash01(a,b,c):
        n=math.sin(a*127.1+b*311.7+c*74.7)*43758.5453
        return n-math.floor(n)

    for y in range(size):
        course,fy=divmod(y,block)
        for x in range(size):
            shifted=(x+block//2*course)%size
            index,fx=divmod(shifted,block)
            stone=(hash01(index,course,11)-.5)*.037
            grain=(hash01(x//2,y//2,19)-.5)*.023
            black_fleck=(hash01(x,y,7)-.5)*.017
            joint=min(fx,block-fx-1,fy,block-fy-1)
            mortar=-.055*max(0,1-joint/2.2)
            shade=stone+grain+black_fleck+mortar
            pixels.extend((.294+shade,.305+shade,.294+shade,1))
    image=bpy.data.images.new("Granite | staggered dark ashlar courses",
                              width=size,height=size,alpha=False)
    image.file_format="PNG"
    image.pixels.foreach_set(pixels)
    image.pack()
    return image


def mat(name, rgb, image=None, roughness_image=None, normal_image=None):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*rgb, 1.0)
    m.use_nodes = True
    node = m.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*rgb, 1.0)
    node.inputs["Roughness"].default_value = .93
    if image:
        tex = m.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.interpolation = "Linear"
        tex.extension = "REPEAT"
        m.node_tree.links.new(tex.outputs["Color"], node.inputs["Base Color"])
    if roughness_image:
        texture=m.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image=roughness_image
        texture.extension="REPEAT"
        m.node_tree.links.new(texture.outputs["Color"],node.inputs["Roughness"])
    if normal_image:
        texture=m.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image=normal_image
        texture.extension="REPEAT"
        bump=m.node_tree.nodes.new("ShaderNodeNormalMap")
        bump.inputs["Strength"].default_value=.65
        m.node_tree.links.new(texture.outputs["Color"],bump.inputs["Color"])
        m.node_tree.links.new(bump.outputs["Normal"],node.inputs["Normal"])
    return m


def loft(name, sections, material=None, invert=False):
    n = len(sections[0])
    assert len(sections) >= 2 and all(len(r) == n for r in sections)
    faces = [(i+j*n, (i+1)%n+j*n, (i+1)%n+(j+1)*n, i+(j+1)*n)
             for j in range(len(sections)-1) for i in range(n)]
    faces += [tuple(reversed(range(n))), tuple((len(sections)-1)*n+i for i in range(n))]
    if invert:
        faces = [tuple(reversed(face)) for face in faces]
    data = bpy.data.meshes.new(name)
    data.from_pydata([p for row in sections for p in row], [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material or STONE)
    for f in data.polygons:
        f.use_smooth = len(f.vertices) == 4
    # Per-corner arc-length UVs avoid stretching and are exportable as glTF UVs.
    uv = data.uv_layers.new(name="Metric mosaic UV")
    cumulative = []
    for ring in sections:
        lengths = [0.0]
        for k in range(n):
            lengths.append(lengths[-1] + (Vector(ring[(k+1) % n]) -
                                          Vector(ring[k])).length)
        cumulative.append(lengths)
    centers = [sum((Vector(p) for p in ring), Vector()) / n for ring in sections]
    along = [0.0]
    for j in range(1, len(centers)):
        along.append(along[-1] + (centers[j] - centers[j-1]).length)
    for poly in data.polygons:
        j = poly.index // n
        if len(poly.vertices) == 4:
            k = poly.index % n
            coords = ((cumulative[j][k], along[j]),
                      (cumulative[j][k+1], along[j]),
                      (cumulative[j+1][k+1], along[j+1]),
                      (cumulative[j+1][k], along[j+1]))
        else:
            coords = [(data.vertices[idx].co.x, data.vertices[idx].co.y)
                      for idx in poly.vertices]
        for li, coords_m in zip(poly.loop_indices, coords):
            uv.data[li].uv = tuple(v / .72 for v in coords_m)
    return obj


def sphere(name, position, scale, material=None, segments=32, rings=16):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material or STONE)
    for f in obj.data.polygons:
        f.use_smooth = True
    for layer in obj.data.uv_layers:
        for item in layer.data:
            item.uv.x *= TAU * max(scale[0], scale[1]) / .72
            item.uv.y *= math.pi * scale[2] / .72
    return obj


def shaped_path(name, points, widths, material=None, sides=9):
    """Loft individually tapered curved strands rather than stacking spheres."""
    centers = [Vector(p) for p in points]
    rings = []
    for i, p in enumerate(centers):
        tangent = (centers[min(i+1, len(centers)-1)] - centers[max(0,i-1)]).normalized()
        u = tangent.cross(Vector((0,1,0)))
        if u.length < .01:
            u = tangent.cross(Vector((0,0,1)))
        u.normalize()
        v = tangent.cross(u).normalized()
        rings.append([tuple(p + widths[i]*(math.cos(TAU*k/sides)*u+math.sin(TAU*k/sides)*v))
                      for k in range(sides)])
    # Orient outward based on a surface radial vector.
    obj = loft(name, rings, material)
    face = next(f for f in obj.data.polygons if len(f.vertices) == 4)
    midpoint = sum((Vector(p) for p in rings[0]), Vector()) / sides
    if face.normal.dot(face.center-midpoint) < 0:
        for poly in obj.data.polygons:
            poly.flip()
    return obj


ROBE = [
    (8.0, 3.01, 1.72), (9.0, 3.05, 1.76), (12, 3.01, 1.73),
    (16, 2.83, 1.66), (20, 2.83, 1.74), (24, 3.02, 1.84),
    (27, 3.32, 1.90), (29.5, 3.63, 1.95), (31.2, 3.75, 1.91),
    (32.3, 3.16, 1.65), (33.40, 1.26, .99),
]


def torso_profile(z):
    for low, high in zip(ROBE[:-1], ROBE[1:]):
        if low[0] <= z <= high[0]:
            t = (z-low[0])/(high[0]-low[0])
            return (low[1]*(1-t)+high[1]*t, low[2]*(1-t)+high[2]*t)
    raise ValueError(z)


def cloth_front(x, z):
    width, depth = torso_profile(z)
    a = TAU - math.acos(max(-.995, min(.995, x/width)))
    forward = max(0, -math.sin(a))**5
    lower = max(0, min(1, (29.5-z)/19.5))
    pleats = forward * ((.10 + .19*lower)*math.cos(11*a + .11*z + .32*math.sin(z*.24))
                        +(.04 + .08*lower)*math.cos(19*a - .13*z))
    return depth*math.sin(a) - pleats - .12*forward


def draped_panel(name, z0, z1, center, halfwidth, relief):
    """A thin overlapping mantle sheet with physical cloth depth and metric UVs."""
    nz, nx = 66, 18
    vertices = []
    for j in range(nz):
        t = j/(nz-1)
        z = z0+(z1-z0)*t
        mid = center(t)
        half = halfwidth(t)
        for k in range(nx):
            s = k/(nx-1)
            x = mid+(2*s-1)*half
            shape = .08*math.cos(5*math.pi*s+.6*t)*math.sin(math.pi*t)
            # The upper edge sinks into the underlying sculpted cloth; otherwise
            # the panel reads as an upright rectangle glued onto the chest.
            blend=1-t**5
            y = (cloth_front(x,z)-relief*blend+.09*t**5
                 -(.04*math.sin(math.pi*s)+shape)*blend)
            vertices.append((x,y,z))
    polygons = [(j*nx+k,j*nx+k+1,(j+1)*nx+k+1,(j+1)*nx+k)
                for j in range(nz-1) for k in range(nx-1)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], polygons)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(STONE)
    uv = mesh.uv_layers.new(name="Mantle metric UV")
    for poly in mesh.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv = (co.x/.72, co.z/.72)
    solid = obj.modifiers.new("Carved cloth thickness", "SOLIDIFY")
    solid.thickness = .09
    solid.offset = 1
    solid.use_rim = True
    for edge in (-1, 1):
        pts = []
        for j in range(35):
            t = j/34
            z = z0+(z1-z0)*t
            x = center(t)+edge*halfwidth(t)
            blend=1-t**5
            pts.append((x,cloth_front(x,z)-relief*blend+.09*t**5-.017,z))
        shaped_path(name+" | carved selvedge %s"%edge,pts,
                    [.026+.027*math.sin(math.pi*j/34) for j in range(35)])
    return obj


def shoulder_shawl():
    """A folded U-shaped, weighted upper edge, rather than a planar bib."""
    ns,nt=84,48

    def position(s,t):
        x=-3.65+7.30*s
        lower=30.08-2.50*math.sin(math.pi*s)**.36-.54*(s-.5)
        upper=lower+.64+.43*math.sin(math.pi*s)
        z=lower+(upper-lower)*t
        belly=.065*math.sin(math.pi*t)*math.sin(math.pi*s)
        fold=.045*math.cos(7.5*math.pi*s-1.8*t)*math.sin(math.pi*t)
        # This layer needs clearance over the inner mantle; intersecting
        # displacement surfaces created false round lobes across the chest.
        y=cloth_front(x,z)-.37-belly-fold
        return (x,y,z)

    vertices=[position(k/(ns-1),j/(nt-1))
              for j in range(nt) for k in range(ns)]
    faces=[(j*ns+k,j*ns+k+1,(j+1)*ns+k+1,(j+1)*ns+k)
           for j in range(nt-1) for k in range(ns-1)]
    mesh=bpy.data.meshes.new("Mantle | deep shoulder shawl")
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    obj=bpy.data.objects.new("Mantle | continuous hanging shoulder shawl",mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(STONE)
    uv=mesh.uv_layers.new(name="Shawl metric UV")
    for poly in mesh.polygons:
        poly.use_smooth=True
        for li in poly.loop_indices:
            co=mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv=(co.x/.72,co.z/.72)
    solid=obj.modifiers.new("Weight of carved shawl","SOLIDIFY")
    solid.thickness=.13
    solid.offset=1
    solid.use_rim=True
    hem=[position(j/72,0) for j in range(73)]
    shaped_path("Mantle | gently rolled hanging shawl hem",hem,
                [.012+.023*math.sin(math.pi*j/72) for j in range(73)],sides=11)


def mantle_overcloth():
    """Broad asymmetric hanging fabric crossing the front of the actual statue.

    Unequal hem lengths and long, slanting carved folds make this read as a
    weighted cloth sheet, not a geometric panel pasted on the chest.
    """
    ns,nt=61,89

    def position(s,t):
        # A heavy sheet is pulled diagonally from the shoulder, so its lower
        # folds migrate sideways instead of forming parallel vertical pipes.
        x=(-2.90+5.97*s+.45*(1-t)**1.3*math.sin(math.pi*s)
           +.10*math.sin(math.pi*t)*math.sin(2*math.pi*s))
        low=18.8-3.28*s+.72*math.sin(2.5*math.pi*s+.34)
        high=29.40+.45*s
        z=low+(high-low)*t
        # Broader folds curl toward the hem; a deep local inverted fold along
        # the left hip catches light without any glued-on tube geometry.
        sweep=5.2*math.pi*s+2.9*t+1.1*math.sin(2*math.pi*s)
        edge=math.sin(math.pi*s)**.48
        fold=(.12+.23*(1-t))*math.cos(sweep)*edge*(1-t**3)
        seam=.09*math.sin(3*math.pi*s-.4)*math.sin(math.pi*t)
        relief=(.055+.22*edge)*(1-t**5)
        billow=.25*math.exp(-((s-.29)/.22)**2-((t-.30)/.26)**2)
        inverted=.23*math.exp(-((s-.15-.14*t)/.065)**2-((t-.40)/.32)**2)
        y=cloth_front(x,z)-relief+.085*t**5-fold-seam-billow+inverted
        return (x,y,z)

    vertices=[position(k/(ns-1),j/(nt-1))
              for j in range(nt) for k in range(ns)]
    polygons=[(j*ns+k,j*ns+k+1,(j+1)*ns+k+1,(j+1)*ns+k)
              for j in range(nt-1) for k in range(ns-1)]
    mesh=bpy.data.meshes.new("Mantle | asymmetric flowing forecloth")
    mesh.from_pydata(vertices,[],polygons)
    mesh.update()
    obj=bpy.data.objects.new("Mantle | weighted flowing overcloth",mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(STONE)
    uv=mesh.uv_layers.new(name="Overcloth metric UV")
    for polygon in mesh.polygons:
        polygon.use_smooth=True
        for li in polygon.loop_indices:
            co=mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv=(co.x/.72,co.z/.72)
    solid=obj.modifiers.new("Overcloth edge thickness","SOLIDIFY")
    solid.thickness=.080
    solid.offset=0
    solid.use_rim=True
    hem=[position(j/72,0) for j in range(73)]
    shaped_path("Mantle | irregular weighted lower hem",hem,
                [.019+.047*math.sin(math.pi*j/72) for j in range(73)],sides=9)


def robe():
    sections = []
    for j in range(129):
        z = 8.0+(33.4-8.0)*j/128
        width, depth = torso_profile(z)
        row = []
        for k in range(160):
            a = TAU*k/160
            x = width*math.cos(a)
            forward = max(0, -math.sin(a))**5
            back = max(0, math.sin(a))**5
            row.append((x,cloth_front(x,z) if forward>.001 else
                        depth*math.sin(a)+.07*back*math.cos(9*a+.16*z),z))
        sections.append(row)
    loft("Statue | single continuous robe volume", sections)

    # The broad forecloth has an angled hem; a long narrow stole crosses it.
    mantle_overcloth()
    draped_panel("Mantle | long chest stole",17.7,31.0,
                 lambda t: 1.13+.36*t+.08*math.sin(math.pi*t),
                 lambda t: .34+.56*(1-t)+.10*math.sin(math.pi*t),.37)
    shoulder_shawl()
    for k in range(15):
        x0=-2.66+k*.380
        pts=[]
        for j in range(18):
            t=j/17
            z=8.16+t*(8.5+.5*math.cos(k))
            x=x0+.10*math.sin(t*2.3+k*.45)
            pts.append((x,cloth_front(x,z)-.065,z))
        shaped_path("Cloth | lower pleat %02d"%k, pts,
                    [.009+.075*math.sin(math.pi*j/17) for j in range(18)])
    sphere("Collar | shoulder transition",(0,0,32.95),(1.23,1.01,.50))


def arms(side):
    # A hanging sleeve instead of a featureless tapering tube: the lower hem
    # drops several metres below the shoulder line before gathering at cuffs.
    measurements = [
        (2.65,30.75,1.50,1.00),(3.35,30.10,1.93,1.12),
        (4.45,29.40,2.28,1.18),(5.65,29.20,2.23,1.16),
        (7.20,29.10,2.00,1.08),(8.30,29.14,1.70,.90),
        (9.30,29.28,1.37,.75),(10.30,29.48,.98,.55),
        (10.90,29.60,.68,.40),(11.13,29.70,.39,.34),
    ]
    def sleeve_dimensions(x):
        for left,right in zip(measurements[:-1],measurements[1:]):
            if left[0]<=x<=right[0]:
                t=(x-left[0])/(right[0]-left[0])
                return tuple(left[i]*(1-t)+right[i]*t for i in (1,2,3))
        raise ValueError(x)

    sections=[]
    for j in range(97):
        x=2.65+(11.13-2.65)*j/96
        z,r,d=sleeve_dimensions(x)
        ring=[]
        for k in range(64):
            a=TAU*k/64
            lower=max(0,-math.cos(a))
            fluting=.18*math.cos(5*a+x*.42)+.10*math.cos(9*a-.23*x)
            front=max(0,-math.sin(a))**2
            gather=math.sin(math.pi*(x-2.65)/(11.13-2.65))**.7
            large_fold=.15*math.cos(6.5*a+.41*x+.25*math.sin(.7*x))
            small_fold=.065*math.sin(12.5*a-.19*x)
            y=d*math.sin(a)-front*gather*(large_fold+small_fold)
            dz=z+r*math.cos(a)-lower*(.12+fluting)
            ring.append((side*x,y,dz))
        sections.append(ring)
    loft("Sleeve | %s | physically closed carved volume"%side,sections,invert=side==1)
    sphere("Cuff | %s | carved fold joining wrist"%side,
           (side*11.12,0,29.7),(.095,.31,.32))
    hands(side)


def hands(side):
    existing=set(bpy.data.objects)
    z=29.68
    sphere("Hand | %s | broad wrist"%side,(side*11.37,0,z),(.38,.245,.325))
    sphere("Hand | %s | convex palmar mass"%side,
           (side*12.01,-.015,z+.025),(.80,.29,.43),segments=40,rings=24)
    # Individual digits depart from the hand's continuous palm, but their
    # joint lines curve slightly and their distal ends are round, not prongs.
    for k,tip in enumerate((13.79,13.972,13.90,13.70)):
        finger_z=z+.27-.17*k
        points=[]
        widths=[]
        for j in range(15):
            t=j/14
            x=12.37+(tip-12.37)*t
            finger_curve=.032*math.sin(math.pi*t)+.025*t*t
            y=-.018-.045*math.sin(math.pi*t)+.036*t*t
            points.append((side*x,y,finger_z-finger_curve))
            widths.append(.145*(1-t)**1.25+.068*t+.006*math.sin(math.pi*t))
        shaped_path("Hand | %s | gently flexed digit %s"%(side,k),points,
                    widths,sides=14)
        sphere("Hand | %s | softly rounded fingertip %s"%(side,k),
               points[-1],(.067,.074,.068),segments=18,rings=10)
    thumb=[]
    widths=[]
    for j in range(17):
        t=j/16
        x=11.71+1.20*t
        y=-.13-.046*math.sin(math.pi*t)+.045*t*t
        h=z-.24-.34*math.sin(.80*math.pi*t)-.035*t
        thumb.append((side*x,y,h))
        widths.append(.17*(1-t)**.85+.062*t)
    shaped_path("Hand | %s | bent opposable thumb"%side,thumb,widths,sides=15)
    sphere("Hand | %s | thumb pad"%side,
           thumb[-1],(.062,.071,.069),segments=18,rings=10)

    # Join the component masses with a genuine volumetric union. Otherwise
    # the thumbs and individual fingers visibly intersect the palm as rods.
    parts=[obj for obj in bpy.data.objects if obj not in existing and obj.type=="MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active=parts[0]
    bpy.ops.object.join()
    hand=bpy.context.view_layer.objects.active
    hand.name="Hand | %s | fused anatomical stone sculpt"%side
    hand.data.remesh_voxel_size=.019
    bpy.ops.object.voxel_remesh()
    relax=hand.modifiers.new("Smooth sculpted finger junctions","SMOOTH")
    relax.factor=.43
    relax.iterations=3
    bpy.ops.object.modifier_apply(modifier=relax.name)
    for polygon in hand.data.polygons:
        polygon.use_smooth=True
    uv=hand.data.uv_layers.new(name="Hand stone metric UV")
    for polygon in hand.data.polygons:
        for li in polygon.loop_indices:
            co=hand.data.vertices[hand.data.loops[li].vertex_index].co
            uv.data[li].uv=(co.x/.72,co.z/.72)


HEAD = [
    (34.02,.20,.27,-.03),(34.23,.48,.59,-.05),(34.47,.71,.78,-.08),
    (34.82,.91,.96,-.10),(35.30,1.09,1.12,-.08),
    (35.85,1.37,1.22,0),(36.50,1.52,1.24,.05),
    (37.15,1.43,1.18,.10),(37.65,1.03,.91,.11),
    (37.95,.38,.40,.12),(38.0,.045,.07,.12),
]


def head_profile(z):
    # A piecewise-linear loft left visible facet bands around the jaw and
    # temples despite a dense vertex mesh. Interpolate a C1-continuous cage:
    # both the face and its framing hair share this anatomical silhouette.
    for i,(low,high) in enumerate(zip(HEAD[:-1],HEAD[1:])):
        if low[0]<=z<=high[0]:
            span=high[0]-low[0]
            t=(z-low[0])/span
            t2,t3=t*t,t*t*t
            h00=2*t3-3*t2+1
            h10=t3-2*t2+t
            h01=-2*t3+3*t2
            h11=t3-t2
            def slope(j,k):
                before=HEAD[max(0,j-1)]
                after=HEAD[min(len(HEAD)-1,j+1)]
                return (after[k]-before[k])/(after[0]-before[0])
            return tuple(h00*low[k]+h10*span*slope(i,k)+
                         h01*high[k]+h11*span*slope(i+1,k)
                         for k in (1,2,3))
    raise ValueError(z)


def mouth_contour_z(x):
    """Landowski's downturned, arched closed mouth: calm, not smiling."""
    u=min(1.,abs(x)/.56)
    return 35.64-.125*u**1.55


def face():
    sphere("Head | broad tendinous neck",(0,.20,33.52),(.87,.79,1.29))
    # Densify the face loft to sculpt cheekbones, chin and jaw as one surface.
    # Surface folds are smooth Gaussian volumes, not glued-on primitive cheeks.
    sampled=[]
    for low,high in zip(HEAD[:-1],HEAD[1:]):
        for j in range(24):
            z=low[0]+(high[0]-low[0])*j/24
            sampled.append((z,*head_profile(z)))
    sampled.append((HEAD[-1][0],*head_profile(HEAD[-1][0])))
    rings=[]
    for z,rx,ry,cy in sampled:
        ring=[]
        for i in range(160):
            a=TAU*i/160
            x=rx*math.cos(a)
            y=cy+ry*math.sin(a)
            if math.sin(a)<0:
                # Integrate the full facial relief into the head surface.
                # Features are smooth local displacements of the same manifold,
                # eliminating the stacked cylinder/ball appearance in profile.
                cheek=.24*math.exp(-((abs(x)-.83)/.47)**2-((z-35.97)/.64)**2)
                jaw=.13*math.exp(-((abs(x)-.82)/.46)**2-((z-35.24)/.57)**2)
                cheek_hollow=.085*math.exp(-((abs(x)-.82)/.33)**2-((z-35.48)/.36)**2)
                chin=.25*math.exp(-(x/.74)**4-((z-34.73)/.40)**2)
                # Broad eyebrow planes and a shallow closed-eye sculpt. A
                # separate arch-relative brow band was tested in clay but
                # read like floating eyebrows, so the mass stays integrated.
                brow=.18*math.exp(-((abs(x)-.70)/.53)**4-((z-37.18)/.23)**2)
                upper_lid=.055*math.exp(-((abs(x)-.68)/.40)**4-((z-36.985)/.095)**2)
                lower_lid=.035*math.exp(-((abs(x)-.68)/.39)**4-((z-36.75)/.115)**2)
                eye_socket=.21*math.exp(-((abs(x)-.69)/.44)**4-((z-36.915)/.18)**2)
                closed_eye=.065*math.exp(-((abs(x)-.69)/.43)**4-((z-36.805)/.12)**2)
                # A tapered nasal ridge flows into the alae. The previous
                # narrow Gaussian tip looked like a hooked, glued-on beak.
                bridge_width=.20+.08*max(0.,min(1.,(36.8-z)/.65))
                nose_bridge=.33*math.exp(-(x/bridge_width)**2-((z-36.59)/.56)**2)
                nose_tip=.25*math.exp(-(x/.28)**2-((z-36.17)/.27)**2)
                nostril_wing=.125*math.exp(-((abs(x)-.29)/.17)**2-((z-36.065)/.15)**2)
                nostril_cavity=.045*math.exp(-((abs(x)-.28)/.075)**2-((z-36.015)/.065)**2)
                septum=.070*math.exp(-(x/.11)**2-((z-36.005)/.085)**2)
                philtrum=.032*math.exp(-((abs(x)-.145)/.10)**2-((z-35.86)/.16)**2)
                # Sculpt both lips around one shared anatomical mouth contour.
                # Independent horizontal Gaussians produced a pasted-on smile
                # even when the actual carved mouth groove curved downward.
                mouth_z=mouth_contour_z(x)
                lip_weight=max(0.,1-(abs(x)/.58)**4)
                top_lip=(.070*lip_weight*
                         math.exp(-((z-(mouth_z+.085))/.10)**2))
                bottom_lip=(.047*lip_weight*
                            math.exp(-((z-(mouth_z-.115))/.115)**2))
                mouth_crease=(.063*lip_weight*
                              math.exp(-((z-mouth_z)/.045)**2))
                # Reference close-ups show a clean, uninterrupted smooth
                # Art Deco chin. Old beard ridges generated a false combed
                # texture and a moustache-shaped shadow below the nostrils.
                y-= (cheek+jaw+chin+brow+upper_lid+lower_lid+
                     nose_bridge+nose_tip+nostril_wing+septum+philtrum+top_lip+bottom_lip+
                     closed_eye)
                y+=eye_socket+mouth_crease+nostril_cavity+cheek_hollow
            ring.append((x,y,z))
        rings.append(ring)
    head_surface=loft("Head | integrated cheekbone jaw and cranium",rings)

    def face_y(x,z,lift=.012):
        # Ray-cast the actual sculpted surface: grooves follow the cheek/eye
        # relief instead of floating as straight rods in profile.
        hit,point,normal,index=head_surface.ray_cast(
            Vector((x,-4,z)),Vector((0,1,0)))
        assert hit, "Facial detail must project onto the sculpted surface"
        return point.y-lift

    for side in (-1,1):
        # Reference photographs show a continuous semicircular upper lid
        # set into a recessed almond-shaped eye, not an angular dark slit.
        top=[]
        widths=[]
        for j in range(29):
            t=j/28
            x=side*(.30+.80*t)
            z=36.80+.215*math.sin(math.pi*t)**.83-.015*t
            top.append((x,face_y(x,z,-.006),z))
            widths.append(.003+.014*math.sin(math.pi*t)**.7)
        shaped_path("Face | %s | upper eyelid contour"%side,
                    top,widths,CARVING,sides=9)
        # The temple and ear are hidden by the stone hair curtain in the
        # original. Isolated ear spheres read as buttons in profile.
    lip=[(x,mouth_contour_z(x)) for x in (-.55,-.30,0,.30,.55)]
    shaped_path("Face | carved closed mouth",
                [(x,face_y(x,z,-.003),z) for x,z in lip],
                (.002,.006,.008,.006,.002),CARVING)
    # The real monument has a nearly continuous, heavy hair mantle, not
    # individual dangling tubular strands. Construct an open-front shell
    # wrapping the crown and back of the skull, with variable length at the
    # nape and carved shallow flowing channels.
    def hair_position(a,t):
        base=33.16+.30*max(0,math.sin(a))**2+.08*math.sin(5*a+.5)
        z=base+(37.98-base)*t
        rx,ry,cy=head_profile(max(34.02,z))
        # Hair is a substantial curtain framing the cheeks and jaw. Its
        # earlier edge nearly coincided with the skin, giving a bare egg head.
        gap=.10+.18*(1-t)**1.4
        extent=max(rx+gap,1.53*(1-t))
        wave=(.060+.06*(1-t))*math.cos(10*a+2.3*t)+.036*math.sin(17*a-1.2*t)
        return (extent*math.cos(a),cy+(ry-.025+.69*(1-t)+wave)*math.sin(a),z)

    nz,na=120,128
    vertices=[]
    for j in range(nz):
        t=j/(nz-1)
        for k in range(na):
            a=.13+(math.pi-.26)*k/(na-1)
            vertices.append(hair_position(a,t))
    quads=[(j*na+k,j*na+k+1,(j+1)*na+k+1,(j+1)*na+k)
           for j in range(nz-1) for k in range(na-1)]
    mesh=bpy.data.meshes.new("Hair | continuous draped stone mantle")
    mesh.from_pydata(vertices,[],quads)
    mesh.update()
    hood=bpy.data.objects.new("Hair | continuous parted mantle",mesh)
    bpy.context.collection.objects.link(hood)
    hood.data.materials.append(STONE)
    uv=mesh.uv_layers.new(name="Hair metric UV")
    for poly in mesh.polygons:
        poly.use_smooth=True
        for li in poly.loop_indices:
            co=mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv=(co.x/.72,co.z/.72)
    thickness=hood.modifiers.new("Carved hair mantle thickness","SOLIDIFY")
    thickness.thickness=.045
    thickness.offset=1
    thickness.use_rim=True

    for side in (-1,1):
        # Incised parallel channels flow around the temple, ending naturally
        # at the draped lower edge; no free-floating rod or exposed spike tips.
        for k in range(7):
            pts=[]
            for j in range(27):
                t=.075+.85*j/26
                a=(.15+.10*k) if side==1 else (math.pi-.15-.10*k)
                a+=side*.023*math.sin(5*t+k*.4)
                p=Vector(hair_position(a,t))
                normal=Vector((math.cos(a),math.sin(a),0))
                pts.append(tuple(p+normal*.035))
            shaped_path("Hair | %s | engraved sweep %s"%(side,k),pts,
                        [.004+.012*math.sin(math.pi*j/26) for j in range(27)],
                        CARVING,sides=7)
        # The shell itself establishes the hairline. Additional tubes placed
        # on the forehead looked like disconnected pencil marks in close-ups.


def pedestal():
    # Survey/OSM geometry shows a chamfered square chapel footprint, roughly
    # 8 m across, with clipped corners (eight edges), not a regular hexagon.
    def ring(extent,cut,z):
        e=extent
        c=cut
        return [(-e+c,-e,z),(e-c,-e,z),(e,-e+c,z),(e,e-c,z),
                (e-c,e,z),(-e+c,e,z),(-e,e-c,z),(-e,-e+c,z)]

    def granite_loft(name,sections):
        obj=loft(name,sections,GRANITE)
        # One image covers 5.76 m and 16 distinct staggered blocks, avoiding
        # an obvious two-by-two checkerboard repeat on the chapel walls.
        for item in obj.data.uv_layers.active.data:
            item.uv*=.125
        return obj

    granite_loft("Pedestal | chamfered square granite shaft",
                 [ring(4.26,1.06,0),ring(4.16,1.04,.5),
                  ring(3.80,1.0,7.35),ring(3.72,.98,8.0)])
    granite_loft("Granite | lower projecting stone course",
                 [ring(4.38,1.08,0),ring(4.38,1.08,.24)])
    granite_loft("Granite | chapel cornice band",
                 [ring(3.97,1.0,7.34),ring(3.97,1.0,7.73)])
    granite_loft("Granite | upper chamfered coping",
                 [ring(3.83,.98,7.74),ring(3.83,.98,8.0)])

    # The lower pedestal houses the chapel of Nossa Senhora Aparecida.
    # Its iron entry grille is small in relation to the eight-metre plinth.
    def face_front(z,offset=.04):
        return -4.26+.0675*z-offset

    z0,z1=.69,3.48
    vertices=[(-.81,face_front(z0),z0),(.81,face_front(z0),z0),
              (.81,face_front(z1),z1),(-.81,face_front(z1),z1)]
    mesh=bpy.data.meshes.new("Chapel | inset dark entry plane")
    mesh.from_pydata(vertices,[],[(0,1,2,3)])
    mesh.update()
    obj=bpy.data.objects.new("Chapel | recessed entry",mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(mat("Chapel | unlit doorway",(.105,.112,.108)))
    uv=mesh.uv_layers.new(name="Portal metric UV")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            co=mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv=(co.x/.72,co.z/.72)

    framing=[(-.90,face_front(z0,.10),z0),
             (-.90,face_front(z1+.14,.10),z1+.14),
             (.90,face_front(z1+.14,.10),z1+.14),
             (.90,face_front(z0,.10),z0)]
    shaped_path("Chapel | granite door jamb and lintel",framing,
                (.10,.12,.12,.10),GRANITE,sides=11)
    gate=mat("Chapel | black iron lattice",(.19,.205,.20))
    for k in range(9):
        x=-.72+.18*k
        shaped_path("Chapel | wrought-iron vertical %02d"%k,
                    [(x,face_front(z,.095),z) for z in (z0+.09,z1-.05)],
                    (.022,.022),gate,sides=8)
    for k,z in enumerate((1.15,1.91,2.63,3.31)):
        shaped_path("Chapel | forged crossbar %02d"%k,
                    [(x,face_front(z,.108),z) for x in (-.72,.72)],
                    (.022,.022),gate,sides=8)


def orient_head(head_objects):
    """Apply one measured gaze transform to every sculpt and hair detail."""
    pivot=Matrix.Translation((0,0,34.20))
    tilt=pivot @ Matrix.Rotation(math.radians(HEAD_TILT_DEG),4,'X') @ pivot.inverted()
    for obj in head_objects:
        obj.matrix_world=tilt @ obj.matrix_world
    peak=max((obj.matrix_world@v.co).z
             for obj in head_objects for v in obj.data.vertices)
    height_correction=Matrix.Translation((0,0,38.0-peak))
    for obj in head_objects:
        obj.matrix_world=height_correction @ obj.matrix_world


def build_head_study():
    """Fast, neutral clay anatomy proof with no pedestal, body or PBR noise."""
    global STONE,CARVING
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    STONE=mat("Clay | head-only anatomy study",(.68,.68,.66))
    CARVING=mat("Clay | recessed facial carving",(.65,.65,.63))
    existing=set(bpy.data.objects)
    face()
    orient_head(set(bpy.data.objects)-existing)


def build():
    global STONE,CARVING,GRANITE
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    mosaic_color,mosaic_roughness,mosaic_normal=soapstone_mosaic()
    STONE=mat("Warm light grey soapstone | 3 cm tesserae",(.665,.67,.635),
              mosaic_color,mosaic_roughness,mosaic_normal)
    CARVING=mat("Subtle tonal shifts in carved stone recesses",(.64,.65,.63))
    GRANITE=mat("Pedestal | weathered granite ashlar",(.30,.31,.30),granite_blocks())
    pedestal()
    robe()
    existing=set(bpy.data.objects)
    for side in (-1,1):
        arms(side)
    # A measured photogrammetric survey reports 29.4 m fingertip span,
    # compared with the widely repeated rounded 28 m figure.
    arm_objects=set(bpy.data.objects)-existing
    half_span=max(abs((obj.matrix_world@v.co).x)
                  for obj in arm_objects for v in obj.data.vertices)
    arm_scale=14.70/half_span
    for obj in arm_objects:
        obj.matrix_world=Matrix.Scale(arm_scale,4,Vector((1,0,0))) @ obj.matrix_world
    existing=set(bpy.data.objects)
    face()
    # Side photographs show the gaze tilted downward. Apply one rigid
    # anatomical transform to face, neck, and the complete hair mantle.
    orient_head(set(bpy.data.objects)-existing)


def verify():
    objs=[ob for ob in bpy.data.objects if ob.type=="MESH"]
    points=[ob.matrix_world@v.co for ob in objs for v in ob.data.vertices]
    minimum=[min(p[i] for p in points) for i in range(3)]
    maximum=[max(p[i] for p in points) for i in range(3)]
    pedestal_ob=next(ob for ob in objs if ob.name.startswith("Pedestal |"))
    torso_ob=next(ob for ob in objs if ob.name.startswith("Statue |"))
    head_ob=next(ob for ob in objs if ob.name.startswith("Head | integrated cheekbone"))
    ptop=max((pedestal_ob.matrix_world@v.co).z for v in pedestal_ob.data.vertices)
    rbot=min((torso_ob.matrix_world@v.co).z for v in torso_ob.data.vertices)
    winding={}
    open_edges={}
    hand_open_edges={}
    hand_vertex_counts={}
    hand_reach_m={}
    for side in (-1,1):
        obj=next(ob for ob in objs if ob.name.startswith("Sleeve | %s | physically"%side))
        faces=[p for p in obj.data.polygons if len(p.vertices)==4]
        centers=[sum(obj.data.vertices[j*64+k].co.z for k in range(64))/64
                 for j in range(len(obj.data.vertices)//64)]
        checks=[]
        for p in faces:
            j=p.index//64
            center_z=(centers[j]+centers[j+1])/2
            radial=Vector((0,p.center.y,p.center.z-center_z))
            if radial.length>.08:
                checks.append(p.normal.dot(radial.normalized())>0)
        winding[side]=round(sum(checks)/len(checks),3)
        edge_uses={}
        for polygon in obj.data.polygons:
            for edge in polygon.edge_keys:
                edge_uses[edge]=edge_uses.get(edge,0)+1
        open_edges[side]=sum(uses!=2 for uses in edge_uses.values())
        hand=next(ob for ob in objs
                  if ob.name.startswith("Hand | %s | fused anatomical"%side))
        hand_vertex_counts[side]=len(hand.data.vertices)
        tips=[(hand.matrix_world @ vertex.co).x for vertex in hand.data.vertices]
        hand_reach_m[side]=round(max(tips)-min(tips),3)
        hand_edge_uses={}
        for polygon in hand.data.polygons:
            for edge in polygon.edge_keys:
                hand_edge_uses[edge]=hand_edge_uses.get(edge,0)+1
        hand_open_edges[side]=sum(uses!=2 for uses in hand_edge_uses.values())
    result={"bounds_min":minimum,"bounds_max":maximum,"pedestal_top":ptop,
            "robe_bottom":rbot,"sleeve_outward_fraction":winding,
            "sleeve_nonmanifold_edges":open_edges,
            "hand_nonmanifold_edges":hand_open_edges,
            "fused_hand_vertices":hand_vertex_counts,
            "hand_reach_m":hand_reach_m,
            "pedestal_sides":8,
            "mosaic_images":len([im for im in bpy.data.images if im.name.startswith("Soapstone |")]),
            "uv_mesh_count":sum(bool(ob.data.uv_layers) for ob in objs),
            "head_surface_vertices":len(head_ob.data.vertices),
            "head_tilt_deg":HEAD_TILT_DEG,
            "mesh_count":len(objs),"vertex_count":sum(len(ob.data.vertices) for ob in objs)}
    assert abs(ptop-rbot)<.001,"Statue must meet pedestal without floating"
    assert abs(maximum[0]-minimum[0]-29.40)<.015,"Survey fingertip span should be 29.4 m"
    assert maximum[2]>37.9 and minimum[2]<.01,"Scale should be 38 m including pedestal"
    assert min(winding.values())>.60,"Both sleeve normals must face outward"
    assert not any(open_edges.values()),"Sleeve meshes must be closed and manifold"
    assert not any(hand_open_edges.values()),"Fused hands must be closed and manifold"
    assert min(hand_vertex_counts.values())>1500,"Hands must have sculpted union detail"
    assert all(3.10 < reach < 3.30 for reach in hand_reach_m.values()), (
        "Each sculpted hand should match Landowski's recorded 3.20 m length")
    assert len(head_ob.data.vertices)>5000,"Face needs a dense sculptable surface"
    assert (result["mosaic_images"]==3 and result["uv_mesh_count"]>=60 and
            result["uv_mesh_count"]==result["mesh_count"]), (
        "Every exported mesh must carry metric UVs for the PBR mosaic")
    return result


def make_camera(name, location, target, scale):
    bpy.ops.object.camera_add(location=location)
    ob=bpy.context.object
    ob.name="QA camera "+name
    ob.rotation_euler=(Vector(target)-ob.location).to_track_quat("-Z","Y").to_euler()
    ob.data.type="ORTHO"
    ob.data.ortho_scale=scale
    return ob


def views(path,size,samples,selected=None):
    os.makedirs(path,exist_ok=True)
    scene=bpy.context.scene
    scene.render.engine="BLENDER_EEVEE_NEXT"
    scene.render.resolution_x=size
    scene.render.resolution_y=size
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    scene.render.film_transparent=False
    scene.world.use_nodes=True
    background=scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value=(.12,.145,.18,1)
    background.inputs["Strength"].default_value=.7
    scene.eevee.taa_render_samples=samples
    # Balanced front key/fill make anatomical asymmetries visible without
    # throwing the left nostril into the nearly black rear-lit shadow that
    # obscured our earlier comparisons. A separate rim keeps the silhouette.
    lights=[((8,-15,43),4800,9),((-10,-11,39),3300,10),((9,12,25),2600,7)]
    for i,(position,energy,width) in enumerate(lights):
        bpy.ops.object.light_add(type="AREA",location=position)
        light=bpy.context.object
        light.name="QA studio light %d"%i
        light.data.energy=energy
        light.data.shape="DISK"
        light.data.size=width
        light.rotation_euler=(Vector((0,0,27))-light.location).to_track_quat("-Z","Y").to_euler()
    shots=[
        ("front",(0,-70,19),(0,0,19),43),
        ("profile",(70,0,19),(0,0,19),43),
        ("three_quarter",(47,-63,22),(0,0,19),43),
        ("head_front",(0,-19,36),(0,0,36),6.4),
        ("head_low",(0,-18,32.5),(0,0,36),6.4),
        ("head_profile",(19,0,36),(0,0,36),6.4),
        ("head_three_quarter",(14,-15,36),(0,0,36),6.4),
        ("hand_detail",(16,-13,30),(12.5,0,29.7),5.5),
    ]
    for name,position,target,scale in shots:
        if selected and name not in selected:
            continue
        camera_ob=make_camera(name,position,target,scale)
        scene.camera=camera_ob
        scene.render.filepath=str(Path(path)/(name+".png"))
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(camera_ob,do_unlink=True)
    for ob in list(bpy.data.objects):
        if ob.type=="LIGHT" and ob.name.startswith("QA studio light"):
            bpy.data.objects.remove(ob,do_unlink=True)


def export(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for ob in bpy.data.objects:
        if ob.type=="MESH":
            ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path,export_format="GLB",
                              use_selection=True,export_apply=True)
    bpy.ops.object.select_all(action="DESELECT")


def main():
    global HEAD_TILT_DEG
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser()
    p.add_argument("--output")
    p.add_argument("--blend")
    p.add_argument("--views")
    p.add_argument("--metrics")
    p.add_argument("--verify",action="store_true")
    p.add_argument("--head-study",action="store_true",
                   help="Clay-only head study for rapid multi-view anatomical QA")
    p.add_argument("--head-tilt",type=float,default=22.,
                   help="Head bow angle in degrees for scan/profile comparison")
    p.add_argument("--samples",type=int,default=16)
    p.add_argument("--size",type=int,default=720)
    p.add_argument("--shot",action="append",choices=("front","profile","three_quarter",
                                                   "head_front","head_low","head_profile",
                                                   "head_three_quarter","hand_detail"),
                   help="Render only selected camera(s); repeat for multiple shots")
    args=p.parse_args(argv)
    if not 0 <= args.head_tilt <= 50:
        p.error("--head-tilt must be between 0 and 50 degrees")
    HEAD_TILT_DEG=args.head_tilt
    if args.head_study:
        build_head_study()
        head=next(ob for ob in bpy.data.objects
                  if ob.name.startswith("Head | integrated cheekbone"))
        metrics={"study":"head","head_surface_vertices":len(head.data.vertices),
                 "head_tilt_deg":HEAD_TILT_DEG,
                 "mesh_count":sum(ob.type=="MESH" for ob in bpy.data.objects)}
        assert metrics["head_surface_vertices"]>15000
    else:
        build()
        metrics=verify()
    if args.verify:
        print("SCULPTURE_AUDIT "+json.dumps(metrics,sort_keys=True))
    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True,exist_ok=True)
        Path(args.metrics).write_text(json.dumps(metrics,indent=2)+"\n")
    if args.output:
        export(args.output)
    if args.views:
        views(args.views,args.size,args.samples,args.shot)
    if args.blend:
        Path(args.blend).parent.mkdir(parents=True,exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=args.blend)


if __name__=="__main__":
    main()
