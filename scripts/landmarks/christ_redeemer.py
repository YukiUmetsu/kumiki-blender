"""Christ the Redeemer: original procedural, dimensioned Art Deco sculpture.

30 m figure / 29.4 m surveyed arm span / 8 m pedestal.
The default export uses original geometry. Optional anatomy-clay studies can
use a separately identified, rights-cleared generic sculpt base, never an
authentic scan or a claim of target-specific facial likeness.
Run: blender -b --python-exit-code 1 --python scripts/landmarks/christ_redeemer.py --
     --output .review/christ.glb --views .review/christ --verify
"""
import argparse
from array import array
import gzip
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

# The Blender entry script lives in scripts/landmarks, while the shared
# sculpting primitives are part of the installable src/ package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from carved_hair import sculpt_carved_hair_from_anatomy
from registered_face_sculpt import apply_candidate_face_fit
from kumiki_blender.sculpt_hair_flow import anatomical_hairline_height

TAU = math.tau
STONE = None
CARVING = None
GRANITE = None
HEAD_TILT_DEG = 22.0  # Measured against authentic side-view scan previews.
FACE_METHOD = "gaussian"  # Approved approximation; alternative studies held.
ANATOMY_BASE = (Path(__file__).resolve().parents[2] /
                "assets/landmarks/christ_redeemer/head_anatomical_base_cc0.obj.gz")
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


# Signed front-of-face depth and half-width at key anatomical levels. This
# original cage is based on observed facial planes, not geometry copied from
# an unavailable/rights-unverified photogrammetric scan. Its visible accuracy
# must be checked from three views before becoming the default export.
NOSE_CAGE = [
    (35.95,.00,.11), (36.035,.115,.18), (36.10,.31,.30),
    (36.195,.49,.335), (36.275,.49,.28), (36.42,.43,.20),
    (36.68,.39,.17), (36.95,.15,.20), (37.25,.005,.23),
]


def nasal_cage_relief(x,z):
    """Shape a triangular nasal ridge via anatomical z/depth/width landmarks."""
    if z < NOSE_CAGE[0][0] or z > NOSE_CAGE[-1][0]:
        return 0.
    for index,(lo,hi) in enumerate(zip(NOSE_CAGE[:-1],NOSE_CAGE[1:])):
        if lo[0] <= z <= hi[0]:
            span=hi[0]-lo[0]
            t=(z-lo[0])/span
            # Continuous Hermite tangents avoid the stationary terraces at
            # every profile landmark caused by independent smoothstep spans.
            def sample(component):
                prev=NOSE_CAGE[max(0,index-1)]
                nxt=NOSE_CAGE[min(len(NOSE_CAGE)-1,index+2)]
                d0=(hi[component]-prev[component])/(hi[0]-prev[0])
                d1=(nxt[component]-lo[component])/(nxt[0]-lo[0])
                return ((2*t**3-3*t*t+1)*lo[component]+
                        (t**3-2*t*t+t)*span*d0+
                        (-2*t**3+3*t*t)*hi[component]+
                        (t**3-t*t)*span*d1)
            depth=sample(1)
            width=sample(2)
            u=abs(x)/width
            return max(0.,depth)*max(0.,1-u*u)**1.25
    return 0.


def orbital_cage_relief(x,z):
    """A recessed orbital bowl bounded below by its integrated upper eyelid."""
    u=(abs(x)-.30)/.80
    if u <= 0 or u >= 1:
        return 0.
    influence=math.sin(math.pi*u)**.65
    arch=36.79+.17*math.sin(math.pi*u)**.83-.07*u
    ridge=.23*math.exp(-((z-(arch-.02))/.045)**2)
    lower=.065*math.exp(-((z-(arch-.14))/.19)**2)
    cavity=.36*math.exp(-((z-(arch+.10))/.20)**2)
    return influence*(ridge+lower-cavity)


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
                cheek_strength=.28 if FACE_METHOD == "landmark" else .24
                cheek=cheek_strength*math.exp(-((abs(x)-.83)/.47)**2-((z-35.97)/.64)**2)
                jaw=.13*math.exp(-((abs(x)-.82)/.46)**2-((z-35.24)/.57)**2)
                hollow_strength=.11 if FACE_METHOD == "landmark" else .085
                cheek_hollow=hollow_strength*math.exp(-((abs(x)-.82)/.33)**2-((z-35.48)/.36)**2)
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
                upper_strength=.105 if FACE_METHOD == "landmark" else .070
                lower_strength=.076 if FACE_METHOD == "landmark" else .047
                crease_strength=.087 if FACE_METHOD == "landmark" else .063
                top_lip=(upper_strength*lip_weight*
                         math.exp(-((z-(mouth_z+.085))/.10)**2))
                bottom_lip=(lower_strength*lip_weight*
                            math.exp(-((z-(mouth_z-.115))/.115)**2))
                mouth_crease=(crease_strength*lip_weight*
                              math.exp(-((z-mouth_z)/.045)**2))
                landmark_relief=0.
                if FACE_METHOD == "landmark":
                    # Substitution, not a second layer over old spheres:
                    # eye and nasal geometry follows explicit guide curves.
                    brow=upper_lid=lower_lid=eye_socket=closed_eye=0.
                    nose_bridge=nose_tip=septum=0.
                    nostril_wing=.17*math.exp(
                        -((abs(x)-.29)/.12)**2-((z-36.13)/.095)**2)
                    nostril_cavity=.14*math.exp(
                        -((abs(x)-.23)/.078)**2-((z-36.055)/.052)**2)
                    landmark_relief=(nasal_cage_relief(x,z)+
                                     orbital_cage_relief(x,z))
                # Reference close-ups show a clean, uninterrupted smooth
                # Art Deco chin. Old beard ridges generated a false combed
                # texture and a moustache-shaped shadow below the nostrils.
                y-= (cheek+jaw+chin+brow+upper_lid+lower_lid+
                     nose_bridge+nose_tip+nostril_wing+septum+philtrum+top_lip+bottom_lip+
                     closed_eye+landmark_relief)
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

    if FACE_METHOD == "gaussian":
        for side in (-1,1):
            # Retain original removable guides only on the legacy variant;
            # landmark eyelids are part of the continuous head surface.
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


def sculpted_anatomical_hair_mantle():
    """Broad volumetric stone sweeps, rooted at the real head cage.

    This is a deliberately separate, author-directed hairstyle foundation:
    overlapping solid back and temple volumes pass through the scalp and
    posterior cervical surface so their roots cannot hover like the rejected
    hollow hood. The silhouette is inferred from genuine Rio monument photos,
    not a recovered metric scan, and remains an anatomical clay QA candidate.
    """
    objects=[]

    def sweep(name, profiles, side=0):
        # Profile records: cage z, lateral center, sagittal center, lateral
        # radius, sagittal radius. Cubic continuity avoids the stacked
        # scallops seen when each ring-to-ring interval eased independently.
        sections=[]
        for j in range(100):
            z=profiles[0][0]+(profiles[-1][0]-profiles[0][0])*j/99
            for index,(lower,upper) in enumerate(zip(profiles[:-1],profiles[1:])):
                if upper[0]<=z<=lower[0]:
                    weight=(z-lower[0])/(upper[0]-lower[0])
                    previous=profiles[max(0,index-1)]
                    following=profiles[min(len(profiles)-1,index+2)]
                    def cubic(k):
                        a,b,c,d=(previous[k],lower[k],upper[k],following[k])
                        t=weight
                        return .5*((2*b)+(-a+c)*t+
                                   (2*a-5*b+4*c-d)*t*t+
                                   (-a+3*b-3*c+d)*t*t*t)
                    cx,cy,rx,ry=(cubic(k) for k in (1,2,3,4))
                    break
            ring=[]
            for k in range(64):
                angle=TAU*k/64
                x=(side*cx if side else 0.)+rx*math.cos(angle)
                y=cy+ry*math.sin(angle)
                # A gently flattened frontal section gives the temple
                # curtains a broad carved surface rather than a rope shape.
                if side and math.sin(angle)<0:
                    y+=.12*ry*(-math.sin(angle))**2
                ring.append((x,y,z))
            sections.append(ring)
        ob=loft(name,sections,STONE,invert=True)
        ob.scale=(11.,11.,11.)
        ob.location=(0,0,34.03)
        objects.append(ob)

    # This substantial nape volume overlaps the skull and both long side
    # sweeps. It fills the neck indentation without a hollow, sheet-like hood.
    sweep("Hair | solid joined-back stone volume",[
        (.387,0.,.012,.006,.009),
        (.351,0.,.043,.076,.047),
        (.280,0.,.065,.102,.062),
        (.210,0.,.085,.113,.073),
        (.120,0.,.098,.124,.074),
        (.048,0.,.106,.137,.070),
        (.008,0.,.103,.125,.050),
    ])
    for side in (-1,1):
        # Broad, overlapping temple curtains following the jaw to the
        # shoulders. Their crown and side roots deliberately intersect the
        # anatomical mesh and rear hair volume.
        sweep("Hair | %s | scalp-rooted side stone drape"%side,[
            (.381,.012,-.030,.007,.012),
            (.352,.068,-.060,.020,.042),
            (.311,.099,-.038,.029,.049),
            (.251,.111,-.009,.032,.050),
            (.181,.116,.018,.033,.055),
            (.105,.119,.038,.033,.058),
            (.037,.125,.058,.028,.055),
            (.010,.137,.073,.010,.028),
        ],side)
    return objects


def sculpt_scalp_surface_from_anatomy(head, trim_nape=False):
    """Extract the donor's *real scalp topology* as a sculptable stone layer.

    Unlike the failed free-form hood lofts, every source surface point comes
    from the anatomical quad mesh. A rearward neck drape is added to that
    source topology while the forehead hairline is kept on the skull.
    This is a candidate 3D sculpt, not a registered monument reconstruction.
    """
    def ease(value):
        t=max(0.,min(1.,value))
        return t*t*(3-2*t)

    vertices=head.data.vertices
    def hairline(co):
        base=anatomical_hairline_height(co.x,co.y)
        if not trim_nape:
            return base
        # The old extracted scalp included the whole posterior neck and
        # rendered as a huge rolled stone cape. The crown must end at the
        # nape; long hair is a separate sculpted curtain, not neck skin.
        rear=ease((co.y+.015)/.085)
        return max(base,.215*rear)

    selected=[]
    for polygon in head.data.polygons:
        center=sum((vertices[i].co for i in polygon.vertices),Vector())/len(polygon.vertices)
        if center.z>hairline(center):
            selected.append(tuple(polygon.vertices))
    if not selected:
        raise ValueError("No anatomical scalp faces in the selected hairline")

    # Retain the largest connected patch. Separate little islands at the ear
    # or undercut must not turn into free-floating pieces of stone hair.
    vertex_to_faces={}
    for index,face in enumerate(selected):
        for vertex_id in face:
            vertex_to_faces.setdefault(vertex_id,[]).append(index)
    visited=set()
    components=[]
    for start in range(len(selected)):
        if start in visited:
            continue
        stack=[start]
        visited.add(start)
        group=[]
        while stack:
            index=stack.pop()
            group.append(index)
            for vertex_id in selected[index]:
                for adjacent in vertex_to_faces[vertex_id]:
                    if adjacent not in visited:
                        visited.add(adjacent)
                        stack.append(adjacent)
        components.append(group)
    selected=[selected[index] for index in max(components,key=len)]
    mapped={}
    positions=[]
    faces=[]
    for face in selected:
        remapped=[]
        for original_id in face:
            if original_id not in mapped:
                source=vertices[original_id]
                co=source.co.copy()
                rear=ease((co.y+.070)/.120)
                side=ease((abs(co.x)-.050)/.07)
                lower=ease((.315-co.z)/.19)
                attachment=ease((co.z-hairline(co))/.028)
                # Project a separate scalp skin out along the true head
                # normal. The front hairline stays close to the forehead;
                # the lower rear region drapes over, rather than deforms,
                # the donor's cervical column.
                co+=source.normal*(.004+.008*attachment)
                drape=rear*lower*attachment
                co.y+=.034*drape
                co.x+=math.copysign((.014+.012*side)*drape,co.x)
                mapped[original_id]=len(positions)
                positions.append(tuple(co))
            remapped.append(mapped[original_id])
        faces.append(tuple(remapped))
    mesh=bpy.data.meshes.new("Hair | extracted scalp quad topology")
    mesh.from_pydata(positions,[],faces)
    mesh.update()
    # A raw per-face selection creates stair-stepped outlines at the
    # forehead/temples. Relax only the boundary graph, preserving interior
    # anatomical correspondence and a clean sculptable hairline contour.
    counts={}
    for face in faces:
        for a,b in zip(face,face[1:]+face[:1]):
            edge=tuple(sorted((a,b)))
            counts[edge]=counts.get(edge,0)+1
    boundary={}
    for (a,b),count in counts.items():
        if count==1:
            boundary.setdefault(a,set()).add(b)
            boundary.setdefault(b,set()).add(a)
    for _ in range(10):
        previous=[v.co.copy() for v in mesh.vertices]
        for index,neighbors in boundary.items():
            average=sum((previous[j] for j in neighbors),Vector())/len(neighbors)
            mesh.vertices[index].co=previous[index].lerp(average,.46)
    mesh.update()
    ob=bpy.data.objects.new("Hair | anatomy-derived sculptable scalp drape",mesh)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(STONE)
    ob["nape_trimmed"] = bool(trim_nape)
    for poly in mesh.polygons:
        poly.use_smooth=True
    solid=ob.modifiers.new("Physical stone hairline and drape thickness","SOLIDIFY")
    solid.thickness=.009
    solid.offset=-1
    solid.use_rim=True
    ob.scale=(11.,11.,11.)
    ob.location=(0,0,34.03)
    return ob


def sculpt_reference_guided_hair(head, crown_locks=True):
    """Artist-traced, surface-following stone locks; no procedural hair mask.

    Guides are hand-placed against genuine monument front/low oblique photos:
    short, center-parted crown striations and long narrow temple curtains.
    Each crown lock is projected onto the anatomical surface, not an oval
    loft or an expanded scalp mask. This is still an *unregistered* study.
    """
    objects = []

    def catmull(points, resolution):
        result = []
        for i in range(len(points)-1):
            a = points[max(0, i-1)]
            b = points[i]
            c = points[i+1]
            d = points[min(len(points)-1, i+2)]
            for step in range(resolution):
                t = step/resolution
                result.append(tuple(.5*((2*b[j])+(-a[j]+c[j])*t+
                                      (2*a[j]-5*b[j]+4*c[j]-d[j])*t*t+
                                      (-a[j]+3*b[j]-3*c[j]+d[j])*t*t*t)
                                    for j in range(len(b))))
        result.append(tuple(points[-1]))
        return result

    def mesh_object(name, positions, faces):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(positions, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(STONE)
        for face in mesh.polygons:
            face.use_smooth = True
        obj.scale = (11., 11., 11.)
        obj.location = (0, 0, 34.03)
        objects.append(obj)
        return obj

    # The monument's ears are largely concealed by continuous carved hair.
    # Retain their anatomical understructure, but bury the donor's projecting
    # pinnae beneath the new curtain instead of adding decorative ear tubes.
    for v in head.data.vertices:
        p = v.co
        side = min(1., max(0., (abs(p.x)-.083)/.026))
        height = min(1., max(0., (p.z-.205)/.025),
                     max(0., (.304-p.z)/.034))
        rear = min(1., max(0., (p.y+.068)/.080))
        weight = side*side*(3-2*side)*height*rear
        p.x -= math.copysign(.010*weight, p.x)
        p.y += .007*weight
    head.data.update()

    # Hand-authored 3D flow lines: x (lateral), y (front/rear), z (height).
    # These are not claimed to be surveyed points. The shallow sculpt strips
    # follow the actual skin surface through a BVH projection.
    crown_guides = [
        [(.008,-.010,.378), (.027,-.078,.363), (.052,-.110,.351),
         (.080,-.100,.334), (.102,-.078,.300), (.112,-.053,.270)],
        [(.005,.018,.374), (.023,-.045,.369), (.050,-.091,.354),
         (.077,-.082,.334), (.102,-.056,.298), (.115,-.031,.262)],
        [(.011,.049,.359), (.028,-.013,.360), (.056,-.062,.351),
         (.086,-.060,.327), (.109,-.025,.288), (.123,-.005,.253)],
    ]
    bvh = BVHTree.FromObject(head, bpy.context.evaluated_depsgraph_get())
    for sign in (-1, 1):
        for line_no, guide in enumerate(crown_guides if crown_locks else ()):
            path = catmull([(sign*x,y,z) for x,y,z in guide], 18)
            anchors = []
            normals = []
            for pt in path:
                result = bvh.find_nearest(Vector(pt))
                if result[0] is None:
                    raise ValueError("Crown guide left the head surface")
                anchors.append(result[0])
                normals.append(result[1].normalized())
            vertices = []
            sides = 10
            for i, (surface, normal) in enumerate(zip(anchors, normals)):
                tangent = (anchors[min(i+1,len(anchors)-1)] -
                           anchors[max(i-1,0)]).normalized()
                across = normal.cross(tangent).normalized()
                end_taper = min(1., i/9, (len(anchors)-1-i)/12)
                for k in range(sides+1):
                    u = 2*k/sides-1
                    offset = across*(.009*end_taper*u)
                    nearby = bvh.find_nearest(surface+offset)
                    basal = nearby[0] if nearby[0] is not None else surface+offset
                    uplift = (.0005 + .0045*(1-u*u)**2)*end_taper
                    vertices.append(tuple(basal+normal*uplift))
            faces = [(i*(sides+1)+k, (i+1)*(sides+1)+k,
                      (i+1)*(sides+1)+k+1, i*(sides+1)+k+1)
                     for i in range(len(path)-1) for k in range(sides)]
            mesh_object("Hair | %s | traced crown lock %d"%(sign,line_no+1),
                        vertices,faces)

        # These longitudinal control edges describe the narrow temple-to-
        # nape outline observed in real photographs. Unlike the rejected
        # closed-ellipse lofts this is one ruled stone sheet per side, with
        # nonperiodic carved flutes and edges tucked into the skull/neck.
        # (z, anterior x/y, posterior x/y), in donor metres.
        sections = [
            # Both edges begin together below the crown. The first clay
            # iteration erroneously opened a 1.3 m sagittal cape and two
            # horn-like upper corners; genuine photographs show narrow,
            # scalp-hugging locks instead.
            (.345, .093,-.074, .094,-.073),
            (.315, .100,-.062, .110,-.038),
            (.286, .103,-.058, .118,-.020),
            (.249, .103,-.052, .126,+.003),
            (.204, .101,-.039, .134,+.018),
            (.160, .104,-.029, .136,+.036),
            (.113, .108,-.014, .138,+.055),
            (.066, .111,+.017, .138,+.073),
            (.029, .124,+.070, .132,+.081),
        ]
        rows = catmull(sections, 13)
        width_segments = 36
        vertices = []
        for row_index, (z,fx,fy,bx,by) in enumerate(rows):
            longitudinal = row_index/(len(rows)-1)
            tip = min(1., (1-longitudinal)/.055)
            for j in range(width_segments+1):
                u = j/width_segments
                x = sign*((1-u)*fx+u*bx)
                y = (1-u)*fy+u*by
                # A slight front-side sculpt belly follows the temples;
                # 3 uneven groove tracks reproduce carved hair, not waves.
                broad = .008*math.sin(math.pi*u)**2
                channel = sum(.0035*math.exp(-((u-(center+
                             .025*math.sin(longitudinal*3+center*5)))/.027)**2)
                              for center in (.19,.47,.76))
                y -= (broad-channel)*tip
                x *= 1.-(.035*u*(1-tip))
                vertices.append((x,y,z+.002*math.sin(math.pi*u)))
        faces = [(i*(width_segments+1)+k, (i+1)*(width_segments+1)+k,
                  (i+1)*(width_segments+1)+k+1, i*(width_segments+1)+k+1)
                 for i in range(len(rows)-1) for k in range(width_segments)]
        panel = mesh_object("Hair | %s | hand-guided temple curtain"%sign,
                            vertices,faces)
        thickness = panel.modifiers.new("Thin continuous carved stone", "SOLIDIFY")
        thickness.thickness = .008
        thickness.offset = -1
    return objects


def build_anatomical_head_study(path, integrated_hair=False, scalp_surface_hair=False,
                                guided_hair=False, carved_hair=False):
    """Prototype a real quad-topology head instead of more 2D face relief.

    The selected path must be a rights-cleared anatomical OBJ with one
    head and optionally two sclera meshes, all centered as one anatomical
    assembly. This is a *donor anatomy scaffold*, not the monument's scan.
    No OBJ is downloaded or imported into the accepted full-body export.
    """
    if path is None or not path.is_file():
        raise FileNotFoundError("--face-method anatomical requires --anatomy-base OBJ")
    global STONE,CARVING
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    STONE=mat("Clay | anatomical reconstruction study",(.68,.68,.66))
    CARVING=STONE
    if path.suffix == ".gz":
        # Keep the CC0 anatomy asset small in the repository. Use a scoped
        # temporary OBJ for Blender's native importer and discard it safely.
        with tempfile.TemporaryDirectory(prefix="christ-anatomy-") as folder:
            extracted=Path(folder)/"anatomical_head.obj"
            with gzip.open(path,"rb") as source, extracted.open("wb") as target:
                shutil.copyfileobj(source,target)
            bpy.ops.wm.obj_import(filepath=str(extracted))
    else:
        bpy.ops.wm.obj_import(filepath=str(path))
    parts=[obj for obj in bpy.context.selected_objects if obj.type=="MESH"]
    if not parts:
        raise ValueError("The anatomy base OBJ must contain actual mesh geometry")
    # OBJ uses a Y-up convention; Blender imports with a 90-degree X object
    # rotation while vertices remain in OBJ-local axes. Bake this for *all*
    # components before using anatomical X/lateral, Y/depth, Z/height guides.
    # Without this, an apparent jaw-width sculpt mistakenly modifies the
    # sagittal coordinate, a failure visible only in multi-view clay renders.
    for obj in parts:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active=obj
        bpy.ops.object.transform_apply(location=False,rotation=True,scale=False)
    heads=[obj for obj in parts if "head_sculpting_realistic" in obj.name and
           "sclera" not in obj.name and "iris" not in obj.name]
    if len(heads)!=1:
        raise ValueError("Expected one Blender Studio realistic-head scaffold")

    # Broad-form sculpt on a continuous, anatomically organized quad mesh.
    # Coordinates are from the actual CC0 head control cage (metres), before
    # enlarging it to monument scale. Unlike the old face-heightfield this
    # alters skull, mandible, chin and three-dimensional neck transitions.
    # All edits remain *artist inferred* until camera/scan fitting is solved.
    head=heads[0]
    for vertex in head.data.vertices:
        p=vertex.co
        source_x,source_y,source_z=p.x,p.y,p.z
        # The reference sculpture has a heavier, wider Art Deco cranium and
        # mandible than this contemporary human donor. Keep the shoulder
        # root narrower and transition to the full head above the neck.
        neck_blend=max(0.,min(1.,(source_z-.075)/.105))
        neck_blend=neck_blend*neck_blend*(3-2*neck_blend)
        p.x=source_x*(.88+.39*neck_blend)
        # Lengthen the lower-face planes below the eye line; extend the chin
        # in real 3D, while the posterior cervical column stays attached.
        anterior=max(0.,min(1.,(-source_y+.015)/.115))
        lower_face=max(0.,min(1.,(.278-source_z)/.17))
        chin_join=max(0.,min(1.,(source_z-.035)/.055))
        p.z-=.046*anterior*lower_face*chin_join
        # Lift the lower cheek and project the jaw as a single substantial
        # anatomical plane, without gluing on an independent chin sphere.
        jaw_zone=max(0.,1.-((source_z-.137)/.079)**2)
        if jaw_zone>0:
            p.y-=.009*jaw_zone*anterior
        # Sculpt the anatomical upper eyelid downward into the physical eye
        # aperture and lift the lower edge a little. This moves the eyelid
        # topology itself, not a detached black arc placed on an ellipsoid.
        eye_lateral=max(0.,1.-((abs(source_x)-.0316)/.033)**4)
        eye_front=max(0.,min(1.,(-source_y-.065)/.064))
        upper=max(0.,1.-((source_z-.282)/.017)**2)
        lower=max(0.,1.-((source_z-.257)/.014)**2)
        p.z+=eye_lateral*eye_front*(-.0085*upper+.0025*lower)
        # The monument's mouth turns down through both commissures; moving
        # the lip/jaw vertices gives a real three-dimensional curve.
        corner=max(0.,1.-((abs(source_x)-.050)/.021)**2)
        lip_level=max(0.,1.-((source_z-.168)/.023)**2)
        p.z-=.009*corner*lip_level*anterior
        # Landowski's nose reads as a long squared bridge above a substantial
        # alar tip in the authentic scan's side/front previews. Sculpt the
        # *actual anatomical mesh* in 3D: the tip projects along the sagittal
        # axis, but its root withdraws continuously into the glabella.
        # These are artist-fitted planes, not measured scan coordinates.
        nasal_front=max(0.,min(1.,(-source_y-.110)/.045))
        nose_lateral=max(0.,1.-(abs(source_x)/.034)**2)
        # A shallow, broad bridge *plane* preserves the near-vertical nasal
        # dorsum seen in the scan profile. A steep localized projection made
        # an incorrect wedge/beak in the first anatomical trial (r07).
        ridge_z=max(0.,1.-((source_z-.255)/.105)**4)
        tip_z=max(0.,1.-((source_z-.207)/.035)**2)
        p.y-=nasal_front*nose_lateral*(.008*ridge_z+.009*tip_z)
    for obj in parts:
        obj.name=("Head | anatomical quad-topology clay sculpt" if obj == heads[0]
                  else "Head | stone eyeball reference %s"%obj.name)
        obj.scale=(11.,11.,11.)
        obj.location=(0,0,34.03)
        obj.data.materials.clear()
        obj.data.materials.append(STONE)
        if obj == heads[0]:
            subdiv=obj.modifiers.new("Editable anatomical sculpt density","SUBSURF")
            subdiv.levels=2
            subdiv.render_levels=2
            bpy.context.view_layer.objects.active=obj
            bpy.ops.object.modifier_apply(modifier=subdiv.name)
        for polygon in obj.data.polygons:
            polygon.use_smooth=True
    if integrated_hair:
        parts.extend(sculpted_anatomical_hair_mantle())
    if scalp_surface_hair:
        parts.append(sculpt_scalp_surface_from_anatomy(head))
    if guided_hair:
        parts.extend(sculpt_reference_guided_hair(head))
    if carved_hair:
        parts.extend(sculpt_carved_hair_from_anatomy(
            head, curtain_builder=sculpt_reference_guided_hair,
            scalp_builder=sculpt_scalp_surface_from_anatomy))
    # Newly created loft objects can carry stale matrix_world values until
    # a dependency-graph update. Applying the gaze tilt first accidentally
    # put the entire hair volume near the scene origin, outside every camera.
    bpy.context.view_layer.update()
    orient_head(parts)


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
            "head_tilt_deg":HEAD_TILT_DEG,"face_method":FACE_METHOD,
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
    global HEAD_TILT_DEG,FACE_METHOD
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
    p.add_argument("--face-method",choices=("gaussian","landmark","anatomical"),
                   default="gaussian",
                   help="Select old export, relief experiment, or 3D anatomy scaffold")
    p.add_argument("--anatomy-base",type=Path,
                   default=ANATOMY_BASE,
                   help="CC0 anatomical OBJ, used only in --head-study anatomical mode")
    p.add_argument("--integrated-hair",action="store_true",
                   help="Clay-only continuous scalp and neck hair-mass study")
    p.add_argument("--scalp-surface-hair",action="store_true",
                   help="Clay-only anatomy-surface retopology study (not loft hair)")
    p.add_argument("--guided-hair",action="store_true",
                   help="Clay-only artist-traced crown and neck hair study")
    p.add_argument("--carved-hair",action="store_true",
                   help="Clay-only scalp-topology hair with real carved crown grooves")
    p.add_argument("--experimental-face-corrections",type=Path,
                   help="Apply unapproved multi-photo triangulated facial offsets to clay only")
    p.add_argument("--model-revision",type=str,
                   help="Exact model revision recorded by 3D landmark probes")
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
    FACE_METHOD=args.face_method
    if FACE_METHOD == "anatomical" and not args.head_study:
        p.error("Anatomical mode is a clay-only reconstruction HOLD, not a full export")
    if args.integrated_hair and (not args.head_study or FACE_METHOD != "anatomical"):
        p.error("--integrated-hair requires --head-study --face-method anatomical")
    if args.scalp_surface_hair and (not args.head_study or FACE_METHOD != "anatomical"):
        p.error("--scalp-surface-hair requires --head-study --face-method anatomical")
    if args.guided_hair and (not args.head_study or FACE_METHOD != "anatomical"):
        p.error("--guided-hair requires --head-study --face-method anatomical")
    if args.carved_hair and (not args.head_study or FACE_METHOD != "anatomical"):
        p.error("--carved-hair requires --head-study --face-method anatomical")
    if sum((args.scalp_surface_hair,args.integrated_hair,
            args.guided_hair,args.carved_hair)) > 1:
        p.error("Choose one hair reconstruction study")
    if args.experimental_face_corrections:
        if not args.head_study or FACE_METHOD != "anatomical":
            p.error("Facial corrections require anatomical head-study clay mode")
        if not args.model_revision:
            p.error("--model-revision is required with facial corrections")
        if any((args.scalp_surface_hair,args.integrated_hair,
                args.guided_hair,args.carved_hair)):
            p.error("Facial corrections require a bare head until hair collision QA passes")
    if args.head_study:
        if FACE_METHOD == "anatomical":
            build_anatomical_head_study(args.anatomy_base,args.integrated_hair,
                                       args.scalp_surface_hair,args.guided_hair,
                                       args.carved_hair)
        else:
            build_head_study()
        head=next(ob for ob in bpy.data.objects
                  if ob.name.startswith(("Head | integrated cheekbone",
                                          "Head | anatomical quad-topology")))
        face_fit_metrics={}
        if args.experimental_face_corrections:
            report=json.loads(args.experimental_face_corrections.read_text())
            face_fit_metrics=apply_candidate_face_fit(head,report,args.model_revision)
        head_points=[head.matrix_world @ vertex.co for vertex in head.data.vertices]
        metrics={"study":"head","head_surface_vertices":len(head.data.vertices),
                 "head_tilt_deg":HEAD_TILT_DEG,"face_method":FACE_METHOD,
                 "head_width_m":round(max(p.x for p in head_points)-
                                      min(p.x for p in head_points),3),
                 "head_depth_m":round(max(p.y for p in head_points)-
                                      min(p.y for p in head_points),3),
                 "mesh_count":sum(ob.type=="MESH" for ob in bpy.data.objects)}
        metrics.update(face_fit_metrics)
        if FACE_METHOD == "anatomical":
            metrics["integrated_hair"]=args.integrated_hair
            metrics["scalp_surface_hair"]=args.scalp_surface_hair
            metrics["guided_hair"]=args.guided_hair
            metrics["carved_hair"]=args.carved_hair
            if args.integrated_hair or args.scalp_surface_hair or args.guided_hair or args.carved_hair:
                hair_objs=[obj for obj in bpy.data.objects
                           if obj.name.startswith("Hair | ")]
                hair_points=[obj.matrix_world @ v.co
                             for obj in hair_objs for v in obj.data.vertices]
                count=(1 if args.scalp_surface_hair else
                       8 if args.guided_hair else 3)
                assert len(hair_objs)==count and hair_points
                metrics["hair_surface_vertices"]=len(hair_points)
                if args.carved_hair:
                    scalp=next(ob for ob in hair_objs
                               if "anatomy-derived sculptable scalp" in ob.name)
                    metrics["carved_hair_vertices"]=int(scalp["carved_vertices"])
                    metrics["carved_hair_grooves"]=int(scalp["groove_count"])
                    metrics["rooted_curtain_vertices"]=int(
                        scalp["rooted_curtain_vertices"])
                    metrics["max_root_snap_distance_m"]=round(
                        float(scalp["max_root_snap_distance_m"]),5)
                    metrics["groove_projection"]=str(scalp["groove_projection"])
                    metrics["scalp_nape_trimmed"]=bool(scalp["nape_trimmed"])
                    metrics["scalp_min_z_m"]=round(min(
                        (scalp.matrix_world @ v.co).z for v in scalp.data.vertices),3)
                metrics["hair_bounds_min_z_m"]=round(min(p.z for p in hair_points),3)
                metrics["hair_bounds_max_z_m"]=round(max(p.z for p in hair_points),3)
                metrics["hair_bounds_width_m"]=round(
                    max(p.x for p in hair_points)-min(p.x for p in hair_points),3)
            # The local cage has metric human-scale dimensions; its object
            # scale is 11. A physical nose/cheek depth check guards against
            # reverting to an entirely flat mask. This is *not* a photo fit.
            nasal=[v.co.y for v in head.data.vertices
                   if abs(v.co.x)<.018 and .205<v.co.z<.238]
            cheek=[v.co.y for v in head.data.vertices
                   if .055<abs(v.co.x)<.085 and .205<v.co.z<.238]
            assert nasal and cheek
            metrics["nose_advance_vs_cheek_mm"]=round(
                (min(cheek)-min(nasal))*11000)
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
