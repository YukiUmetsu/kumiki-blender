"""Christ the Redeemer: original procedural, dimensioned Art Deco sculpture.

30 m figure / 28 m arm span / 8 m pedestal. No external scan or mesh is used.
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
from mathutils import Vector

TAU = math.tau
STONE = None
CARVING = None
GRANITE = None
MOSAIC_PITCH = .72 / 24  # 3 cm squares split into ~4.2 cm triangular tesserae.


def soapstone_mosaic():
    """Create an original tileable mosaic color map embedded in the GLB."""
    size = 768
    cells = 24
    tile_size = size // cells
    image = bpy.data.images.new("Soapstone | triangular mosaic, 3 cm pitch",
                                width=size, height=size, alpha=False)
    image.file_format = "PNG"
    pixels = array("f")

    def hash01(a, b, c):
        n = math.sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5453
        return n - math.floor(n)

    for y in range(size):
        cy, fy = divmod(y, tile_size)
        for x in range(size):
            cx, fx = divmod(x, tile_size)
            upper = int(fx + fy >= tile_size)
            variation = (hash01(cx, cy, upper) - .5) * .024
            fine = (hash01(x // 3, y // 3, 7) - .5) * .004
            # Fine joints between the triangles, without hard dark grid lines.
            joint = min(fx, fy, tile_size - 1 - fx,
                        tile_size - 1 - fy, abs(fx + fy - tile_size) / 1.414)
            grout = -.014 * max(0, 1 - joint / 1.3)
            shade = variation + fine + grout
            pixels.extend((.665 + shade, .673 + shade, .650 + shade, 1))
    image.pixels.foreach_set(pixels)
    image.pack()
    return image


def mat(name, rgb, image=None):
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
    (8.0, 2.35, 1.36), (9.0, 2.40, 1.39), (12, 2.34, 1.37),
    (16, 2.18, 1.30), (20, 2.18, 1.39), (24, 2.47, 1.49),
    (27, 2.86, 1.58), (29.5, 3.27, 1.68), (31.2, 3.58, 1.73),
    (32.3, 3.03, 1.50), (33.40, 1.26, .99),
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
            y = cloth_front(x,z)-relief-.04*math.sin(math.pi*s)-shape
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
            pts.append((x,cloth_front(x,z)-relief-.055,z))
        shaped_path(name+" | carved selvedge %s"%edge,pts,
                    [.026+.027*math.sin(math.pi*j/34) for j in range(35)])
    return obj


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

    # The asymmetrically overlapping cloak flows off one shoulder, crosses the
    # breast, then drops as a weighted mantle; geometry, not painted lines.
    draped_panel("Mantle | oblique shoulder fold",27.3,32.0,
                 lambda t: .68-2.76*t,
                 lambda t: .34+.60*math.sin(math.pi*t)+.43*t,.20)
    draped_panel("Mantle | descending left drape",10.1,27.5,
                 lambda t: -.90+1.58*t+.13*math.sin(math.pi*t),
                 lambda t: .025+.325*t+.27*math.sin(math.pi*t),.17)

    # Curved secondary drapery follows the curved body; end widths taper to zero.
    for side in (-1,1):
        for strand in range(3):
            pts = []
            for j in range(18):
                t=j/17
                z=15.8+14.2*t
                x=side*(.45+strand*.31 + (.34-.05*strand)*math.sin(t*math.pi))
                pts.append((x,cloth_front(x,z)-.11,z))
            shaped_path("Cloth | sweeping chest panel %s %s"%(side,strand),pts,
                        [.015+.09*math.sin(math.pi*j/17) for j in range(18)])
    for k in range(15):
        x0=-1.93+k*.273
        pts=[]
        for j in range(18):
            t=j/17
            z=8.16+t*(8.5+.5*math.cos(k))
            x=x0+.10*math.sin(t*2.3+k*.45)
            pts.append((x,cloth_front(x,z)-.065,z))
        shaped_path("Cloth | lower pleat %02d"%k, pts,
                    [.009+.075*math.sin(math.pi*j/17) for j in range(18)])
    for k in range(6):
        pts=[]
        for j in range(14):
            t=j/13
            x=-1.38+t*(2.37-.13*k)
            z=27.15+.55*k+.19*math.sin(t*math.pi)
            pts.append((x,cloth_front(x,z)-.07,z))
        shaped_path("Cloth | diagonal chest fold %02d"%k,pts,
                    [.008+.067*math.sin(math.pi*j/13) for j in range(14)])
    sphere("Collar | shoulder transition",(0,0,32.95),(1.23,1.01,.50))


def arms(side):
    # Long tapering mass, upper surface nearly horizontal, weighted lower hem.
    measurements = [
        (2.65,31.28,1.26,1.05),(3.35,31.13,1.30,1.07),
        (4.45,30.85,1.25,1.03),(5.65,30.59,1.15,.96),
        (7.20,30.34,1.00,.83),(8.30,30.12,.87,.74),
        (9.30,29.93,.76,.64),(10.30,29.78,.62,.49),
        (10.90,29.72,.46,.38),(11.13,29.70,.38,.34),
    ]
    sections=[]
    for x,z,r,d in measurements:
        ring=[]
        for k in range(64):
            a=TAU*k/64
            lower=max(0,-math.cos(a))
            fluting=.065*math.cos(5*a+x*.4)+.03*math.cos(9*a)
            ring.append((side*x, d*math.sin(a), z+r*math.cos(a)-lower*(.12+fluting)))
        sections.append(ring)
    loft("Sleeve | %s | physically closed carved volume"%side,sections,invert=side==1)
    for k in range(2):
        pts=[]
        for x,z,r,d in measurements[1:7]:
            angle=1.2+.36*k+.38*math.sin((x-3.3)*.39)
            pts.append((side*x,-d*math.sin(angle)-.018,z-r*abs(math.cos(angle))-.04))
        shaped_path("Sleeve | %s | flow groove %s"%(side,k),pts,
                    [.009+.037*math.sin(math.pi*j/(len(pts)-1)) for j in range(len(pts))])
    sphere("Cuff | %s | recessed thickness"%side,(side*11.14,0,29.7),(.075,.32,.34),CARVING)
    hands(side)


def hands(side):
    z=29.68
    sphere("Hand | %s | wrist"%side,(side*11.36,0,z),(.31,.22,.30))
    sphere("Hand | %s | broad sculpted palm"%side,(side*11.86,0,z),(.73,.21,.43))
    for k,tip in enumerate((13.57,13.96,14.0,13.56)):
        finger_z=z+.31-.205*k
        pts=[(side*12.22,-.015,finger_z),
             (side*12.65,-.028,finger_z-.018),
             (side*(12.65+.52*(tip-12.65)),-.042,finger_z-.032),
             (side*tip,-.055,finger_z-.055)]
        shaped_path("Hand | %s | articulated finger %s"%(side,k),pts,
                    (.115,.12,.096,.032),sides=12)
    shaped_path("Hand | %s | thumb"%side,
                [(side*11.58,-.12,z-.28),(side*11.99,-.16,z-.51),
                 (side*12.65,-.16,z-.63),(side*12.82,-.10,z-.55)],
                (.155,.14,.105,.038),sides=12)


HEAD = [
    (34.02,.18,.27,-.03),(34.23,.54,.56,-.05),(34.47,.81,.75,-.08),
    (34.82,1.04,.92,-.10),(35.30,1.28,1.10,-.08),
    (35.85,1.47,1.22,0),(36.50,1.52,1.24,.05),
    (37.15,1.43,1.18,.10),(37.65,1.03,.91,.11),
    (37.95,.38,.40,.12),(38.0,.045,.07,.12),
]


def face():
    sphere("Head | strong neck",(0,.18,33.75),(.54,.57,.94))
    # Densify the face loft to sculpt cheekbones, chin and jaw as one surface.
    # Surface folds are smooth Gaussian volumes, not glued-on primitive cheeks.
    sampled=[]
    for low,high in zip(HEAD[:-1],HEAD[1:]):
        for j in range(6):
            t=j/6
            sampled.append(tuple(low[k]*(1-t)+high[k]*t for k in range(4)))
    sampled.append(HEAD[-1])
    rings=[]
    for z,rx,ry,cy in sampled:
        ring=[]
        for i in range(96):
            a=TAU*i/96
            x=rx*math.cos(a)
            y=cy+ry*math.sin(a)
            if math.sin(a)<0:
                # Integrate the full facial relief into the head surface.
                # Features are smooth local displacements of the same manifold,
                # eliminating the stacked cylinder/ball appearance in profile.
                cheek=.17*math.exp(-((abs(x)-.82)/.33)**2-((z-35.43)/.46)**2)
                jaw=.07*math.exp(-((abs(x)-.66)/.42)**2-((z-34.90)/.35)**2)
                chin=.18*math.exp(-(x/.57)**2-((z-34.46)/.32)**2)
                brow=.07*math.exp(-((abs(x)-.65)/.37)**2-((z-36.77)/.115)**2)
                upper_lid=.052*math.exp(-((abs(x)-.66)/.32)**2-((z-36.55)/.052)**2)
                lower_lid=.029*math.exp(-((abs(x)-.67)/.32)**2-((z-36.37)/.067)**2)
                eye_socket=.08*math.exp(-((abs(x)-.67)/.36)**2-((z-36.46)/.13)**2)
                nose_bridge=.13*math.exp(-(x/.18)**2-((z-36.18)/.48)**2)
                nose_tip=.31*math.exp(-(x/.22)**2-((z-35.74)/.19)**2)
                nostril_wing=.12*math.exp(-((abs(x)-.23)/.16)**2-((z-35.64)/.13)**2)
                top_lip=.057*math.exp(-(x/.40)**2-((z-35.20)/.072)**2)
                bottom_lip=.049*math.exp(-(x/.37)**2-((z-35.075)/.068)**2)
                mouth_crease=.045*math.exp(-(x/.38)**2-((z-35.13)/.030)**2)
                jaw_beard=.13*math.exp(-((z-34.77)/.37)**2)*math.exp(-(x/1.12)**6)
                side_beard=.15*math.exp(-((abs(x)-.98)/.28)**2-((z-35.23)/.56)**2)
                beard_mask=min(1.,(jaw_beard+side_beard)/.16)
                carved_grain=.026*beard_mask*(.6*math.cos(33*x+8*z)+
                                                .4*math.cos(21*x-12*z))
                y-= (cheek+jaw+chin+brow+upper_lid+lower_lid+
                     nose_bridge+nose_tip+nostril_wing+top_lip+bottom_lip+
                     jaw_beard+side_beard+carved_grain)
                y+=eye_socket+mouth_crease
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
        # Almond eyelid lines conform to the sculpted surface, even in profile.
        top=[(side*.32,36.48),(side*.53,36.50),(side*.78,36.485),
             (side*1.04,36.46)]
        bottom=[(side*.32,36.44),(side*.58,36.395),(side*.84,36.415),
                (side*1.04,36.46)]
        shaped_path("Face | %s | upper eyelid incision"%side,
                    [(x,face_y(x,z),z) for x,z in top],
                    (.008,.022,.019,.006),CARVING,sides=8)
        shaped_path("Face | %s | lower eyelid incision"%side,
                    [(x,face_y(x,z,.009),z) for x,z in bottom],
                    (.007,.012,.011,.004),CARVING,sides=8)
        sphere("Ear | %s | helix"%side,(side*1.41,-.035,36.08),(.11,.15,.36))
        sphere("Ear | %s | concha shadow"%side,(side*1.50,-.16,36.06),(.027,.066,.17),CARVING)
        sphere("Ear | %s | lobule"%side,(side*1.41,-.07,35.78),(.08,.10,.11))
    for side in (-1,1):
        x=side*.16
        sphere("Face | %s | carved nostril"%side,
               (x,face_y(x,35.61,.008),35.61),(.032,.020,.014),CARVING)
    lip=[(-.38,35.14),(-.17,35.13),(0,35.13),(.17,35.13),(.38,35.14)]
    shaped_path("Face | carved closed mouth",
                [(x,face_y(x,z,.009),z) for x,z in lip],
                (.006,.011,.012,.011,.006),CARVING)
    for side in (-1,1):
        # Soft moustache falls away from the philtrum, framing the closed lips.
        pts=[]
        for j in range(10):
            t=j/9
            x=side*(.08+.52*t)
            z=35.34-.095*t+.027*math.sin(math.pi*t)
            pts.append((x,face_y(x,z,.024),z))
        shaped_path("Beard | %s | carved moustache"%side,pts,
                    [.012+.032*math.sin(math.pi*j/9) for j in range(10)])
        for k in range(4):
            # Follow the lower cheek down to a tapered central chin rather
            # than applying an unstructured black beard material.
            pts=[]
            for j in range(13):
                t=j/12
                x=side*(.84+.045*k-(.19+.04*k)*t+.025*math.sin(math.pi*t))
                z=35.35-.145*k-(.32+.04*k)*t
                pts.append((x,face_y(x,z,.012),z))
            shaped_path("Beard | %s | etched curl %s"%(side,k),pts,
                        [.003+.009*math.sin(math.pi*j/12) for j in range(13)],
                        CARVING,sides=7)
    # Center-parted, gravity-led waves; lengths descend behind the cheeks.
    for side in (-1,1):
        for k in range(2):
            # Forehead locks sweep from the center part to the temples. The
            # starts sit on the scalp, not as exposed spike tips above it.
            pts=[]
            for j in range(18):
                t=j/17
                x=side*(.12+1.08*t+.055*k)
                z=37.69-.61*t-.045*k+.035*math.sin(math.pi*t)
                y=face_y(x,z,.07)
                pts.append((x,y,z))
            shaped_path("Hair | %s | swept forehead wave %s"%(side,k),pts,
                        [.009+.044*math.sin(math.pi*j/17) for j in range(18)],sides=13)
        for k in range(2):
            pts=[]
            for j in range(24):
                t=j/23
                pts.append((side*(1.25+.16*k+.18*math.sin(2.5*t)),
                            -.18-.23*math.sin(math.pi*t)+.2*k,
                            37.42-(2.75+.28*k)*t))
            shaped_path("Hair | %s | temple curtain %s"%(side,k),pts,
                        [.055+.18*math.sin(math.pi*j/23) for j in range(24)],sides=13)
        for k in range(3):
            pts=[]
            radius=[]
            for j in range(19):
                t=j/18
                z=37.82-(3.42+.17*k)*t
                x=side*(.21+.43*math.sin(t*math.pi/2)+.25*k+.10*math.sin(4*t+.6*k))
                y=.31+.27*k-.18*math.sin(t*math.pi)
                pts.append((x,y,z))
                radius.append(.075+(.22-.018*k)*math.sin(math.pi*t))
            shaped_path("Hair | %s | flowing lock %s"%(side,k),pts,radius,sides=13)
            sphere("Hair | %s | rounded lock end %s"%(side,k),pts[-1],(.076,.078,.085))
            if k<3:
                groove=[(x+side*.115,y-.1,z) for x,y,z in pts[2:-1]]
                shaped_path("Hair | %s | recessed strand %s"%(side,k),groove,
                            [.012+.021*math.sin(math.pi*j/(len(groove)-1)) for j in range(len(groove))],
                            CARVING,sides=7)
        sphere("Hair | %s | crown"%side,(side*.55,.39,37.49),(.66,.62,.32))


def pedestal():
    bpy.ops.mesh.primitive_cylinder_add(vertices=6,radius=3.2,depth=8.0,
                                        location=(0,0,4),rotation=(0,0,math.pi/6))
    obj=bpy.context.object
    obj.name="Pedestal | hexagonal granite"
    obj.data.materials.append(GRANITE)
    bevel=obj.modifiers.new("Cut stone edges","BEVEL")
    bevel.width=.085
    bevel.segments=3
    obj.modifiers.new("Weighted pedestal normals","WEIGHTED_NORMAL")
    for title,z,depth,radius in (("ground course",.24,.48,3.36),
                                 ("upper cornice",7.83,.34,3.30)):
        bpy.ops.mesh.primitive_cylinder_add(vertices=6,radius=radius,depth=depth,
                                            location=(0,0,z),rotation=(0,0,math.pi/6))
        course=bpy.context.object
        course.name="Granite | hexagonal "+title
        course.data.materials.append(GRANITE)
        mod=course.modifiers.new("Chiselled stone arris","BEVEL")
        mod.width=.04
        mod.segments=2
        course.modifiers.new("Weighted cornice normals","WEIGHTED_NORMAL")


def build():
    global STONE,CARVING,GRANITE
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    STONE=mat("Warm light grey soapstone | tesserae",(.665,.67,.635),
              soapstone_mosaic())
    CARVING=mat("Soft shadow in carved recesses",(.43,.45,.42))
    GRANITE=mat("Pedestal | grey granite",(.33,.34,.33))
    pedestal()
    robe()
    for side in (-1,1):
        arms(side)
    face()


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
    for side in (-1,1):
        obj=next(ob for ob in objs if ob.name.startswith("Sleeve | %s | physically"%side))
        faces=[p for p in obj.data.polygons if len(p.vertices)==4]
        checks=[p.normal.dot(Vector((0,p.center.y,p.center.z-30.0)).normalized()) > 0
                for p in faces if abs(p.center.z-30)>0.05]
        winding[side]=round(sum(checks)/len(checks),3)
        edge_uses={}
        for polygon in obj.data.polygons:
            for edge in polygon.edge_keys:
                edge_uses[edge]=edge_uses.get(edge,0)+1
        open_edges[side]=sum(uses!=2 for uses in edge_uses.values())
    result={"bounds_min":minimum,"bounds_max":maximum,"pedestal_top":ptop,
            "robe_bottom":rbot,"sleeve_outward_fraction":winding,
            "sleeve_nonmanifold_edges":open_edges,
            "pedestal_sides":6,
            "mosaic_images":len([im for im in bpy.data.images if im.name.startswith("Soapstone |")]),
            "uv_mesh_count":sum(bool(ob.data.uv_layers) for ob in objs),
            "head_surface_vertices":len(head_ob.data.vertices),
            "mesh_count":len(objs),"vertex_count":sum(len(ob.data.vertices) for ob in objs)}
    assert abs(ptop-rbot)<.001,"Statue must meet pedestal without floating"
    assert maximum[0]-minimum[0]>27.6,"Hands should span 28 m"
    assert maximum[2]>37.9 and minimum[2]<.01,"Scale should be 38 m including pedestal"
    assert min(winding.values())>.60,"Both sleeve normals must face outward"
    assert not any(open_edges.values()),"Sleeve meshes must be closed and manifold"
    assert len(head_ob.data.vertices)>5000,"Face needs a dense sculptable surface"
    assert result["mosaic_images"]==1 and result["uv_mesh_count"]>70, "Mosaic needs glTF UVs"
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
    lights=[((8,-15,43),4800,9),((-10,3,36),5600,8),((9,12,25),2600,7)]
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
        ("head_profile",(19,0,36),(0,0,36),6.4),
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
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser()
    p.add_argument("--output")
    p.add_argument("--blend")
    p.add_argument("--views")
    p.add_argument("--metrics")
    p.add_argument("--verify",action="store_true")
    p.add_argument("--samples",type=int,default=16)
    p.add_argument("--size",type=int,default=720)
    p.add_argument("--shot",action="append",choices=("front","profile","three_quarter",
                                                   "head_front","head_profile","hand_detail"),
                   help="Render only selected camera(s); repeat for multiple shots")
    args=p.parse_args(argv)
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
