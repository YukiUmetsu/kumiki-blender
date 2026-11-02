"""A deliberately non-geographic Fuji miniature: silhouette, spring and shrine.

Everything visible is modelled as portable geometry with simple PBR materials.
There are no billboards, external textures, fonts or linked dependencies.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from ...blender_output import create_model_scene, export_revision
from ...glb_audit import audit_glb
from ...shots import QualityShot
from .blender_scene import _studio


# Deliberately more legible at a small thumbnail than the geographic colorway:
# cold volcanic blue-grey, evergreen green and obvious multi-pink sakura.
ICONIC_BLUE_BASALT=(.115,.182,.278)
ICONIC_WEATHERED_ROCK=(.205,.220,.265)
ICONIC_TREE_LINE=(.082,.213,.145)
SAKURA_PETAL_PALETTE=((.98,.46,.63),(.99,.67,.78),(.997,.83,.87),
                      (.92,.34,.51),(.97,.70,.80))


def plan_iconic_shots() -> tuple[QualityShot, ...]:
    """Hero plus scrutiny of the emblematic geometry and small-footprint layout."""
    return (
        QualityShot("hero", (9.8, -17.4, 9), (0, 0, 2.0), 11.9),
        QualityShot("front", (0, -18, 6.3), (0, 0, 2.05), 11.5),
        QualityShot("opposite", (-13, -11, 7.5), (0, .2, 1.95), 12.1),
        QualityShot("torii-and-sakura", (5.8, -8.8, 4.7), (1.55, -2.25, 1.55), 5.0),
        QualityShot("sakura-botanical", (-6.4, -5.3, 3.7), (-3.1, -1.45, 2.0), 3.65),
        QualityShot("snow-crown", (6.5, -6, 8.2), (0, .65, 3.68), 4.2),
        QualityShot("footprint", (0, -1, 19), (0, 0, .35), 11.5),
    )


def iconic_spec(detail: str = "standard") -> dict[str, Any]:
    if detail not in ("preview", "standard", "high"):
        raise ValueError(f"Unknown iconic detail: {detail}")
    return {
        "asset_id": "mount-fuji-iconic-diorama",
        "style": "iconic",
        "detail": detail,
        "footprint_units": [10.2, 8.6],
        "vertical_scale": "intentionally exaggerated and non-geographic",
        "landmarks": ["Fuji snow crown", "vermillion torii", "flowering sakura"],
        "terrain_divisions": {"preview": 144, "standard": 240, "high": 360}[detail],
        "sakura_blossom_target": {"preview": 1500, "standard": 3700, "high": 8500}[detail],
    }


def _material(bpy: Any, name: str, rgb: tuple[float, float, float], *,
              roughness: float = .76, metallic: float = 0) -> Any:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1)
    mat.use_nodes = True
    shader = next(node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*rgb, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return mat


def _blend(a: tuple[float,float,float], b: tuple[float,float,float], t: float) -> tuple[float,float,float]:
    t=max(0.,min(1.,t))
    return tuple(a[i]*(1-t)+b[i]*t for i in range(3))


def _ease(start: float, end: float, value: float) -> float:
    t=max(0.,min(1.,(value-start)/(end-start)))
    return t*t*(3-2*t)


def _mesh(bpy: Any, collection: Any, name: str, vertices: list, faces: list,
          material: Any, *, colors: list | None = None, smooth: bool = False) -> Any:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.materials.append(material)
    if colors:
        attr = mesh.color_attributes.new(name="Albedo", type="FLOAT_COLOR", domain="POINT")
        for i, value in enumerate(colors):
            attr.data[i].color = (*value, 1)
    if smooth:
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj["kumiki_part"] = name.split(" / ")[0]
    return obj


def _vertex_material(bpy: Any, name: str) -> Any:
    mat = _material(bpy, name, (.6, .6, .6))
    node = mat.node_tree.nodes.new("ShaderNodeVertexColor")
    node.layer_name = "Albedo"
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    mat.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _bake_portable_palette_textures(bpy: Any, objects: list[Any],
                                    mats: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    """Bake iconic vertex colors as ordinary glTF PBR albedo textures.

    Some GLB viewers silently ignore COLOR_0, making our pink sakura and
    blue-green Fuji grey or white. Instead of storing a costly per-triangle
    atlas for thousands of blossoms, project each material's colors onto its
    best-fit 2D color plane. UV coordinates *encode pigment*, not position;
    a small linear color-plane texture reproduces gradients without relying
    on optional vertex-color rendering. Original pigments remain as a custom
    point attribute in the editable .blend, but no COLOR_0 multiplies the
    exported texture a second time in standards-compliant glTF viewers.
    """
    import numpy as np  # Blender ships NumPy.

    report: dict[str, dict[str, float | int]] = {}
    for key in ("ground_vertex", "flank_vertex", "snow_vertex", "blossom_vertex"):
        mat=mats[key]
        owners=[obj for obj in objects if obj.type=="MESH" and
                any(candidate == mat for candidate in obj.data.materials)]
        if not owners:
            raise ValueError(f"Iconic material {key} has no mesh to color")
        source=[]
        for obj in owners:
            attribute=obj.data.color_attributes.get("Albedo")
            if attribute is None or attribute.domain!="POINT":
                raise ValueError(f"{obj.name} is missing point-domain Albedo colors")
            samples=np.empty(len(attribute.data)*4,dtype=np.float32)
            attribute.data.foreach_get("color",samples)
            source.append(samples.reshape(-1,4)[:,:3].copy())
        colors=np.concatenate(source,axis=0).astype(np.float64)
        mean=colors.mean(axis=0)
        centered=colors-mean
        covariance=(centered.T @ centered)/len(colors)
        _,vectors=np.linalg.eigh(covariance)
        basis=vectors[:,1:3]
        projected=centered @ basis
        minimum=projected.min(axis=0)
        span=np.maximum(projected.max(axis=0)-minimum,1e-5)
        reconstructed=mean+projected @ basis.T
        max_error=float(np.max(np.abs(reconstructed-colors)))
        rms_error=float(np.sqrt(np.mean((reconstructed-colors)**2)))
        if rms_error>.055:
            raise ValueError(f"Iconic color plane loses too much {key} pigment: {rms_error:.4f}")

        # A linear color plane has no photographic pixel detail: 256² pixels
        # suffice even for 8500 five-petal blossoms and their subtle gradients.
        size=256
        margin=.025
        u=np.linspace(0,1,size,dtype=np.float64)
        uv_u,uv_v=np.meshgrid(u,u)
        position=np.stack((uv_u,uv_v),axis=-1)
        projected_pixels=minimum+(position-margin)/(1-2*margin)*span
        linear=np.clip(mean+projected_pixels @ basis.T,0,1)
        srgb=np.where(linear<=.0031308,linear*12.92,
                      1.055*np.power(linear,1/2.4)-.055)
        pixels=np.empty((size,size,4),dtype=np.float32)
        pixels[:,:,:3]=srgb
        pixels[:,:,3]=1
        image=bpy.data.images.new(f"Japan / embedded {key} PBR albedo",
                                  width=size,height=size,alpha=True,float_buffer=False)
        image.colorspace_settings.name="sRGB"
        image.pixels.foreach_set(np.ascontiguousarray(pixels).ravel())
        image.update()
        image.pack()

        offset=0
        for obj,values in zip(owners,source):
            count=len(values)
            point_uv=margin+(projected[offset:offset+count]-minimum)/span*(1-2*margin)
            offset+=count
            mesh=obj.data
            uv_layer=mesh.uv_layers.new(name="JapanColorUV")
            indices=np.empty(len(mesh.loops),dtype=np.int32)
            mesh.loops.foreach_get("vertex_index",indices)
            uv_layer.data.foreach_set("uv",np.ascontiguousarray(
                point_uv[indices],dtype=np.float32).ravel())
            original=mesh.attributes.new(name="JapanOriginalAlbedoRGB",
                                         type="FLOAT_VECTOR",domain="POINT")
            original.data.foreach_set("vector",np.ascontiguousarray(
                values,dtype=np.float32).ravel())
            # A glTF baseColorTexture is *multiplied* by COLOR_0, so leaving
            # the former vertex-color channel exported darkens the model.
            mesh.color_attributes.remove(mesh.color_attributes["Albedo"])

        bsdf=next(n for n in mat.node_tree.nodes if n.type=="BSDF_PRINCIPLED")
        tex=mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image=image
        tex.interpolation="Linear"
        tex.extension="EXTEND"
        uvmap=mat.node_tree.nodes.new("ShaderNodeUVMap")
        uvmap.uv_map="JapanColorUV"
        mat.node_tree.links.new(uvmap.outputs["UV"],tex.inputs["Vector"])
        mat.node_tree.links.new(tex.outputs["Color"],bsdf.inputs["Base Color"])
        for node in list(mat.node_tree.nodes):
            if node.type=="VERTEX_COLOR":
                mat.node_tree.nodes.remove(node)
        report[key]={"vertices":len(colors),"texture_px":size,
                     "rms_linear_rgb_error":round(rms_error,5),
                     "max_linear_rgb_error":round(max_error,5)}
    return report


def _rounded_rectangle(t: float, width: float, depth: float, radius: float) -> tuple[float, float]:
    """Constant-speed-ish closed rounded rectangle in four circular corners."""
    quarter = int(t*4) % 4
    local = (t*4 % 1)*(math.pi/2)
    corners = ((width/2-radius, depth/2-radius),
               (-width/2+radius, depth/2-radius),
               (-width/2+radius, -depth/2+radius),
               (width/2-radius, -depth/2+radius))
    angles = (0, math.pi/2, math.pi, 3*math.pi/2)
    # Starting at east of northeast corner, traverse counterclockwise.
    cx, cy = corners[quarter]
    return (cx+radius*math.cos(angles[quarter]+local),
            cy+radius*math.sin(angles[quarter]+local))


def _plinth(bpy: Any, model: Any, mats: dict) -> list:
    """Layered round-corner display plaque; actual 10.2 by 8.6 footprint."""
    n = 160
    levels = ((10.2, 8.6, 1.05, .06), (10.2, 8.6, 1.05, .34),
              (10.02, 8.42, 1.05, .46), (9.75, 8.17, 1.06, .54))
    verts = [(*_rounded_rectangle(j/n, w, d, rad), z)
             for w, d, rad, z in levels for j in range(n)]
    faces = [(r*n+j, r*n+(j+1)%n, (r+1)*n+(j+1)%n, (r+1)*n+j)
             for r in range(len(levels)-1) for j in range(n)]
    faces += [tuple(reversed(tuple(range(n)))), tuple((len(levels)-1)*n+j for j in range(n))]
    plaque = _mesh(bpy, model, "Pedestal / beveled smoked basalt display plaque",
                   verts, faces, mats["obsidian"])
    # Multiple inset contour rings support a dappled, vertex-painted ground.
    # The upper plane is flat; only its albedo is heterogeneous.
    def spring_color(x: float, y: float) -> tuple[float,float,float]:
        moss=.45+.19*math.sin(x*1.8+y*2.5)*math.sin(y*2.7-x*1.9)
        moss+=.13*math.sin(x*5.7+y*4.2)
        base=_blend((.145,.267,.142),(.30,.328,.16),moss)
        leaf_litter=.18+.16*math.sin(x*2.3+y*.5)
        base=_blend(base,(.34,.242,.14),max(0,leaf_litter)*.50)
        path=math.exp(-((x-1.76-.08*math.sin(y*1.5))/.43)**2)
        path*=_ease(-4.0,-3.15,y)*(1-_ease(-1.7,-.75,y))
        return _blend(base,(.39,.322,.226),path*.76)
    bands=8
    ground_verts=[(0,0,.565)]
    ground_colors=[spring_color(0,0)]
    for k in range(1,bands+1):
        fraction=k/bands
        for j in range(n):
            x,y=_rounded_rectangle(j/n,9.62*fraction,8.02*fraction,1.0*fraction)
            ground_verts.append((x,y,.565))
            ground_colors.append(spring_color(x,y))
    ground_faces=[(0,j+1,(j+1)%n+1) for j in range(n)]
    for k in range(bands-1):
        inner=1+k*n;outer=inner+n
        for j in range(n):
            neighbor=(j+1)%n
            ground_faces.append((inner+j,outer+j,outer+neighbor,inner+neighbor))
    ground = _mesh(bpy, model, "Ground / dappled moss, ochre earth and pilgrim approach",
                   ground_verts,ground_faces,mats["ground_vertex"],colors=ground_colors)
    return [plaque, ground]


def _fuji_profile(u: float) -> float:
    """Compressed conical upper slopes with a smooth, calm lower shoulder."""
    if u >= 1:
        return .57
    return .57 + 4.10*(1-u)**1.22


def _fuji_radius(a: float) -> float:
    return 3.53*(1+.028*math.sin(a*3+.2)+.021*math.sin(a*7-.8))


def _fuji_height(u: float, a: float) -> float:
    # Subtle sculpted gullies, not a mathematically perfect rubber cone.
    ridges = (.055*math.sin(13*a+.7) + .03*math.sin(29*a+u*5.2))
    ridge = ridges * max(0, u-.08) * (1-min(1.,u))**.28
    return _fuji_profile(u)+ridge


def _snow_limit(a: float) -> float:
    # Snow tongues reach down the valleys: distinguishable even as a thumbnail.
    return .335 + .034*math.sin(9*a+.6) + .028*math.sin(15*a-1.1) + .015*math.sin(25*a)


def _mountain(bpy: Any, model: Any, mats: dict, divisions: int) -> list:
    # A single watertight crater-like summit surface, no primitive cone.
    n = divisions
    radial = max(45, divisions//3)
    verts, faces, colors = [], [], []
    for k in range(radial+1):
        u = .068 + .932*k/radial
        for j in range(n):
            a = j*math.tau/n
            radius = _fuji_radius(a)*u
            h = _fuji_height(u, a)
            verts.append((radius*math.cos(a), .65+radius*.89*math.sin(a), h))
            value=math.sin(a*13+u*24)*math.sin(a*27-u*15)
            # Blue-slate volcanic cone and dark evergreen foot: unmistakable
            # iconic Fuji colors even in a small thumbnail or glTF viewer.
            base=_blend(ICONIC_BLUE_BASALT,ICONIC_WEATHERED_ROCK,_ease(.24,.64,u))
            tree_line=(.70+.053*math.sin(7*a+.6)+.03*math.sin(17*a-1.4)
                       +.024*math.sin(4*a+.7))
            base=_blend(base,ICONIC_TREE_LINE,_ease(tree_line,tree_line+.15,u))
            groove=max(0.,math.sin(23*a+2*math.sin(3*a)+u*2.8))**8
            t=.88+.20*value+.09*math.sin(7*a+u*11)-.20*groove
            colors.append(tuple(min(.93,max(.018,v*t)) for v in base))
    for k in range(radial):
        for j in range(n):
            nx = (j+1)%n
            faces.append((k*n+j, (k+1)*n+j, (k+1)*n+nx, k*n+nx))
    verts.append((0,.65,4.54))
    colors.append((.17,.19,.21))
    for j in range(n):
        faces.append((len(verts)-1,j,(j+1)%n))
    flank = _mesh(bpy, model, "Mountain / hand-shaped iconic Fuji flanks",
                  verts, faces, mats["flank_vertex"], colors=colors, smooth=True)
    # Physical cap has long jagged skirt, pale blue shadowing, snowy ridges.
    cap_verts, cap_faces, cap_colors = [], [], []
    snow_rows = max(26, divisions//7)
    for k in range(snow_rows+1):
        v = k/snow_rows
        for j in range(n):
            a = j*math.tau/n
            u = .068 + v*(_snow_limit(a)-.068)
            radius = _fuji_radius(a)*u
            z = _fuji_height(u,a)+.015
            cap_verts.append((radius*math.cos(a), .65+radius*.89*math.sin(a), z))
            shade=.97-.11*v+.047*math.sin(7*a+v*3)+.018*math.sin(23*a)
            ice=_blend((.83,.90,.96),(.96,.96,.925),.55+.25*math.sin(a*4.2+v*6))
            cap_colors.append(tuple(min(1.,max(.4,c*shade)) for c in ice))
    for k in range(snow_rows):
        for j in range(n):
            nx=(j+1)%n
            cap_faces.append((k*n+j,(k+1)*n+j,(k+1)*n+nx,k*n+nx))
    cap = _mesh(bpy, model, "Snow / sculpted five-point alpine snow crown",
                cap_verts,cap_faces,mats["snow_vertex"],colors=cap_colors,smooth=True)
    # Tiny summit caldera: a recessed dark bowl within the snow-silver crest.
    rim_verts, rim_faces = [], []
    for rad,z in ((.255,4.355),(.21,4.415),(.105,4.305),(.0,4.28)):
        for j in range(n):
            a=j*math.tau/n
            z_warp = .010*math.sin(7*a+.3) + .008*math.sin(13*a)
            rim_verts.append((rad*math.cos(a),.65+rad*math.sin(a),z+z_warp*(rad/.255)))
    for k in range(3):
        for j in range(n):
            rim_faces.append((k*n+j,(k+1)*n+j,(k+1)*n+(j+1)%n,k*n+(j+1)%n))
    rim=_mesh(bpy,model,"Summit / recessed volcanic rim and dark caldera",
              rim_verts,rim_faces,mats["crater"],smooth=True)
    return [flank,cap,rim]


def _box(bpy: Any, model: Any, name: str, center: tuple, size: tuple, mat: Any) -> Any:
    x,y,z=center; sx,sy,sz=size
    v=[(x+dx*sx/2,y+dy*sy/2,z+dz*sz/2)
       for dz in (-1,1) for dy in (-1,1) for dx in (-1,1)]
    f=[(0,1,3,2),(4,6,7,5),(0,4,5,1),
       (2,3,7,6),(0,2,6,4),(1,5,7,3)]
    obj=_mesh(bpy,model,name,v,f,mat)
    if name.startswith("Torii / ") and "insignia" not in name:
        bevel=obj.modifiers.new("Softly rounded lacquer and timber edges", "BEVEL")
        bevel.width=.014
        bevel.segments=2
    return obj


def _tube(bpy: Any, collection: Any, name: str, points: list[tuple],
          radii: list[float], material: Any, *, sides: int=8) -> Any:
    """Swept branch / post tube with a stable travel frame."""
    from mathutils import Vector
    verts=[]
    p=[Vector(v) for v in points]
    for k,point in enumerate(p):
        tangent = (p[min(k+1,len(p)-1)] - p[max(0,k-1)]).normalized()
        other=Vector((0,1,0)) if abs(tangent.y)<.8 else Vector((1,0,0))
        normal=tangent.cross(other).normalized()
        side=tangent.cross(normal).normalized()
        for j in range(sides):
            a=j*math.tau/sides
            v=point+radii[k]*(math.cos(a)*normal+math.sin(a)*side)
            verts.append(tuple(v))
    faces=[]
    for k in range(len(p)-1):
        for j in range(sides):
            faces.append((k*sides+j,k*sides+(j+1)%sides,
                          (k+1)*sides+(j+1)%sides,(k+1)*sides+j))
    faces.extend((tuple(reversed(range(sides))),tuple((len(p)-1)*sides+j for j in range(sides))))
    return _mesh(bpy,collection,name,verts,faces,material,smooth=True)


def _torii(bpy: Any, model: Any, mats: dict) -> list:
    """Two splayed hashira, penetrated nuki, curved kasagi and shimaki lintels."""
    objects=[]
    x,y=1.75,-3.02
    for side in (-1,1):
        X=x+side*.77
        objects.append(_tube(bpy,model,"Torii / vermilion hashira pillar",
                    [(X,y,.68),(X+side*.035,y,1.25),(X+side*.015,y,2.32)],
                    [.119,.109,.107],mats["vermilion"],sides=12))
        objects.append(_box(bpy,model,"Torii / blackened stone post shoe",
                            (X,y,.67),(.33,.32,.17),mats["charcoal"]))
        objects.append(_box(bpy,model,"Torii / gilt collar at post foot",
                            (X,y,.79),(.24,.245,.034),mats["gold"]))
    objects.append(_box(bpy,model,"Torii / horizontal nuki crossbeam",
                        (x,y,1.76),(2.12,.17,.138),mats["vermilion"]))
    objects.append(_box(bpy,model,"Torii / upper shimaki crossbar",
                        (x,y,2.34),(2.31,.26,.167),mats["vermilion"]))
    # The black polished kasagi overhangs and curls gently at both tips.
    verts=[]; faces=[]; seg=28
    for j in range(seg+1):
        dx=-1.31+2.62*j/seg
        bend=.082*(abs(dx)/1.31)**3
        verts.extend([(x+dx,y-.157,2.425+bend),
                      (x+dx,y+.157,2.425+bend),
                      (x+dx,y-.157,2.563+bend),
                      (x+dx,y+.157,2.563+bend)])
        if j:
            p=(j-1)*4;q=j*4
            faces.extend([(p,p+1,q+1,q),(p+2,q+2,q+3,p+3),
                          (p,q,q+2,p+2),(p+1,p+3,q+3,q+1)])
    faces.extend([(0,2,3,1),tuple((seg*4+i) for i in (0,1,3,2))])
    objects.append(_mesh(bpy,model,"Torii / sweeping black kasagi crown",
                         verts,faces,mats["charcoal"]))
    # Central hanging shrine plaque, dark panel with warm brass border.
    objects.append(_box(bpy,model,"Torii / engraved brass plaque surround",
                        (x,y-.142,2.04),(.30,.055,.33),mats["gold"]))
    objects.append(_box(bpy,model,"Torii / midnight lacquer inscription panel",
                        (x,y-.176,2.04),(.237,.041,.264),mats["charcoal"]))
    for z in (1.97,2.04,2.11):
        objects.append(_box(bpy,model,"Torii / abstract brass shrine insignia",
                            (x,y-.204,z),(.118,.015,.014),mats["gold"]))
    return objects


def _sakura(bpy: Any, model: Any, mats: dict, *, origin: tuple[float,float],
            scale: float, seed: int, blossoms: int) -> list:
    """Branched timber with flowering cluster volumes and actual five-petal blooms."""
    rng=random.Random(seed)
    ox,oy=origin
    objects=[]
    def point(x,y,z):
        return (ox+x*scale,oy+y*scale,.56+z*scale)
    root=[point(0,0,0),point(.07,.015,.38),point(.15,.055,.82),
          point(.14,.10,1.21),point(.09,.13,1.58)]
    objects.append(_tube(bpy,model,"Sakura / curved mature bark trunk",root,
                         [r*scale for r in (.175,.159,.125,.083,.045)],mats["bark"],sides=12))
    # Fissures follow the leaning trunk instead of a flat wood-grain texture.
    for stripe in range(7):
        angle=stripe*math.tau/7+.18
        positions=[]
        for k,location in enumerate(root[1:4]):
            rad=(.161,.127,.087)[k]*scale
            positions.append((location[0]+rad*math.cos(angle+k*.04),
                              location[1]+rad*math.sin(angle+k*.04),location[2]))
        objects.append(_tube(bpy,model,"Sakura / fine aged-bark vertical fissure",
                            positions,[.006*scale,.007*scale,.003*scale],
                            mats["bark_dark"],sides=5))
    tips=[]
    angles=(.06,.69,1.27,1.95,2.67,3.35,4.07,4.77,5.56)
    for i,base_angle in enumerate(angles):
        a=base_angle+rng.uniform(-.19,.19)
        reach=rng.uniform(.90,1.48)
        h=rng.uniform(1.78,2.48)
        start=root[2] if i%3==0 else (root[3] if i%3==1 else root[4])
        via=point(.14+.29*math.cos(a-.14),.10+.27*math.sin(a-.14),1.47)
        elbow=point(.12+.72*reach*math.cos(a+.05),
                    .13+.68*reach*math.sin(a+.05),h-.08)
        end=point(.09+reach*math.cos(a),.13+reach*math.sin(a),h)
        objects.append(_tube(bpy,model,"Sakura / tapering sculptural branch",
                             [start,via,elbow,end],
                             [r*scale for r in (.078,.060,.036,.018)],
                             mats["bark"],sides=8))
        tips.append(end)
        # Secondary limbs sag and then turn upward into flowering tips.
        for side in (-1,1):
            angle=a+side*rng.uniform(.23,.58)
            length=reach*rng.uniform(.79,1.14)
            terminal=point(.09+length*math.cos(angle),
                           .13+length*math.sin(angle),
                           h+rng.uniform(-.22,.23))
            bend=point(.10+(.88*length)*math.cos(angle),
                       .13+(.88*length)*math.sin(angle),
                       h-.17+rng.uniform(-.12,.09))
            objects.append(_tube(bpy,model,"Sakura / graceful secondary flowering limb",
                                 [elbow,bend,terminal],
                                 [.035*scale,.019*scale,.007*scale],mats["bark"],sides=6))
            tips.append(terminal)
            if rng.random()<.72:
                tips.append(point(.09+length*math.cos(angle+side*.16),
                                  .13+length*math.sin(angle+side*.16),
                                  h+rng.uniform(-.25,.17)))
    # Broad umbrella growth resolves into many small, ragged blossom sprays.
    blobs=[]
    for tip in tips:
        if rng.random()<.075: continue
        radius=rng.uniform(.17,.28)*scale
        center=(tip[0]+rng.uniform(-.075,.075)*scale,
                tip[1]+rng.uniform(-.075,.075)*scale,
                tip[2]+rng.uniform(-.085,.09)*scale)
        blobs.append((center,radius))
        if rng.random()<.38:
            secondary=(center[0]+rng.uniform(-.24,.24)*scale,
                       center[1]+rng.uniform(-.20,.20)*scale,
                       center[2]+rng.uniform(-.10,.10)*scale)
            blobs.append((secondary,radius*rng.uniform(.61,.83)))
    if scale<.9:
        # Smaller companion tree should have finer secondary flowering twigs,
        # not a visibly hollow crown when viewed opposite the main hero.
        for _ in range(16):
            center,rad=blobs[rng.randrange(len(blobs))]
            a=rng.uniform(0,math.tau)
            shift=rng.uniform(.22,.73)*rad
            blobs.append(((center[0]+shift*math.cos(a),
                           center[1]+shift*math.sin(a),
                           center[2]+rng.uniform(-.16,.16)*rad),
                          rad*rng.uniform(.66,.89)))
    cloud_v=[];cloud_f=[];cloud_colors=[]
    palette=SAKURA_PETAL_PALETTE
    for index,(center,rad) in enumerate(blobs):
        base=len(cloud_v)
        lat_count=8;lon_count=13
        tone=_blend(palette[index%len(palette)],(.63,.30,.39),.35)
        for k in range(lat_count+1):
            theta=math.pi*k/lat_count
            for j in range(lon_count):
                phi=j*math.tau/lon_count
                grain=(1+.18*math.sin(5*phi+index*.71)
                       +.11*math.cos(4*theta+3*phi+index*.31))
                # The solid core is small: hundreds of actual flower rosettes
                # form the visible outer contour, not opaque cotton-like blobs.
                rx=rad*.38*grain*math.sin(theta)*math.cos(phi)
                ry=rad*.33*grain*math.sin(theta)*math.sin(phi)
                rz=rad*.25*grain*math.cos(theta)
                cloud_v.append((center[0]+rx,center[1]+ry,center[2]+rz))
                m=.91+.08*math.sin(phi*4+theta*5)
                cloud_colors.append(tuple(min(1,c*m) for c in tone))
        for k in range(lat_count):
            for j in range(lon_count):
                next_j=(j+1)%lon_count
                cloud_f.append((base+k*lon_count+j,base+k*lon_count+next_j,
                                base+(k+1)*lon_count+next_j,base+(k+1)*lon_count+j))
    objects.append(_mesh(bpy,model,"Sakura / small concealed rosebud cores",
                         cloud_v,cloud_f,mats["blossom_vertex"],colors=cloud_colors,smooth=True))
    # Five actual lobed petals per blossom. Combined by color into only one mesh.
    flower_v=[];flower_f=[];flower_colors=[]
    for k in range(blossoms):
        center,rad=blobs[rng.randrange(len(blobs))]
        a=rng.uniform(0,math.tau)
        h=rng.uniform(-.77,.96)
        spread=math.sqrt(1-h*h)
        nx,ny,nz=spread*math.cos(a),spread*math.sin(a),h
        spread_radius=.26+.95*(rng.random()**.49)
        pos=(center[0]+rad*1.16*spread_radius*nx,
             center[1]+rad*1.02*spread_radius*ny,
             center[2]+rad*.78*spread_radius*nz)
        # Tangent frame for tiny outward-facing flower with five scalloped petals.
        tx,ty,tz=(-ny,nx,0)
        norm=math.hypot(tx,ty)
        tx,ty=tx/max(.001,norm),ty/max(.001,norm)
        ux,uy,uz=(ny*tz-nz*ty,nz*tx-nx*tz,nx*ty-ny*tx)
        rad_p=rng.uniform(.025,.045)*scale
        color=palette[rng.randrange(len(palette))]
        base=len(flower_v)
        flower_v.append(pos)
        flower_colors.append((.96,.70,.64))
        for j in range(5):
            theta=j*math.tau/5
            for offset,weight in ((-.46,.87),(-.19,1.13),(.19,1.13),(.46,.87)):
                angle=theta+offset
                outward=rad_p*weight
                flower_v.append(tuple(pos[i]+outward*(math.cos(angle)*(tx,ty,tz)[i]
                                                  + math.sin(angle)*(ux,uy,uz)[i])
                                      + .006*scale*(nx,ny,nz)[i] for i in range(3)))
                flower_colors.append(color)
            j0=base+1+j*4
            flower_f.append((base,j0,j0+1))
            flower_f.append((base,j0+1,j0+2))
            flower_f.append((base,j0+2,j0+3))
            flower_f.append((base,j0+3,base+1+((j+1)%5)*4))
    objects.append(_mesh(bpy,model,"Sakura / hundreds of individually modelled five-petal flowers",
                         flower_v,flower_f,mats["blossom_vertex"],colors=flower_colors))
    # A few young leaf-buds behind blossom clusters bring spring tree color
    # and the characteristic serrated, lanceolate leaf silhouette.
    leaf_v=[];leaf_f=[];leaf_colors=[]
    for k in range(105):
        center,rad=blobs[rng.randrange(len(blobs))]
        a=rng.uniform(0,math.tau)
        z=rng.uniform(-.38,.38)
        x=center[0]+rad*1.03*math.cos(a)
        y=center[1]+rad*.80*math.sin(a)
        z=center[2]+rad*.60*z
        leaf_len=rng.uniform(.045,.11)*scale
        direction=(math.cos(a),math.sin(a),rng.uniform(-.4,.5))
        side=(-direction[1],direction[0],0)
        base=len(leaf_v)
        leaf_v.extend([(x,y,z),
                       (x+side[0]*leaf_len*.29,y+side[1]*leaf_len*.29,z+.016*scale),
                       (x+direction[0]*leaf_len,y+direction[1]*leaf_len,
                        z+direction[2]*leaf_len),
                       (x-side[0]*leaf_len*.29,y-side[1]*leaf_len*.29,z+.016*scale)])
        hue=.85+.18*rng.random()
        leaf_colors.extend([(.11*hue,.28*hue,.09*hue)]*4)
        leaf_f.extend([(base,base+1,base+2),(base,base+2,base+3)])
    objects.append(_mesh(bpy,model,"Sakura / scattered olive spring leaf buds",
                         leaf_v,leaf_f,mats["ground_vertex"],colors=leaf_colors))
    return objects


def _path_and_rocks(bpy: Any, model: Any, mats: dict) -> list:
    rng=random.Random(236)
    objects=[]
    # Individual polygonal flagstones, each with softened outline, irregular
    # joint and height. Unlike identical boxes these read as hand-set slate.
    for i in range(8):
        y=-3.92+i*.17
        x=1.72+.07*math.sin(i*.85)
        rx=.23-.065*i/8
        ry=.068+rng.uniform(-.006,.012)
        stone_v=[];stone_f=[];count=10
        for z in (.575+rng.uniform(0,.012),.623+rng.uniform(0,.012)):
            for j in range(count):
                a=j*math.tau/count
                jitter=1+rng.uniform(-.10,.10)
                stone_v.append((x+rx*math.cos(a)*jitter,
                                y+ry*math.sin(a)*jitter,z))
        stone_f.append(tuple(range(count,2*count)))
        for j in range(count):
            stone_f.append((j,(j+1)%count,count+(j+1)%count,count+j))
        objects.append(_mesh(bpy,model,"Garden / individually shaped shrine flagstone",
                             stone_v,stone_f,mats["stone"]))
    # Scatter deliberate small basalt accents beyond the path.
    for i in range(23):
        a=rng.uniform(0,math.tau)
        r=rng.uniform(3.05,4.2)
        x=r*math.cos(a);y=r*.82*math.sin(a)
        if abs(x-1.75)<1.1 and y< -2.5: continue
        s=rng.uniform(.07,.18)
        objects.append(_box(bpy,model,"Garden / weathered volcanic stone",
                            (x,y,.591),(s*1.7,s*1.2,s*.45),mats["stone"]))
    # Fallen petals are individually modelled as two triangular planes and
    # scattered around the two flowering trees, not baked into a photograph.
    pinks=((.95,.63,.71),(.97,.79,.81),(.85,.42,.53),(.99,.86,.84))
    v=[];f=[];colors=[]
    for i in range(280):
        ox,oy=((-3.1,-1.45) if i%2==0 else (3.48,.35))
        a=rng.uniform(0,math.tau)
        r=rng.uniform(.12,1.28)
        x,y=ox+r*math.cos(a),oy+.72*r*math.sin(a)
        if abs(x)>4.4 or abs(y)>3.7:continue
        length=rng.uniform(.022,.066)
        nx,ny=math.cos(a),math.sin(a)
        b=len(v)
        v.extend([(x,y,.576),(x+length*nx,y+length*ny,.577),
                  (x+.32*length*ny,y-.32*length*nx,.578),
                  (x-.32*length*ny,y+.32*length*nx,.578)])
        colors.extend([pinks[rng.randrange(len(pinks))]]*4)
        f.extend([(b,b+2,b+1),(b,b+1,b+3)])
    objects.append(_mesh(bpy,model,"Garden / fallen blush and ivory sakura petals",
                         v,f,mats["blossom_vertex"],colors=colors))
    # Small grass tufts are shaded tapered geometry, balanced away from the
    # shrine path; their close-range profile breaks the tabletop flatness.
    stems=[];stemfaces=[];stemcolors=[]
    for i in range(170):
        a=rng.uniform(0,math.tau)
        r=rng.uniform(3.4,4.35)
        x,y=r*math.cos(a),r*.83*math.sin(a)
        if abs(x)>4.5 or abs(y)>3.7 or (abs(x-1.75)<1 and y< -2.7):continue
        s=rng.uniform(.055,.16)
        for j in range(3):
            direction=a+j*math.tau/3
            b=len(stems)
            stems.extend([(x,y,.578),(x+.14*s*math.cos(direction),y+.14*s*math.sin(direction),.58),
                          (x+.9*s*math.cos(direction),y+.9*s*math.sin(direction),.58+s*.85)])
            stemfaces.append((b,b+1,b+2))
            hue=rng.uniform(.8,1.15)
            stemcolors.extend([(.15*hue,.28*hue,.09*hue)]*3)
    objects.append(_mesh(bpy,model,"Garden / low alpine-grass tufts and dry sedges",
                         stems,stemfaces,mats["ground_vertex"],colors=stemcolors))
    return objects


def build_iconic_fuji_scene(destination: Path, *, detail: str = "standard",
                           bpy_module: Any = None, preview_resolution: int = 900) -> dict[str, Any]:
    """Export a compact, self-contained miniature that communicates 'Japan'."""
    if bpy_module is None:
        raise RuntimeError("The iconic Fuji builder requires Blender bpy")
    bpy=bpy_module
    if tuple(bpy.app.version)<(4,2,0):
        raise RuntimeError("Iconic Fuji export needs Blender 4.2+")
    destination=Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Output revision already exists: {destination}")
    spec=iconic_spec(detail)
    scene=create_model_scene(bpy,spec["asset_id"])
    scene.render.resolution_x=preview_resolution
    scene.render.resolution_y=round(preview_resolution*.78)
    scene.render.resolution_percentage=100
    model=bpy.data.collections.new("Japan / Fuji, shrine, cherry blossom miniature")
    studio=bpy.data.collections.new("Japan / studio lights and cameras")
    scene.collection.children.link(model)
    scene.collection.children.link(studio)
    camera=_studio(bpy,scene,studio)
    # Softer warm spring daylight rather than the survey model's cold winter.
    sun=next(o for o in studio.objects if o.type=="LIGHT")
    sun.data.energy=3.0
    sun.rotation_euler=(math.radians(28),math.radians(-28),math.radians(-25))
    bg=next(n for n in scene.world.node_tree.nodes if n.type=="BACKGROUND")
    bg.inputs["Color"].default_value=(.37,.52,.68,1)
    bg.inputs["Strength"].default_value=.70
    mats={
        "obsidian":_material(bpy,"Japan / dark basalt and lacquer",(.055,.067,.077),roughness=.55,metallic=.16),
        "ground_vertex":_vertex_material(bpy,"Japan / moss, volcanic scree and spring garden"),
        "flank_vertex":_vertex_material(bpy,"Japan / blue-grey volcanic Fuji ridges"),
        "snow_vertex":_vertex_material(bpy,"Japan / alpine snow ivory and blue"),
        "crater":_material(bpy,"Japan / hollow summit scree",(.19,.23,.26)),
        "vermilion":_material(bpy,"Japan / warm vermilion painted timber",(.83,.115,.062),roughness=.40),
        "charcoal":_material(bpy,"Japan / black polished kasagi",(.055,.045,.05),roughness=.32),
        "gold":_material(bpy,"Japan / burnished brass details",(.55,.35,.11),roughness=.35,metallic=.68),
        "bark":_material(bpy,"Japan / aged cherry-bark umber",(.21,.116,.082),roughness=.94),
        "bark_dark":_material(bpy,"Japan / old cherry trunk bark fissures",(.095,.056,.051),roughness=.97),
        "blossom_vertex":_vertex_material(bpy,"Japan / multi-hue sakura petals"),
        "stone":_material(bpy,"Japan / moss-flecked granite",(.33,.33,.31),roughness=.95),
    }
    objects=[]
    objects.extend(_plinth(bpy,model,mats))
    objects.extend(_mountain(bpy,model,mats,spec["terrain_divisions"]))
    objects.extend(_torii(bpy,model,mats))
    flower_count=spec["sakura_blossom_target"]
    objects.extend(_sakura(bpy,model,mats,origin=(-3.1,-1.45),scale=1.05,seed=41,
                           blossoms=int(flower_count*.56)))
    objects.extend(_sakura(bpy,model,mats,origin=(3.48,.35),scale=.79,seed=89,
                           blossoms=flower_count-int(flower_count*.56)))
    objects.extend(_path_and_rocks(bpy,model,mats))
    palette_report=_bake_portable_palette_textures(bpy,objects,mats)
    manifest=export_revision(bpy,scene,camera,objects,plan_iconic_shots(),destination,
                           samples=12,metadata={
                               **spec,"generator":"hand-sculpted procedural Japan miniature",
                               "portable_palette_albedo":palette_report,
                               "named_parts":{"mountain":1,"snow_crown":1,"summit_crater":1,
                                              "torii_gate":1,"flowering_sakura":2,"display_plinth":1},
                               "sakura_count":2,
                               "geographic_accuracy":False,
                               "editorial_intent":"Read as Fuji and Japan at thumbnail scale; not to survey scale",
                           })
    audit_glb(destination / "model.glb",require_base_color_texture=True,
              required_albedo_materials=tuple(mats[k].name for k in (
                  "ground_vertex","flank_vertex","snow_vertex","blossom_vertex")),
              forbid_color_texture_multiplication=True)
    return manifest
