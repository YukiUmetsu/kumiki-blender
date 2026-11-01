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
from ...shots import QualityShot
from .blender_scene import _studio


def plan_iconic_shots() -> tuple[QualityShot, ...]:
    """Hero plus scrutiny of the emblematic geometry and small-footprint layout."""
    return (
        QualityShot("hero", (9.8, -17.4, 9), (0, 0, 2.0), 11.9),
        QualityShot("front", (0, -18, 6.3), (0, 0, 2.05), 11.5),
        QualityShot("opposite", (-13, -11, 7.5), (0, .2, 1.95), 12.1),
        QualityShot("torii-and-sakura", (5.8, -8.8, 4.7), (1.55, -2.25, 1.55), 5.0),
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
        "sakura_blossom_target": {"preview": 260, "standard": 670, "high": 1250}[detail],
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
    # Top is rough mossy spring ground; thin inset leaves a dark lacquer rim.
    ground_verts = [(0, 0, .565)]
    ground_verts.extend((*_rounded_rectangle(j/n, 9.62, 8.02, 1.0), .565)
                        for j in range(n))
    ground = _mesh(bpy, model, "Ground / Japanese spring-green diorama floor",
                   ground_verts, [(0,j+1, (j+1)%n+1) for j in range(n)], mats["ground"])
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
            value = math.sin(a*13+u*24)*math.sin(a*27-u*15)
            if u > .75:
                base = (.205,.266,.232)
            else:
                base = (.215,.243,.282)
            t = .8 + .13*value + .07*math.sin(a*7+u*11)
            colors.append(tuple(min(.9,max(0.,v*t)) for v in base))
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
            shade = .96-.14*v + .065*math.sin(7*a+v*3) + .025*math.sin(23*a)
            cap_colors.append((min(1.,shade),min(1.,shade*.979),min(1.,shade*1.01)))
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
    return _mesh(bpy,model,name,v,f,mat)


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
    root=[point(0,0,0),point(.065,.02,.43),point(.13,.055,.96),point(.09,.06,1.42)]
    objects.append(_tube(bpy,model,"Sakura / curved mature bark trunk",root,
                         [r*scale for r in (.16,.142,.109,.067)],mats["bark"],sides=10))
    tips=[]
    for i in range(9):
        a=i*math.tau/9 + rng.uniform(-.15,.15)
        reach=rng.uniform(.68,1.35)
        h=rng.uniform(1.76,2.28)
        start=root[2] if i<6 else root[3]
        via=point(.13+.31*math.cos(a), .06+.27*math.sin(a),1.52)
        end=point(.12+reach*math.cos(a), .05+reach*math.sin(a),h)
        objects.append(_tube(bpy,model,"Sakura / tapering sculptural branch",
                             [start,via,end],[.067*scale,.044*scale,.018*scale],
                             mats["bark"],sides=7))
        tips.append(end)
        for side in (-1,1):
            x=.12+(reach*.82)*math.cos(a+side*.27)
            y=.05+(reach*.82)*math.sin(a+side*.27)
            tips.append(point(x,y,h+rng.uniform(-.15,.24)))
    # Broad groups are deliberately fractured and varied, not a pink UV sphere.
    blobs=[]
    for tip in tips:
        if rng.random()<.20: continue
        radius=rng.uniform(.23,.40)*scale
        center=(tip[0],tip[1],tip[2]+rng.uniform(-.06,.12)*scale)
        blobs.append((center,radius))
    cloud_v=[];cloud_f=[];cloud_colors=[]
    palette=((.99,.52,.66),(.98,.69,.78),(.99,.81,.86),(.93,.44,.59),(.96,.72,.79))
    for index,(center,rad) in enumerate(blobs):
        base=len(cloud_v)
        lat_count=7;lon_count=12
        tone=palette[index%len(palette)]
        for k in range(lat_count+1):
            theta=math.pi*k/lat_count
            for j in range(lon_count):
                phi=j*math.tau/lon_count
                grain=1+.13*math.sin(5*phi+index)+.09*math.cos(4*theta+3*phi)
                rx=rad*1.2*grain*math.sin(theta)*math.cos(phi)
                ry=rad*.92*grain*math.sin(theta)*math.sin(phi)
                rz=rad*.66*grain*math.cos(theta)
                cloud_v.append((center[0]+rx,center[1]+ry,center[2]+rz))
                m=.91+.08*math.sin(phi*4+theta*5)
                cloud_colors.append(tuple(min(1,c*m) for c in tone))
        for k in range(lat_count):
            for j in range(lon_count):
                next_j=(j+1)%lon_count
                cloud_f.append((base+k*lon_count+j,base+k*lon_count+next_j,
                                base+(k+1)*lon_count+next_j,base+(k+1)*lon_count+j))
    objects.append(_mesh(bpy,model,"Sakura / overlapping rose and blush blossom clouds",
                         cloud_v,cloud_f,mats["blossom_vertex"],colors=cloud_colors,smooth=True))
    # Five actual lobed petals per blossom. Combined by color into only one mesh.
    flower_v=[];flower_f=[];flower_colors=[]
    for k in range(blossoms):
        center,rad=blobs[rng.randrange(len(blobs))]
        a=rng.uniform(0,math.tau)
        h=rng.uniform(-.77,.96)
        spread=math.sqrt(1-h*h)
        nx,ny,nz=spread*math.cos(a),spread*math.sin(a),h
        pos=(center[0]+rad*1.17*nx,center[1]+rad*.94*ny,center[2]+rad*.72*nz)
        # Tangent frame for tiny outward-facing flower with five scalloped petals.
        tx,ty,tz=(-ny,nx,0)
        norm=math.hypot(tx,ty)
        tx,ty=tx/max(.001,norm),ty/max(.001,norm)
        ux,uy,uz=(ny*tz-nz*ty,nz*tx-nx*tz,nx*ty-ny*tx)
        rad_p=rng.uniform(.029,.051)*scale
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
    return objects


def _path_and_rocks(bpy: Any, model: Any, mats: dict) -> list:
    rng=random.Random(236)
    objects=[]
    # A disciplined, small-scale pilgrim stepping-stone route to the torii.
    for i in range(7):
        y=-3.88+i*.20
        x=1.72+.055*math.sin(i*.8)
        w=.48-.035*i
        objects.append(_box(bpy,model,"Garden / hand-set shrine approach stone",
                            (x,y,.596),(w,.155,.061),mats["stone"]))
    # Scatter deliberate small basalt accents beyond the path.
    for i in range(23):
        a=rng.uniform(0,math.tau)
        r=rng.uniform(3.05,4.2)
        x=r*math.cos(a);y=r*.82*math.sin(a)
        if abs(x-1.75)<1.1 and y< -2.5: continue
        s=rng.uniform(.07,.18)
        objects.append(_box(bpy,model,"Garden / weathered volcanic stone",
                            (x,y,.591),(s*1.7,s*1.2,s*.45),mats["stone"]))
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
    sun.data.energy=2.7
    sun.rotation_euler=(math.radians(28),math.radians(-28),math.radians(-25))
    mats={
        "obsidian":_material(bpy,"Japan / dark basalt and lacquer",(.055,.067,.077),roughness=.55,metallic=.16),
        "ground":_material(bpy,"Japan / warm moss and pale grass",(.22,.29,.20),roughness=.94),
        "flank_vertex":_vertex_material(bpy,"Japan / blue-grey volcanic Fuji ridges"),
        "snow_vertex":_vertex_material(bpy,"Japan / alpine snow ivory and blue"),
        "crater":_material(bpy,"Japan / hollow summit scree",(.19,.23,.26)),
        "vermilion":_material(bpy,"Japan / shuiro vermilion lacquer",(.73,.085,.052),roughness=.33),
        "charcoal":_material(bpy,"Japan / black polished kasagi",(.055,.045,.05),roughness=.32),
        "gold":_material(bpy,"Japan / burnished brass details",(.55,.35,.11),roughness=.35,metallic=.68),
        "bark":_material(bpy,"Japan / aged cherry-bark umber",(.205,.114,.119),roughness=.94),
        "blossom_vertex":_vertex_material(bpy,"Japan / multi-hue sakura petals"),
        "stone":_material(bpy,"Japan / moss-flecked granite",(.33,.33,.31),roughness=.95),
    }
    objects=[]
    objects.extend(_plinth(bpy,model,mats))
    objects.extend(_mountain(bpy,model,mats,spec["terrain_divisions"]))
    objects.extend(_torii(bpy,model,mats))
    flower_count=spec["sakura_blossom_target"]
    objects.extend(_sakura(bpy,model,mats,origin=(-3.1,-1.45),scale=1.05,seed=41,
                           blossoms=int(flower_count*.63)))
    objects.extend(_sakura(bpy,model,mats,origin=(3.48,.35),scale=.79,seed=89,
                           blossoms=flower_count-int(flower_count*.63)))
    objects.extend(_path_and_rocks(bpy,model,mats))
    return export_revision(bpy,scene,camera,objects,plan_iconic_shots(),destination,
                           samples=12,metadata={
                               **spec,"generator":"hand-sculpted procedural Japan miniature",
                               "named_parts":{"mountain":1,"snow_crown":1,"summit_crater":1,
                                              "torii_gate":1,"flowering_sakura":2,"display_plinth":1},
                               "sakura_count":2,
                               "geographic_accuracy":False,
                               "editorial_intent":"Read as Fuji and Japan at thumbnail scale; not to survey scale",
                           })
