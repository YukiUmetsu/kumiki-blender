"""Mount Fuji: survey-based, three-dimensional volcano terrain in Blender.

The actual flanks, ridge drainage, summit rim and Hōei flank crater are
preserved from the input DEM; there is no cone primitive or procedural
replacement of surveyed elevation.
"""

from __future__ import annotations

import math
from array import array
from pathlib import Path
from typing import Any

from ...blender_output import create_model_scene, export_revision
from ...shots import QualityShot
from .dem import FujiDem, SUMMIT, load_dem
from .patches import FujiDetailPatch, load_patch, validate_patches


ROCK_NORMAL_TILES = 6  # 1024² shading detail across 26 km, not 26-m rock pixels.


def plan_fuji_shots() -> tuple[QualityShot, ...]:
    """Opposing profiles plus the two named crater checkpoints."""
    return (
        QualityShot("hero", (11, -23, 8.0), (0, 0, 2.25), 12.5),
        QualityShot("north-profile", (-12, 32, 9), (0, 0, 2.05), 18.0),
        QualityShot("summit-crater", (0.85, -1.55, 5.9), (0, 0, 3.55), 2.0),
        QualityShot("hoei-context", (8.5, -10.5, 5.1), (0.8, -1.2, 2.85), 5.4),
        QualityShot("hoei-crater", (2.25, -3.25, 6.4), (1.61, -2.18, 2.55), 2.5),
        QualityShot("overhead", (0, 0, 36), (0, 0, 2), 27.0),
    )


def _mix(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return tuple(a[i]*(1-t)+b[i]*t for i in range(3))


def _smoothstep(start: float, end: float, value: float) -> float:
    t = max(0., min(1., (value-start)/(end-start)))
    return t*t*(3-2*t)


def _value_noise(x: float, y: float) -> float:
    """Smooth deterministic 2-D noise, avoiding visible trigonometric grids.

    A small procedural albedo texture, never an elevation perturbation. It is
    evaluated into vertices so GLBs need no external texture or shader nodes.
    """
    ix, iy = math.floor(x), math.floor(y)
    u, v = x-ix, y-iy
    u, v = u*u*(3-2*u), v*v*(3-2*v)

    def corner(a: int, b: int) -> float:
        n = (a*374761393+b*668265263+1442695041) & 0xFFFFFFFF
        n = ((n ^ (n >> 13))*1274126177) & 0xFFFFFFFF
        return (n ^ (n >> 16))/4294967295

    north=corner(ix,iy)*(1-u)+corner(ix+1,iy)*u
    south=corner(ix,iy+1)*(1-u)+corner(ix+1,iy+1)*u
    return north*(1-v)+south*v


def terrain_color(height_m: float, slope: float, x: float, y: float,
                  *, season: str = "winter") -> tuple[float, float, float, float]:
    """Photo-informed seasonal albedo; all elevation geometry remains GSI.

    Environment Ministry field photographs describe black scoria above the
    fifth station; Fujinomiya and Hōei images show *localized* red cinders,
    not an entire pastel brown cone. Lower woods are dense dark conifers.
    Summer models August/September when the summit is normally snow-free;
    winter retains the original late-winter/early-spring snow interpretation.
    Neither albedo variant claims satellite-accurate seasonal vegetation.
    """
    if season not in ("winter", "summer"):
        raise ValueError(f"Unsupported Fuji albedo season: {season}")
    clamp=lambda value: max(0.,min(1.,value))
    large=_value_noise(x*.28+7.4,y*.28-4.1)
    canopy=_value_noise(x*1.05-13.7,y*1.05+9.4)
    woodland_detail=_value_noise(x*4.4+2.3,y*4.4-1.7)
    woods=.46*large+.36*canopy+.18*woodland_detail

    # Dark cedar/cypress forest. Broad pockets of olive mountain scrub are
    # spatial, rather than one continuous sage-green elevation stripe.
    dense_conifers=(.008,.021,.013)
    sunlit_conifers=(.034,.077,.029)
    c=_mix(dense_conifers,sunlit_conifers,clamp(.14+woods*.98))
    # Sub-grid canopy shadows help the forest read as densely packed crowns,
    # not a continuous green lawn. These pigments never alter survey geometry.
    canopy_crowns=_value_noise(x*14.6+3.3,y*14.6-8.2)
    canopy_gaps=_value_noise(x*30.7-6.1,y*30.7+2.8)
    canopy_tone=.73+.47*canopy_crowns+.20*canopy_gaps
    c=tuple(channel*canopy_tone for channel in c)
    scrub=(.081,.113,.044)
    scrub_weight=.31*_smoothstep(1700,2070,height_m)*clamp((woods-.32)*2.1)
    c=_mix(c,scrub,scrub_weight)

    # The real tree line varies by exposure, local gullies and elevation.
    # 2,200–2,400m is a broad reference, not a perfectly circular contour.
    tree_line=2230+125*(large-.5)+90*(canopy-.5)+55*clamp(y/6)
    exposed=_smoothstep(tree_line-140,tree_line+155,height_m)
    rock_pattern=_value_noise(x*.90-3.1,y*.90+8.4)
    iron_pattern=_value_noise(x*3.8+21.2,y*3.8-7.6)
    charcoal=(.052,.043,.055)
    gray_scoria=(.113,.081,.099)
    iron_scoria=(.205,.064,.077)
    rock=_mix(charcoal,gray_scoria,.16+.59*rock_pattern)
    oxidation=clamp((iron_pattern-.43)*1.25)*(.25+.40*rock_pattern)
    rock=_mix(rock,iron_scoria,oxidation)
    # The summit is a composite of black/grey basalt and narrower oxidized
    # radial scoria seams; high-altitude photos do not show a uniform tint.
    bearing=math.atan2(y,x)
    radial_iron=max(0.,math.sin(bearing*24.0+large*3.2+rock_pattern*1.5))**8
    outcrop_breakup=_smoothstep(.27,.76,_value_noise(x*1.9+18.1,y*1.9-23.6))
    radial_weight=.28*radial_iron*outcrop_breakup*_smoothstep(2440,2940,height_m)
    rock=_mix(rock,iron_scoria,radial_weight)
    rock=_mix(rock,charcoal,.20*clamp(slope*1.5))
    c=_mix(c,rock,exposed)
    if season=="summer":
        # Late-summer deciduous/birch scrub is slightly more olive than the
        # evergreen conifer belt. Never paint the barren summit green.
        green_scrub=(.089,.129,.052)
        foliage=.16*(1-exposed)*_smoothstep(1800,2250,height_m)*(.3+.7*large)
        c=_mix(c,green_scrub,foliage)

    # Hōei 1707 pyroclastic deposits show rusty-grey grit distinct from the
    # darker upper summit. Modulate by the measured crater wall steepness.
    hoei_distance=math.hypot(x-1.61,y+2.18)
    if hoei_distance<1.8 and height_m>2020:
        influence=_smoothstep(1.8,.35,hoei_distance)
        warm_scree=(.214,.101,.103)
        c=_mix(c,warm_scree,exposed*influence*(.37+.32*clamp(slope*1.8)))
        # The interior is dark glassy cinder, separate from the lighter outer
        # lapilli fan. Keep the mask broad enough to respect irregular rim.
        inner_bowl=math.exp(-.5*(hoei_distance/.53)**4)
        c=_mix(c,(.066,.038,.052),.19*inner_bowl*exposed)
    # A restricted ochre/ash shoulder on Hōei's southeastern ejecta apron.
    # This is an independently drawn geological color interpretation, not a
    # claim to classify every deposit from a soil survey.
    if season=="summer" and height_m>2250:
        dx=(x-2.30)/.58
        dy=(y+1.84)/.48
        golden_shoulder=math.exp(-1.65*(dx*dx+dy*dy))
        golden_shoulder*=(_smoothstep(2250,2500,height_m)
                          *(1-_smoothstep(3290,3460,height_m)))
        c=_mix(c,(.274,.186,.093),.74*golden_shoulder)

    # Grain represents pigment variation only. It cannot create or erase any
    # measured erosion grooves, cliffs or other DEM geometry.
    grain=(.26*(_value_noise(x*7.3+5.2,y*7.3-8.2)-.5)
           +.14*(_value_noise(x*24.1,y*24.1)-.5))
    c=tuple(clamp(channel*(1+grain)) for channel in c)

    # A separate snow-free GLB must contain absolutely no painted snow,
    # including the inner summit crater and wind-sheltered gullies.
    if season=="summer":
        return (*c,1.)

    # Seasonal snow accumulates in irregular fingers and sheltered northern
    # gullies; steep crater cliffs retain exposed dark scoria.
    fingers=(math.sin(bearing*9+math.sin(bearing*3.1))*112
             +math.sin(bearing*17+y*.8)*57
             +math.sin(x*2.2+y*2.7)*24)
    sheltered=clamp(y/6)
    snowline=2885+fingers+min(.45,slope)*195-115*sheltered
    coverage=_smoothstep(snowline-55,snowline+415,height_m)
    coverage*=max(.31,1-slope*.57)
    volcanic_seam=max(0.,math.sin(bearing*31+math.sin(bearing*7)+x*.38))**7
    coverage*=1-.32*volcanic_seam
    snow=(.86,.898,.918)
    c=_mix(c,snow,coverage*.98)
    return (*c, 1.)


def _stone_normal_image(bpy: Any, size: int) -> Any:
    """Bake a unique, tileable synthetic weathering normal into a PNG image.

    This adds *shading only*. DEM vertex locations remain survey samples.
    Blender ships NumPy, so all image math can run vectorized in background
    mode with no network requests, photos, custom render plugins or UV seams.
    """
    import numpy as np  # Available in Blender's bundled Python runtime.

    if size < 128 or size > 2048 or size & (size-1):
        raise ValueError("Fuji surface normal texture requires a 128..2048 power of two")
    rng=np.random.default_rng(3776)
    height=np.zeros((size,size),dtype=np.float32)
    for cells,weight in ((12,.46),(32,.28),(96,.17),(256,.09)):
        # Wrapped lattice values ensure no seam when the texture is repeated.
        lattice=rng.random((cells,cells),dtype=np.float32)
        coord=np.arange(size,dtype=np.float32)*cells/size
        lower=np.floor(coord).astype(np.int32)
        upper=(lower+1)%cells
        transition=coord-lower
        transition=transition*transition*(3-2*transition)
        x0,y0=lower[None,:],lower[:,None]
        x1,y1=upper[None,:],upper[:,None]
        tx,ty=transition[None,:],transition[:,None]
        a=lattice[y0,x0]*(1-tx)+lattice[y0,x1]*tx
        b=lattice[y1,x0]*(1-tx)+lattice[y1,x1]*tx
        height+=weight*(a*(1-ty)+b*ty)
    dx=np.roll(height,-1,axis=1)-np.roll(height,1,axis=1)
    dy=np.roll(height,-1,axis=0)-np.roll(height,1,axis=0)
    nx,ny=-dx*5.5,-dy*5.5
    length=np.sqrt(nx*nx+ny*ny+1)
    rgba=np.empty((size,size,4),dtype=np.float32)
    rgba[:,:,0]=.5+.5*nx/length
    rgba[:,:,1]=.5+.5*ny/length
    rgba[:,:,2]=.5+.5/length
    rgba[:,:,3]=1
    image=bpy.data.images.new("Fuji / original weathered-rock surface normals",
                              width=size,height=size,alpha=True,float_buffer=False)
    image.colorspace_settings.name="Non-Color"
    image.pixels.foreach_set(rgba.ravel())
    image.update()
    image.pack()
    return image


def _terrain_object(bpy: Any, model: Any, dem: FujiDem, divisions: int,
                    patches: tuple[FujiDetailPatch, ...] = (),
                    *, season: str = "winter") -> Any:
    spec = dem.spec
    if (divisions < 32 or divisions > spec.samples-1 or
            (spec.samples-1) % divisions):
        raise ValueError("Terrain divisions must evenly divide DEM intervals")
    row_stride = (spec.samples-1)//divisions
    if patches and row_stride != 1:
        raise ValueError("Survey patches require the full 1024-division parent mesh")
    validate_patches(patches,dem,divisions)
    count = divisions+1
    heights = dem.elevations
    verts: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    for row in range(count):
        src_row = row*row_stride
        lat = spec.latitude_max - row/divisions*(spec.latitude_max-spec.latitude_min)
        for col in range(count):
            src_col = col*row_stride
            lon = spec.longitude_min + col/divisions*(spec.longitude_max-spec.longitude_min)
            x, y = spec.xy(lon, lat)
            height = heights[src_row*spec.samples+src_col]
            verts.append((x, y, height/1000))
            # Derivatives in real-world kilometres (metres/metres).
            left = heights[src_row*spec.samples+max(0,src_col-row_stride)]
            right = heights[src_row*spec.samples+min(spec.samples-1,src_col+row_stride)]
            up = heights[max(0,src_row-row_stride)*spec.samples+src_col]
            down = heights[min(spec.samples-1,src_row+row_stride)*spec.samples+src_col]
            grad_x = (right-left)/1000/max(1e-6, spec.width_km*(2 if 0<col<divisions else 1)/divisions)
            grad_y = (down-up)/1000/max(1e-6, spec.depth_km*(2 if 0<row<divisions else 1)/divisions)
            colors.append(terrain_color(height, math.hypot(grad_x,grad_y), x, y, season=season))

    # Refine only the craters; keep their fine boundary vertices shared with
    # adjacent coarse polygons. Explicit edge splits eliminate T-junctions,
    # holes, z-fighting and overlapping coarse terrain underneath the patches.
    edge_splits: dict[tuple[int,int], list[int]] = {}
    per_patch_indices: dict[tuple[str,int,int], int] = {}
    skipped_cells: set[tuple[int,int]] = set()
    for patch in patches:
        s=patch.spec
        for r in range(s.row_start,s.row_end):
            for c in range(s.col_start,s.col_end):
                skipped_cells.add((r,c))
        ids=[]
        dx=spec.width_km/(spec.samples-1)/s.factor
        dy=spec.depth_km/(spec.samples-1)/s.factor
        for r in range(s.rows):
            row=s.row_start+r/s.factor
            lat=spec.latitude_max-row/(spec.samples-1)*(spec.latitude_max-spec.latitude_min)
            for c in range(s.cols):
                col=s.col_start+c/s.factor
                lon=spec.longitude_min+col/(spec.samples-1)*(spec.longitude_max-spec.longitude_min)
                x,y=spec.xy(lon,lat)
                h=patch.elevations[r*s.cols+c]
                l=patch.elevations[r*s.cols+max(0,c-1)]
                ri=patch.elevations[r*s.cols+min(s.cols-1,c+1)]
                up=patch.elevations[max(0,r-1)*s.cols+c]
                down=patch.elevations[min(s.rows-1,r+1)*s.cols+c]
                slope=math.hypot((ri-l)/(1000*dx*(2 if 0<c<s.cols-1 else 1)),
                                 (down-up)/(1000*dy*(2 if 0<r<s.rows-1 else 1)))
                location=(x,y,h/1000)
                color=terrain_color(h,slope,x,y,season=season)
                if r%s.factor==0 and c%s.factor==0:
                    idx=(s.row_start+r//s.factor)*count+s.col_start+c//s.factor
                    verts[idx]=location
                    colors[idx]=color
                else:
                    idx=len(verts)
                    verts.append(location)
                    colors.append(color)
                ids.append(idx)
                per_patch_indices[(s.name,r,c)]=idx
        def split(start: int, end: int, between: list[int]) -> None:
            edge_splits[(start,end)]=between
            edge_splits[(end,start)]=list(reversed(between))
        for c in range(s.col_start,s.col_end):
            k=(c-s.col_start)*s.factor
            a=s.row_start*count+c
            b=a+1
            split(a,b,[ids[k+i] for i in range(1,s.factor)])
            k=(s.rows-1)*s.cols+(c-s.col_start)*s.factor
            a=s.row_end*count+c
            split(a,a+1,[ids[k+i] for i in range(1,s.factor)])
        for r in range(s.row_start,s.row_end):
            k=(r-s.row_start)*s.factor*s.cols
            a=r*count+s.col_start
            split(a,a+count,[ids[k+i*s.cols] for i in range(1,s.factor)])
            k=(r-s.row_start)*s.factor*s.cols+s.cols-1
            a=r*count+s.col_end
            split(a,a+count,[ids[k+i*s.cols] for i in range(1,s.factor)])

    faces=[]
    for r in range(divisions):
        for c in range(divisions):
            if (r,c) in skipped_cells:
                continue
            ring=(r*count+c,(r+1)*count+c,(r+1)*count+c+1,r*count+c+1)
            polygon=[]
            for j,idx in enumerate(ring):
                polygon.append(idx)
                polygon.extend(edge_splits.get((idx,ring[(j+1)%4]),()))
            faces.append(tuple(polygon))
    for patch in patches:
        s=patch.spec
        # Original sampled coarse vertices are reused inside the refined mesh.
        def index(r: int,c: int) -> int:
            if r%s.factor==0 and c%s.factor==0:
                return (s.row_start+r//s.factor)*count+s.col_start+c//s.factor
            # Fine vertices are stored in the order assembled above. A lookup
            # avoids relying on unrelated patches' vertex offsets.
            return per_patch_indices[(s.name,r,c)]

        for r in range(s.rows-1):
            for c in range(s.cols-1):
                faces.append((index(r,c),index(r+1,c),index(r+1,c+1),index(r,c+1)))
    mesh = bpy.data.meshes.new("Fuji / sampled GSI relief mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    attr = mesh.color_attributes.new(name="FujiAlbedo", type="FLOAT_COLOR", domain="POINT")
    for i, color in enumerate(colors):
        attr.data[i].color = color
    # UV orientation follows geographic north-to-south DEM row order. All
    # normals live in the GLB as a packed PNG, with zero external references.
    uv_layer=mesh.uv_layers.new(name="FujiGeographicUV")
    uv_coords=array("f")
    west,south=spec.xy(spec.longitude_min,spec.latitude_min)
    for polygon in faces:
        for index in polygon:
            x,y,_=verts[index]
            uv_coords.extend((ROCK_NORMAL_TILES*(x-west)/spec.width_km,
                              ROCK_NORMAL_TILES*(y-south)/spec.depth_km))
    uv_layer.data.foreach_set("uv",uv_coords)
    mat = bpy.data.materials.new("Fuji / alpine forest, basalt and seasonal snow")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = .86
    vertex = nodes.new("ShaderNodeVertexColor")
    vertex.layer_name = "FujiAlbedo"
    mat.node_tree.links.new(vertex.outputs["Color"], bsdf.inputs["Base Color"])
    normal_texture=nodes.new("ShaderNodeTexImage")
    normal_texture.image=_stone_normal_image(bpy,512 if divisions<1024 else 1024)
    normal_texture.interpolation="Linear"
    normal_texture.extension="REPEAT"
    normal_node=nodes.new("ShaderNodeNormalMap")
    normal_node.space="TANGENT"
    normal_node.uv_map=uv_layer.name
    normal_node.inputs["Strength"].default_value=.53
    mat.node_tree.links.new(normal_texture.outputs["Color"],normal_node.inputs["Color"])
    mat.node_tree.links.new(normal_node.outputs["Normal"],bsdf.inputs["Normal"])
    mesh.materials.append(mat)
    obj = bpy.data.objects.new("Mount Fuji / measured terrain surface", mesh)
    model.objects.link(obj)
    obj["elevation_source"] = dem.source
    obj["vertical_exaggeration"] = 1.0
    obj["metres_per_scene_unit"] = 1000
    obj["kumiki_part"] = "terrain-surface"
    obj["detail_patches"] = ", ".join(p.spec.name for p in patches)
    obj["surface_season"] = season
    obj["packed_normal_texture_tiles"] = ROCK_NORMAL_TILES
    return obj


def _plinth(bpy: Any, model: Any, dem: FujiDem, divisions: int) -> tuple[Any, Any]:
    """A surveyed perimeter skirt closes every relief edge down to its plinth."""
    spec = dem.spec
    min_height = min(dem.elevations)/1000
    x0, y0 = spec.xy(spec.longitude_min, spec.latitude_min)
    x1, y1 = spec.xy(spec.longitude_max, spec.latitude_max)
    floor = min_height-.22
    stride = (spec.samples-1)//divisions
    count = divisions+1
    ring = ([(0,c) for c in range(divisions)] +
            [(r,divisions) for r in range(divisions)] +
            [(divisions,c) for c in range(divisions,0,-1)] +
            [(r,0) for r in range(divisions,0,-1)])
    wall_vertices = []
    for r,c in ring:
        lon = spec.longitude_min + c/divisions*(spec.longitude_max-spec.longitude_min)
        lat = spec.latitude_max - r/divisions*(spec.latitude_max-spec.latitude_min)
        x,y = spec.xy(lon,lat)
        z = dem.elevations[(r*stride)*spec.samples+c*stride]/1000
        wall_vertices.append((x,y,z))
    wall_vertices += [(x,y,floor) for x,y,_ in wall_vertices]
    n = len(ring)
    wall_mesh = bpy.data.meshes.new("Fuji / contiguous perimeter cliff")
    wall_mesh.from_pydata(wall_vertices, [],
                          [(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)])
    wall_mesh.update()
    wall_mat = bpy.data.materials.new("Fuji / cross-section basalt")
    wall_mat.diffuse_color = (.08,.073,.079,1)
    wall_mat.use_nodes = True
    shader = next(n for n in wall_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = wall_mat.diffuse_color
    shader.inputs["Roughness"].default_value = .94
    wall_mesh.materials.append(wall_mat)
    wall = bpy.data.objects.new("Fuji / elevation-aligned perimeter", wall_mesh)
    model.objects.link(wall)
    wall["kumiki_part"] = "terrain-perimeter"
    # The base is below the entire sampled envelope, with no hanging surface.
    mesh = bpy.data.meshes.new("Fuji / below-terrain display slab")
    coords = [(x0,y0,floor), (x1,y0,floor), (x1,y1,floor), (x0,y1,floor),
              (x0,y0,floor-.13), (x1,y0,floor-.13), (x1,y1,floor-.13), (x0,y1,floor-.13)]
    mesh.from_pydata(coords, [], [(0,1,2,3),(4,7,6,5),(0,4,5,1),
                                  (1,5,6,2),(2,6,7,3),(3,7,4,0)])
    mesh.update()
    mat = bpy.data.materials.new("Pedestal / dark volcanic iron")
    mat.diffuse_color = (.042,.052,.060,1)
    mat.use_nodes = True
    node = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    node.inputs["Base Color"].default_value = mat.diffuse_color
    node.inputs["Metallic"].default_value = .20
    node.inputs["Roughness"].default_value = .56
    mesh.materials.append(mat)
    obj = bpy.data.objects.new("Fuji / relief mounting base", mesh)
    model.objects.link(obj)
    obj["kumiki_part"] = "terrain-pedestal"
    return wall, obj


def _studio(bpy: Any, scene: Any, stage: Any) -> Any:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.new("Fuji / clear high-altitude daylight")
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (.32,.44,.58,1)
    bg.inputs["Strength"].default_value = .61
    scene.world = world
    sun_data = bpy.data.lights.new("Fuji / clear-afternoon survey sunlight", "SUN")
    sun_data.energy = 2.7
    sun_data.angle = math.radians(4.5)
    sun = bpy.data.objects.new(sun_data.name, sun_data)
    stage.objects.link(sun)
    sun.rotation_euler = (math.radians(26), math.radians(-32), math.radians(-22))
    camera_data = bpy.data.cameras.new("Fuji / orbital study camera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    stage.objects.link(camera)
    return camera


def build_fuji_scene(destination: Path, dem_path: Path, *, divisions: int = 512,
                     detail_patches: tuple[Path, ...] = (),
                     season: str = "winter",
                     bpy_module: Any = None, preview_resolution: int = 1150) -> dict[str, Any]:
    if bpy_module is None:
        raise RuntimeError("Building Mount Fuji requires Blender bpy")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4,2,0):
        raise RuntimeError("Mount Fuji export needs Blender 4.2+ (4.5 LTS recommended)")
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Output revision already exists: {destination}")
    dem = load_dem(Path(dem_path))
    patches=tuple(load_patch(path) for path in detail_patches)
    scene = create_model_scene(bpy, "mount-fuji-survey-terrain")
    scene.render.resolution_x = preview_resolution
    scene.render.resolution_y = round(preview_resolution*.75)
    scene.render.resolution_percentage = 100
    model = bpy.data.collections.new("Fuji / measured geographic model")
    stage = bpy.data.collections.new("Fuji / documentary studio only")
    scene.collection.children.link(model)
    scene.collection.children.link(stage)
    camera = _studio(bpy, scene, stage)
    terrain = _terrain_object(bpy, model, dem, divisions, patches, season=season)
    wall, base = _plinth(bpy, model, dem, divisions)
    return export_revision(bpy, scene, camera, (terrain,wall,base), plan_fuji_shots(), destination,
                           samples=48 if divisions>=1024 else 16, metadata={
                               "asset_id": "mount-fuji-survey-terrain",
                               "generator": "GSI elevation-mesh",
                               "vertical_exaggeration": 1.0,
                               "surface_season": season,
                               "snow_cover": "none" if season=="summer" else "late-winter artistic snow",
                               "terrain_divisions": divisions,
                               "terrain_vertices": len(terrain.data.vertices),
                               "terrain_faces": len(terrain.data.polygons),
                               "detail_patches": [
                                   {"name":p.spec.name,"grid":[p.spec.rows,p.spec.cols],
                                    "factor":p.spec.factor,"surveyed_fraction":round(p.surveyed_fraction,5),
                                    "source":p.source} for p in patches],
                               "baked_normal_texture_px": 512 if divisions<1024 else 1024,
                               "packed_normal_texture_tiles": ROCK_NORMAL_TILES,
                               "named_parts": {"terrain_surface": 1,
                                               "terrain_perimeter": 1,
                                               "terrain_pedestal": 1},
                               "data_source": dem.source,
                               "dem_statistics": dem.statistics(),
                               "summit_lon_lat": SUMMIT,
                               "craters": ["summit", "Hoei / 1707 eruption"],
                           })
