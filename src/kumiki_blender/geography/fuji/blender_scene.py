"""Mount Fuji: survey-based, three-dimensional volcano terrain in Blender.

The actual flanks, ridge drainage, summit rim and Hōei flank crater are
preserved from the input DEM; there is no cone primitive or procedural
replacement of surveyed elevation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ...blender_output import create_model_scene, export_revision
from ...shots import QualityShot
from .dem import FujiDem, SUMMIT, load_dem


def plan_fuji_shots() -> tuple[QualityShot, ...]:
    """Opposing profiles plus the two named crater checkpoints."""
    return (
        QualityShot("hero", (13, -31, 8.0), (0, 0, 2.1), 16.5),
        QualityShot("north-profile", (-12, 32, 9), (0, 0, 2.05), 18.0),
        QualityShot("summit-crater", (1.8, -3.7, 7.2), (0, 0, 3.52), 4.7),
        QualityShot("hoei-context", (9, -10, 8.7), (1.61, -2.18, 2.43), 7.2),
        QualityShot("hoei-crater", (1.61, -2.18, 12), (1.61, -2.18, 2.43), 4.2),
        QualityShot("overhead", (0, 0, 36), (0, 0, 2), 27.0),
    )


def _mix(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return tuple(a[i]*(1-t)+b[i]*t for i in range(3))


def terrain_color(height_m: float, slope: float, x: float, y: float) -> tuple[float, float, float, float]:
    """Elevation and exposure-dependent albedo inspired by actual Fuji photos.

    Rocky radial gullies expose darker basalt between winter snow tongues.
    This is stylized albedo, not a claim of satellite/snow-cover accuracy.
    """
    # Shared gradients rather than hard rings; no flat-color cut at the tree line.
    forest = (.078, .148, .114)
    subalpine = (.21, .245, .18)
    basalt = (.115, .123, .137)
    oxidized = (.305, .258, .245)
    snow = (.84, .88, .91)
    if height_m < 1800:
        shade = max(0, min(1, (height_m-1000)/800))
        c = _mix(forest, subalpine, shade)
    elif height_m < 2750:
        c = _mix(subalpine, basalt, min(1, (height_m-1800)/950))
    else:
        c = _mix(basalt, oxidized, min(1, (height_m-2750)/1050)*.45)

    # First Hōei crater: exposed reddish-grey pyroclastic scree on steeper
    # walls, not the vegetated tone of the surrounding lower flanks.
    hoei_distance = math.hypot(x-1.61,y+2.18)
    if hoei_distance < 1.9 and height_m > 2050:
        influence = min(1., (1.9-hoei_distance)/.75)
        scree = (.235,.196,.183)
        c = _mix(c, scree, .35*influence*min(1.,slope*1.7+.20))

    # Fine, deterministic pigmentation: small enough not to falsify geometry.
    grain = (math.sin(x*31.3+y*13.1)*math.sin(y*37.2-x*19.1))*.05
    c = tuple(max(.0, min(1., v*(1+grain))) for v in c)
    # Gully-facing slopes tend to expose volcanic scree; irregular snow boundary
    # has radial fingers instead of a flat painted white circular band.
    bearing = math.atan2(y, x)
    fingers = (math.sin(bearing*9 + math.sin(bearing*3.1)) * 110 +
               math.sin(bearing*17 + y*.8)*65 + math.sin(x*2.2+y*2.7)*28)
    snowline = 2820 + fingers + min(.45, slope)*260
    coverage = max(0., min(1., (height_m-snowline)/390))
    coverage *= max(.25, 1-slope*.7)
    c = _mix(c, snow, coverage*.95)
    return (*c, 1.)


def _terrain_object(bpy: Any, model: Any, dem: FujiDem, divisions: int) -> Any:
    spec = dem.spec
    if (divisions < 32 or divisions > spec.samples-1 or
            (spec.samples-1) % divisions):
        raise ValueError("Terrain divisions must evenly divide DEM intervals")
    row_stride = (spec.samples-1)//divisions
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
            colors.append(terrain_color(height, math.hypot(grad_x,grad_y), x, y))

    faces = [(r*count+c, (r+1)*count+c, (r+1)*count+c+1, r*count+c+1)
             for r in range(divisions) for c in range(divisions)]
    mesh = bpy.data.meshes.new("Fuji / sampled GSI relief mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    attr = mesh.color_attributes.new(name="FujiAlbedo", type="FLOAT_COLOR", domain="POINT")
    for i, color in enumerate(colors):
        attr.data[i].color = color
    mat = bpy.data.materials.new("Fuji / alpine forest, basalt and seasonal snow")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = .86
    vertex = nodes.new("ShaderNodeVertexColor")
    vertex.layer_name = "FujiAlbedo"
    mat.node_tree.links.new(vertex.outputs["Color"], bsdf.inputs["Base Color"])
    mesh.materials.append(mat)
    obj = bpy.data.objects.new("Mount Fuji / measured terrain surface", mesh)
    model.objects.link(obj)
    obj["elevation_source"] = dem.source
    obj["vertical_exaggeration"] = 1.0
    obj["metres_per_scene_unit"] = 1000
    obj["kumiki_part"] = "terrain-surface"
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
    bg.inputs["Color"].default_value = (.28,.40,.55,1)
    bg.inputs["Strength"].default_value = .50
    scene.world = world
    sun_data = bpy.data.lights.new("Fuji / winter afternoon sun", "SUN")
    sun_data.energy = 2.2
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
                     bpy_module: Any = None, preview_resolution: int = 1000) -> dict[str, Any]:
    if bpy_module is None:
        raise RuntimeError("Building Mount Fuji requires Blender bpy")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4,2,0):
        raise RuntimeError("Mount Fuji export needs Blender 4.2+ (4.5 LTS recommended)")
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Output revision already exists: {destination}")
    dem = load_dem(Path(dem_path))
    scene = create_model_scene(bpy, "mount-fuji-survey-terrain")
    scene.render.resolution_x = preview_resolution
    scene.render.resolution_y = round(preview_resolution*.75)
    scene.render.resolution_percentage = 100
    model = bpy.data.collections.new("Fuji / measured geographic model")
    stage = bpy.data.collections.new("Fuji / documentary studio only")
    scene.collection.children.link(model)
    scene.collection.children.link(stage)
    camera = _studio(bpy, scene, stage)
    terrain = _terrain_object(bpy, model, dem, divisions)
    wall, base = _plinth(bpy, model, dem, divisions)
    return export_revision(bpy, scene, camera, (terrain,wall,base), plan_fuji_shots(), destination,
                           samples=12, metadata={
                               "asset_id": "mount-fuji-survey-terrain",
                               "generator": "GSI elevation-mesh",
                               "vertical_exaggeration": 1.0,
                               "terrain_divisions": divisions,
                               "terrain_vertices": (divisions+1)**2,
                               "named_parts": {"terrain_surface": 1,
                                               "terrain_perimeter": 1,
                                               "terrain_pedestal": 1},
                               "data_source": dem.source,
                               "dem_statistics": dem.statistics(),
                               "summit_lon_lat": SUMMIT,
                               "craters": ["summit", "Hoei / 1707 eruption"],
                           })
