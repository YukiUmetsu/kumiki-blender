"""Blender 4.2+ adapter for a layered, editorial Egypt relief-table diorama.

The export is actual terrain geometry, embedded geographic albedo, separate
river ribbons, landmark miniatures and converted typography—not a photo plane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...blender_output import create_model_scene, export_revision
from ...blender_scene import _color, _create_box, _material
from ...blender_textures import _packed_png
from .terrain import (
    BORDER_LINES, CITY_MARKERS, DEPTH, EAST, EgyptMapSpec, LABELS, NORTH, RIVER_CHANNELS,
    SOUTH, WEST, WIDTH, geo_to_xy, sample_terrain,
)
from .blender_support import (
    _add_terrain, _attach, _color_atlas_png, _make_plain_material, _marker,
    _river_ribbon, _serif_font, _text_mesh, _xy_ground,
)
from .blender_landmarks import (
    _add_alexandria_harbor, _add_aswan_dam, _add_cairo_mosque,
    _add_city_district, _add_nile_fields, _add_palm_groves, _add_sharm_marina,
    _pyramid, _temple,
)
from ...render_profiles import select_render_profile
from ...shots import QualityShot
from ...surfaces import get_surface


def plan_egypt_shots() -> tuple[QualityShot, ...]:
    """A reference-like hero and editorial/macro shots to audit actual craft."""
    delta = geo_to_xy(31.1, 30.4)
    redsea = geo_to_xy(34.45, 27.2)
    cairo = geo_to_xy(31.43, 30.04)
    luxor = geo_to_xy(32.73, 25.70)
    alexandria = geo_to_xy(29.98, 31.16)
    aswan = geo_to_xy(32.90, 24.09)
    sharm = geo_to_xy(34.33, 27.92)
    qattara = geo_to_xy(28.50, 29.42)
    nasser = geo_to_xy(32.75, 22.85)
    eastern_peaks = geo_to_xy(34.15, 26.10)
    return (
        QualityShot("hero", (5.7, -18.8, 15.0), (0, -1.05, -.10), 21.7),
        QualityShot("atlas", (0, -2.0, 25.0), (0, 0, .18), 18.4),
        QualityShot("delta", (delta[0]+3.1, delta[1]-4.0, 11.7),
                    (delta[0], delta[1], .15), 8.5),
        QualityShot("red-sea", (redsea[0]+6.1, redsea[1]-6.2, 11.4),
                    (redsea[0], redsea[1], .44), 9.2),
        QualityShot("cairo", (cairo[0]+1.5, cairo[1]-2.5, 4.2),
                    (cairo[0], cairo[1], .23), 2.9),
        QualityShot("luxor", (luxor[0]-1.5, luxor[1]-1.65, 3.3),
                    (luxor[0]+.15, luxor[1], .20), 2.1),
        QualityShot("alexandria", (alexandria[0]+1.2, alexandria[1]-1.5, 3.1),
                    (alexandria[0], alexandria[1], .15), 1.9),
        QualityShot("aswan", (aswan[0]-1.5, aswan[1]-1.35, 3.2),
                    (aswan[0]+.12, aswan[1]+.12, .14), 2.1),
        QualityShot("sharm", (sharm[0]+.15, sharm[1]-.50, 4.6),
                    (sharm[0]-.20, sharm[1]+.20, .17), 1.55),
        # A low raking angle exposes the saltpan-to-escarpment height step.
        QualityShot("qattara", (qattara[0]-3.44, qattara[1]-4.27, 2.3),
                    (qattara[0]-.04, qattara[1]-.67, .17), 4.5),
        QualityShot("lake-nasser", (nasser[0]-2.4, nasser[1]-2.7, 5.0),
                    (nasser[0], nasser[1], .13), 4.5),
        QualityShot("eastern-peaks", (eastern_peaks[0]-2.7,
                                      eastern_peaks[1]-3.4, 5.0),
                    (eastern_peaks[0], eastern_peaks[1], .91), 4.1),
    )


def _studio(bpy: Any, scene: Any, stage: Any) -> Any:
    backdrop = _make_plain_material(bpy, "Studio / midnight slate", "#1C2931", roughness=.96)
    _create_box(bpy, stage, "Studio / seamless dark floor", (34, 26, .18),
                (0, 0, -1.11), backdrop)
    world = bpy.data.worlds.new("Egypt / blue-hour environment")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = _color("#566675")
    background.inputs["Strength"].default_value = .52
    scene.world = world
    for name, pos, energy, size, tint in (
        ("Key / warm Mediterranean sun", (-8, 1, 15), 2600, 11, (1.0, .83, .65)),
        ("Fill / broad cool softbox", (9, 4, 13), 1950, 12, (.68, .82, 1.0)),
        ("Rim / pale glint", (0, 12, 11), 1450, 9, (.86, .94, 1.0)),
    ):
        lamp = bpy.data.lights.new(name, "AREA")
        lamp.energy = energy
        lamp.shape = "DISK"
        lamp.size = size
        lamp.color = tint
        obj = bpy.data.objects.new(name, lamp)
        stage.objects.link(obj)
        from ...blender_output import aim_at
        obj.location = pos
        aim_at(obj, (0, 0, 0))
    camera_data = bpy.data.cameras.new("Camera / Egypt editorial relief")
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    stage.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.lens = 42
    return camera


def build_egypt_map_scene(destination: Path, *, spec: EgyptMapSpec = EgyptMapSpec(),
                          bpy_module: Any = None) -> dict[str, Any]:
    """Produce immutable .blend/.glb plus twelve Cycles inspection views."""
    if bpy_module is None:
        raise RuntimeError("Building the Egypt map requires Blender bpy")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError("Blender 4.2 or newer is required")
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Output revision already exists: {destination}")
    profile = select_render_profile(1920, 1200, 72)
    scene = create_model_scene(bpy, "egypt-relief-atlas")
    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile.samples
    scene.render.resolution_x, scene.render.resolution_y = profile.width, profile.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    model = bpy.data.collections.new("Egypt / portable relief, rivers, and map type")
    scene.collection.children.link(model)
    stage = bpy.data.collections.new("Egypt / studio, not exported")
    scene.collection.children.link(stage)
    # A smoke render may reduce *preview* resolution, never the exported
    # asset's texture fidelity. Match the repository-wide immutable-asset
    # contract by packing the identical map material in every profile.
    # Closer editorial zooms need finer albedo detail as well as additional
    # geometry. Still small enough to remain a self-contained GLB texture.
    atlas_size = (1536, 1200)
    atlas = _packed_png(bpy, "Egypt shaded geographic albedo",
                        _color_atlas_png(*atlas_size), non_color=False)

    land = _material(bpy, get_surface("weathered-stone", color="#DCC294", roughness=.89,
                                      weathering_strength=.32))
    water = _make_plain_material(bpy, "Egypt / inlaid lapis water", "#216477",
                                 roughness=.29, metallic=.08)
    for surface in (land, water):
        nodes = surface.node_tree.nodes
        shader = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.label = "Packed / hand-authored Egypt geographic palette"
        image_node.image = atlas
        surface.node_tree.links.new(image_node.outputs["Color"], shader.inputs["Base Color"])

    side = _material(bpy, get_surface("weathered-stone", color="#665748", roughness=.78,
                                      weathering_strength=.18))
    sandstone = _material(bpy, get_surface("natural-stone", color="#B2946E", roughness=.88))
    temple_shadow = _make_plain_material(bpy, "Temple / dark course", "#776047", roughness=.9)
    dark_ink = _make_plain_material(bpy, "Map type / carved umber", "#423D36", roughness=.74)
    pale_ink = _make_plain_material(bpy, "Map type / silver-blue", "#CEDFE7", roughness=.48)
    river_ink = _make_plain_material(bpy, "Map type / river blue", "#2E6480", roughness=.4)
    sage_ink = _make_plain_material(bpy, "Map type / light sage", "#E3EDCD", roughness=.8)
    city_type = _make_plain_material(bpy, "City legend / chalk", "#F5F1E6", roughness=.6)
    bronze = _make_plain_material(bpy, "Title / antique silver", "#C3BBB1", roughness=.31,
                                   metallic=.62)
    marker_gold = _make_plain_material(bpy, "City locator / warm gold", "#FFCF8A",
                                       roughness=.2, metallic=.3)
    nile_blue = _make_plain_material(bpy, "River / azure enamel", "#1D7495",
                                     roughness=.24, metallic=.12)
    border_stone = _make_plain_material(bpy, "Map boundary / graphite engraving", "#555452",
                                        roughness=.59, metallic=.09)
    limestone = _make_plain_material(bpy, "Architecture / pale chalk limestone",
                                     "#E8D0A1", roughness=.88)
    city_walls = tuple(_make_plain_material(bpy, f"City / courtyard sandstone {i}", color,
                                             roughness=.87)
                       for i, color in enumerate(("#D9B88D", "#C99F77", "#E5CEAA")))
    roof_clay = _make_plain_material(bpy, "City / clay tile parapets", "#916751", roughness=.9)
    window_ink = _make_plain_material(bpy, "City / indigo window recesses",
                                      "#344C54", roughness=.59)
    copper = _make_plain_material(bpy, "Architecture / aged turquoise copper",
                                  "#3F8987", roughness=.49, metallic=.35)
    concrete = _make_plain_material(bpy, "Aswan / pale granite dam",
                                    "#A9A295", roughness=.88)
    field_materials = tuple(_make_plain_material(bpy, f"Nile crop / {i}", color,
                                                  roughness=.94)
                            for i, color in enumerate(("#47784B", "#729447", "#B4A366", "#316649")))
    palm_bark = _make_plain_material(bpy, "Palm / fiber bark", "#876A4B", roughness=.94)
    palm_leaf = _make_plain_material(bpy, "Palm / leaf crowns", "#527646", roughness=.78)

    objects: list[Any] = []
    slab = _create_box(bpy, model, "Egypt / solid sand-colored atlas plinth",
                       (WIDTH, DEPTH, .57), (0, 0, -.285), side)
    slab["kumiki_part"] = "atlas-plinth"
    objects.append(slab)
    objects.append(_add_terrain(bpy, model, spec, atlas, land, water))
    for name, route in RIVER_CHANNELS.items():
        vertices, faces = _river_ribbon(name, route)
        objects.append(_attach(bpy, model, f"Waterway / {name}", vertices, faces,
                               nile_blue, "waterway"))
    for name, route in BORDER_LINES.items():
        vertices, faces = _river_ribbon("border", route)
        objects.append(_attach(bpy, model, f"Cartographic border / {name}",
                               vertices, faces, border_stone, "illustrated-border"))
    for name, lon, lat, width, height in (
        ("Khufu", 30.73, 29.92, .29, .27),
        ("Khafre", 30.49, 29.99, .235, .225),
        ("Menkaure", 30.31, 29.91, .17, .155),
    ):
        objects.append(_pyramid(bpy, model, f"Giza / {name} miniature", lon, lat,
                                width, height, sandstone))
    objects.extend(_temple(bpy, model, sandstone, temple_shadow, limestone))
    objects.extend(_add_cairo_mosque(bpy, model, limestone, copper))
    objects.extend(_add_alexandria_harbor(bpy, model, sandstone, copper))
    objects.extend(_add_aswan_dam(bpy, model, concrete, temple_shadow))
    objects.extend(_add_sharm_marina(bpy, model, limestone, copper))

    city_total = 0
    for name, lon, lat, offset, grid, seed in (
        ("Cairo", 31.24, 30.04, (.49, -.12), (10, 9, .078), 18),
        ("Alexandria", 29.92, 31.20, (.24, -.18), (8, 5, .068), 49),
        ("Luxor", 32.65, 25.70, (.59, -.12), (8, 5, .068), 76),
        ("Aswan", 32.90, 24.09, (.42, .22), (7, 5, .065), 93),
        ("Sharm El-Sheikh", 34.33, 27.92, (-.39, .22), (7, 4, .060), 132),
    ):
        components, count = _add_city_district(bpy, model, name, lon, lat, offset,
                                               grid, seed, city_walls,
                                               roof_clay, window_ink)
        objects.extend(components)
        city_total += count
    farmland, field_total = _add_nile_fields(bpy, model, field_materials)
    objects.extend(farmland)
    palms, palm_total = _add_palm_groves(bpy, model, palm_bark, palm_leaf)
    objects.extend(palms)

    serif = _serif_font(bpy)
    for label in LABELS:
        x, y = geo_to_xy(label.lon, label.lat)
        z = sample_terrain(label.lon, label.lat).height_m + (.16 if label.style != "sea" else .035)
        material = (pale_ink if label.style == "sea" else
                    river_ink if label.style == "river" else
                    sage_ink if label.style == "sage" else dark_ink)
        obj = _text_mesh(bpy, model, label.text, f"Atlas type / {label.text}",
                         (x, y, z), label.size, material,
                         angle=label.angle, font=serif, clear_relief=True)
        objects.append(obj)
    for name, (lon, lat) in CITY_MARKERS.items():
        objects.append(_marker(bpy, model, name, lon, lat, marker_gold))
        x, y = geo_to_xy(lon, lat)
        # Keep legends out of the sculpted street grids. In the first macro
        # review 'Alexandria' obscured the entire lighthouse district.
        offset = {
            "Alexandria": (-.43, .34), "Cairo": (-.32, .32),
            "Luxor": (-.32, .28), "Aswan": (-.32, .25),
            "Sharm El-Sheikh": (.80, -.30),
        }[name]
        tx, ty = x+offset[0], y+offset[1]
        z = _xy_ground(tx, ty, .105)
        objects.append(_text_mesh(bpy, model, name, f"City legend / {name}",
                                  (tx, ty, z), .167, city_type,
                                  depth=.006, clear_relief=True))

    # Frontispiece: actual raised metal glyphs on the dark stage in front of
    # the block. In the reference they are far more visible than side letters.
    objects.append(_text_mesh(bpy, model, "EGYPT", "Frontispiece / EGYPT",
                              (0, -DEPTH/2-1.25, -.997), 1.24, bronze,
                              font=serif, depth=.044))
    objects.append(_text_mesh(bpy, model, "SUDAN", "Front edge / SUDAN",
                              (0, -DEPTH/2-.014, -.335), .32, pale_ink,
                              upright=True, font=serif, depth=.009))
    camera = _studio(bpy, scene, stage)
    return export_revision(
        bpy, scene, camera, objects, plan_egypt_shots(), destination,
        samples=profile.samples,
        metadata={
            "asset_id": "egypt-relief-atlas",
            "generator": "egypt-relief-map",
            "render_profile": profile.name,
            "terrain_grid": [spec.columns, spec.rows],
            "geographic_albedo_pixels": list(atlas_size),
            "geographic_extent_lon_lat": [[WEST, SOUTH], [EAST, NORTH]],
            "geography_provenance": "Original illustrated interpretation of supplied reference; not surveyed terrain or authoritative country boundaries",
            "named_parts": {"terrain": 1, "waterways": len(RIVER_CHANNELS),
                            "borders": len(BORDER_LINES),
                            "city_markers": len(CITY_MARKERS), "geography_labels": len(LABELS),
                            "pyramids": 3, "temple_components": 4,
                            "city_buildings": city_total,
                            "farm_parcels": field_total, "palm_trees": palm_total,
                            "additional_landmarks": 4},
            "additional_landmark_names": ("Cairo citadel mosque", "Alexandria Pharos",
                                          "Aswan dam", "Sharm marina"),
        },
    )
