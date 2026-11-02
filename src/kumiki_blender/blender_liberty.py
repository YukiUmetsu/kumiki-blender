"""Blender 4.2+ Cycles staging and revision export for the Liberty study."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .blender_output import create_model_scene, export_revision
from .blender_scene import _create_box, _material
from .blender_textures import ensure_metric_uvs
from .liberty import LibertyStudy
from .liberty_tablet import tablet_center_x, tablet_front_y
from .liberty_torch_placement import (detailed_torch_placement,
                                      detailed_torch_shots,
                                      replace_legacy_torch_part)
from .render_profiles import select_render_profile
from .shots import QualityShot
from .surfaces import PBRSurface, get_surface


def plan_liberty_shots(study: LibertyStudy) -> tuple[QualityShot, ...]:
    """Front, back, profile, face, torch and tablet all receive dedicated QA."""
    scale = dict(study.parameters)["scale"]
    if dict(study.parameters).get("monument_proportions"):
        offset = 0 if dict(study.parameters)["include_pedestal"] else -46.94

        def measured(x: float, y: float, z: float) -> tuple[float, float, float]:
            return (x*scale, y*scale, (z+offset)*scale)

        return (
            # 16:9 renders need ortho width ~height*16/9, plus breathing room.
            QualityShot("hero", measured(48, -94, 66), measured(0, 0, 46.5), 210*scale),
            QualityShot("front", measured(0, -115, 47), measured(0, 0, 46.5), 200*scale),
            QualityShot("side", measured(115, 0, 47), measured(0, 0, 46.5), 200*scale),
            QualityShot("rear", measured(0, 115, 47), measured(0, 0, 46.5), 200*scale),
            QualityShot("face", measured(4, -26, 79.4), measured(0, 0, 79.4), 12.5*scale),
            QualityShot("tablet", measured(9.2, -27, 69.8), measured(7.0, -1, 70), 14*scale),
            QualityShot("torch", measured(-7, -26, 90), measured(-7.15, 0, 90), 13*scale),
        )
    offset = 0 if dict(study.parameters)["include_pedestal"] else -9.30

    def pos(x: float, y: float, z: float) -> tuple[float, float, float]:
        return (x*scale, y*scale, (z+offset)*scale)

    return (
        QualityShot("hero", pos(19, -44, 24), pos(0, 0, 18), 61*scale),
        QualityShot("front", pos(0, -48, 18), pos(0, 0, 18), 58*scale),
        QualityShot("side", pos(48, 0, 18), pos(0, 0, 18), 58*scale),
        QualityShot("rear", pos(0, 48, 18), pos(0, 0, 18), 58*scale),
        QualityShot("face", pos(3.0, -11, 27.25), pos(0, 0, 27.1), 7.4*scale),
        QualityShot("tablet", pos(5.2, -8.5, 23), pos(3.20, -.7, 22.9), 6.3*scale),
        QualityShot("torch", pos(-1.1, -8.0, 34.6), pos(-3.45, 0, 34.6), 5.6*scale),
    )


def _tablet_lettering(bpy: Any, collection: Any, material: Any, scale: float,
                      base_offset: float, detailed: bool = False) -> list[Any]:
    """Emboss the Roman-date inscription as genuine editable 3D text meshes."""
    objects = []
    font = None
    if detailed:
        # Converted into editable meshes before export: no external font is
        # needed to open the GLB or source .blend on another computer.
        for candidate in (
            "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        ):
            if Path(candidate).exists():
                font = bpy.data.fonts.load(candidate)
                break
    lines = (("JULY IV", 23.49, .41), ("MDCCLXXVI", 22.78, .325)) if detailed else (
        ("JULY IV", 23.42, .31), ("MDCCLXXVI", 22.66, .27))
    for index, (label, z, size) in enumerate(lines):
        glyph = bpy.data.curves.new(f"Tablet engraving / {label}", "FONT")
        glyph.body = label
        if font is not None:
            glyph.font = font
        glyph.align_x = "CENTER"
        glyph.align_y = "CENTER"
        glyph.size = size*scale
        glyph.extrude = (.006 if detailed else .012)*scale
        glyph.bevel_depth = (.001 if detailed else .0025)*scale
        glyph.bevel_resolution = 1
        font_obj = bpy.data.objects.new(f"liberty-tablet-inscription-{index}", glyph)
        collection.objects.link(font_obj)
        if detailed:
            font_obj.location = (tablet_center_x(z)*scale,
                                 (tablet_front_y(z)-.017)*scale,
                                 (z+base_offset)*scale)
            font_obj.rotation_euler = (math.pi/2-.035, 0, 0)
        else:
            font_obj.location = (3.32*scale, -.993*scale, (z+base_offset)*scale)
            font_obj.rotation_euler = (math.pi/2, 0, -.09)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = font_obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True,
                                               depsgraph=depsgraph)
        if mesh is None or not mesh.vertices or not mesh.polygons:
            raise RuntimeError(f"Could not convert tablet inscription {label} into a 3D mesh")
        shape = bpy.data.objects.new(font_obj.name+"-mesh", mesh)
        collection.objects.link(shape)
        shape.matrix_world = font_obj.matrix_world.copy()
        shape.data.materials.append(material)
        shape["kumiki_part"] = "tablet-inscription"
        objects.append(shape)
        bpy.data.objects.remove(font_obj, do_unlink=True)
    return objects


def build_liberty_scene(study: LibertyStudy, destination: Path, *,
                        bpy_module: Any = None,
                        detailed_torch: bool = False) -> dict[str, Any]:
    """Author an editable .blend, portable GLB and 7 camera inspections."""
    if bpy_module is None:
        raise RuntimeError("Blender bpy is required to build Liberty")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError("Blender 4.2 or newer is required")
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Output revision exists: {destination}")

    profile = select_render_profile(1280, 800, 48)
    scene = create_model_scene(bpy, study.asset_id)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile.samples
    scene.render.resolution_x = profile.width
    scene.render.resolution_y = profile.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.render.film_transparent = False

    model = bpy.data.collections.new("Liberty / named editable copper and masonry")
    scene.collection.children.link(model)
    stage = bpy.data.collections.new("Liberty / studio - exclude from GLB")
    scene.collection.children.link(stage)

    # Palette uses PBR textures packaged in the GLB, with a cooler shadow
    # variant to reveal carved cloth and a contrasting real golden flame.
    specs = {
        "patina": PBRSurface("Liberty / oxidized copper", "#79AA9C", .31, .70,
                              texture_kind="galvanized-steel"),
        "patina-light": PBRSurface("Liberty / sunlit verdigris", "#92BCB0", .22, .77,
                                    texture_kind="galvanized-steel"),
        "patina-shadow": PBRSurface("Liberty / dark green recess", "#487D72", .28, .76,
                                     texture_kind="galvanized-steel"),
        "flame-gold": PBRSurface("Liberty / fire-gilded copper", "#E5B64E", .78, .37,
                                  texture_kind="galvanized-steel"),
        "masonry": get_surface("weathered-stone", color="#AB9F8B", roughness=.89,
                               weathering_strength=.62),
        "limestone": get_surface("weathered-stone", color="#C1B7A3", roughness=.83,
                                 weathering_strength=.38),
        "granite": get_surface("weathered-stone", color="#88867E", roughness=.9,
                               weathering_strength=.49),
        "recess": get_surface("natural-stone", color="#393E3A", roughness=.91),
    }
    palette = {key: _material(bpy, spec) for key, spec in specs.items()}
    objects = []
    for part in study.instances:
        if detailed_torch and replace_legacy_torch_part(part.name):
            continue
        mesh = bpy.data.meshes.new(f"Geometry / {part.name}")
        mesh.from_pydata(part.mesh.vertices, [], part.mesh.faces)
        mesh.update()
        ensure_metric_uvs(mesh, tile_size_m=1.30)
        mesh.materials.append(palette[part.material])
        obj = bpy.data.objects.new(part.name, mesh)
        model.objects.link(obj)
        obj["kumiki_part"] = part.name.removeprefix("liberty-").split("-")[0]
        obj["kumiki_source"] = "original procedural geometry / photo reference"
        for polygon in mesh.polygons:
            polygon.use_smooth = not any(word in part.name for word in (
                "pedestal", "tablet-slab", "crown-ray", "crown-diadem", "window-dark"))
        if part.name.startswith("liberty-pedestal-") or part.name == "liberty-tablet-slab":
            modifier = obj.modifiers.new("Stone or copper edge softness", "BEVEL")
            modifier.width = .014*dict(study.parameters)["scale"]
            modifier.segments = 2
            obj.modifiers.new("Weighted flat face normals", "WEIGHTED_NORMAL")
        objects.append(obj)

    parameters = dict(study.parameters)
    offset = (0 if parameters["include_pedestal"] else
              -46.94 if parameters.get("monument_proportions") else -9.30)
    objects.extend(_tablet_lettering(bpy, model, palette["patina-shadow"],
                                     parameters["scale"], offset,
                                     parameters.get("detailed_tablet", False)))
    torch_fit = None
    if detailed_torch:
        from .blender_liberty_torch_detail import build_torch_detail

        torch_fit = detailed_torch_placement(study)
        torch_root = build_torch_detail(origin=torch_fit.origin)
        torch_root.scale = (torch_fit.radial_scale, torch_fit.radial_scale,
                            torch_fit.height_scale)
        torch_root["kumiki_part"] = "torch-assembly-control"
        torch_root["kumiki_source"] = "original editable geometric detail study"
        torch_parts = [obj for obj in bpy.data.objects if obj.parent is torch_root]
        if len(torch_parts) < 40:
            raise RuntimeError("Detailed torch assembly is missing expected cast components")
        for obj in [torch_root, *torch_parts]:
            for old_collection in tuple(obj.users_collection):
                old_collection.objects.unlink(obj)
            model.objects.link(obj)
            if obj.type != "MESH":
                continue
            surface = (palette["flame-gold"] if obj.name.startswith("Flame /")
                       else palette["patina-shadow"] if "dark recessed" in obj.name.lower()
                       else palette["patina"])
            obj.data.materials.clear()
            obj.data.materials.append(surface)
            ensure_metric_uvs(obj.data, tile_size_m=1.30)
            obj["kumiki_part"] = "torch"
            obj["kumiki_source"] = "original detailed torch geometry / photographic reference"
            objects.append(obj)
    stage_floor = get_surface("studio-matte", color="#363A3C", roughness=.97)
    extent = 54*parameters["scale"]
    _create_box(bpy, stage, "Studio / charcoal cyclorama", (extent, extent, .35*parameters["scale"]),
                (0, 0, -.23*parameters["scale"]), _material(bpy, stage_floor))

    world = bpy.data.worlds.new("Liberty / blue-grey ambience")
    world.use_nodes = True
    background = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    background.inputs["Color"].default_value = (.19, .24, .27, 1)
    background.inputs["Strength"].default_value = .55
    scene.world = world

    measured_frame = bool(parameters.get("monument_proportions"))

    def area(name: str, location: tuple[float, float, float], power: float,
             color: tuple[float, float, float], size: float) -> None:
        from .blender_output import aim_at
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = power
        light_data.shape = "DISK"
        light_data.size = size*parameters["scale"]
        light_data.color = color
        light = bpy.data.objects.new(name, light_data)
        stage.objects.link(light)
        light.location = tuple(p*parameters["scale"] for p in location)
        aim_at(light, (0, 0, (47 if measured_frame else 18+offset)*parameters["scale"]))

    if measured_frame:
        area("Key / broad overcast studio", (-36, -52, 79), 22000, (0.86, .94, 1.0), 27)
        area("Fill / pale aqua bounce", (36, -19, 54), 15000, (.61, .80, .87), 24)
        area("Rim / warm autumn", (22, 34, 74), 27000, (1.0, .81, .60), 27)
    else:
        area("Key / broad overcast studio", (-16, -23, 32+offset), 6800, (0.86, .94, 1.0), 15)
        area("Fill / pale aqua bounce", (18, -8, 21+offset), 4200, (.61, .80, .87), 11)
        area("Rim / warm autumn", (8, 14, 30+offset), 7800, (1.0, .81, .60), 12)

    camera_data = bpy.data.cameras.new("Camera / Liberty cinematic inspection")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    stage.objects.link(camera)
    scene.camera = camera
    shots = plan_liberty_shots(study)
    if detailed_torch:
        shots += detailed_torch_shots(study)
    if parameters.get("detailed_tablet"):
        def tablet_pos(x: float, y: float, z: float) -> tuple[float, float, float]:
            return (x*parameters["scale"], y*parameters["scale"],
                    (z+offset)*parameters["scale"])
        shots += (
            QualityShot("tablet-detail-front", tablet_pos(3.22, -9, 22.5),
                        tablet_pos(3.27, -.9, 22.4), 7.1*parameters["scale"]),
            QualityShot("tablet-detail-profile", tablet_pos(9, -.1, 22.4),
                        tablet_pos(4.06, -.85, 22.4), 5.1*parameters["scale"]),
            QualityShot("tablet-grip-detail", tablet_pos(5.8, -5.8, 22.3),
                        tablet_pos(4.27, -.9, 22.35), 2.35*parameters["scale"]),
        )
    if parameters.get("detailed_crown") and not measured_frame:
        def crown_pos(x: float, y: float, z: float) -> tuple[float, float, float]:
            return (x*parameters["scale"], y*parameters["scale"],
                    (z+offset)*parameters["scale"])
        shots += (
            QualityShot("crown-rays-front", crown_pos(0, -11, 30.15),
                        crown_pos(0, 0, 30.0), 9.0*parameters["scale"]),
            QualityShot("crown-band-front", crown_pos(0, -7.5, 28.95),
                        crown_pos(0, 0, 28.89), 4.7*parameters["scale"]),
            QualityShot("crown-band-oblique", crown_pos(4.8, -7.5, 29.0),
                        crown_pos(0, 0, 28.9), 5.0*parameters["scale"]),
            QualityShot("crown-ray-profile", crown_pos(8, -.75, 30.05),
                        crown_pos(0, 0, 30.0), 8.0*parameters["scale"]),
        )
    if parameters.get("detailed_face") and not measured_frame:
        def face_pos(x: float, y: float, z: float) -> tuple[float, float, float]:
            return (x*parameters["scale"], y*parameters["scale"],
                    (z+offset)*parameters["scale"])
        shots += (
            QualityShot("face-detail-front", face_pos(0, -7.2, 26.67),
                        face_pos(0, 0, 26.67), 3.7*parameters["scale"]),
            QualityShot("face-detail-threequarter", face_pos(4.8, -6.7, 26.67),
                        face_pos(0, 0, 26.67), 3.7*parameters["scale"]),
            QualityShot("face-detail-profile", face_pos(7.2, -.3, 26.67),
                        face_pos(0, 0, 26.67), 3.7*parameters["scale"]),
        )
    if parameters.get("detailed_feet") and measured_frame:
        def feet_pos(x: float, y: float, z: float) -> tuple[float, float, float]:
            return (x*parameters["scale"], y*parameters["scale"],
                    (z+offset)*parameters["scale"])
        shots += (
            QualityShot("feet-shackle-front", feet_pos(0, -19, 50),
                        feet_pos(0, -3, 48.4), 13.5*parameters["scale"]),
            QualityShot("feet-shackle-oblique", feet_pos(-14, -17, 52),
                        feet_pos(-1.1, -3.4, 49), 13.5*parameters["scale"]),
        )
    elif parameters.get("detailed_feet"):
        def feet_pos(x: float, y: float, z: float) -> tuple[float, float, float]:
            return (x*parameters["scale"], y*parameters["scale"],
                    (z+offset)*parameters["scale"])
        shots += (
            QualityShot("feet-shackle-front", feet_pos(0, -8.5, 11.4),
                        feet_pos(0, -2.04, 9.87), 5.6*parameters["scale"]),
            QualityShot("feet-shackle-oblique", feet_pos(-5.1, -7.3, 12.6),
                        feet_pos(-.35, -2.04, 9.85), 5.5*parameters["scale"]),
            QualityShot("feet-left-broken-link", feet_pos(5.0, -5.4, 11.35),
                        feet_pos(1.80, -2.43, 9.78), 2.9*parameters["scale"]),
            QualityShot("feet-right-broken-shackle", feet_pos(-4.9, -5.6, 11.6),
                        feet_pos(-.82, -2.27, 9.80), 3.3*parameters["scale"]),
        )
    if parameters.get("detailed_drapery") and measured_frame:
        def drape_pos(x: float, y: float, z: float) -> tuple[float, float, float]:
            return (x*parameters["scale"], y*parameters["scale"],
                    (z+offset)*parameters["scale"])
        shots += (
            QualityShot("drapery-front", drape_pos(0, -40, 64),
                        drape_pos(0, 0, 65), 31*parameters["scale"]),
            QualityShot("drapery-threequarter", drape_pos(26, -35, 64),
                        drape_pos(0, 0, 65), 31*parameters["scale"]),
            QualityShot("drapery-profile", drape_pos(43, 0, 64),
                        drape_pos(0, 0, 65), 31*parameters["scale"]),
            QualityShot("drapery-raised-sleeve", drape_pos(-13, -24, 72),
                        drape_pos(-5, 0, 73), 13.5*parameters["scale"]),
        )
    elif parameters.get("detailed_drapery"):
        def drape_pos(x: float, y: float, z: float) -> tuple[float, float, float]:
            return (x*parameters["scale"], y*parameters["scale"],
                    (z+offset)*parameters["scale"])
        shots += (
            QualityShot("drapery-front", drape_pos(0, -22, 16.9),
                        drape_pos(0, 0, 17.0), 18.2*parameters["scale"]),
            QualityShot("drapery-threequarter", drape_pos(14.4, -18.2, 17.2),
                        drape_pos(0, 0, 17.0), 18.2*parameters["scale"]),
            QualityShot("drapery-profile", drape_pos(22, 0, 17.0),
                        drape_pos(0, 0, 17.0), 18.2*parameters["scale"]),
            QualityShot("drapery-raised-sleeve", drape_pos(-5.8, -9.2, 24.3),
                        drape_pos(-2.33, -.04, 23.6), 5.4*parameters["scale"]),
        )
    metadata = {
        "asset_id": study.asset_id,
        "generator": "kumiki-blender/procedural-statue-of-liberty",
        "status": "reference-driven sculpture / requires human likeness review",
        "provenance": "Original procedural sculpture derived from user-supplied photos; not a surveyed photogrammetry scan",
        "render_profile": {"name": profile.name, "width": profile.width,
                           "height": profile.height, "samples": profile.samples},
        "units": ("meters (NPS dimension-constrained approximation)"
                  if measured_frame else "meters (illustrative proportions)"),
        "native_axes": "Blender Z-up, front -Y",
        "named_parts": dict(study.part_counts),
        "bounds_m": [study.bounds_min, study.bounds_max],
        "parameters": parameters,
        "detailed_torch": detailed_torch,
        "detailed_tablet": bool(parameters.get("detailed_tablet")),
        "detailed_crown": bool(parameters.get("detailed_crown")),
        "detailed_feet": bool(parameters.get("detailed_feet")),
        "detailed_drapery": bool(parameters.get("detailed_drapery")),
        "monument_proportions": measured_frame,
        "detailed_torch_fit": ({"origin": torch_fit.origin,
                               "radial_scale": torch_fit.radial_scale,
                               "height_scale": torch_fit.height_scale}
                              if torch_fit else None),
        "detail_cameras": [shot.name for shot in shots],
    }
    return export_revision(bpy, scene, camera, objects, shots, destination,
                           samples=profile.samples, metadata=metadata)
