"""Blender 4.2+ adapter: editable red biplane source, rig and cinematic QA.

The geometry is authored in vintage_biplane.py and remains testable without
bpy. This module exclusively handles materials, scene organization, animation,
lighting, shot composition and revision export.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .blender_output import create_model_scene, export_revision
from .blender_scene import _attach_mesh, _material
from .render_profiles import select_render_profile
from .shots import QualityShot
from .surfaces import get_surface
from .vintage_biplane import BiplaneModel, animation_keyframes


def plan_biplane_shots() -> tuple[QualityShot, ...]:
    """True reference-like front, rear, side and top inspection cameras."""
    return (
        QualityShot("hero", (12, -15, 8.3), (0, -.18, 1.47), 13.9),
        QualityShot("front", (0, -18, 2.12), (0, -.1, 1.52), 11.6),
        QualityShot("side", (18, 0, 2.18), (0, -.12, 1.46), 9.1),
        QualityShot("top", (0, -.30, 19), (0, -.30, 1.48), 15.3),
        QualityShot("rear", (0, 18, 2.23), (0, .05, 1.56), 11.6),
        QualityShot("cockpit", (5.2, -.16, 3.55), (0, -.54, 1.56), 4.15),
    )


def _palette(bpy: Any) -> dict[str, Any]:
    """Pigmented linen, oxblood enamel, aged steel, black rubber and oiled wood.

    Generated PBR maps stay packed in .blend and GLB and require no downloaded
    texture licenses. Each material is authored once and reused across parts.
    """
    specs = {
        "fabric-red": ("painted-steel", "#AE211C", .65),
        "rib-red": ("painted-steel", "#B92C23", .69),
        "deep-red": ("painted-steel", "#78211D", .42),
        "red-lacquer": ("painted-steel", "#AC251D", .40),
        "cockpit-dark": ("studio-matte", "#242222", .95),
        "engine-dark": ("painted-steel", "#353A3E", .65),
        "aged-steel": ("galvanized-steel", "#777772", .56),
        "brass": ("galvanized-steel", "#C19A5B", .32),
        "varnished-wood": ("timber", "#A56A3D", .42),
        "prop-wood": ("timber", "#C58F55", .32),
        "aged-rubber": ("studio-matte", "#17191B", .94),
        "saddle-leather": ("studio-matte", "#422D22", .82),
        "cable-steel": ("galvanized-steel", "#54585A", .59),
        "windscreen-glass": ("optical-glass", "#C3DAE4", .13),
    }
    return {role: _material(bpy, get_surface(preset, color=color, roughness=roughness))
            for role, (preset, color, roughness) in specs.items()}


def _rig_controllers(bpy: Any, model_collection: Any,
                     model: BiplaneModel) -> tuple[Any, dict[str, Any]]:
    """Every independent hinge rotates about its actual world-space axis."""
    root = bpy.data.objects.new("RIG / aircraft-root", None)
    model_collection.objects.link(root)
    root.empty_display_type = "CIRCLE"
    root.empty_display_size = .8
    root["kumiki_rig"] = "aircraft-root"
    controls = {}
    for channel, point in model.pivots.items():
        handle = bpy.data.objects.new(f"RIG / {channel}", None)
        model_collection.objects.link(handle)
        handle.parent = root
        handle.location = point
        handle.empty_display_type = "ARROWS"
        handle.empty_display_size = .30
        handle["kumiki_rig"] = channel
        handle["kumiki_control_axis"] = ("Y" if channel == "propeller" else
                                           "Z" if channel == "rudder" else "X")
        controls[channel] = handle
    return root, controls


def _animate_controls(bpy: Any, scene: Any, controls: dict[str, Any]) -> None:
    scene.frame_start, scene.frame_end = 1, 97
    scene.render.fps = 24
    channels = animation_keyframes()
    for channel, keys in channels.items():
        obj = controls[channel]
        axis = 1 if channel == "propeller" else 2 if channel == "rudder" else 0
        offset = math.radians(35) if channel == "propeller" else 0.
        obj.rotation_mode = "XYZ"
        for frame, angle in keys:
            obj.rotation_euler[axis] = angle + offset
            obj.keyframe_insert(data_path="rotation_euler", index=axis, frame=frame,
                                group="Airplane mechanical controls")
        action = obj.animation_data.action
        if action is not None:
            action.name = f"Kumiki / {channel} / flight-loop"
            # Blender 4.5 uses slotted actions and can store the curve in a
            # channelbag rather than the legacy action.fcurves collection.
            curves = []
            if hasattr(action, "fcurves"):
                curves = list(action.fcurves)
            if not curves and hasattr(action, "layers") and action.layers:
                for strip in action.layers[0].strips:
                    if obj.animation_data.action_slot is not None:
                        bag = strip.channelbag(obj.animation_data.action_slot, ensure=False)
                        if bag is not None:
                            curves.extend(bag.fcurves)
            for curve in curves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR" if channel in (
                        "propeller", "left-wheel", "right-wheel") else "BEZIER"
        obj["kumiki_animation"] = "flight-loop / 96 frames @ 24fps"
    scene.frame_set(1)


def _stage(bpy: Any, scene: Any, collection: Any) -> Any:
    floor = _material(bpy, get_surface("studio-matte", color="#A9B1B7", roughness=.91))
    # A single planar cyc surface cannot show the dark edge of a giant cube.
    mesh = bpy.data.meshes.new("STUDIO / matte infinity ground")
    mesh.from_pydata([(-180, -180, -.022), (180, -180, -.022),
                      (180, 180, -.022), (-180, 180, -.022)], [], [(0, 1, 2, 3)])
    mesh.update()
    mesh.materials.append(floor)
    collection.objects.link(bpy.data.objects.new("STUDIO / seamless floor", mesh))
    world = bpy.data.worlds.new("STUDIO / balanced neutral ambient")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (.52, .56, .60, 1.)
    background.inputs["Strength"].default_value = .78
    scene.world = world

    def area(name: str, location: tuple[float, float, float],
             power: float, size: float, color: tuple[float, float, float]):
        lamp_data = bpy.data.lights.new(name, type="AREA")
        lamp_data.energy, lamp_data.shape = power, "DISK"
        lamp_data.size, lamp_data.color = size, color
        lamp = bpy.data.objects.new(name, lamp_data)
        collection.objects.link(lamp)
        lamp.location = location
        from .blender_output import aim_at
        aim_at(lamp, (0, -.1, 1.35))

    area("STUDIO / broad warm key", (-5, -7, 10), 1450, 8, (1, .91, .83))
    area("STUDIO / cool wing fill", (6, -1, 7), 1000, 9, (.80, .89, 1))
    area("STUDIO / rear enamel rim", (0, 7, 8), 1300, 7, (1, .94, .88))
    camera_data = bpy.data.cameras.new("STUDIO / six-view inspection camera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    collection.objects.link(camera)
    scene.camera = camera
    return camera


def build_biplane_scene(model: BiplaneModel, destination: Path, *,
                        bpy_module: Any = None) -> dict[str, Any]:
    if bpy_module is None:
        raise RuntimeError("Biplane builds require Blender bpy")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError("Blender 4.2 or newer is required")
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("Choose a new output directory; revisions are immutable")

    profile = select_render_profile(1440, 810, 72)
    scene = create_model_scene(bpy, model.asset_id)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile.samples
    scene.render.resolution_x = profile.width
    scene.render.resolution_y = profile.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -.15
    scene.render.film_transparent = False
    model_collection = bpy.data.collections.new("AIRCRAFT / exportable rig and geometry")
    scene.collection.children.link(model_collection)
    studio_collection = bpy.data.collections.new("STUDIO / excluded from GLB")
    scene.collection.children.link(studio_collection)
    materials = _palette(bpy)
    root, controls = _rig_controllers(bpy, model_collection, model)

    objects = []
    for part in model.parts:
        # Mesh coordinates are world-space in pure Python. A rigged mesh stores
        # them relative to its hinge; the rest remain relative to the root.
        pivot = model.pivots[part.rig] if part.rig else (0., 0., 0.)
        local = [tuple(vertex[i]-pivot[i] for i in range(3))
                 for vertex in part.mesh.vertices]
        obj = _attach_mesh(bpy, model_collection, part.name, local,
                           list(part.mesh.faces), materials[part.material])
        obj.parent = controls[part.rig] if part.rig else root
        obj["kumiki_part"] = part.group
        obj["kumiki_material_role"] = part.material
        if part.rig:
            obj["kumiki_rig_parent"] = part.rig
        # Fabric, cowl and molded propeller benefit from continuous normals;
        # their closed caps stay flat, preventing edge glow and melted tips.
        if part.group in ("wing", "aileron", "elevator", "tail", "fuselage",
                          "propeller-blade", "wheel"):
            for polygon in obj.data.polygons:
                polygon.use_smooth = len(polygon.vertices) == 4
        objects.append(obj)

    _animate_controls(bpy, scene, controls)
    camera = _stage(bpy, scene, studio_collection)
    metadata = {
        "asset_id": model.asset_id,
        "generator": "kumiki-blender/red-vintage-biplane",
        "status": "original illustrated-aircraft interpretation / rendered QA required",
        "provenance": "Geometry authored from user-supplied multiview red airplane illustration; non-measured, no external mesh assets",
        "units": "meters", "native_axes": "Blender Z-up; nose -Y, wings +/-X",
        "named_parts": model.part_counts,
        "bounds_m": [model.bounds_min, model.bounds_max],
        "rig_controls": {name: list(pivot) for name, pivot in model.pivots.items()},
        "animation": {"name": "flight-loop", "start_frame": 1, "end_frame": 97,
                      "fps": 24, "channels": len(controls)},
        "render_profile": {"name": profile.name, "width": profile.width,
                           "height": profile.height, "samples": profile.samples},
    }
    return export_revision(bpy, scene, camera, objects, plan_biplane_shots(),
                           destination, samples=profile.samples, metadata=metadata,
                           rig_objects=(root, *controls.values()))
