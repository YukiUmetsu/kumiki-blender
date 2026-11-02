"""Quick, neutral-clay facial likeness gate for iterative Liberty sculpting.

Render only the closed face and ears; the full detailed statue takes much
longer to export and can hide anatomy behind hair and the crown. This is an
inspection tool, not a replacement for the final all-angle GLB audit.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# Preserve all sculpt-inspection poses in the saved .blend. Profile faces left;
# opposite-profile faces right. These are *not* calibrated HAER cameras.
FIXED_CLAY_SHOTS = (
    ("front", (0., -8., 26.69)),
    ("threequarter", (5.8, -7.4, 26.69)),
    ("profile", (8., -.12, 26.69)),
    ("opposite-profile", (-8., -.12, 26.69)),
)


def main() -> None:
    import bpy
    from mathutils import Vector

    from kumiki_blender.liberty import make_liberty_study

    args = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--measurement-guides", action="store_true",
                        help="Annotate the frontal clay view with scaled NPS dimensions")
    cfg = parser.parse_args(args)
    if not 256 <= cfg.resolution <= 2048:
        parser.error("resolution must be in [256, 2048]")
    if cfg.output.exists() and any(cfg.output.iterdir()):
        parser.error("output revision already exists; choose a new checkpoint path")
    cfg.output.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.render.resolution_x = cfg.resolution
    scene.render.resolution_y = cfg.resolution
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"

    clay = bpy.data.materials.new("Neutral clay / single anatomy surface")
    clay.diffuse_color = (.55, .64, .61, 1.)
    clay.use_nodes = True
    principled = clay.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (.55, .64, .61, 1.)
    principled.inputs["Roughness"].default_value = .82

    study = make_liberty_study(include_pedestal=True, detailed_face=True)
    selected = {"liberty-head-face", "liberty-head-left-ear", "liberty-head-right-ear"}
    for part in study.instances:
        if part.name not in selected:
            continue
        mesh = bpy.data.meshes.new(part.name)
        mesh.from_pydata(part.mesh.vertices, [], part.mesh.faces)
        mesh.update()
        mesh.materials.append(clay)
        obj = bpy.data.objects.new(part.name, mesh)
        scene.collection.objects.link(obj)
        for polygon in mesh.polygons:
            polygon.use_smooth = True

    world = bpy.data.worlds.new("Clay neutral grey world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.25, .29, .30, 1.)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .65
    scene.world = world

    def area(name: str, location: tuple[float, float, float], power: int, size: float) -> None:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = power
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        direction = Vector((0., 0., 26.65))-obj.location
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    area("Large soft key", (-5., -6., 30.), 850, 4.0)
    area("Narrow raking fill", (4., -1.8, 28.5), 440, 3.0)

    guides = []
    if cfg.measurement_guides:
        from kumiki_blender.liberty_face_measurements import scaled_nps_head_targets

        target = scaled_nps_head_targets()
        scene["nps_reference_head_width_m"] = target.width
        scene["nps_reference_eye_width_m"] = target.eye_width
        scene["nps_reference_mouth_width_m"] = target.mouth_width
        scene["nps_reference_nose_length_m"] = target.nose_length
        scene["measurement_warning"] = "Scaled ratios only; front/profile cameras not calibrated"

        ink = bpy.data.materials.new("Reference guides / deep umber")
        ink.diffuse_color = (.22, .12, .045, 1.)
        ink.use_nodes = True
        ink.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
            .22, .12, .045, 1.)

        def line(name: str, p1: tuple[float, float, float],
                 p2: tuple[float, float, float]) -> None:
            curve = bpy.data.curves.new(name, "CURVE")
            curve.dimensions = "3D"
            curve.bevel_depth = .011
            curve.bevel_resolution = 3
            spline = curve.splines.new("POLY")
            spline.points.add(1)
            for index, point in enumerate((p1, p2)):
                spline.points[index].co = (*point, 1.)
            obj = bpy.data.objects.new(name, curve)
            obj.data.materials.append(ink)
            scene.collection.objects.link(obj)
            guides.append(obj)

        def annotate(label: str, x: float, z: float) -> None:
            font = bpy.data.curves.new("Measurement / " + label, "FONT")
            font.body = label
            font.size = .115
            obj = bpy.data.objects.new("Measurement / " + label, font)
            scene.collection.objects.link(obj)
            obj.location = (x, -1.73, z)
            obj.rotation_euler = (math.pi/2, 0, 0)
            obj.data.materials.append(ink)
            guides.append(obj)

        def bar(label: str, x0: float, x1: float, z: float, caption_x: float) -> None:
            line(label, (x0, -1.71, z), (x1, -1.71, z))
            for x in (x0, x1):
                line(label + " tick", (x, -1.71, z-.05), (x, -1.71, z+.05))
            annotate(label, caption_x, z-.20)

        bar(f"NPS head {target.width:.2f}m", -target.width/2,
            target.width/2, 24.76, -.92)
        for sign, word in ((-1, "left"), (1, "right")):
            cx = sign*.43
            bar(f"{word} eye {target.eye_width:.2f}m", cx-target.eye_width/2,
                cx+target.eye_width/2, 27.22, cx-.23)
        bar(f"NPS mouth {target.mouth_width:.2f}m", -target.mouth_width/2,
            target.mouth_width/2, 25.78, -.48)
        line("NPS nose length", (1.27, -1.71, 27.40),
             (1.27, -1.71, 27.40-target.nose_length))
        annotate(f"nose {target.nose_length:.2f}m", 1.31, 26.57)

    cameras = {}
    for name, position in FIXED_CLAY_SHOTS:
        data = bpy.data.cameras.new(f"Clay QA / {name} / orthographic-unregistered")
        data.type = "ORTHO"
        data.ortho_scale = 4.70 if cfg.measurement_guides else 4.10
        camera = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(camera)
        camera.location = position
        camera.rotation_euler = (Vector((0., 0., 26.69)) - camera.location).to_track_quat(
            "-Z", "Y").to_euler()
        camera["correspondence"] = "Illustrative orthographic, not photo-registered"
        cameras[name] = camera

    for name, _position in FIXED_CLAY_SHOTS:
        for obj in guides:
            obj.hide_render = name != "front"
        scene.camera = cameras[name]
        scene.render.filepath = str(cfg.output / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    for obj in guides:
        obj.hide_render = False
    scene.camera = cameras["front"]
    bpy.ops.wm.save_as_mainfile(filepath=str(cfg.output / "clay.blend"))
    print(f"Neutral facial comparison views: {cfg.output}")


if __name__ == "__main__":
    main()
