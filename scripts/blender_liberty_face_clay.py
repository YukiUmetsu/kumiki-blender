"""Quick, neutral-clay facial likeness gate for iterative Liberty sculpting.

Render only the closed face and ears; the full detailed statue takes much
longer to export and can hide anatomy behind hair and the crown. This is an
inspection tool, not a replacement for the final all-angle GLB audit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import bpy
    from mathutils import Vector

    from kumiki_blender.liberty import make_liberty_study

    args = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=640)
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

    data = bpy.data.cameras.new("Fixed clay inspection orthographic")
    data.type = "ORTHO"
    data.ortho_scale = 4.10
    camera = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    shots = (
        ("front", (0., -8., 26.69)),
        ("threequarter", (5.8, -7.4, 26.69)),
        ("profile", (8., -.12, 26.69)),
    )
    for name, position in shots:
        camera.location = position
        vector = Vector((0., 0., 26.69))-camera.location
        camera.rotation_euler = vector.to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(cfg.output / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(cfg.output / "clay.blend"))
    print(f"Neutral facial comparison views: {cfg.output}")


if __name__ == "__main__":
    main()
