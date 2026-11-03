"""Run within Blender against an audited, saved native `.blend` physics source.

blender --background source.blend --python scripts/blender_bridge_parity_render.py -- \
  --plan parity-plan.json --model model.glb --output new-native-capture
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bpy  # noqa: E402
from mathutils import Quaternion, Vector  # noqa: E402
from kumiki_blender.bridge_parity import (BridgeParityError, camera_pose, frame_name,
                                          load_plan, make_receipt, sha256)  # noqa: E402


def render(plan_path: Path, model_path: Path, output: Path) -> None:
    plan = load_plan(plan_path)
    if sha256(model_path) != plan["model_sha256"]:
        raise BridgeParityError("Model hash does not match capture plan")
    if not bpy.data.filepath:
        raise BridgeParityError("Load the native physics source .blend before capturing parity")
    if output.is_symlink() or output.exists():
        raise BridgeParityError("Native capture destination must be a new directory")
    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and not obj.hide_render]
    if not meshes:
        raise BridgeParityError("No visible native mesh available for silhouette capture")

    # Emit one flat signal for all meshes, independent of PBR, AO, textures and lighting.
    silhouette = bpy.data.materials.new("Kumiki-parity-emission")
    silhouette.use_nodes = True
    nodes = silhouette.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1, 1, 1, 1)
    emission.inputs["Strength"].default_value = 1.0
    output_node = nodes.new("ShaderNodeOutputMaterial")
    silhouette.node_tree.links.new(emission.outputs["Emission"], output_node.inputs["Surface"])
    # Preserve evaluated geometry and rigid-body transforms; swapping slots for this
    # unsaved background session cannot change the source .blend.
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(silhouette)

    camera = bpy.data.cameras.new("Kumiki-parity-camera")
    camera_obj = bpy.data.objects.new("Kumiki-parity-camera", camera)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    position, target, fov, roll = camera_pose(plan)
    camera_obj.location = position
    camera_obj.rotation_mode = "QUATERNION"
    camera_obj.rotation_quaternion = (Vector(target) - camera_obj.location).to_track_quat("-Z", "Y") @ Quaternion((0, 0, 1), math.radians(roll))
    camera.type = "PERSP"
    camera.sensor_fit = "VERTICAL"
    camera.lens = camera.sensor_height / (2 * math.tan(math.radians(fov) / 2))
    camera.clip_start, camera.clip_end = .001, 100000

    world = bpy.data.worlds.new("Kumiki-parity-black")
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Color"].default_value = (0, 0, 0, 1)
    world.node_tree.nodes.get("Background").inputs["Strength"].default_value = 0
    scene.world = world
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x, scene.render.resolution_y = plan["width"], plan["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    # White emission and black background stay binary even with color management.
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    output.mkdir(parents=True, exist_ok=False)
    # Bullet simulation is stateful. Advance strictly forward from the cache start,
    # not by jumping directly to sample frames, to avoid stale physics poses.
    original = scene.frame_start
    first = min(original, plan["frames"][0])
    for frame in range(first, plan["frames"][-1] + 1):
        scene.frame_set(frame)
        if frame in plan["frames"]:
            scene.render.filepath = str(output / frame_name(frame))
            bpy.ops.render.render(write_still=True)
    receipt = make_receipt(plan_path, model_path, output, "blender")
    receipt["source_blend_sha256"] = sha256(Path(bpy.data.filepath))
    (output / "capture.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("KUMIKI_BRIDGE_PARITY_NATIVE_OK", output)


if __name__ == "__main__":
    params = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(params)
    render(args.plan.resolve(), args.model.resolve(), args.output)
