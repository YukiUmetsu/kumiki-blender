"""Blender worker: reference cameras, image empties and blockout; optional silhouette renders.

Arguments (JSON): handoff, output, render (bool), include_blockout (bool), save (bool).
Opened on a user .blend, it adds the references to that file's scene. With
``render`` it renders each REF_CAM camera to ``output/<view>.png`` (Workbench,
flat, transparent), the input of ``kumiki-character-build compare``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import (arguments, clear_startup_scene, import_mesh,  # noqa: E402
                                                           write_report)

COLLECTION = "Character References"


def main() -> None:
    args = arguments()
    root, output = Path(args["handoff"]).resolve(), Path(args["output"]).resolve()
    spec = json.loads((root / "spec.json").read_text(encoding="utf-8"))
    clear_startup_scene()
    scene = bpy.context.scene
    group = bpy.data.collections.get(COLLECTION)
    if group is None:
        group = bpy.data.collections.new(COLLECTION)
        scene.collection.children.link(group)
    scene.unit_settings.system = "METRIC"
    scene.render.resolution_x, scene.render.resolution_y = spec["frame"]["canvas"]
    scene.render.resolution_percentage = 100

    def place(obj, data):
        obj.location = data["location"]
        obj.rotation_euler = [math.radians(v) for v in data["rotation_deg"]]

    for name, cam in spec["cameras"].items():
        obj = bpy.data.objects.get(f"REF_CAM_{name}")
        if obj is None:
            obj = bpy.data.objects.new(f"REF_CAM_{name}", bpy.data.cameras.new(f"REF_CAM_{name}"))
            group.objects.link(obj)
        obj.data.type, obj.data.sensor_fit = "ORTHO", "AUTO"
        obj.data.ortho_scale = cam["ortho_scale"]
        obj.data.clip_end = 20 * spec["height_m"]
        place(obj, cam)
        obj.data.background_images.clear()
        if cam["image"]:
            background = obj.data.background_images.new()
            background.image = bpy.data.images.load(str(root / cam["image"]), check_existing=True)
            background.alpha, background.frame_method = 0.5, "STRETCH"
            obj.data.show_background_images = True

    for name, ref in spec["references"].items():
        obj = bpy.data.objects.get(f"REF_IMG_{name}")
        if obj is None:
            obj = bpy.data.objects.new(f"REF_IMG_{name}", None)
            group.objects.link(obj)
        obj.empty_display_type = "IMAGE"
        obj.data = bpy.data.images.load(str(root / ref["image"]), check_existing=True)
        obj.empty_display_size = ref["size"]
        obj.empty_image_offset = (-0.5, -0.5)
        obj.empty_image_depth, obj.empty_image_side = "BACK", "FRONT"
        obj.show_empty_image_only_axis_aligned = True
        obj.use_empty_image_alpha, obj.color[3] = True, 0.6
        obj.hide_render = True
        place(obj, ref)

    blockout = bpy.data.objects.get("REF_Blockout")
    if blockout is None and args.get("include_blockout", True):
        blockout = import_mesh(str(root / spec["blockout"]["file"]), "REF_Blockout")
        for owner in list(blockout.users_collection):
            owner.objects.unlink(blockout)
        group.objects.link(blockout)
    if blockout is not None:
        blockout.hide_render = not args.get("render_blockout", False)

    rendered = []
    if args.get("render"):
        output.mkdir(parents=True, exist_ok=True)
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.film_transparent = True
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "SINGLE"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        for name in spec["cameras"]:
            if spec["cameras"][name]["image"] is None:
                continue
            scene.camera = bpy.data.objects[f"REF_CAM_{name}"]
            scene.render.filepath = str(output / f"{name}.png")
            bpy.ops.render.render(write_still=True)
            rendered.append(name)
    saved = None
    if args.get("save"):  # always a copy in the output folder; a loaded user .blend is never written
        output.mkdir(parents=True, exist_ok=True)
        saved = str(output / "references.blend")
        bpy.ops.wm.save_as_mainfile(filepath=saved, copy=bool(bpy.data.filepath))
    write_report(output, {"rendered": rendered, "saved": saved, "cameras": sorted(spec["cameras"]),
                          "note": "Blender's numpad 3 Right view shows the character's LEFT side."})


main()
