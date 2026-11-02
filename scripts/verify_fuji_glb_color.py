"""Render a Fuji GLB after *re-import*, using the source studio's camera.

This catches a material export that looks colored in source.blend but becomes
white or too dark when opened in a glTF viewer. It is an inspection render,
not a geometry-generation step; the existing source.blend is never modified.

    blender --background revision/source.blend --python scripts/verify_fuji_glb_color.py \
      -- --root revision
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    root = args.root.resolve()
    scene = bpy.context.scene
    if scene.camera is None:
        raise RuntimeError("Source .blend must preserve the quality-review camera")
    prior = [obj for obj in scene.objects if obj.get("kumiki_part") in
             ("terrain-surface", "terrain-perimeter", "terrain-pedestal")]
    if len(prior) != 3:
        raise RuntimeError(f"Expected three native Fuji meshes, found {len(prior)}")
    for obj in prior:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    # A loaded source.blend already owns materials and packed images with the
    # same names. Purge them before import to inspect the *GLB's* images, not
    # a stale or lazily unloaded duplicate from the original Blender scene.
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)
    for image in list(bpy.data.images):
        if image.source not in ("VIEWER",):
            bpy.data.images.remove(image, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=str(root / "model.glb"))
    imported = [obj for obj in scene.objects if obj.type == "MESH" and
                obj.get("kumiki_part") == "terrain-surface"]
    if len(imported) != 1:
        raise RuntimeError("Re-imported GLB is missing the named Fuji terrain")
    material = imported[0].active_material
    albedo = [node.image for node in material.node_tree.nodes
              if node.type == "TEX_IMAGE" and node.image is not None and
              "albedo" in node.image.name.lower()]
    if len(albedo) != 1 or not albedo[0].packed_file:
        raise RuntimeError("Re-imported Fuji terrain has no decoded color texture")
    # Packed FILE images can have has_data=False until the first pixel read;
    # force decoding instead of falsely rejecting lazy-loaded embedded data.
    _ = albedo[0].pixels[0]
    if not albedo[0].has_data:
        raise RuntimeError("Embedded Fuji albedo PNG could not be decoded")
    scene.render.filepath = str(root / "glb-reimport-preview.png")
    scene.render.image_settings.file_format = "PNG"
    scene.cycles.samples = 16
    bpy.ops.render.render(write_still=True, scene=scene.name)
    print("FUJI_GLB_COLOR_REIMPORT=" + json.dumps({
        "mesh_vertices": len(imported[0].data.vertices),
        "albedo_size": list(albedo[0].size),
        "albedo_image": albedo[0].name,
        "render": scene.render.filepath,
    }))


if __name__ == "__main__":
    main()
