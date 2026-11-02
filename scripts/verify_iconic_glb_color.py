"""Independently re-import and render the colorful iconic Fuji GLB.

Usage:
  blender --background REV/source.blend --python scripts/verify_iconic_glb_color.py \
    -- --root REV

Native meshes/materials are removed before importing to ensure all pigment
comes from the exported GLB's embedded standard PBR base-color textures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kumiki_blender.geography.fuji.iconic_scene import plan_iconic_shots


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    root=args.root.resolve()
    scene=bpy.context.scene
    camera=scene.camera
    if camera is None:
        raise RuntimeError("Expected the saved iconic Fuji studio camera")
    original=[obj for obj in scene.objects if obj.type=="MESH" and
              isinstance(obj.get("kumiki_part"),str)]
    if len(original)<20:
        raise RuntimeError(f"Expected original iconic source meshes, got {len(original)}")
    for obj in original:
        bpy.data.objects.remove(obj,do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material,do_unlink=True)
    for image in list(bpy.data.images):
        if image.source!="VIEWER":
            bpy.data.images.remove(image,do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=str(root/"model.glb"))
    imported=[obj for obj in scene.objects if obj.type=="MESH" and
              isinstance(obj.get("kumiki_part"),str)]
    if len(imported)<20:
        raise RuntimeError("GLB re-import lost most iconic geometry")
    targets=("Japan / moss, volcanic scree and spring garden",
             "Japan / blue-grey volcanic Fuji ridges",
             "Japan / alpine snow ivory and blue",
             "Japan / multi-hue sakura petals")
    embedded={}
    for name in targets:
        materials=[mat for mat in bpy.data.materials if mat.name==name]
        if len(materials)!=1:
            raise RuntimeError(f"Missing unique iconic palette material {name}")
        images=[n.image for n in materials[0].node_tree.nodes if
                n.type=="TEX_IMAGE" and n.image is not None]
        if len(images)!=1 or not images[0].packed_file:
            raise RuntimeError(f"GLB failed to decode packed albedo for {name}")
        _=images[0].pixels[0]
        if not images[0].has_data or min(images[0].size)<128:
            raise RuntimeError(f"Invalid packed albedo for {name}")
        embedded[name]=list(images[0].size)
    scene.cycles.samples=16
    scene.render.image_settings.file_format="PNG"
    shot_map={shot.name:shot for shot in plan_iconic_shots()}
    rendered=[]
    for shot_name,output in (("hero","glb-reimport-preview.png"),
                             ("sakura-botanical","glb-reimport-sakura-close.png"),
                             ("snow-crown","glb-reimport-fuji-close.png")):
        shot=shot_map[shot_name]
        camera.location=shot.position
        camera.rotation_euler=(Vector(shot.target)-camera.location).to_track_quat("-Z","Y").to_euler()
        camera.data.ortho_scale=shot.ortho_scale
        scene.render.filepath=str(root/output)
        bpy.ops.render.render(write_still=True,scene=scene.name)
        rendered.append(output)
    print("FUJI_ICONIC_GLB_COLOR_REIMPORT="+json.dumps({
        "mesh_objects":len(imported),"embedded_albedos":embedded,"rendered":rendered}))


if __name__=="__main__":
    main()
