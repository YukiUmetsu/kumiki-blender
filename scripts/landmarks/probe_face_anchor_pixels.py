"""Resolve model screenshot landmarks to actual head-mesh 3D surface points.

Run in Blender with the experimental `.blend` opened. The input is an image
annotation JSON with `model_uv` for pixels on a square orthographic QA render
and `photo_uv` for the matching real monument photo points. The output gets
Blender world_xyz ray casts, suitable for provisional camera fitting.

    blender -b head.blend --python scripts/landmarks/probe_face_anchor_pixels.py \
      -- --input annotations.json --output fitted-anchors.json

This is **not** a landmark detector: user annotations must be checked by eye.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def main():
    args=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--ortho-size-m",type=float,default=6.4)
    parser.add_argument("--head-center-z-m",type=float,default=36.0)
    parser.add_argument("--render-size-px",type=int,default=700)
    options=parser.parse_args(args)
    source=json.loads(options.input.read_text())
    head=next((obj for obj in bpy.data.objects if
               obj.name.startswith("Head | anatomical quad-topology")),None)
    if head is None:
        raise ValueError("Load the anatomical clay .blend before probing anchors")
    transform=head.matrix_world.copy()
    inverse=transform.inverted()
    output=[]
    for entry in source["landmarks"]:
        px,py=entry["model_uv"]
        x=(px/options.render_size_px-.5)*options.ortho_size_m
        z=options.head_center_z_m+(.5-py/options.render_size_px)*options.ortho_size_m
        origin=inverse@Vector((x,-100.,z))
        direction=(inverse.to_3x3()@Vector((0,1,0))).normalized()
        success, hit, _normal, _face=head.ray_cast(origin,direction)
        if not success:
            raise ValueError("No head-mesh surface under %s at %s"%
                             (entry["name"],entry["model_uv"]))
        output.append({"name":entry["name"],
                       "world_xyz":[round(float(v),6) for v in transform@hit],
                       "photo_uv":entry["photo_uv"]})
    report={"model_revision":source.get("model_revision"),
            "subject_photo_url":source.get("subject_photo_url"),
            "image_size":source["image_size"],"landmarks":output}
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(report,indent=2)+"\n")
    print("HEAD_ANCHORS "+json.dumps({"count":len(output),
                                      "output":str(options.output)}))


if __name__=="__main__":
    main()
