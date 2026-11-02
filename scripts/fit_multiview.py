"""Cross-check independent real-subject photos against one unchanged 3D head.

  PYTHONPATH=src python scripts/fit_multiview.py \
    front-anchors-3d.json oblique-anchors-3d.json \
    --output .review/christ_redeemer/multiview-fit.json

Input JSONs come from probe_face_anchor_pixels.py. This diagnostic fits
separate perspective cameras and identifies persistent pixel residuals. It is
NOT photogrammetric bundle adjustment and does not edit the 3D sculpture.
"""

import argparse
import json
from pathlib import Path

from kumiki_blender.multiview_registration import assess_multiview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",type=Path,nargs="+",help="Two or more 3D anchor JSONs")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-error-px",type=float,default=12.)
    args=parser.parse_args()
    if len(args.inputs)<2 or args.max_error_px<=0:
        parser.error("Need two or more independent views and a positive threshold")
    views=[]
    for path in args.inputs:
        data=json.loads(path.read_text())
        views.append({"view_id":data.get("reference_view_id",path.stem),
                      "model_revision":data.get("model_revision"),
                      "subject_photo_url":data.get("subject_photo_url"),
                      "image_size":data["image_size"],
                      "landmarks":data["landmarks"]})
    report=assess_multiview(views,max_error_px=args.max_error_px)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print("MULTIVIEW_FIT "+json.dumps({"status":report["status"],
                                       "views":report["view_count"],
                                       "shared":report["common_landmark_count"]}))


if __name__=="__main__":
    main()
