"""Generate *unapproved* facial-sculpt anchors from independent photographs.

PYTHONPATH=src python scripts/fit_face_geometry.py front.json oblique.json \
    --output .review/christ_redeemer/face-corrections-HOLD.json

Each input is the output of probe_face_anchor_pixels.py, with real-photo
landmarks, a shared model_revision, distinct source-photo URLs, and >=12
features. Each camera fit withholds the feature it is triangulating.
"""
import argparse
import json
from pathlib import Path

from kumiki_blender.face_reconstruction import propose_face_corrections


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",type=Path,nargs="+",help="At least two independent reference-photo annotation JSON files")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-camera-rms-px",type=float,default=5.)
    parser.add_argument("--max-target-error-px",type=float,default=3.)
    args=parser.parse_args()
    if len(args.inputs)<2:
        parser.error("At least two independent photographs required")
    views=[]
    for path in args.inputs:
        source=json.loads(path.read_text())
        views.append({"view_id":source.get("reference_view_id",path.stem),
                      "subject_photo_url":source.get("subject_photo_url"),
                      "model_revision":source.get("model_revision"),
                      "image_size":source["image_size"],
                      "landmarks":source["landmarks"]})
    result=propose_face_corrections(views,max_camera_rms_px=args.max_camera_rms_px,
                                    max_target_error_px=args.max_target_error_px)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("FACIAL_SCULPT_CANDIDATES "+json.dumps({
        "status":result["status"],"candidates":result["candidate_count"],
        "held":result["held_count"],"output":str(args.output)}))


if __name__=='__main__':
    main()
