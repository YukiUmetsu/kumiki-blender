"""Fit a provisional perspective camera from actual-subject photo landmarks.

Input JSON contains `image_size` and `landmarks`: each with a unique `name`,
`world_xyz` from the sculpt, and `photo_uv` from an actual monument photograph.
No copyrighted photograph is embedded in the JSON or resulting report.

    PYTHONPATH=src python scripts/fit_reference_camera.py anchors.json --output fit.json
"""

import argparse
import json
from pathlib import Path

from kumiki_blender.camera_registration import fit_landmarks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--max-error-px",type=float,default=12.)
    args=parser.parse_args()
    if args.max_error_px<=0:
        parser.error("--max-error-px must be positive")
    data=json.loads(args.input.read_text())
    result=fit_landmarks(data["landmarks"],data["image_size"],
                         max_error_px=args.max_error_px)
    result["subject_photo_url"]=data.get("subject_photo_url")
    result["model_revision"]=data.get("model_revision")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("CAMERA_FIT "+json.dumps({k:v for k,v in result.items()
                                   if k in ("status","rms_reprojection_px",
                                            "max_reprojection_px","success")}))


if __name__=="__main__":
    main()
