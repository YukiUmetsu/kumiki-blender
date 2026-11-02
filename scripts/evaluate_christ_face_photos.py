"""Compare GLB head revisions against hand-annotated real-monument photos.

No photo bytes are redistributed. Reports provisional 2D/3D reprojection
residuals, NOT metric scan accuracy. Both GLBs must share vertex topology.

  python scripts/evaluate_christ_face_photos.py --baseline r59.glb \
    --candidate r61.glb --annotations face-photo-landmarks.json \
    --output photo-fit-HOLD.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def _head_world(scene):
    node="Head | anatomical quad-topology HOLD"
    if node not in scene.graph.nodes_geometry:
        raise ValueError("Both GLBs must contain the same anatomical donor head node")
    t,g=scene.graph[node]
    m=scene.geometry[g].copy()
    m.apply_transform(t)
    return m


def compare_photos(baseline,candidate,annotations):
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    from kumiki_blender.orthographic_registration import fit_ortho_landmarks
    base=_head_world(trimesh.load(baseline,force="scene"))
    trial=_head_world(trimesh.load(candidate,force="scene"))
    if base.vertices.shape!=trial.vertices.shape or not np.array_equal(base.faces,trial.faces):
        raise ValueError("HOLD_topology_mismatch: vertex IDs cannot be transferred")
    views=annotations.get("views",[])
    if len(views)<2 or len({v.get("subject_photo_url") for v in views})!=len(views):
        raise ValueError("At least two distinct real photo annotation sets required")
    tree=cKDTree(base.vertices)
    reports=[]
    for view in views:
        if len(view["landmarks"])<8:
            raise ValueError("Eight real photo landmarks per view required")
        candidate_points=[]
        gaps=[]
        for entry in view["landmarks"]:
            xyz=np.asarray(entry["world_xyz"],float)
            if xyz.shape!=(3,) or not np.isfinite(xyz).all():
                raise ValueError("Nonfinite or invalid 3D annotation")
            glb=np.array((xyz[0],xyz[2],-xyz[1]))
            gap,index=tree.query(glb)
            gaps.append(float(gap))
            if gap>.025:
                raise ValueError("HOLD_anchor_mesh_mismatch: "+entry["name"])
            vertex=trial.vertices[index]
            candidate_points.append({**entry,"world_xyz":[float(vertex[0]),
                                      float(-vertex[2]),float(vertex[1])]})
        base_report=fit_ortho_landmarks(view["landmarks"],view["image_size"])
        candidate_report=fit_ortho_landmarks(candidate_points,view["image_size"])
        def chin(report):
            return next((p["error_px"] for p in report["landmarks"]
                         if p["name"]=="chin_tip"),None)
        reports.append({"view_id":view["view_id"],
                        "subject_photo_url":view["subject_photo_url"],
                        "baseline_rms_px":base_report["rms_reprojection_px"],
                        "candidate_rms_px":candidate_report["rms_reprojection_px"],
                        "baseline_chin_error_px":chin(base_report),
                        "candidate_chin_error_px":chin(candidate_report),
                        "baseline_status":base_report["status"],
                        "candidate_status":candidate_report["status"],
                        "max_anchor_mesh_gap_m":round(max(gaps),6),
                        "candidate_landmark_errors":candidate_report["landmarks"]})
    return {"status":"HOLD_photo_likeness_not_verified",
            "baseline":str(baseline),"candidate":str(candidate),
            "model_revision":annotations.get("model_revision"),
            "views":reports,
            "limitations":"Manually marked photo points and approximate weak-perspective camera; errors do not establish metric scan accuracy."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",required=True,type=Path)
    p.add_argument("--candidate",required=True,type=Path)
    p.add_argument("--annotations",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    report=compare_photos(args.baseline,args.candidate,
                          json.loads(args.annotations.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print("PHOTO_LIKENESS_HOLD",[(v["view_id"],v["baseline_rms_px"],
                                 v["candidate_rms_px"]) for v in report["views"]])


if __name__=="__main__":
    main()
