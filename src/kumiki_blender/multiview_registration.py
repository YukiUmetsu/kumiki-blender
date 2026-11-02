"""Conservative multi-view landmark gate for monument facial reconstruction.

Each photograph receives its own perspective-camera fit against the SAME 3D
head surface. A shared landmark must resolve to the same world coordinate in
all views; otherwise the inputs are not a consistent 3D registration dataset.
This does not adjust the sculpt, infer missing geometry or assert survey truth.
"""

import math

from .camera_registration import fit_landmarks


def assess_multiview(views, *, max_error_px=12., min_shared=6, max_anchor_spread_m=.035):
    """Fit independent cameras and report persistent per-feature mismatches.

    views is a list of {view_id, image_size, landmarks, initial?} objects.
    Each landmark is {name, world_xyz, photo_uv}. At least two photographs and
    six shared features are needed to distinguish persistent sculpt mismatch
    from single-view camera/annotation errors. A provisional result still
    requires visual validation and real-source provenance.
    """
    if len(views)<2:
        raise ValueError("At least two independent reference views are required")
    ids=[view["view_id"] for view in views]
    if len(set(ids))!=len(ids):
        raise ValueError("View IDs must be unique")
    revisions=[v.get("model_revision") for v in views]
    if any(revisions) and (not all(revisions) or len(set(revisions))!=1):
        raise ValueError("All views must reference the same model_revision")
    photos=[v.get("subject_photo_url") for v in views]
    if any(photos) and len(set(p for p in photos if p))!=len([p for p in photos if p]):
        raise ValueError("Independent views cannot reuse the same source photograph")
    provenance_verified=all(photos) and len(set(photos))==len(views)
    world={}
    names=[]
    for view in views:
        if len(view["landmarks"])<8:
            raise ValueError("At least eight landmarks per view are required")
        current=set()
        for point in view["landmarks"]:
            name=point["name"]
            if name in current:
                raise ValueError("Duplicate landmark %s in %s"%(name,view["view_id"]))
            current.add(name)
            xyz=tuple(float(v) for v in point["world_xyz"])
            if len(xyz)!=3 or not all(math.isfinite(v) for v in xyz):
                raise ValueError("Invalid 3D landmark " + name)
            world.setdefault(name,[]).append(xyz)
        names.append(current)
    shared=set.intersection(*names)
    if len(shared)<min_shared:
        raise ValueError("At least %d common facial landmarks are required"%min_shared)
    # Independently clicked pixels on the same sculpt may hit neighboring
    # vertices. Tolerate small annotation scatter, then use one *canonical*
    # 3D point for every view; never pretend the independent hits are exact.
    spread={name:max((math.dist(a,b) for a in samples for b in samples),default=0.)
            for name,samples in world.items()}
    for name,meters in spread.items():
        if meters>max_anchor_spread_m:
            raise ValueError("Inconsistent world_xyz for shared landmark %s (%.3f m)"%
                             (name,meters))
    canonical={name:tuple(sum(v[i] for v in samples)/len(samples)
                          for i in range(3)) for name,samples in world.items()}
    fitted=[]
    errors={}
    for view in views:
        consistent=[{**point,"world_xyz":canonical[point["name"]]}
                    for point in view["landmarks"]]
        report=fit_landmarks(consistent,view["image_size"],
                             initial=view.get("initial"),max_error_px=max_error_px)
        fitted.append({"view_id":view["view_id"],**report})
        for point in report["landmarks"]:
            errors.setdefault(point["name"],[]).append({
                "view_id":view["view_id"],"error_px":point["error_px"]})
    persistent=[]
    for name in sorted(shared):
        measurements=errors[name]
        if all(item["error_px"]>max_error_px for item in measurements):
            persistent.append({"name":name,"per_view":measurements,
                               "warning":"Possible shared geometry mismatch; "
                               "verify annotations and camera calibration first"})
    status=("HOLD_unverified_source_independence" if not provenance_verified else
        "HOLD_camera_registration" if any(
        view["status"]!="provisional_fit" for view in fitted) else
        "HOLD_persistent_landmark_mismatch" if persistent else
        "provisional_multiview")
    return {"status":status,"view_count":len(views),
            "source_independence_documented":bool(provenance_verified),
            "common_landmark_count":len(shared),
            "max_shared_anchor_spread_m":round(max(spread[name] for name in shared),6),
            "persistent_mismatch_candidates":persistent,
            "views":fitted,
            "limitations":"Independent camera fits, not bundle adjustment. "
                          "Camera calibration, occlusion, correspondence and "
                          "real-monument likeness require separate validation."}
