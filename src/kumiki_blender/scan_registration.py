"""Conservative 3D landmark registration for rights-cleared monument scans.

No bounding-box-only alignment, mirrored axes, anisotropic scale or shear.
Successful registration never authenticates a scan or approves a sculpt.
"""
from __future__ import annotations

import itertools
import math
from typing import Any


class ScanRegistrationError(ValueError):
    """The reference/scan is not geometrically safe to register."""


def _points(anchors: list[dict[str, Any]]):
    import numpy as np
    if not isinstance(anchors, list) or not 6 <= len(anchors) <= 200:
        raise ScanRegistrationError("6..200 independent 3D landmarks required")
    if any(not isinstance(item, dict) for item in anchors):
        raise ScanRegistrationError("Landmarks must be objects")
    names=[item.get("name") for item in anchors]
    if (any(not isinstance(n,str) or not n or len(n)>96 for n in names)
            or len(set(names)) != len(names)):
        raise ScanRegistrationError("Landmark names must be unique nonempty strings")
    try:
        source=np.asarray([item["source_xyz"] for item in anchors],dtype=float)
        target=np.asarray([item["target_xyz"] for item in anchors],dtype=float)
    except (KeyError,TypeError,ValueError) as exc:
        raise ScanRegistrationError("Landmarks require source_xyz and target_xyz") from exc
    if (source.shape != (len(anchors),3) or target.shape != source.shape or
            not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ScanRegistrationError("Landmark coordinates must be finite XYZ triples")
    if (np.linalg.matrix_rank(source-source.mean(axis=0),tol=1e-7)<3 or
            np.linalg.matrix_rank(target-target.mean(axis=0),tol=1e-7)<3):
        raise ScanRegistrationError("3D registration requires non-coplanar landmarks")
    return names,source,target


def _umeyama(source,target,*,allow_reflection=False):
    import numpy as np
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3:
        raise ScanRegistrationError("Expected matching Nx3 point sets")
    a=source-source.mean(axis=0)
    b=target-target.mean(axis=0)
    variance=float((a*a).sum()/len(a))
    if variance<1e-12:
        raise ScanRegistrationError("Landmark configuration is degenerate")
    u,s,vt=np.linalg.svd((b.T@a)/len(a))
    correction=np.eye(3)
    if np.linalg.det(u@vt)<0 and not allow_reflection:
        correction[2,2]=-1.
    rotation=u@correction@vt
    scale=float(np.sum(s*np.diag(correction))/variance)
    if not math.isfinite(scale) or scale<=0:
        raise ScanRegistrationError("Invalid uniform scan scale")
    translation=target.mean(axis=0)-scale*(rotation@source.mean(axis=0))
    return scale,rotation,translation


def fit_scan_landmarks(anchors: list[dict[str,Any]], *,
                       inlier_tolerance_m: float=.06,
                       max_rms_m: float=.045,
                       min_inlier_fraction: float=.75,
                       max_iterations: int=512) -> dict[str,Any]:
    """Deterministic RANSAC over non-coplanar 3D quartets; never photo approval."""
    import numpy as np
    names,source,target=_points(anchors)
    if (not all(math.isfinite(v) and v>0 for v in
                (inlier_tolerance_m,max_rms_m,min_inlier_fraction)) or
            min_inlier_fraction>1 or not 16<=max_iterations<=10000):
        raise ScanRegistrationError("Invalid scan registration thresholds")
    n=len(source)
    # A mirrored source cannot produce enough proper-rotation RANSAC inliers.
    # Detect a clear handedness mismatch before robust fitting.
    p_scale,p_rot,p_trans=_umeyama(source,target)
    f_scale,f_rot,f_trans=_umeyama(source,target,allow_reflection=True)
    proper_rms=float(np.sqrt(np.mean(np.sum((p_scale*source@p_rot.T+p_trans-target)**2,axis=1))))
    reflected_rms=float(np.sqrt(np.mean(np.sum((f_scale*source@f_rot.T+f_trans-target)**2,axis=1))))
    if (np.linalg.det(f_rot)<0 and reflected_rms<inlier_tolerance_m
            and reflected_rms<proper_rms*.25):
        raise ScanRegistrationError("HOLD_mirrored_scan_coordinate_system")
    rng=np.random.default_rng(20261102)
    if n<=10:
        samples=itertools.combinations(range(n),4)
    else:
        samples=(rng.choice(n,4,replace=False) for _ in range(max_iterations))
    best=None
    for ids in samples:
        ids=np.asarray(tuple(ids),int)
        if np.linalg.matrix_rank(source[ids]-source[ids].mean(axis=0),tol=1e-7)<3:
            continue
        try:
            scale,rotation,translation=_umeyama(source[ids],target[ids])
        except (ScanRegistrationError,np.linalg.LinAlgError):
            continue
        distances=np.linalg.norm((scale*source@rotation.T+translation)-target,axis=1)
        mask=distances<=inlier_tolerance_m
        if mask.sum()<6:
            continue
        score=(int(mask.sum()),-float(np.median(distances[mask])))
        if best is None or score>best[0]:
            best=(score,mask)
    if best is None or best[1].sum()<max(6,math.ceil(n*min_inlier_fraction)):
        raise ScanRegistrationError("HOLD_insufficient_inlier_landmarks")
    mask=best[1]
    for _ in range(4):
        scale,rotation,translation=_umeyama(source[mask],target[mask])
        distances=np.linalg.norm((scale*source@rotation.T+translation)-target,axis=1)
        new_mask=distances<=inlier_tolerance_m
        if np.array_equal(mask,new_mask):
            break
        if new_mask.sum()<max(6,math.ceil(n*min_inlier_fraction)):
            raise ScanRegistrationError("HOLD_unstable_registration_inliers")
        mask=new_mask
    scale,rotation,translation=_umeyama(source[mask],target[mask])
    distances=np.linalg.norm((scale*source@rotation.T+translation)-target,axis=1)
    rms=float(np.sqrt(np.mean(distances[mask]**2)))
    if rms>max_rms_m:
        raise ScanRegistrationError(f"HOLD_high_3d_residual: {rms:.4f} m")
    mirror_scale,mirror_rot,mirror_trans=_umeyama(source[mask],target[mask],allow_reflection=True)
    mirror_res=np.linalg.norm((mirror_scale*source[mask]@mirror_rot.T+mirror_trans)-target[mask],axis=1)
    mirror_rms=float(np.sqrt(np.mean(mirror_res**2)))
    if np.linalg.det(mirror_rot)<0 and mirror_rms<max(.005,rms*.40):
        raise ScanRegistrationError("HOLD_mirrored_scan_coordinate_system")
    if not 1e-4<=scale<=1e4:
        raise ScanRegistrationError("HOLD_implausible_uniform_scale")
    transform=np.eye(4)
    transform[:3,:3]=scale*rotation
    transform[:3,3]=translation
    return {
        "status":"HOLD_requires_authenticated_scan_and_multiview_qa",
        "method":"deterministic robust 3D similarity registration (no mirror/shear)",
        "uniform_scale":round(scale,9),
        "rotation_determinant":round(float(np.linalg.det(rotation)),8),
        "matrix_source_to_target":np.round(transform,9).tolist(),
        "rms_inlier_m":round(rms,6),
        "max_inlier_error_m":round(float(distances[mask].max()),6),
        "inlier_count":int(mask.sum()),"total_count":n,
        "inlier_names":[name for name,ok in zip(names,mask) if ok],
        "outlier_names":[name for name,ok in zip(names,mask) if not ok],
        "per_landmark_error_m":{name:round(float(err),6) for name,err in zip(names,distances)},
        "limitations":"3D alignment is not proof of monument authenticity, source rights, facial likeness or scan completeness.",
    }
