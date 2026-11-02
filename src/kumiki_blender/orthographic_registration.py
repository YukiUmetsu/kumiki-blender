"""Stable weak-perspective photo alignment when camera EXIF/intrinsics are unknown.

Unlike free focal+distance+target perspective fitting, this avoids focal/
distance degeneracy. Residuals are for provisional likeness QA, NOT a survey.
"""
import math


def basis(azimuth,elevation,roll):
    import numpy as np
    d=np.array((math.sin(azimuth)*math.cos(elevation),
                -math.cos(azimuth)*math.cos(elevation),math.sin(elevation)))
    forward=-d
    right=np.cross(forward,np.array((0.,0.,1.)))
    right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    old=right.copy()
    right=math.cos(roll)*old+math.sin(roll)*up
    up=-math.sin(roll)*old+math.cos(roll)*up
    return d,right,up


def project_ortho(points, *, target, azimuth,elevation,roll,scale_px_per_m,offset_uv):
    import numpy as np
    pts=np.asarray(points,float)
    center=np.asarray(target,float)
    if pts.ndim!=2 or pts.shape[1]!=3 or center.shape!=(3,):
        raise ValueError("Expected Nx3 points and 3D target")
    if not (math.isfinite(scale_px_per_m) and scale_px_per_m>0):
        raise ValueError("Scale must be positive and finite")
    _,right,up=basis(azimuth,elevation,roll)
    v=pts-center
    return np.column_stack((offset_uv[0]+scale_px_per_m*(v@right),
                            offset_uv[1]-scale_px_per_m*(v@up)))


def fit_ortho_landmarks(landmarks,image_size,*,max_error_px=15.,initial_elevation=0.):
    import numpy as np
    from scipy.optimize import least_squares
    if len(landmarks)<8 or len({p["name"] for p in landmarks})!=len(landmarks):
        raise ValueError("Eight unique 2D/3D correspondences required")
    xyz=np.asarray([p["world_xyz"] for p in landmarks],float)
    uv=np.asarray([p["photo_uv"] for p in landmarks],float)
    if (xyz.shape!=(len(landmarks),3) or uv.shape!=(len(landmarks),2) or
            not np.isfinite(xyz).all() or not np.isfinite(uv).all()):
        raise ValueError("Finite 3D/2D landmarks required")
    if min(image_size)<=0 or not math.isfinite(max_error_px) or max_error_px<=0:
        raise ValueError("Invalid image dimensions or error threshold")
    center=xyz.mean(axis=0)
    w,h=map(float,image_size)
    scale=max(50.,min(1000.,w/3.))
    x0=[0.,initial_elevation,0.,math.log(scale),w/2,h/2]
    lower=[-.7,-.85,-.4,math.log(30),-w,-h]
    upper=[.7,.85,.4,math.log(1500),2*w,2*h]
    def unpack(x):
        return dict(target=center,azimuth=x[0],elevation=x[1],roll=x[2],
                    scale_px_per_m=math.exp(x[3]),offset_uv=x[4:6])
    def residual(x):
        return (project_ortho(xyz,**unpack(x))-uv).ravel()
    result=least_squares(residual,x0,bounds=(lower,upper),loss="soft_l1",f_scale=8.,max_nfev=1500)
    camera=unpack(result.x)
    reproj=project_ortho(xyz,**camera)
    error=np.linalg.norm(reproj-uv,axis=1)
    rms=math.sqrt(float(np.mean(error**2)))
    bound_hit=any(abs(value-lo)<(hi-lo)*.001 or abs(value-hi)<(hi-lo)*.001
                  for value,lo,hi in zip(result.x,lower,upper))
    status=("HOLD_camera_bounds" if bound_hit else
            "provisional_ortho_fit" if result.success and rms<=max_error_px else
            "HOLD_high_reprojection_error")
    return {"status":status,"success":bool(result.success),
            "rms_reprojection_px":round(rms,3),
            "max_reprojection_px":round(float(error.max()),3),
            "camera":{"target_world_m":[round(float(v),6) for v in center],
                      "azimuth_deg":round(math.degrees(result.x[0]),4),
                      "elevation_deg":round(math.degrees(result.x[1]),4),
                      "roll_deg":round(math.degrees(result.x[2]),4),
                      "scale_px_per_m":round(camera["scale_px_per_m"],5),
                      "offset_uv":[round(float(v),4) for v in camera["offset_uv"]]},
            "landmarks":[{"name":p["name"],"photo_uv":p["photo_uv"],
                          "reprojected_uv":[round(float(v),2) for v in q],
                          "error_px":round(float(e),2)}
                         for p,q,e in zip(landmarks,reproj,error)],
            "limitations":"Weak-perspective approximation; hand annotations, lens distortion and donor mismatch unverified."}
