"""Perspective landmark registration for honest monument-to-sculpt comparisons.

This estimates a source camera from hand-annotated 2D photo landmarks and 3D
sculpt landmarks. It never rescales the sculpture or silently marks unmatched
photographs as calibrated. NumPy/SciPy are optional reconstruction dependencies.
"""

import math


def _array_module():
    try:
        import numpy
    except ImportError as error:
        raise RuntimeError("Install kumiki-blender[reconstruction] for photo fitting") from error
    return numpy


def project(points, *, target, azimuth, elevation, distance, focal_px,
            image_size, roll=0.0):
    """Project world 3D points, using Blender-compatible +Z world up.

    The camera orbits `target`: zero azimuth is on the world's negative Y axis;
    negative elevation is a viewpoint below the object. `focal_px` is the
    focal length in pixels, NOT a physical camera-lens claim. Pixels start at
    the photograph's upper-left corner.
    """
    np = _array_module()
    p = np.asarray(points, dtype=float)
    target = np.asarray(target, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if distance <= 0 or focal_px <= 0 or min(image_size) <= 0:
        raise ValueError("Camera distance, focal length and dimensions must be positive")
    direction = np.array((math.sin(azimuth)*math.cos(elevation),
                          -math.cos(azimuth)*math.cos(elevation),
                          math.sin(elevation)))
    camera = target + direction*distance
    forward = -direction
    right = np.cross(forward, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    old_right = right.copy()
    right = math.cos(roll)*old_right + math.sin(roll)*up
    up = -math.sin(roll)*old_right + math.cos(roll)*up
    delta = p-camera
    depth = delta@forward
    # An invalid pose should not produce NaNs that a numeric optimizer could
    # mistake for a solution. Penalize camera-behind-surface configurations.
    safe_depth = np.maximum(depth, .0001)
    projected = np.column_stack((image_size[0]/2 + focal_px*(delta@right)/safe_depth,
                                 image_size[1]/2 - focal_px*(delta@up)/safe_depth))
    return projected, depth


def fit_landmarks(landmarks, image_size, *, initial=None, max_error_px=12.0):
    """Fit a provisional perspective camera and return per-point residuals.

    Each landmark supplies `name`, `world_xyz` and `photo_uv`. At least eight
    well-spread correspondences are required. This minimizes 2D reprojection
    error only; it cannot establish source mesh authenticity, physical scale,
    accuracy of manually matched landmarks, or the true lens calibration.
    """
    np = _array_module()
    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise RuntimeError("Install kumiki-blender[reconstruction] for photo fitting") from error
    if len(landmarks) < 8:
        raise ValueError("At least eight facial landmarks are required")
    if len({p["name"] for p in landmarks}) != len(landmarks):
        raise ValueError("Landmark names must be unique")
    xyz = np.asarray([p["world_xyz"] for p in landmarks], dtype=float)
    uv = np.asarray([p["photo_uv"] for p in landmarks], dtype=float)
    if xyz.shape != (len(landmarks), 3) or uv.shape != (len(landmarks), 2):
        raise ValueError("Each landmark needs a 3D point and 2D photograph pixel")
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("Landmarks must have finite numbers")
    if min(image_size) <= 0:
        raise ValueError("Image width and height must be positive")
    if float(np.linalg.matrix_rank(xyz-xyz.mean(axis=0))) < 2:
        raise ValueError("Landmarks must cover a 2D or 3D facial surface")
    center = xyz.mean(axis=0)
    if initial is None:
        x0 = np.array((*center, 0., math.radians(-35), math.log(11.),
                       math.log(float(max(image_size))), 0.))
    else:
        x0 = np.array((*initial["target"], initial["azimuth"],
                       initial["elevation"], math.log(initial["distance"]),
                       math.log(initial["focal_px"]), initial.get("roll", 0.)))
    bounds_min = [center[0]-4, center[1]-4, center[2]-4,
                  -1.3, -1.3, math.log(2), math.log(150), -.8]
    bounds_max = [center[0]+4, center[1]+4, center[2]+4,
                  +1.3, +1.3, math.log(80), math.log(30000), +.8]

    def unpack(vector):
        return dict(target=vector[:3],azimuth=vector[3],elevation=vector[4],
                    distance=math.exp(vector[5]),focal_px=math.exp(vector[6]),
                    roll=vector[7],image_size=image_size)

    def residual(vector):
        projected, depth = project(xyz, **unpack(vector))
        reprojection = (projected-uv).ravel()
        # Favor fits where the entire head remains in front of the camera.
        penalty = np.minimum(depth-0.1, 0.)*1000
        return np.concatenate((reprojection, penalty))

    result = least_squares(residual, x0, bounds=(bounds_min,bounds_max),
                           loss="soft_l1", f_scale=12., max_nfev=1500)
    params = unpack(result.x)
    projected, depth = project(xyz, **params)
    per_point = np.linalg.norm(projected-uv,axis=1)
    rms = math.sqrt(float(np.mean(per_point**2)))
    bound_hit = any(abs(value-low) < (high-low)*.001 or
                    abs(value-high) < (high-low)*.001
                    for value,low,high in zip(result.x,bounds_min,bounds_max))
    # Geometric fit + manual landmark accuracy + reference provenance are
    # independent quality gates. A low RMS is *never* photo fidelity proof.
    return {
        "status":("HOLD_camera_bounds" if bound_hit else
                  "provisional_fit" if rms<=max_error_px and min(depth)>.1 else
                  "HOLD_high_reprojection_error"),
        "success":bool(result.success),
        "camera_bounds_hit":bool(bound_hit),
        "rms_reprojection_px":round(rms,3),
        "max_reprojection_px":round(float(per_point.max()),3),
        "image_size":list(image_size),
        "camera":{
            "target_world_m":[round(float(v),6) for v in params["target"]],
            "azimuth_deg":round(math.degrees(params["azimuth"]),5),
            "elevation_deg":round(math.degrees(params["elevation"]),5),
            "roll_deg":round(math.degrees(params["roll"]),5),
            "distance_m":round(params["distance"],5),
            "focal_px":round(params["focal_px"],4),
        },
        "landmarks":[{"name":p["name"],
                      "photo_uv":list(p["photo_uv"]),
                      "reprojected_uv":[round(float(v),2) for v in q],
                      "error_px":round(float(e),2)}
                     for p,q,e in zip(landmarks,projected,per_point)],
        "limitations":"Hand-annotated image points and generic-donor 3D features; "
                      "camera lens/distortion unverified. Never treat as survey data."
    }
