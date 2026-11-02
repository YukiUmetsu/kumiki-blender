"""Triangulate real-photo face anchors before proposing a sculpt correction.

Every camera is fitted with the *target feature withheld*, so the feature's
reprojection error cannot be made artificially small by its own calibration.
The resulting 3D point is a proposal only, never a monument survey result.
"""

import math

from .camera_registration import fit_landmarks, project


def _camera_ray(camera, uv, image_size):
    """Return a world-space pinhole ray consistent with camera_registration.project."""
    import numpy as np

    target = np.asarray(camera["target_world_m"], dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Camera target must be a finite 3D point")
    azimuth, elevation, roll = (math.radians(float(camera[k])) for k in
                                ("azimuth_deg", "elevation_deg", "roll_deg"))
    distance, focal = (float(camera[k]) for k in ("distance_m", "focal_px"))
    if (not all(math.isfinite(v) for v in (azimuth, elevation, roll, distance, focal))
            or distance <= 0 or focal <= 0 or abs(elevation) >= math.pi / 2):
        raise ValueError("Camera orientation, distance or focal length is invalid")
    w, h = map(float, image_size)
    u, v = map(float, uv)
    if not all(math.isfinite(x) for x in (w, h, u, v)) or w <= 0 or h <= 0:
        raise ValueError("Image dimensions and pixel coordinates must be finite")
    direction = np.array((math.sin(azimuth) * math.cos(elevation),
                          -math.cos(azimuth) * math.cos(elevation),
                          math.sin(elevation)))
    origin = target + direction * distance
    forward = -direction
    right = np.cross(forward, np.array((0., 0., 1.)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    original_right = right.copy()
    right = math.cos(roll) * original_right + math.sin(roll) * up
    up = -math.sin(roll) * original_right + math.cos(roll) * up
    ray = forward + ((u - w / 2) / focal) * right - ((v - h / 2) / focal) * up
    ray /= np.linalg.norm(ray)
    return origin, ray


def _triangulate_rays(rays, *, min_angle_deg=12., max_ray_gap_m=.035):
    """Least-squares intersection of two or more calibrated world-space rays."""
    import numpy as np

    if len(rays) < 2 or min_angle_deg <= 0 or max_ray_gap_m <= 0:
        raise ValueError("Need two rays and positive triangulation thresholds")
    origins, directions = [], []
    for origin, direction in rays:
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        if o.shape != (3,) or d.shape != (3,) or not np.isfinite(o).all() or not np.isfinite(d).all():
            raise ValueError("Ray origin/direction must be finite 3D vectors")
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            raise ValueError("Cannot triangulate a zero-length ray")
        origins.append(o)
        directions.append(d / norm)
    angles = [math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), -1, 1))))
              for i, a in enumerate(directions) for b in directions[i + 1:]]
    angle = max(angles)
    if angle < min_angle_deg:
        raise ValueError("HOLD_parallel_camera_rays: %.2f deg" % angle)
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, directions):
        projector = np.eye(3) - np.outer(d, d)
        a += projector
        b += projector @ o
    point = np.linalg.solve(a, b)
    depths = [float((point - o) @ d) for o, d in zip(origins, directions)]
    if min(depths) <= 0:
        raise ValueError("HOLD_behind_camera")
    gaps = [float(np.linalg.norm(np.cross(point - o, d))) for o, d in zip(origins, directions)]
    if max(gaps) > max_ray_gap_m:
        raise ValueError("HOLD_inconsistent_camera_rays: %.4f m" % max(gaps))
    return point, {"largest_intersection_angle_deg": round(angle, 3),
                   "max_ray_gap_m": round(max(gaps), 6)}


def propose_face_corrections(views, *, max_camera_rms_px=5.,
                             max_target_error_px=3., max_ray_gap_m=.035,
                             min_angle_deg=12., max_displacement_m=.25):
    """Leave-one-landmark-out camera fit and cross-view 3D reconstruction.

    Inputs are independent real-subject photographs annotated with named
    world_xyz/photo_uv correspondences and a model_revision. Every view must
    have >=12 anchors, >=10 shared names and distinct source-photo URLs.
    Candidate corrections remain unapproved; the returned status is always
    HOLD_needs_visual_review. No geometry is modified by this function.
    """
    import numpy as np

    if len(views) < 2:
        raise ValueError("Two independent real-subject views required")
    ids = [v.get("view_id") for v in views]
    photos = [v.get("subject_photo_url") for v in views]
    revisions = [v.get("model_revision") for v in views]
    if (not all(ids) or len(set(ids)) != len(ids) or not all(photos)
            or len(set(photos)) != len(photos)):
        raise ValueError("Unique view IDs and distinct source photographs are required")
    if not all(revisions) or len(set(revisions)) != 1:
        raise ValueError("All views must use one explicit model_revision")
    named = []
    for view in views:
        pts = {p["name"]: p for p in view["landmarks"]}
        if len(pts) != len(view["landmarks"]) or len(pts) < 12:
            raise ValueError("At least twelve unique features per photograph required")
        named.append(pts)
    shared = set.intersection(*(set(p) for p in named))
    if len(shared) < 10:
        raise ValueError("At least ten shared facial landmarks required")
    corrections, holds = [], []
    for feature in sorted(shared):
        cameras, rays, donor_positions = [], [], []
        blocked = None
        for view, points in zip(views, named):
            # The held-out landmark cannot influence the fitted camera.
            training = [point for name, point in points.items() if name != feature]
            report = fit_landmarks(training, view["image_size"],
                                   initial=view.get("initial"),
                                   max_error_px=max_camera_rms_px)
            if report["status"] != "provisional_fit" or not report["success"]:
                blocked = "HOLD_camera_fit_" + view["view_id"]
                break
            camera = report["camera"]
            rays.append(_camera_ray(camera, points[feature]["photo_uv"], view["image_size"]))
            cameras.append((camera, view["image_size"], points[feature]["photo_uv"], report))
            donor_positions.append(points[feature]["world_xyz"])
        if blocked:
            holds.append({"name": feature, "reason": blocked})
            continue
        donor_positions = np.asarray(donor_positions, float)
        spread = float(np.max(np.linalg.norm(donor_positions - donor_positions.mean(axis=0), axis=1)))
        if spread > .035:
            holds.append({"name": feature, "reason": "HOLD_inconsistent_model_anchors",
                          "max_anchor_spread_m": round(spread, 6)})
            continue
        try:
            point, diagnostic = _triangulate_rays(
                rays, min_angle_deg=min_angle_deg, max_ray_gap_m=max_ray_gap_m)
        except (ValueError, np.linalg.LinAlgError) as exc:
            holds.append({"name": feature, "reason": str(exc)})
            continue
        errors = []
        for camera, size, photo_uv, _ in cameras:
            uv, depth = project(np.asarray([point]), target=camera["target_world_m"],
                                azimuth=math.radians(camera["azimuth_deg"]),
                                elevation=math.radians(camera["elevation_deg"]),
                                roll=math.radians(camera["roll_deg"]),
                                distance=camera["distance_m"], focal_px=camera["focal_px"],
                                image_size=size)
            if depth[0] <= 0:
                errors.append(math.inf)
            else:
                errors.append(float(np.linalg.norm(uv[0] - photo_uv)))
        if max(errors) > max_target_error_px:
            holds.append({"name": feature, "reason": "HOLD_target_reprojection",
                          "max_reprojection_px": round(max(errors), 3)})
            continue
        donor = donor_positions.mean(axis=0)
        delta = point - donor
        if np.linalg.norm(delta) > max_displacement_m:
            holds.append({"name": feature, "reason": "HOLD_excessive_sculpt_displacement",
                          "offset_m": round(float(np.linalg.norm(delta)), 4)})
            continue
        corrections.append({"name": feature,
                            "source_world_xyz": [round(float(v), 6) for v in donor],
                            "target_world_xyz": [round(float(v), 6) for v in point],
                            "offset_world_m": [round(float(v), 6) for v in delta],
                            "max_reprojection_px": round(max(errors), 3),
                            **diagnostic})
    return {"status": "HOLD_needs_visual_review",
            "model_revision": revisions[0],
            "source_photographs": photos,
            "candidate_count": len(corrections),
            "held_count": len(holds),
            "candidates": corrections,
            "held_features": holds,
            "method": "leave-one-out independent camera fits, ray intersection",
            "limitations": "Requires actual calibrated photographs, reviewed correspondences, "
                           "and independent clay render comparison. Never treat as a survey scan."}
