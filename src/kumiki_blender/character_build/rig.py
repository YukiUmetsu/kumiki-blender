"""Estimate rig joints from a fitted character mesh and the handoff landmarks.

The result is an editable ``joints.json``. ``blender_rig.py`` builds the
armature from it. Heights come from ``spec.json`` landmarks (``crown``,
``face_bottom``, ``shoulder``, ``hip``, ``knee``) when they exist. Missing
heights use chibi-proportion defaults and are listed under ``estimated``:
review those joints before rigging.

Positions come from mesh cross-sections: the spine follows the body centre
at each height, the legs are the two vertex clusters at knee height, the
arms or wings are the lateral mass between knee and shoulder, and the tail
is the mass behind the body.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh

# Height of each landmark as a fraction of the figure height, used only when spec.json has none.
DEFAULT_FRACTION = {"crown": 0.97, "face_bottom": 0.60, "shoulder": 0.55, "hip": 0.32, "knee": 0.14}


def _band(points: np.ndarray, z: float, half: float) -> np.ndarray:
    return points[np.abs(points[:, 2] - z) <= half]


def estimate_joints(mesh: Mesh, height_m: float, landmarks_z: dict | None = None, *, limbs: str = "arms",
                    tail: bool = False) -> dict:
    """Return joints (metres, Blender axes) and a bone list for a biped creature."""
    if limbs not in ("arms", "wings"):
        raise ValueError("limbs must be arms or wings")
    points = mesh.vertices
    landmarks_z = landmarks_z or {}
    z, estimated = {}, []
    for key, fraction in DEFAULT_FRACTION.items():
        if key in landmarks_z:
            z[key] = float(landmarks_z[key])
        else:
            z[key], _ = fraction * height_m, estimated.append(key)
    if not z["knee"] < z["hip"] < z["shoulder"] <= z["face_bottom"] < z["crown"]:
        raise ValueError(f"Landmark heights must rise knee < hip < shoulder <= face_bottom < crown: {z}")
    half = 0.02 * height_m
    top = float(points[:, 2].max())
    if "crown" in estimated and z["face_bottom"] + half < top < z["crown"]:
        # A model that came out shorter than the drawing has no surface at the drawn crown.
        z["crown"] = top - half / 2

    # The handoff frame centres each view's bounding box, so a one-sided accessory moves x = 0
    # off the body. The midline is the median cross-section centre between knee and crown.
    centres_x = [(b[:, 0].min() + b[:, 0].max()) / 2 for at in np.linspace(z["knee"], z["crown"], 25)
                 if len(b := _band(points, at, half))]
    if not centres_x:
        raise ValueError("No mesh between the knee and crown heights")
    mid = float(np.median(centres_x))

    def centre(at: float) -> np.ndarray:
        band = _band(points, at, half)
        if len(band) == 0:
            raise ValueError(f"No mesh at height {at:.3f} m")
        return np.array([mid, float((band[:, 1].min() + band[:, 1].max()) / 2), at])

    joints = {"root": np.array([mid, centre(z["hip"])[1], 0.0]), "hips": centre(z["hip"]),
              "chest": centre(z["shoulder"]), "neck": centre(z["face_bottom"]), "head_top": centre(z["crown"])}
    joints["spine"] = (joints["hips"] + joints["chest"]) / 2

    # Legs sit in the inner part of the body. Wing tips or hands that hang to knee height are lateral.
    inner = 0.55 * np.abs(points[:, 0] - mid).max()
    knee_band = _band(points, z["knee"], half)
    knee_band = knee_band[np.abs(knee_band[:, 0] - mid) <= inner]
    for side, sign in (("L", 1), ("R", -1)):
        foot = points[(points[:, 2] < 0.06 * height_m) & ((points[:, 0] - mid) * sign > 0)
                      & (np.abs(points[:, 0] - mid) <= inner)]
        if len(foot) < 3:
            raise ValueError(f"No {side} foot found near the soles")
        ankle = np.array([(foot[:, 0].min() + foot[:, 0].max()) / 2, (foot[:, 1].min() + foot[:, 1].max()) / 2,
                          0.04 * height_m])
        # The knee is the mass straight above the foot. Wing tips, a tail or a coat hem that hang
        # to knee height lie away from it, so only points near the ankle in plan view count.
        reach = np.hypot(knee_band[:, 0] - ankle[0], knee_band[:, 1] - ankle[1])
        leg = knee_band[((knee_band[:, 0] - mid) * sign > 0) & (reach <= 0.08 * height_m)]
        if len(leg) < 3:
            raise ValueError(f"No {side} leg found above the foot at knee height; set the knee landmark")
        knee = np.array([(leg[:, 0].min() + leg[:, 0].max()) / 2, (leg[:, 1].min() + leg[:, 1].max()) / 2, z["knee"]])
        toe = np.array([ankle[0], foot[:, 1].min(), 0.01 * height_m])
        joints[f"thigh.{side}"] = np.array([knee[0], joints["hips"][1], z["hip"]])
        joints[f"knee.{side}"], joints[f"ankle.{side}"], joints[f"toe.{side}"] = knee, ankle, toe

        shoulder_band = _band(points, z["shoulder"], half)
        side_band = shoulder_band[(shoulder_band[:, 0] - mid) * sign > 0]
        width = np.abs(side_band[:, 0] - mid).max() if len(side_band) else 0.1 * height_m
        shoulder = np.array([mid + sign * 0.75 * width, joints["chest"][1], z["shoulder"]])
        span = points[(points[:, 2] > z["knee"]) & (points[:, 2] < z["shoulder"]) & ((points[:, 0] - mid) * sign > 0)]
        reach = np.abs(span[:, 0] - mid)
        lateral = span[reach >= 0.8 * reach.max()] if len(span) else span
        if len(lateral) == 0:
            raise ValueError(f"No {side} {limbs[:-1]} mass found beside the body")
        hand = lateral[np.argmax(np.linalg.norm(lateral - shoulder, axis=1))]
        elbow = (shoulder + hand) / 2 + np.array([0, 0.02 * height_m, 0])  # a slight bend toward the back
        joints[f"shoulder.{side}"], joints[f"elbow.{side}"], joints[f"hand.{side}"] = shoulder, elbow, hand

    bones = [("root", "root", "hips", None), ("spine", "hips", "spine", "root"), ("chest", "spine", "chest", "spine"),
             ("neck", "chest", "neck", "chest"), ("head", "neck", "head_top", "neck")]
    limb = "wing" if limbs == "wings" else "arm"
    for side in ("L", "R"):
        bones += [(f"thigh.{side}", f"thigh.{side}", f"knee.{side}", "root"),
                  (f"shin.{side}", f"knee.{side}", f"ankle.{side}", f"thigh.{side}"),
                  (f"foot.{side}", f"ankle.{side}", f"toe.{side}", f"shin.{side}"),
                  (f"upper_{limb}.{side}", f"shoulder.{side}", f"elbow.{side}", "chest"),
                  (f"fore_{limb}.{side}", f"elbow.{side}", f"hand.{side}", f"upper_{limb}.{side}")]
    if tail:
        back = joints["hips"][1] + 0.5 * (points[:, 1].max() - joints["hips"][1])
        mass = points[(points[:, 1] > back) & (points[:, 2] > z["knee"] * 0.5)]
        if len(mass) < 10:
            raise ValueError("No tail mass found behind the body")
        base = np.array([mid, joints["hips"][1] + 0.6 * (back - joints["hips"][1]), z["hip"] * 0.9])
        # Joints at centroids of distance bands from the root follow the tail's spine. The single
        # farthest point would land on the edge of a fanned tail.
        distance = np.linalg.norm(mass - base, axis=1)
        reach = distance.max()
        previous = "root"
        for i, (low, high) in enumerate(((0.25, 0.42), (0.55, 0.72), (0.85, 1.0)), start=1):
            band = mass[(distance >= low * reach) & (distance <= high * reach)]
            joints[f"tail{i}"] = band.mean(axis=0) if len(band) else base + (mass.mean(axis=0) - base) * i / 3
        joints["tail0"] = base
        for i in range(1, 4):
            bones.append((f"tail{i}", f"tail{i - 1}", f"tail{i}", previous))
            previous = f"tail{i}"
    return {"schema_version": 1, "height_m": height_m, "limbs": limbs, "estimated_landmarks": estimated,
            "midline_x_m": round(mid, 5),
            "landmarks_z_m": {k: round(v, 5) for k, v in z.items()},
            "joints": {k: [round(float(c), 5) for c in v] for k, v in joints.items()},
            "bones": [{"name": n, "head": h, "tail": t, "parent": p} for n, h, t, p in bones]}


def validate_joints(data: dict) -> None:
    joints, names = data.get("joints"), set()
    if not isinstance(joints, dict) or not isinstance(data.get("bones"), list) or not data["bones"]:
        raise ValueError("joints.json needs joints and bones")
    for bone in data["bones"]:
        for key in ("head", "tail"):
            if bone.get(key) not in joints:
                raise ValueError(f"Bone {bone.get('name')} uses unknown joint {bone.get(key)}")
        if np.linalg.norm(np.subtract(joints[bone["tail"]], joints[bone["head"]])) < 1e-4:
            raise ValueError(f"Bone {bone['name']} has zero length")
        if bone.get("parent") not in names | {None}:
            raise ValueError(f"Bone {bone['name']} needs its parent listed before it")
        names.add(bone["name"])
