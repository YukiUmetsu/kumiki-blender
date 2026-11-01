"""Refit *original* Sphinx facial geometry from corresponding frontal landmarks.

The photo remains external. Fitting uses a uniform camera alignment, then a
monotone vertical map on the continuous facial surface. Horizontal
corrections are *reported but not applied*: uncalibrated photographic cameras
can suggest physically impossible eye/cheek displacements. This is an
intentionally reversible step toward survey-based 3D reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from .architecture_geometry import Mesh
from .sphinx_landmarks import LandmarkError, LandmarkView, compare_landmark_views


# Locations on the sculpt's Y/Z landmark grid, in its existing *source*
# coordinates. Front camera looks along +X: image left is positive world Y.
# These are editable model correspondences, NOT surveyed monument positions.
_FACE_POINTS = {
    "left-eye": (.292, 3.965),
    "right-eye": (-.292, 3.965),
    "nose-break": (0., 3.719),
    "mouth-midline": (0., 3.275),
    "chin-bottom": (0., 2.934),
    "left-cheek-outline": (.43, 3.725),
    "right-cheek-outline": (-.43, 3.725),
}
_WINGS = ("left-nemes-wing-tip", "right-nemes-wing-tip")


@dataclass(frozen=True)
class FaceControl:
    name: str
    y_m: float
    z_m: float
    delta_y_m: float
    delta_z_m: float


@dataclass(frozen=True)
class SphinxFaceFit:
    controls: tuple[FaceControl, ...]
    reference_sha256: str
    max_shift_m: float


def prepare_sphinx_face_fit(reference: LandmarkView, candidate: LandmarkView, *,
                            scale_y: float = 1., scale_z: float = 3.6/4.6,
                            max_shift_m: float = .29) -> SphinxFaceFit:
    """Derive modest physical Y/Z shifts from registered images.

    Width calibration is the known *procedural* 3.14 m source headcloth wing
    span. It does NOT imply that the photographed statue is that size.
    The reference/candidate must depict the same front-oriented landmark
    semantics and use the two wing tips and crown as stable alignment anchors.
    """
    if reference.view != "frontal":
        raise LandmarkError("Sphinx face refitting currently requires a frontal view")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (scale_y, scale_z, max_shift_m)):
        raise LandmarkError("Face fit scales and displacement limit must be positive and finite")
    if max_shift_m > 1.0:
        raise LandmarkError("Facial shifts exceeding one meter are unsafe")
    if not set(_WINGS).issubset(reference.anchors):
        raise LandmarkError("Face fit needs both nemes wing tips among registration anchors")
    missing = set(_FACE_POINTS) - set(reference.landmarks)
    if missing:
        raise LandmarkError(f"Face fit is missing features: {', '.join(sorted(missing))}")
    comparison = compare_landmark_views(reference, candidate)
    transform = comparison["alignment"]
    uniform_scale = transform["uniform_scale"]
    shift_x, shift_y = transform["translation_px"]
    cand_span = abs(candidate.landmarks[_WINGS[1]][0] -
                    candidate.landmarks[_WINGS[0]][0])
    pixels_per_meter = cand_span*uniform_scale/(3.14*scale_y)
    if pixels_per_meter < 1e-3:
        raise LandmarkError("Reference wing tips do not provide a usable pixel scale")

    controls = []
    for name, (source_y, source_z) in _FACE_POINTS.items():
        rx, ry = reference.landmarks[name]
        cx, cy = candidate.landmarks[name]
        dx = rx-(cx*uniform_scale+shift_x)
        dy = ry-(cy*uniform_scale+shift_y)
        delta_y = -dx/pixels_per_meter
        delta_z = -dy/pixels_per_meter
        if math.hypot(delta_y, delta_z) > max_shift_m:
            raise LandmarkError(
                f"Photo fit for {name} requires excessive movement; "
                "check image perspective and landmark annotations")
        controls.append(FaceControl(name, source_y*scale_y, source_z*scale_z,
                                    delta_y, delta_z))
    serialized = json.dumps(
        {"reference": reference.landmarks, "candidate": candidate.landmarks,
         "reference_size": reference.size, "candidate_size": candidate.size,
         "anchors": reference.anchors, "scale_y": scale_y, "scale_z": scale_z},
        sort_keys=True, separators=(",", ":")).encode()
    return SphinxFaceFit(
        tuple(controls), hashlib.sha256(serialized).hexdigest(), max_shift_m)


def deform_sphinx_face(mesh: Mesh, fit: SphinxFaceFit, *,
                       scale_y: float = 1., scale_z: float = 3.6/4.6) -> Mesh:
    """Monotone vertical face refit that cannot fold the modeled Y/Z grid.

    Unlike a free 2D radial-basis warp, interpolation here is strictly
    order-preserving from chin to mouth to nose to eyes to crown. Modelers can
    inspect untrusted horizontal measurements without using them to invert
    the facial surface. Both front and rear surfaces follow the same vertical
    deformation so their watertight side walls remain physically coherent.
    """
    if not isinstance(fit, SphinxFaceFit) or not fit.controls:
        raise LandmarkError("Face deformation requires a prepared SphinxFaceFit")
    if len(mesh.vertices) < 1000 or len(mesh.vertices) % 2:
        raise LandmarkError("Expected paired front/back grid facial topology")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (scale_y, scale_z)):
        raise LandmarkError("Face mesh scales must be positive and finite")
    controls = {control.name: control for control in fit.controls}
    required = ("chin-bottom", "mouth-midline", "nose-break", "left-eye", "right-eye")
    if any(name not in controls for name in required):
        raise LandmarkError("Vertical face refit requires chin, mouth, nose and both eyes")
    left, right = controls["left-eye"], controls["right-eye"]
    if abs(left.delta_z_m-right.delta_z_m) > .10*scale_z:
        raise LandmarkError("Eye line differs too much across the two reference images")
    if all(abs(control.delta_z_m) < 1e-12 for control in fit.controls):
        return mesh
    # Fixed crown and lower edge prevent headcloth/neck disconnection.
    nodes = [
        (2.69*scale_z, 2.69*scale_z),
        (controls["chin-bottom"].z_m,
         controls["chin-bottom"].z_m+controls["chin-bottom"].delta_z_m),
        (controls["mouth-midline"].z_m,
         controls["mouth-midline"].z_m+controls["mouth-midline"].delta_z_m),
        (controls["nose-break"].z_m,
         controls["nose-break"].z_m+controls["nose-break"].delta_z_m),
        ((left.z_m+right.z_m)/2.,
         (left.z_m+right.z_m+left.delta_z_m+right.delta_z_m)/2.),
        (4.51*scale_z, 4.51*scale_z),
    ]
    if any(b[0] <= a[0] or b[1] <= a[1]
           for a, b in zip(nodes, nodes[1:])):
        raise LandmarkError(
            "Reference would invert chin/mouth/nose/eyes; recalibrate cameras")
    xs = [node[0] for node in nodes]
    ys = [node[1] for node in nodes]
    hs = [b-a for a, b in zip(xs, xs[1:])]
    ds = [(b-a)/h for a, b, h in zip(ys, ys[1:], hs)]
    # Shape-preserving Fritsch-Carlson derivatives; all secants are positive.
    slopes = [0.]*len(xs)
    for i in range(1, len(xs)-1):
        w1, w2 = 2*hs[i]+hs[i-1], hs[i]+2*hs[i-1]
        slopes[i] = (w1+w2)/(w1/ds[i-1]+w2/ds[i])

    def endpoint(h0: float, h1: float, d0: float, d1: float) -> float:
        value = ((2*h0+h1)*d0-h0*d1)/(h0+h1)
        return max(0., min(3*d0, value))

    slopes[0] = endpoint(hs[0], hs[1], ds[0], ds[1])
    slopes[-1] = endpoint(hs[-1], hs[-2], ds[-1], ds[-2])

    def remap(z: float) -> float:
        if z <= xs[0] or z >= xs[-1]:
            return z
        index = next(i for i in range(len(hs)) if z <= xs[i+1])
        t = (z-xs[index])/hs[index]
        t2, t3 = t*t, t*t*t
        return ((2*t3-3*t2+1)*ys[index] +
                (t3-2*t2+t)*hs[index]*slopes[index] +
                (-2*t3+3*t2)*ys[index+1] +
                (t3-t2)*hs[index]*slopes[index+1])

    warped = Mesh(tuple((x, y, remap(z)) for x, y, z in mesh.vertices), mesh.faces)
    # A monotone 1D mapping preserves the row ordering by construction.
    # Still verify it explicitly: future changes to face topology or control
    # placement must not quietly reintroduce the inverted grid seen in QA.
    cols = 97
    if len(mesh.vertices)//2 % cols:
        raise LandmarkError("Face topology changed; refit row layout must be reviewed")
    rows = len(mesh.vertices)//(2*cols)
    for row in range(rows-1):
        for col in range(cols):
            i = row*cols+col
            if warped.vertices[i+cols][2] <= warped.vertices[i][2]:
                raise LandmarkError("Photographic warp folded the facial surface")
    return warped
