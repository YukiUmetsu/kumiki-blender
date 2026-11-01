"""Refit *original* Sphinx facial geometry from corresponding frontal landmarks.

The photo remains external. Fitting uses a uniform camera alignment, then a
regularized radial-basis displacement field on the continuous front face.
This is an intentionally reversible intermediate step toward survey-based 3D
reconstruction: a single image cannot recover profile depth or fine anatomy.
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


def _solve_dense(matrix: list[list[float]], right: list[float]) -> list[float]:
    """Pivoted Gauss-Jordan solve for a small Gaussian-RBF interpolant."""
    n = len(right)
    a = [row[:] + [value] for row, value in zip(matrix, right)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-10:
            raise LandmarkError("Facial landmark warp is ill-conditioned")
        a[pivot], a[col] = a[col], a[pivot]
        inv = 1./a[col][col]
        for j in range(col, n+1):
            a[col][j] *= inv
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor:
                for j in range(col, n+1):
                    a[row][j] -= factor*a[col][j]
    return [row[n] for row in a]


def deform_sphinx_face(mesh: Mesh, fit: SphinxFaceFit, *,
                       scale_y: float = 1., scale_z: float = 3.6/4.6) -> Mesh:
    """Move features on one watertight facial skin, fixing peripheral points.

    Rear vertices stay unchanged. The rear face and side-wall topology remain
    intact so Blender can voxel-union this head with the hood as before.
    """
    if not isinstance(fit, SphinxFaceFit) or not fit.controls:
        raise LandmarkError("Face deformation requires a prepared SphinxFaceFit")
    if len(mesh.vertices) < 1000 or len(mesh.vertices) % 2:
        raise LandmarkError("Expected paired front/back grid facial topology")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (scale_y, scale_z)):
        raise LandmarkError("Face mesh scales must be positive and finite")
    # Pin nearby points on upper, lower, and lateral margins so the face
    # still joins the royal headdress and neck after voxel remeshing.
    boundary = [
        (0., 2.69*scale_z), (0., 4.51*scale_z),
        (-.30*scale_y, 2.69*scale_z), (.30*scale_y, 2.69*scale_z),
        (-.47*scale_y, 4.51*scale_z), (.47*scale_y, 4.51*scale_z),
    ]
    for z, width in ((3.31, .465), (3.61, .525), (3.91, .559), (4.14, .551)):
        boundary.extend([(-width*scale_y, z*scale_z), (width*scale_y, z*scale_z)])
    knots = [(p.y_m, p.z_m, p.delta_y_m, p.delta_z_m) for p in fit.controls]
    knots.extend((y, z, 0., 0.) for y, z in boundary)
    # Scaled to model dimensions, not raw user-image pixels.
    radius_y = .30*scale_y
    radius_z = .31*scale_z

    def kernel(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return math.exp(-.5*((a[0]-b[0])/radius_y)**2 -
                        .5*((a[1]-b[1])/radius_z)**2)

    matrix = [[kernel(a, b) + (1e-5 if i == j else 0.)
               for j, b in enumerate(knots)] for i, a in enumerate(knots)]
    weights_y = _solve_dense(matrix, [knot[2] for knot in knots])
    weights_z = _solve_dense(matrix, [knot[3] for knot in knots])
    count_front = len(mesh.vertices)//2
    vertices = []
    for vertex in mesh.vertices[:count_front]:
        x, y, z = vertex
        k_values = [kernel(vertex[1:], knot) for knot in knots]
        delta_y = sum(k*w for k, w in zip(k_values, weights_y))
        delta_z = sum(k*w for k, w in zip(k_values, weights_z))
        vertices.append((x, y+delta_y, z+delta_z))
    vertices.extend(mesh.vertices[count_front:])
    return Mesh(tuple(vertices), mesh.faces)
