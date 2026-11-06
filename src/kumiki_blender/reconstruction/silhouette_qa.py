"""Compare registered binary silhouettes without asserting geometric accuracy.

The caller is responsible for rendering the mesh with the calibrated camera and
for establishing that both masks use the same rectified pixel coordinate system.
"""
from __future__ import annotations

from typing import Any


def compare_silhouettes(reference: Any, rendered: Any, *, boundary_tolerance: int = 2) -> tuple[dict, Any]:
    """Return pixel counts and an RGB discrepancy map.

    Red is reference-only, blue is render-only, white is agreement and yellow
    marks disagreement close to the reference boundary. The tolerance band is
    reported separately; strict metrics always include every pixel.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    ref = np.asarray(reference)
    mesh = np.asarray(rendered)
    if ref.ndim != 2 or mesh.shape != ref.shape or not ref.size:
        raise ValueError("Silhouettes must be nonempty, equally sized 2D masks in the same camera frame")
    if not np.isin(ref, (0, 1, 255)).all() or not np.isin(mesh, (0, 1, 255)).all():
        raise ValueError("Silhouettes must contain only binary values (0, 1 or 255)")
    if type(boundary_tolerance) is not int or not 0 <= boundary_tolerance <= 20:
        raise ValueError("boundary_tolerance must be an integer between 0 and 20 pixels")
    ref, mesh = ref.astype(bool), mesh.astype(bool)
    intersection = int(np.count_nonzero(ref & mesh))
    missing = ref & ~mesh
    excess = mesh & ~ref
    union = int(np.count_nonzero(ref | mesh))
    reference_count, rendered_count = int(ref.sum()), int(mesh.sum())
    uncertain = np.zeros(ref.shape, dtype=bool)
    if boundary_tolerance:
        image = Image.fromarray(ref.astype(np.uint8) * 255)
        size = 2 * boundary_tolerance + 1
        dilated = np.asarray(image.filter(ImageFilter.MaxFilter(size))) > 0
        eroded = np.asarray(image.filter(ImageFilter.MinFilter(size))) > 0
        uncertain = dilated & ~eroded & (missing | excess)
    difference = np.zeros((*ref.shape, 3), dtype=np.uint8)
    difference[ref & mesh] = (220, 220, 220)
    difference[missing] = (235, 50, 45)
    difference[excess] = (45, 120, 245)
    difference[uncertain] = (255, 210, 40)
    return ({"intersection_pixels": intersection, "reference_pixels": reference_count,
             "rendered_pixels": rendered_count, "missing_pixels": int(missing.sum()),
             "excess_pixels": int(excess.sum()), "boundary_uncertain_pixels": int(uncertain.sum()),
             "boundary_tolerance_pixels": boundary_tolerance,
             "iou": intersection / union if union else None,
             "precision": intersection / rendered_count if rendered_count else None,
             "recall": intersection / reference_count if reference_count else None}, difference)
