"""Object-independent visible-region discovery for promptable segmentation.

The name of the subject is never used to invent anatomy. Regions are obtained
from image-local mask evidence and optional localized text detections; their
labels are geometric placeholders, not semantic or 3D part identifications.
"""

from __future__ import annotations

import math
from typing import Any


SCHEMA = "visible-object-regions-v1"


def discover_visible_regions(photo: Any, broad: Any, grounded: Any, *,
                             detector_guide: dict[str, Any] | None = None,
                             background_seeds: list[dict] | None = None,
                             max_regions: int = 5) -> tuple[dict[str, Any] | None,
                                                            dict[str, Any]]:
    """Construct bounded SAM2 part boxes/positive points from observed pixels.

    The image mask is merely a proposal for where to *ask* SAM2, never a
    verified foreground label. Large DINO part boxes are discarded because
    they do not independently localize a component. No Sphinx-specific terms,
    image-specific coordinates, or network inference are embedded here.
    """
    import numpy as np
    from PIL import Image
    from .masking import _DEFAULT_BOX, _auto_sam2_guide, _guide

    receipt: dict[str, Any] = {"schema": SCHEMA, "status": "abstained / no-localized-object",
                               "components": [], "limitations": "Image regions are 2D prompt guesses, not verified object anatomy."}
    if not 2 <= max_regions <= 8 or photo.size != broad.size or photo.size != grounded.size:
        return None, receipt
    try:
        from scipy import ndimage as ndi  # type: ignore[import-not-found]
    except ImportError:
        receipt["status"] = "abstained / scipy-not-installed"
        return None, receipt

    scale = min(1.0, 512/max(photo.size))
    dimensions = (max(16, round(photo.width*scale)), max(16, round(photo.height*scale)))
    raw = np.asarray(grounded.convert("L").resize(dimensions, Image.Resampling.NEAREST)) >= 128
    kernel = max(3, round(max(dimensions)*.008)) | 1
    closed = ndi.binary_closing(raw, structure=np.ones((kernel, kernel), dtype=bool))
    groups, count = ndi.label(closed)
    sizes = np.bincount(groups.ravel())
    sizes[0] = 0
    if not count or sizes.max() < .045 * closed.size:
        receipt["status"] = "abstained / no-substantial-connected-foreground"
        return None, receipt
    support = ndi.binary_fill_holes(groups == int(sizes.argmax()))
    if support.mean() > .80:
        receipt["status"] = "abstained / nearly-full-frame-proposal"
        return None, receipt

    yy, xx = np.nonzero(support)
    coords = np.column_stack((xx/max(1, dimensions[0]-1),
                              yy/max(1, dimensions[1]-1)))
    center = coords.mean(axis=0)
    covariance = np.cov((coords-center).T)
    eigval, eigvec = np.linalg.eigh(covariance)
    direction = eigvec[:, int(np.argmax(eigval))]
    # Deterministic orientation for stable cache keys and reproducible regions.
    if direction[int(np.argmax(abs(direction)))] < 0:
        direction = -direction
    projection = (coords-center) @ direction
    bounds = np.quantile(projection, np.linspace(0, 1, max_regions+1))
    distances = ndi.distance_transform_edt(support)
    component_boxes: dict[str, list[float]] = {}
    part_points: dict[str, list[list[float]]] = {}

    def add_region(name: str, box: list[float], point: list[float], source: str) -> None:
        if name in component_boxes or len(component_boxes) >= 8:
            return
        left, top, right, bottom = box
        if right-left < .05 or bottom-top < .05:
            return
        if any(abs(point[0]-p[0][0]) < .035 and abs(point[1]-p[0][1]) < .035
               for p in part_points.values()):
            return
        component_boxes[name] = [round(v, 6) for v in box]
        part_points[name] = [[round(v, 6) for v in point]]
        receipt["components"].append({"id": name, "source": source,
                                       "box": component_boxes[name],
                                       "point": part_points[name][0]})

    # Respect genuinely localizable detector parts; broad full-image boxes
    # were exactly the Sphinx failure mode and provide no part evidence.
    if isinstance(detector_guide, dict):
        for name, box in (detector_guide.get("component_boxes") or {}).items():
            if name == "subject" or not isinstance(box, list) or len(box) != 4:
                continue
            if (any(type(value) not in (float, int) or not math.isfinite(value)
                    for value in box) or not (0 <= box[0] < box[2] <= 1) or
                    not (0 <= box[1] < box[3] <= 1)):
                continue
            area = (box[2]-box[0]) * (box[3]-box[1])
            if not .005 <= area <= .35:
                continue
            region = ((coords[:, 0] >= box[0]) & (coords[:, 0] <= box[2]) &
                      (coords[:, 1] >= box[1]) & (coords[:, 1] <= box[3]))
            if region.sum() < 16:
                continue
            samples = coords[region]
            point = samples[np.argmin(np.sum((samples-np.mean(samples, axis=0))**2, axis=1))]
            add_region("detected_"+str(len(component_boxes)+1).zfill(2),
                       [float(v) for v in box], point.tolist(), "localized-text-detection")

    for index in range(max_regions):
        selected = (projection >= bounds[index]) & (projection <= bounds[index+1])
        if selected.sum() < max(20, round(coords.shape[0]*.04)):
            continue
        samples = coords[selected]
        rows = yy[selected]
        columns = xx[selected]
        # Deepest image-internal support point is more reliable than a mean
        # that may land on sky, a hole, or another overlapping component.
        priority = int(np.argmax(distances[rows, columns]))
        point = [float(samples[priority, 0]), float(samples[priority, 1])]
        margin = .055
        box = [max(0., float(np.min(samples[:, 0]))-margin),
               max(0., float(np.min(samples[:, 1]))-margin),
               min(1., float(np.max(samples[:, 0]))+margin),
               min(1., float(np.max(samples[:, 1]))+margin)]
        # SAM2's box must enclose the chosen positive point; 8% minimum
        # dimensions preserve a usable prompt on thin elongated objects.
        if box[2]-box[0] < .08:
            box[0], box[2] = max(0., point[0]-.05), min(1., point[0]+.05)
        if box[3]-box[1] < .08:
            box[1], box[3] = max(0., point[1]-.05), min(1., point[1]+.05)
        add_region(f"region_{index+1:02d}", box, point, "connected-mask-spatial-partition")

    if len(component_boxes) < 2:
        receipt["status"] = "abstained / insufficient-distinct-regions"
        return None, receipt
    negatives = _auto_sam2_guide(photo, _DEFAULT_BOX)["background_points"]
    for row in background_seeds or []:
        point = row.get("normalized_xy") if isinstance(row, dict) else None
        if (isinstance(point, list) and len(point) == 2 and
                all(type(value) in (float, int) and math.isfinite(value) and
                    0 <= value <= 1 for value in point) and
                point not in negatives and len(negatives) < 60):
            negatives.append(point)
    guide = _guide({"component_boxes": component_boxes, "part_points": part_points,
                    "background_points": negatives}, default_box=_DEFAULT_BOX)
    receipt.update(status="candidate / spatially-discovered-regions",
                   discovered_regions=len(component_boxes),
                   analysis_dimensions=list(dimensions),
                   principal_axis=[round(float(n), 6) for n in direction])
    return guide, receipt
