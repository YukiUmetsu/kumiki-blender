"""Object-independent visible-region discovery for promptable segmentation.

The name of the subject is never used to invent anatomy. Regions are obtained
from image-local mask evidence and optional localized text detections; their
labels are geometric placeholders, not semantic or 3D part identifications.
"""

from __future__ import annotations

import math
from typing import Any


SCHEMA = "visible-object-regions-v3"


def discover_visible_regions(photo: Any, broad: Any, grounded: Any, *,
                             detector_guide: dict[str, Any] | None = None,
                             foreground_seeds: list[dict] | None = None,
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

    # Verified source-view tracks may identify a part missed in *both* masks.
    # Use them as optional, local, bounded prompts only; held-out tracks must
    # never be consumed here. Points far from the observed object abstain.
    exterior_distance = ndi.distance_transform_edt(~support)
    trusted_positives = []
    for row in (foreground_seeds or [])[:12]:
        point = row.get("normalized_xy") if isinstance(row, dict) else None
        if (not isinstance(point, list) or len(point) != 2 or
                any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
                    for v in point)):
            continue
        x, y = point
        trusted_positives.append(point)
        px = min(dimensions[0]-1, round(x*(dimensions[0]-1)))
        py = min(dimensions[1]-1, round(y*(dimensions[1]-1)))
        if exterior_distance[py, px] > .14*max(dimensions):
            continue
        if support[py, px]:
            for name, (left, top, right, bottom) in component_boxes.items():
                if (left <= x <= right and top <= y <= bottom and
                        all(math.dist(point, prior) > .025
                            for prior in part_points[name])):
                    part_points[name].append([round(x, 6), round(y, 6)])
                    receipt["components"].append({"id": name, "source": "cross-view-track-click",
                                                   "point": part_points[name][-1]})
                    break
            continue
        # Keep a small prompt box near the visible silhouette rather than a
        # huge object box that can admit adjacent walls or ground.
        box = [max(0., x-.11), max(0., y-.11), min(1., x+.11), min(1., y+.11)]
        add_region(f"tracked_{len(component_boxes)+1:02d}", box, point,
                   "nearby-cross-view-foreground-track")

    if len(component_boxes) < 2:
        receipt["status"] = "abstained / insufficient-distinct-regions"
        return None, receipt
    negatives = [point for point in
                 _auto_sam2_guide(photo, _DEFAULT_BOX)["background_points"]
                 if all(math.dist(point, positive) >= .08
                        for positive in trusted_positives)]
    # A distinct, distant lower-scene appearance can suggest negative SAM2
    # clicks without being promoted to *verified* ground evidence. Use the
    # intersection of both masks as the visual reference when available.
    rgb = np.asarray(photo.convert("RGB").resize(dimensions, Image.Resampling.BILINEAR),
                     dtype=np.float32)
    broad_raw = np.asarray(broad.convert("L").resize(
        dimensions, Image.Resampling.NEAREST)) >= 128
    exterior_union_distance = ndi.distance_transform_edt(~(broad_raw | support))
    reliable_core = ndi.binary_erosion(broad_raw & support, iterations=max(2, kernel//2))
    reference = reliable_core if reliable_core.sum() >= .015*support.size else support
    reference_rgb = np.median(rgb[reference], axis=0)
    visual_exteriors = []
    for y in (.63, .77, .89, .96):
        for x in (.07, .23, .50, .77, .93):
            px = min(dimensions[0]-1, round(x*(dimensions[0]-1)))
            py = min(dimensions[1]-1, round(y*(dimensions[1]-1)))
            if exterior_union_distance[py, px] < .055*max(dimensions):
                continue
            radius = max(2, round(max(dimensions)*.008))
            neighborhood = rgb[max(0, py-radius):py+radius+1,
                               max(0, px-radius):px+radius+1]
            color = np.mean(neighborhood, axis=(0, 1))
            contrast = float(np.linalg.norm(color-reference_rgb))
            brightness_ratio = float(np.mean(color)/max(1., np.mean(reference_rgb)))
            chroma_difference = float(np.linalg.norm(
                color/max(1., np.sum(color)) -
                reference_rgb/max(1., np.sum(reference_rgb))))
            # Illumination may turn the same stone much lighter, especially
            # on statues. Do not label bright sandstone as background merely
            # because it is absent from an already-incomplete SAM2 mask.
            if (contrast < 55 or
                    (brightness_ratio >= .85 and chroma_difference < .25) or
                    np.mean(np.std(neighborhood, axis=(0, 1))) > 38):
                continue
            point = [x, y]
            if (all(math.dist(point, positive) >= .08
                    for positive in trusted_positives) and
                    all(math.dist(point, other) > .09
                        for other in negatives + visual_exteriors)):
                visual_exteriors.append(point)
            if len(visual_exteriors) >= 6:
                break
        if len(visual_exteriors) >= 6:
            break
    negatives.extend(visual_exteriors)
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
                   heuristic_exterior_negatives=visual_exteriors,
                   analysis_dimensions=list(dimensions),
                   principal_axis=[round(float(n), 6) for n in direction])
    return guide, receipt
