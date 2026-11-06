"""Find image-space discrepancy candidates without inventing 3D positions."""
from __future__ import annotations


def discrepancy_regions(reference, rendered, *, boundary_tolerance=2, min_pixels=4):
    """Return connected reference-only and render-only components, largest first.

    Confidence labels describe distance from the mask edge, not probability that
    a reconstruction defect exists. Camera error and masks remain confounders.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    ref = np.asarray(reference).astype(bool)
    mesh = np.asarray(rendered).astype(bool)
    if ref.ndim != 2 or mesh.shape != ref.shape or not ref.size:
        raise ValueError('Masks must be nonempty and have equal two-dimensional shapes')
    if type(boundary_tolerance) is not int or not 0 <= boundary_tolerance <= 20:
        raise ValueError('boundary_tolerance must be an integer between 0 and 20')
    if type(min_pixels) is not int or min_pixels < 1:
        raise ValueError('min_pixels must be a positive integer')
    band = np.zeros(ref.shape, dtype=bool)
    if boundary_tolerance:
        image = Image.fromarray(ref.astype(np.uint8) * 255)
        size = boundary_tolerance * 2 + 1
        band = ((np.asarray(image.filter(ImageFilter.MaxFilter(size))) > 0) &
                ~(np.asarray(image.filter(ImageFilter.MinFilter(size))) > 0))
    height, width = ref.shape
    regions = []
    for kind, discrepancy in (('reference_only', ref & ~mesh), ('render_only', mesh & ~ref)):
        remaining = discrepancy.copy()
        # Scanline flood fill avoids recursion and handles large contiguous regions.
        for y, x in zip(*np.nonzero(discrepancy)):
            if not remaining[y, x]:
                continue
            stack = [(int(y), int(x))]
            count = uncertain = 0
            left, top, right, bottom = width, height, 0, 0
            while stack:
                row, column = stack.pop()
                if not remaining[row, column]:
                    continue
                lo = column
                while lo > 0 and remaining[row, lo - 1]:
                    lo -= 1
                hi = column
                while hi + 1 < width and remaining[row, hi + 1]:
                    hi += 1
                remaining[row, lo:hi+1] = False
                count += hi - lo + 1
                uncertain += int(band[row, lo:hi+1].sum())
                left, right = min(left, lo), max(right, hi)
                top, bottom = min(top, row), max(bottom, row)
                for neighbor in (row - 1, row + 1):
                    if 0 <= neighbor < height:
                        runs = np.flatnonzero(remaining[neighbor, lo:hi+1]) + lo
                        if len(runs):
                            starts = runs[np.r_[True, np.diff(runs) > 1]]
                            stack.extend((neighbor, int(start)) for start in starts)
            if count >= min_pixels:
                core = count - uncertain
                regions.append({'kind': kind, 'pixels': count, 'bbox_xyxy': [left, top, right+1, bottom+1],
                                'image_fraction': count / (width * height),
                                'boundary_band_pixels': uncertain, 'interior_pixels': core,
                                'evidence': 'interior-discrepancy' if core else 'mask-boundary-only'})
    return sorted(regions, key=lambda item: (-item['interior_pixels'], -item['pixels'], item['kind']))
