"""Make a set of views usable when one view shows a part that the others have no room for.

A front and a back view of one object are mirror images in outline. When a view is
replaced (a new back view with a different tail), the old opposite view still shows the
old part. The shape tools then get two answers, and the texture gets colours with no
surface. ``parts_without_room`` finds such a part, and ``trim_view`` removes it from the
drawing.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from .views import View


def parts_without_room(view_mask: np.ndarray, opposite_mask: np.ndarray, *, slack_px: int = 15,
                       min_thickness_px: int = 40, max_shift_px: int = 200) -> np.ndarray:
    """Aligned-frame pixels of a view that lie outside the mirror image of the opposite view.

    Only thick parts count (a tail, a wing, a hat): an outline that is ``slack_px`` off, or
    an arm that is drawn a little higher, is a normal disagreement and stays. A part is
    thick if a disc of ``min_thickness_px`` fits in it; the whole part outside the mirror
    outline is then returned, with its thin fringe.
    """
    mirror = opposite_mask[:, ::-1]
    # Each view is centred on its own outline, so a part on one side moves the whole figure.
    # Slide the mirror image to where the two outlines overlap most.
    columns_a, columns_b = view_mask.sum(axis=0).astype(float), mirror.sum(axis=0).astype(float)
    shifts = range(-max_shift_px, max_shift_px + 1)
    overlap = [np.minimum(columns_a, np.roll(columns_b, shift)).sum() for shift in shifts]
    mirror = np.roll(mirror, shifts[int(np.argmax(overlap))], axis=1)
    excess = view_mask & ~ndimage.binary_dilation(mirror, iterations=slack_px)
    radius = max(1, min_thickness_px // 2)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    core = ndimage.binary_opening(excess, structure=xx * xx + yy * yy <= radius * radius)
    return ndimage.binary_propagation(core, mask=excess)


def trim_view(view: View, remove: np.ndarray) -> tuple[Image.Image, float]:
    """The view's raw image with ``remove`` (an aligned-frame mask) made transparent.

    Returns the RGBA image (background transparent too) and the removed share of the figure.
    """
    to = view.to_aligned
    height, width = view.raw_mask.shape
    ys, xs = np.nonzero(view.raw_mask)
    col = np.clip(np.round((xs - to["left"]) * to["scale"] + to["x0"]).astype(int), 0, remove.shape[1] - 1)
    row = np.clip(np.round((ys - to["top"]) * to["scale"] + to["y0"]).astype(int), 0, remove.shape[0] - 1)
    gone = remove[row, col]
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[ys[~gone], xs[~gone]] = 255
    image = view.raw.copy()
    image.putalpha(Image.fromarray(alpha))
    return image, float(gone.mean())
