"""Remove the step that is left where two source images meet inside one part of a texture.

``seams.level_seams`` makes the two sides of a seam meet at the scale of its surface
nodes (about 24 aligned pixels). Finer differences stay: along the seam, the texel on
one side comes from one drawing and its neighbour from another, and the eye finds that
line. ``polish_seams`` works on the baked texture itself. Each texel near a seam gets a
small colour offset; the offsets of two texels on opposite sides of the seam must cancel
the colour step between them, and offsets change smoothly inside the area of one source.
That is the same least-squares problem as the leveling, at texel scale, and it needs
only the texture, the source of each texel and the part of each texel. ``smooth_fill``
does the same for the texels that no source painted. No offset crosses
from one part to another, so the edge of a scarf on fur stays as sharp as it was drawn.

Texels are neighbours if they touch in the UV layout, or across the border of two UV
islands (``stitch_islands``).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg, splu
from scipy.spatial import cKDTree

from .checks import lab

Stitches = tuple[np.ndarray, np.ndarray]   # flat texel indices (p, q) of neighbours across a UV island border


def stitch_islands(rows: np.ndarray, cols: np.ndarray, points: np.ndarray, normals: np.ndarray,
                   covered: np.ndarray) -> Stitches:
    """Pairs of texels that touch on the surface but lie on two UV islands.

    In the UV layout, the texels on the two sides of an island border are far apart, so a
    seam or a gap that crosses the border would be cut there. A texel on the edge of an
    island is paired with the nearest edge texels of other places in the layout that lie
    within one and a half texel spacings in space and face the same way.
    """
    height, width = covered.shape
    edge = covered & ~ndimage.binary_erosion(covered)
    on_edge = np.nonzero(edge[rows, cols])[0]
    if len(on_edge) < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    place = points[on_edge]
    tree = cKDTree(place)
    spacing = float(np.median(tree.query(place[:: max(1, len(place) // 5000)], k=2)[0][:, 1]))
    distance, other = tree.query(place, k=min(6, len(place)), distance_upper_bound=1.5 * max(spacing, 1e-12), workers=-1)
    source = np.repeat(np.arange(len(place)), other.shape[1])
    target = other.ravel()
    found = np.isfinite(distance.ravel()) & (target < len(place)) & (source < target)
    source, target = source[found], target[found]
    apart = (np.abs(rows[on_edge[source]] - rows[on_edge[target]]) + np.abs(cols[on_edge[source]] - cols[on_edge[target]])) > 2
    same_way = (normals[on_edge[source]] * normals[on_edge[target]]).sum(axis=1) > 0.3
    keep = apart & same_way
    first = rows[on_edge[source[keep]]] * width + cols[on_edge[source[keep]]]
    second = rows[on_edge[target[keep]]] * width + cols[on_edge[target[keep]]]
    return first.astype(np.int64), second.astype(np.int64)


def _neighbours(shape: tuple[int, int], stitches: Stitches | None) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(shape[0] * shape[1]).reshape(shape)
    first = [index[:, :-1].ravel(), index[:-1].ravel()]
    second = [index[:, 1:].ravel(), index[1:].ravel()]
    if stitches is not None:
        first.append(stitches[0])
        second.append(stitches[1])
    return np.concatenate(first), np.concatenate(second)


def _pairs(owner: np.ndarray, part: np.ndarray, covered: np.ndarray, stitches: Stitches | None = None
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat indices (p, q) of touching covered texels of one part, and whether a seam runs between them.

    A texel without a source (-1: a filled texel) continues the source next to it.
    """
    first, second = _neighbours(owner.shape, stitches)
    flat_owner, flat_part, flat_covered = owner.ravel(), part.ravel(), covered.ravel()
    keep = flat_covered[first] & flat_covered[second] & (flat_part[first] == flat_part[second])
    first, second = first[keep], second[keep]
    seam = (flat_owner[first] >= 0) & (flat_owner[second] >= 0) & (flat_owner[first] != flat_owner[second])
    return first, second, seam


def seam_contrast(image: np.ndarray, owner: np.ndarray, part: np.ndarray, covered: np.ndarray, *,
                  beside_px: int = 3, stitches: Stitches | None = None) -> dict:
    """How much a seam shows: the colour step across it against the step between texels beside it.

    ``across_delta_e`` is the mean CIE76 difference of two touching texels on opposite
    sides of a seam; ``beside_delta_e`` the same for touching texels of one source within
    ``beside_px`` of a seam (the texture's own grain there). A ``ratio`` near 1 means
    the seam is no stronger than the grain.
    """
    first, second, seam = _pairs(owner, part, covered, stitches)
    if not seam.any():
        return {"seam_pairs": 0, "across_delta_e": 0.0, "beside_delta_e": 0.0, "ratio": 1.0}
    flat = np.clip(image, 0, 255).astype(np.uint8).reshape(-1, 3)
    near = np.zeros(owner.size, dtype=bool)
    near[first[seam]] = near[second[seam]] = True
    near = ndimage.binary_dilation(near.reshape(owner.shape), iterations=beside_px).ravel()
    beside = ~seam & near[first] & near[second]

    def step(mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        return float(np.linalg.norm(lab(flat[first[mask]]) - lab(flat[second[mask]]), axis=1).mean())

    across, grain = step(seam), step(beside)
    return {"seam_pairs": int(seam.sum()), "across_delta_e": round(across, 2), "beside_delta_e": round(grain, 2),
            "ratio": round(across / max(grain, 1e-6), 3)}


def smooth_fill(image: np.ndarray, owner: np.ndarray, part: np.ndarray, covered: np.ndarray, *,
                detail: float = 0.7, detail_px: float = 6.0, stitches: Stitches | None = None
                ) -> tuple[np.ndarray, dict]:
    """Texels that no source painted, filled again so that their tone runs smoothly between the painted texels.

    The bake fills such a texel with the colour of the nearest painted one. Over a wide
    gap that gives flat patches with hard borders. Here the tone of the gap is the smooth
    surface that meets the painted texels around it (the Laplace equation, inside one
    part; ``stitches`` carries it across UV island borders). The grain comes from the first fill: its difference to its
    own blur (radius ``detail_px`` texels), scaled by ``detail``. A gap with no painted
    texel around it in the layout keeps the first fill.
    """
    result = np.asarray(image, dtype=np.float64).copy()
    gap = covered & (owner < 0)
    if not gap.any() or not (covered & (owner >= 0)).any():
        return result, {"filled_texels": int(gap.sum())}
    height, width = owner.shape
    first, second = _neighbours(owner.shape, stitches)
    flat_gap, flat_part, flat_covered = gap.ravel(), part.ravel(), covered.ravel()
    keep = flat_covered[first] & flat_covered[second] & (flat_part[first] == flat_part[second]) \
        & (flat_gap[first] | flat_gap[second])
    first, second = first[keep], second[keep]
    unknown = np.full(height * width, -1)
    ids = np.nonzero(flat_gap)[0]
    unknown[ids] = np.arange(len(ids))
    p, q = unknown[first], unknown[second]
    inner = (p >= 0) & (q >= 0)
    edge = ~inner                                   # one side is a painted texel: it fixes the value
    inside = np.where(p[edge] >= 0, p[edge], q[edge])
    outside = np.where(p[edge] >= 0, second[edge], first[edge])
    flat = result.reshape(-1, 3)
    hold = 1e-3                                     # a gap with no painted texel beside it stays as it is
    diagonal = (np.bincount(p[inner], minlength=len(ids)) + np.bincount(q[inner], minlength=len(ids))
                + np.bincount(inside, minlength=len(ids)) + hold)
    matrix = sparse.coo_matrix(
        (np.concatenate([diagonal, -np.ones(inner.sum()), -np.ones(inner.sum())]),
         (np.concatenate([np.arange(len(ids)), p[inner], q[inner]]),
          np.concatenate([np.arange(len(ids)), q[inner], p[inner]]))), shape=(len(ids), len(ids))).tocsc()
    rhs = np.stack([np.bincount(inside, weights=flat[outside, channel], minlength=len(ids))
                    + hold * flat[ids, channel] for channel in range(3)], axis=1)
    if len(ids) <= 400_000:
        tone = splu(matrix).solve(rhs)
    else:
        scale = sparse.diags(1.0 / diagonal)
        tone = np.stack([cg(matrix, rhs[:, channel], x0=flat[ids, channel], M=scale, rtol=1e-4, maxiter=600)[0]
                         for channel in range(3)], axis=1)
    weight = ndimage.gaussian_filter(covered.astype(np.float64), detail_px)
    blurred = np.dstack([ndimage.gaussian_filter(result[..., channel] * covered, detail_px) for channel in range(3)])
    grain = (result - blurred / np.maximum(weight, 1e-6)[..., None]).reshape(-1, 3)[ids]
    flat[ids] = np.clip(tone + detail * grain, 0, 255)
    return result, {"filled_texels": int(len(ids))}


def polish_seams(image: np.ndarray, owner: np.ndarray, part: np.ndarray, covered: np.ndarray, *, reach: int = 12,
                 smooth: float = 0.5, robust: float = 120.0, max_shift: float = 40.0, along: float = 2.0,
                 stitches: Stitches | None = None) -> tuple[np.ndarray, dict]:
    """The texture with the seams inside each part closed, and a short report.

    ``image`` is (H, W, 3) float in the UV layout, ``owner`` the source index of each
    texel (-1: filled), ``part`` its part (texels of different parts are never coupled)
    and ``covered`` the texels that the mesh uses. Offsets are solved for the texels
    within ``reach`` texels of a seam and are zero beyond. ``smooth`` weighs smoothness
    against closing the step: a constant step closes to 1 / (1 + smooth / (2 reach)) of
    itself, 98% with the defaults. The step to close is the mean over the seam texels
    within about ``along`` texels (0: each pair's own step, which also removes the grain
    on the seam line; on Pip the seam was then 0.6 times as strong as the grain beside
    it, with 2 it was 1.2 times, unpolished 2.7 times). A step above ``robust`` (sRGB units) is a drawn edge that happens
    to lie on the seam; it counts less.
    """
    first, second, seam = _pairs(owner, part, covered, stitches)
    result = np.asarray(image, dtype=np.float64).copy()
    if not seam.any():
        return result, {"polished_texels": 0, "polish_mean_shift": 0.0}
    height, width = owner.shape
    on_seam = np.zeros(height * width, dtype=bool)
    on_seam[first[seam]] = on_seam[second[seam]] = True
    band = (ndimage.distance_transform_edt(~on_seam.reshape(height, width)) <= reach).ravel() & covered.ravel()
    if stitches is not None:   # the band continues on the island across the border
        for _ in range(2):
            reached = band[stitches[0]] | band[stitches[1]]
            grown = np.zeros(height * width, dtype=bool)
            grown[stitches[0][reached]] = grown[stitches[1][reached]] = True
            band |= ndimage.binary_dilation(grown.reshape(height, width), iterations=reach // 2).ravel() & covered.ravel()
    unknown = np.full(height * width, -1)
    ids = np.nonzero(band)[0]
    unknown[ids] = np.arange(len(ids))
    flat = result.reshape(-1, 3)
    p, q = unknown[first], unknown[second]
    inner = (p >= 0) & (q >= 0)
    jump = flat[first] - flat[second]
    if along > 0:
        # Only the step that is common to the texels along the seam is closed. The difference from
        # texel to texel is the grain of the texture, and a seam without grain shows as a smooth line.
        flat_owner = owner.ravel()
        sign = np.where(flat_owner[first[seam]] < flat_owner[second[seam]], 1.0, -1.0)   # always a - b for a < b
        at = (first[seam] // width, first[seam] % width)
        total = np.zeros((height, width, 3))
        count = np.zeros((height, width))
        np.add.at(total, at, jump[seam] * sign[:, None])
        np.add.at(count, at, 1.0)
        common = np.dstack([ndimage.gaussian_filter(total[..., c], along) for c in range(3)]) \
            / np.maximum(ndimage.gaussian_filter(count, along), 1e-9)[..., None]
        jump[seam] = common[at] * sign[:, None]
    weight = np.where(seam, 1.0 / (1.0 + (np.linalg.norm(jump, axis=1) / robust) ** 2), smooth)
    # Texels at the rim of the band hold the offset to zero through their neighbours outside it.
    rim = (p >= 0) ^ (q >= 0)
    diagonal = (np.bincount(p[inner], weights=weight[inner], minlength=len(ids))
                + np.bincount(q[inner], weights=weight[inner], minlength=len(ids))
                + np.bincount(np.where(p[rim] >= 0, p[rim], q[rim]), weights=weight[rim], minlength=len(ids)) + 1e-4)
    matrix = sparse.coo_matrix(
        (np.concatenate([diagonal, -weight[inner], -weight[inner]]),
         (np.concatenate([np.arange(len(ids)), p[inner], q[inner]]),
          np.concatenate([np.arange(len(ids)), q[inner], p[inner]]))), shape=(len(ids), len(ids))).tocsc()
    across = inner & seam
    rhs = np.zeros((len(ids), 3))
    for channel in range(3):   # minimise weight * (offset_p - offset_q + jump)^2
        pull = weight[across] * jump[across, channel]
        rhs[:, channel] = (np.bincount(q[across], weights=pull, minlength=len(ids))
                           - np.bincount(p[across], weights=pull, minlength=len(ids)))
    if len(ids) <= 400_000:
        offset = splu(matrix).solve(rhs)
    else:
        scale = sparse.diags(1.0 / diagonal)
        offset = np.stack([cg(matrix, rhs[:, channel], M=scale, rtol=1e-4, maxiter=400)[0] for channel in range(3)], axis=1)
    offset = np.clip(offset, -max_shift, max_shift)
    flat[ids] = np.clip(flat[ids] + offset, 0, 255)
    return result, {"polished_texels": int(len(ids)),
                    "polish_mean_shift": round(float(np.abs(offset).mean()), 3)}
