"""Choose one source image per region of the surface, and level the colours across the seams.

A texture baked from several drawings shows a conflict wherever two drawings meet:
their tones differ (one drawing shows the tail light, another dark), and on a bumpy
surface the better-facing drawing changes from texel to texel, which mixes them in
streaks. The remedy has two parts, both on a coarse graph of surface nodes:

1. ``choose_sources``: how well a source sees a *neighbourhood* decides who paints it,
   not how well it sees one texel. Regions come out compact, with few and short seams.
2. ``level_seams``: a smooth colour offset per region makes the two sides of every seam
   meet (Lempitsky and Ivanov 2007, as used by Waechter et al. 2014). The offset fades
   with distance from the seam, so each drawing keeps its own tone where it is alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .checks import lab

LIGHTNESS_WEIGHT = 0.4   # in comparisons between sources: drawings differ in tone (lightness) far more than in hue
NORMAL_WEIGHT = 1.5   # nodes that face apart by 90 degrees are 2.1 sigma apart: the two sides of a thin flap stay separate
SEAM_EDGES = 8        # neighbours per node that count as adjacent


@dataclass
class Regions:
    cell: np.ndarray        # (N,) node of each texel
    texels: np.ndarray      # (M,) texels per node
    centre: np.ndarray      # (M, 3) node position
    feature: np.ndarray     # (M, 6) node position / sigma and normal * NORMAL_WEIGHT
    normal: np.ndarray      # (M, 3) surface normal averaged over the neighbourhood: the facing without the bumps
    tree: cKDTree
    near: np.ndarray        # (M, k) neighbours of each node, itself first
    gauss: np.ndarray       # (M, k) Gaussian weight of each neighbour
    sigma: float
    spacing: float
    quality: np.ndarray | None = None   # (V, M) how well each source sees the neighbourhood of a node
    lowpass: np.ndarray | None = None   # (V, M, 3) mean sRGB colour a source shows around a node
    has: np.ndarray | None = None       # (V, M) the source shows enough of the neighbourhood for ``lowpass``
    label: np.ndarray | None = None     # (M,) source that paints the node, -1 if none sees it
    group: np.ndarray | None = None     # (M,) semantic owner of each node, or None for legacy behavior

    def per_node(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(self.cell, weights=values, minlength=len(self.texels))

    def add_sources(self, quality: np.ndarray, colours: np.ndarray) -> None:
        """Smooth each source's fields. ``quality`` (V, N) is 0 where a source does not paint a texel;
        ``colours`` is (V, N, 3) uint8."""
        count, sources = len(self.texels), len(quality)
        area = (self.gauss * self.texels[self.near]).sum(axis=1)
        self.quality = np.zeros((sources, count), dtype=np.float32)
        self.lowpass = np.zeros((sources, count, 3), dtype=np.float32)
        self.has = np.zeros((sources, count), dtype=bool)
        for v in range(sources):
            used = quality[v] > 0
            self.quality[v] = (self.gauss * self.per_node(quality[v])[self.near]).sum(axis=1) / area
            seen = (self.gauss * self.per_node(used)[self.near]).sum(axis=1)
            for channel in range(3):
                total = (self.gauss * self.per_node(np.where(used, colours[v][:, channel], 0))[self.near]).sum(axis=1)
                self.lowpass[v, :, channel] = total / np.maximum(seen, 1e-9)
            self.has[v] = seen > 0.05 * area
        self.label = np.full(count, -1)


def build_regions(points: np.ndarray, normals: np.ndarray, *, sigma: float, spacing: float,
                  neighbours: int = 48, group_labels: np.ndarray | None = None) -> Regions:
    """Surface nodes: one per ``spacing`` cell and facing direction, linked to their neighbours."""
    keys = np.floor(points / spacing).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    axis = np.argmax(np.abs(normals), axis=1)
    facing = axis * 2 + (normals[np.arange(len(normals)), axis] > 0)
    if group_labels is None:
        flat = ((keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]) * 6 + facing
        _, cell = np.unique(flat, return_inverse=True)
    else:
        group_labels = np.asarray(group_labels, dtype=np.int64)
        if group_labels.shape != (len(points),):
            raise ValueError("Semantic group labels must match surface sample count")
        # A cell/facing pair occupied by two different physical parts creates
        # two separate nodes. Never average a scarf with the nearby tail.
        bins = np.column_stack([keys, facing, group_labels])
        _, cell = np.unique(bins, axis=0, return_inverse=True)
    count = cell.max() + 1
    texels = np.bincount(cell, minlength=count).astype(np.float64)
    group = None if group_labels is None else (
        np.rint(np.bincount(cell, weights=group_labels, minlength=count) / texels).astype(np.int64))

    def per_node(values):
        return np.bincount(cell, weights=values, minlength=count)

    centre = np.stack([per_node(points[:, i]) for i in range(3)], axis=1) / texels[:, None]
    total = np.stack([per_node(normals[:, i]) for i in range(3)], axis=1)
    # The facing without the bumps. A node's own mean normal leans to the axis that grouped
    # its texels, so average over the nodes around it in space, but only those that face
    # the same half-space: the other side of a thin flap must not cancel it.
    smooth = total / np.maximum(np.linalg.norm(total, axis=1), 1e-12)[:, None]
    distance, around = cKDTree(centre / sigma).query(centre / sigma, k=min(neighbours, count), workers=-1)
    if count == 1:
        distance, around = distance[:, None], around[:, None]
    weight = np.exp(-0.5 * distance ** 2)
    if group is not None:
        weight *= group[around] == group[:, None]
    for _ in range(2):
        same_side = np.clip(np.einsum("mc,mkc->mk", smooth, smooth[around]), 0, 1)
        smooth = np.einsum("mk,mkc->mc", weight * same_side, total[around])
        smooth /= np.maximum(np.linalg.norm(smooth, axis=1), 1e-12)[:, None]
    feature = np.hstack([centre / sigma, smooth * NORMAL_WEIGHT])
    if group is not None:
        feature = np.column_stack([feature, group * 20.0])
    tree = cKDTree(feature)
    distance, near = tree.query(feature, k=min(neighbours, count), workers=-1)
    if count == 1:
        distance, near = distance[:, None], near[:, None]
    gauss = np.exp(-0.5 * distance ** 2)
    if group is not None:
        gauss *= group[near] == group[:, None]
    return Regions(cell, texels, centre, feature, smooth, tree, near, gauss, sigma, spacing, group=group)


@dataclass
class Testimony:
    """What one source shows around each node, whether or not it paints there."""
    picture: np.ndarray     # (h, w, 3) low-pass picture of the source in its camera frame on a coarse grid (weighted Lab)
    valid: np.ndarray       # (h, w) the picture shows the figure there
    col: np.ndarray         # (M,) grid cell of each node in ``picture``
    row: np.ndarray
    weight: np.ndarray      # (M,) how well the source can see the node (0: hidden or grazing)


def slack_match(claims: np.ndarray, testimony: Testimony, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Compare each claimed colour (V, M, 3) with what a witness shows at and around each node.

    Returns (V, M) arrays: the difference at the node itself and the smallest difference
    within ``radius`` cells. Infinite where the witness cannot see the node or shows no
    figure. Two drawings place the same feature a little apart on the surface, so only
    the second number says whether the witness knows the claimed colour at all.
    """
    height, width = testimony.valid.shape
    here, best = np.full(claims.shape[:2], np.inf), np.full(claims.shape[:2], np.inf)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            r, c = testimony.row + dy, testimony.col + dx
            ok = (r >= 0) & (r < height) & (c >= 0) & (c < width) & (testimony.weight > 0)
            ok[ok] = testimony.valid[r[ok], c[ok]]
            distance = np.linalg.norm(claims[:, ok] - testimony.picture[r[ok], c[ok]][None], axis=2)
            if dx == 0 and dy == 0:
                here[:, ok] = distance
            best[:, ok] = np.minimum(best[:, ok], distance)
    return here, best


def _components(label: np.ndarray, rows: np.ndarray, cols: np.ndarray, keep: np.ndarray) -> tuple[int, np.ndarray]:
    count = len(label)
    graph = sparse.coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(count, count))
    return csgraph.connected_components(graph, directed=False)


def choose_sources(regions: Regions, witness: np.ndarray, testimony: list[Testimony] | None = None, *,
                   agree_delta_e: float = 15.0, slack_cells: int = 7, contest_delta_e: float = 20.0,
                   match_delta_e: float = 12.0, min_region_nodes: int = 25) -> dict:
    """Set ``regions.label``; return the score (V, M) that decided it and some counts.

    Colours are compared in Lab with the lightness scaled by ``LIGHTNESS_WEIGHT``.

    1. Misplaced parts. Where the mesh is wider than a drawing shows, the drawing puts
       a part on the wrong surface: the back drawing puts the scarf on a tail that the
       mesh makes too wide. A ``witness`` source (a drawing) that can see a node objects
       to another source's colour there if its own picture shows something else at that
       point (more than ``contest_delta_e``) and nothing like the claimed colour within
       ``slack_cells`` (more than ``match_delta_e``). The slack matters: two drawings
       place the same feature a little apart, and that is no objection. A source does
       not paint a node if most of its witnesses object.
    2. Agreement. The remaining score falls with the smallest difference to the
       witnesses within the slack (scale ``agree_delta_e``).
    3. The source with the best score paints each node.
    4. Contested features. Where the two best sources show different colours at the
       same point, a seam between them would cut a feature in two. Each connected patch
       of such nodes goes whole to the source that sees it best.
    5. Regions smaller than ``min_region_nodes`` join a neighbour.
    """
    score = regions.quality.astype(np.float64).copy()
    sources, count = score.shape
    colour = np.stack([lab(np.clip(regions.lowpass[v], 0, 255).astype(np.uint8)) for v in range(sources)])
    colour[..., 0] *= LIGHTNESS_WEIGHT
    rejected = 0
    misplaced = np.zeros(count, dtype=bool)
    contested = np.zeros(count, dtype=bool)
    if agree_delta_e and sources > 1 and testimony is not None:
        total, weight, against = np.zeros((sources, count)), np.zeros((sources, count)), np.zeros((sources, count))
        for u in range(sources):
            if not witness[u]:
                continue
            here, nearby = slack_match(colour, testimony[u], slack_cells)
            here[u] = nearby[u] = np.inf   # a source is no witness of itself
            heard = np.isfinite(here) & regions.has
            say = testimony[u].weight[None]
            against += (heard & (here > contest_delta_e) & (nearby > match_delta_e)) * say
            total += np.where(heard, np.minimum(nearby, 100), 0) * say
            weight += heard * say
        refuse = (against > 0.5 * weight) & (weight > 0.15) & (score > 0)
        rejected = int(refuse.sum())
        # Worth showing: the refused source would have been among the best painters of the node.
        misplaced = (refuse & (regions.quality >= 0.5 * regions.quality.max(axis=0)[None])).any(axis=0)
        score *= np.exp(-np.where(weight > 0, total / np.maximum(weight, 1e-12), 0) / agree_delta_e)
        score[refuse] = 0
    label = np.where(score.max(axis=0) > 0, np.argmax(score, axis=0), -1)
    rows = np.repeat(np.arange(count), min(SEAM_EDGES, regions.near.shape[1] - 1))
    cols = regions.near[:, 1:SEAM_EDGES + 1].ravel()
    same_group = (np.ones(len(rows), dtype=bool) if regions.group is None
                  else regions.group[rows] == regions.group[cols])
    contested_nodes = 0
    if contest_delta_e and sources > 1:
        order = np.argsort(score, axis=0)
        first, second = order[-1], order[-2]
        every = np.arange(count)
        close = score[second, every] > 0.25 * score[first, every]
        both = regions.has[first, every] & regions.has[second, every] & (score[second, every] > 0)
        apart = np.linalg.norm(colour[first, every] - colour[second, every], axis=1)
        contested = close & both & (apart > contest_delta_e) & (label >= 0)
        contested_nodes = int(contested.sum())
        if contested.any():
            parts, part = _components(label, rows, cols, contested[rows] & contested[cols] & same_group)
            ids = np.nonzero(contested)[0]
            sums = np.zeros((parts, sources))
            for v in range(sources):
                sums[:, v] = np.bincount(part[ids], weights=score[v, ids], minlength=parts)
            winner = np.argmax(sums, axis=1)[part[ids]]
            can = score[winner, ids] > 0
            label[ids[can]] = winner[can]
    merged = 0
    for _ in range(6):
        parts, part = _components(label, rows, cols, (label[rows] == label[cols]) & same_group)
        small = (np.bincount(part, minlength=parts)[part] < min_region_nodes) & (label >= 0)
        if not small.any():
            break
        # A small region joins the neighbouring large region whose source sees it best.
        ids = np.nonzero(small)[0]
        votes = np.zeros((len(ids), sources))
        neighbour = regions.near[ids]
        valid = ~small[neighbour] & (label[neighbour] >= 0)
        if regions.group is not None:
            valid &= regions.group[neighbour] == regions.group[ids, None]
        strength = regions.gauss[ids] * valid * score[np.clip(label[neighbour], 0, None), neighbour]
        np.add.at(votes, (np.repeat(np.arange(len(ids)), neighbour.shape[1]),
                          np.clip(label[neighbour], 0, None).ravel()), strength.ravel())
        votes[score[:, ids].T <= 0] = 0    # the new source must see the node's neighbourhood
        winner = np.argmax(votes, axis=1)
        moved = votes[np.arange(len(ids)), winner] > 0
        merged += int(moved.sum())
        if not moved.any():
            break
        label[ids[moved]] = winner[moved]
    regions.label = label
    parts, part = _components(label, rows, cols, (label[rows] == label[cols]) & same_group)
    return {"score": score, "regions": int(len(np.unique(part[label >= 0]))), "merged_nodes": merged,
            "rejected_claims": rejected, "contested_nodes": contested_nodes,
            "misplaced": misplaced, "contested": contested}


def level_seams(regions: Regions, *, reach: float, robust_delta_e: float = 18.0, max_shift: float = 45.0,
                keep_strength: float = 30.0) -> tuple[np.ndarray, dict]:
    """sRGB offset per node (M, 3) that makes the two sides of each seam meet.

    At a seam between sources a and b, the offsets must cancel the difference between the
    colours that a and b show at that same place. A difference above about
    ``robust_delta_e`` is not a tone difference (the sources show different parts there)
    and is left alone. ``reach`` is the distance over which an offset fades, in the same
    units as the node positions; it is shorter (by up to sqrt(1 + ``keep_strength``)) on
    surface that the painting source sees squarely.
    """
    label, near, count = regions.label, regions.near, len(regions.label)
    width = min(SEAM_EDGES, near.shape[1] - 1)
    i = np.repeat(np.arange(count), width)
    j = near[:, 1:width + 1].ravel()
    g = regions.gauss[:, 1:width + 1].ravel()
    keep = (i < j) & (label[i] >= 0) & (label[j] >= 0)
    # Physical part seams are not color-adjustment seams.
    if regions.group is not None:
        keep &= regions.group[i] == regions.group[j]
    # An edge listed from both ends counts once: sort the pair and drop duplicates.
    reverse = (i > j) & (label[i] >= 0) & (label[j] >= 0)
    if regions.group is not None:
        reverse &= regions.group[i] == regions.group[j]
    pairs = np.concatenate([np.stack([i[keep], j[keep]], axis=1), np.stack([j[reverse], i[reverse]], axis=1)])
    weights = np.concatenate([g[keep], g[reverse]])
    _, first = np.unique(pairs[:, 0] * count + pairs[:, 1], return_index=True)
    i, j, g = pairs[first, 0], pairs[first, 1], weights[first]
    a, b = label[i], label[j]
    seam = a != b
    jump = np.zeros((len(i), 3))
    samples = np.zeros(len(i))
    for at in (i, j):   # the difference a - b where both sources show the same node
        both = seam & regions.has[a, at] & regions.has[b, at]
        jump[both] += regions.lowpass[a[both], at[both]] - regions.lowpass[b[both], at[both]]
        samples[both] += 1
    measured = seam & (samples > 0)
    jump[measured] /= samples[measured][:, None]
    delta = np.zeros(len(i))
    if measured.any():
        mid = np.zeros((len(i), 3))
        for at in (i, j):
            both = measured & regions.has[a, at] & regions.has[b, at]
            mid[both] += regions.lowpass[b[both], at[both]]
        mid[measured] /= samples[measured][:, None]
        delta[measured] = np.linalg.norm(lab(np.clip(mid[measured] + jump[measured], 0, 255).astype(np.uint8))
                                         - lab(np.clip(mid[measured], 0, 255).astype(np.uint8)), axis=1)
    weight = np.where(seam, g * np.exp(-(delta / robust_delta_e) ** 2) * measured, g)
    # A node that its source sees squarely keeps its colour; the correction goes to the oblique
    # surface next to the seams, where a drawing is least reliable.
    held = regions.quality[np.clip(label, 0, None), np.arange(count)] * (label >= 0)
    anchor = (regions.spacing / reach) ** 2 * (1 + keep_strength * held / max(float(regions.quality.max()), 1e-9))
    diagonal = np.bincount(i, weights=weight, minlength=count) + np.bincount(j, weights=weight, minlength=count) + anchor
    matrix = sparse.coo_matrix((np.concatenate([diagonal, -weight, -weight]),
                                (np.concatenate([np.arange(count), i, j]), np.concatenate([np.arange(count), j, i]))),
                               shape=(count, count)).tocsc()
    rhs = np.zeros((count, 3))
    pull = weight[:, None] * jump * seam[:, None]
    for channel in range(3):   # minimise weight * (offset_i - offset_j + jump)^2
        rhs[:, channel] = np.bincount(j, weights=pull[:, channel], minlength=count) \
            - np.bincount(i, weights=pull[:, channel], minlength=count)
    offset = np.clip(splu(matrix).solve(rhs), -max_shift, max_shift)
    after = np.zeros(len(i))
    if measured.any():
        left = jump[measured] + offset[i[measured]] - offset[j[measured]]
        after[measured] = np.linalg.norm(lab(np.clip(mid[measured] + left, 0, 255).astype(np.uint8))
                                         - lab(np.clip(mid[measured], 0, 255).astype(np.uint8)), axis=1)
    tone = measured & (delta <= 2 * robust_delta_e)
    node_jump = np.zeros(count)   # the largest tone difference at a seam that touches the node, before leveling
    for at in (i, j):
        np.maximum.at(node_jump, at[tone], delta[tone])
    report = {"node_jump": node_jump,
              "seam_edges": int(seam.sum()), "seam_edges_measured": int(measured.sum()),
              "seam_delta_e_before": round(float(delta[tone].mean()), 2) if tone.any() else 0.0,
              "seam_delta_e_after": round(float(after[tone].mean()), 2) if tone.any() else 0.0,
              "seam_edges_different_parts": int((measured & ~tone).sum())}
    return offset, report


def per_texel(regions: Regions, points: np.ndarray, score: np.ndarray, offset: np.ndarray | None,
              chunk: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Source (N,) and colour offset (N, 3) of each texel, interpolated from its four nearest nodes.

    A texel may take only a source that paints one of those nodes, so the seams follow
    the smooth score fields, not the node cells, and no new islands appear.
    """
    count = len(points)
    label = np.full(count, -1)
    shift = np.zeros((count, 3), dtype=np.float32)
    node_normal = regions.feature[:, 3:6]
    k = min(4, len(regions.label))
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        feature = np.hstack([points[start:stop] / regions.sigma, node_normal[regions.cell[start:stop]]])
        if regions.group is not None:
            feature = np.column_stack([feature, regions.group[regions.cell[start:stop]] * 20.0])
        distance, idx = regions.tree.query(feature, k=k, workers=-1)
        if k == 1:
            distance, idx = distance[:, None], idx[:, None]
        weight = 1.0 / (distance + 1e-3) ** 2
        if regions.group is not None:
            owners = regions.group[regions.cell[start:stop]]
            weight *= regions.group[idx] == owners[:, None]
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-20)
        node_label = regions.label[idx]
        best = np.full(stop - start, -np.inf)
        chosen = np.full(stop - start, -1)
        for v in np.unique(node_label[node_label >= 0]):
            field = (weight * score[v][idx]).sum(axis=1)
            field[~(node_label == v).any(axis=1)] = -np.inf
            better = field > best
            best[better], chosen[better] = field[better], v
        label[start:stop] = chosen
        if offset is not None:
            match = (node_label == chosen[:, None]) * weight
            total = match.sum(axis=1, keepdims=True)
            shift[start:stop] = np.einsum("nk,nkc->nc", match, offset[idx]) / np.maximum(total, 1e-12)
    return label, shift


def pair_table(regions: Regions, names: list[str], contest_delta_e: float = 20.0) -> list[dict]:
    """For each pair of sources: how much surface both paint, and how they differ there.

    ``tone_delta_e`` is the mean CIE76 difference of their low-pass colours over the
    shared surface. ``different`` is the share of it where they show different things
    (weighted difference above ``contest_delta_e``), not only a different tone.
    """
    area = regions.texels / regions.texels.sum()
    colour = np.stack([lab(np.clip(regions.lowpass[v], 0, 255).astype(np.uint8)) for v in range(len(names))])
    rows = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            both = regions.has[a] & regions.has[b] & (regions.quality[a] > 0) & (regions.quality[b] > 0)
            if area[both].sum() < 0.001:
                continue
            gap = colour[a, both] - colour[b, both]
            plain = np.linalg.norm(gap, axis=1)
            gap[:, 0] *= LIGHTNESS_WEIGHT
            weight = area[both]
            rows.append({"views": [names[a], names[b]], "overlap": round(float(weight.sum()), 4),
                         "tone_delta_e": round(float((plain * weight).sum() / weight.sum()), 2),
                         "different": round(float(((np.linalg.norm(gap, axis=1) > contest_delta_e) * weight).sum()
                                                  / weight.sum()), 4)})
    return sorted(rows, key=lambda row: -row["overlap"] * row["different"])


def suggest_views(regions: Regions, mask: np.ndarray, *, min_nodes: int = 40, step: float = 15.0,
                  limit: int = 6) -> list[dict]:
    """Free cameras (az<deg>_el<deg>) that face the connected patches of ``mask``, largest first.

    A picture from such a camera (a new drawing, or ``repaint``) sees the patch squarely
    and can paint it whole.
    """
    count = len(mask)
    rows = np.repeat(np.arange(count), min(SEAM_EDGES, regions.near.shape[1] - 1))
    cols = regions.near[:, 1:SEAM_EDGES + 1].ravel()
    connected = mask[rows] & mask[cols]
    if regions.group is not None:
        connected &= regions.group[rows] == regions.group[cols]
    parts, part = _components(mask, rows, cols, connected)
    sizes = np.bincount(part[mask], minlength=parts)
    found: dict[str, int] = {}
    for index in np.argsort(-sizes):
        if sizes[index] < min_nodes:
            break
        normal = regions.normal[mask & (part == index)].mean(axis=0)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        azimuth = round(float(np.degrees(np.arctan2(normal[0], -normal[1]))) / step) * step
        elevation = float(np.clip(round(float(np.degrees(np.arcsin(np.clip(normal[2], -1, 1)))) / step) * step, -60, 60))
        azimuth = 180.0 if azimuth == -180 else azimuth
        name = f"az{azimuth:g}_el{elevation:g}"
        found[name] = found.get(name, 0) + int(sizes[index])
    total = float(regions.texels.sum())
    share = regions.texels.mean() / total   # mean share of the surface per node
    return [{"camera": name, "surface": round(nodes * share, 4)}
            for name, nodes in sorted(found.items(), key=lambda item: -item[1])[:limit]]

