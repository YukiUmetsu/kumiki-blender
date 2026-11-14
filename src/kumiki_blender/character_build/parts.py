"""Recognise the parts of a character on its mesh and in each drawing, then bake each part from its own pixels.

The drawings of one character and the mesh never agree exactly. Where the mesh is
wider than a drawing, the drawing's scarf lands on the tail. The view-by-view rules of
``seams.py`` cannot always tell, because they compare colours and do not know what a
part is. Here the parts are found first:

1. ``recognise_parts``: on the surface. A first bake from the drawings gives every
   surface node a colour. Nodes are grouped by colour class, the groups are split into
   connected patches of the surface (two red things that do not touch are two parts),
   and patches join while the colour step at their common border is small. Stripes
   inside a tail join; a scarf on fur does not. Geometry decides what touches: the two
   sides of a thin flap, or a tail that hangs in front of a back, are not neighbours.
2. ``view_part_labels``: in each drawing. The mesh's parts are projected into the
   drawing. That says where each part should be. The drawing's own colours then correct
   it: a pixel that lies on the projected tail but has the scarf's main colour, close to
   the projected scarf, belongs to the scarf.
3. ``bake_parts``: each part is baked only from the pixels that show that part, and the
   results are put together. Inside a part the views are joined as before (regions,
   leveling) and then polished (``polish.py``). Between parts nothing is blended, so
   the edge of a scarf stays crisp.

A part that colour and shape cannot find (a sleeve of the colour of the fur beside it),
or that they find with a ragged border, can be given: ``find_parts(given=...)`` takes
part handoffs (from ``split_views`` or ``box_part``), in which the pixels of the part
are marked in each drawing. ``given_part_texels`` puts them on the mesh.

The recognised parts are estimates, not verified labels: review ``turnaround.png``
before you rely on them. ``semantic.json`` gives the same masks in the format of the
``resolve-conflicts`` policy (authored masks), so a reviewed set can be used there.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.cluster.vq import kmeans2
from scipy.sparse import csgraph

from .checks import lab
from .mesh import Mesh, rasterize
from .seams import LIGHTNESS_WEIGHT, SEAM_EDGES, build_regions
from .texture import OWNER_COLOURS, bake_texture, render_textured, texel_samples, turnaround
from .views import CX, camera

PART_COLOURS = np.array([(66, 135, 245), (240, 90, 60), (70, 190, 110), (245, 200, 50), (170, 90, 210),
                         (40, 200, 210), (250, 140, 40), (140, 140, 140), (130, 80, 40), (250, 140, 200),
                         (120, 160, 40), (90, 60, 160)], dtype=np.uint8)
DOMINANT_SHARE = 0.15   # a colour class with this share of a part is one of its main colours


def _source_signature(handoff: Path, mesh: Mesh, corner_uv: np.ndarray, *,
                      views: list[str] | None, extra: list[dict] | None,
                      mirror_missing: bool) -> str:
    """SHA-256 of geometry/UVs and every source image driving part ownership.

    Without this lock an old UV mask could be reused after the eye, clothing,
    or tail geometry changes, silently painting the wrong anatomy.
    The signature excludes bake output resolution: a reviewed 256px part map
    may be intentionally re-used at 2048px on unchanged source geometry.
    """
    spec_file = handoff / "spec.json"
    spec_bytes = spec_file.read_bytes()
    spec = json.loads(spec_bytes)
    names = [
        name for name in spec["raw"]
        if (views is None or name in views)
        and (handoff / "raw" / f"{name}.png").is_file()
    ]
    digest = hashlib.sha256(b"kumiki-parts-source-v1\n")
    digest.update(spec_bytes)
    digest.update(json.dumps({
        "view_names": names,
        "mirror_missing": bool(mirror_missing),
        "extras": [
            {key: value for key, value in entry.items() if key != "image"}
            for entry in (extra or [])
        ],
    }, sort_keys=True, default=str).encode("utf-8"))
    for name in names:
        path = handoff / "raw" / f"{name}.png"
        digest.update(name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    for index, entry in enumerate(extra or []):
        digest.update(f"extra-image:{index}".encode("ascii"))
        with Path(entry["image"]).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    for name, array, dtype in (
        ("vertices", mesh.vertices, np.float64),
        ("faces", mesh.faces, np.int64),
        ("corner_uv", corner_uv, np.float64),
    ):
        value = np.ascontiguousarray(array, dtype=dtype)
        digest.update(name.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(memoryview(value).cast("B"))
    if mesh.colours is not None:
        colours = np.ascontiguousarray(mesh.colours, dtype=np.uint8)
        digest.update(b"vertex_colours")
        digest.update(str(colours.shape).encode("ascii"))
        digest.update(memoryview(colours).cast("B"))
    return digest.hexdigest()



def weighted_lab(rgb: np.ndarray) -> np.ndarray:
    """CIE Lab with the lightness scaled down: drawings differ in tone far more than in hue."""
    colour = lab(np.asarray(rgb, dtype=np.uint8).reshape(-1, 3))
    colour[:, 0] *= LIGHTNESS_WEIGHT
    return colour.reshape(np.shape(rgb))


def _distance_to(colour: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Smallest distance from each colour (N, 3) to any of the centres (C, 3)."""
    if len(centres) == 0:
        return np.full(len(colour), np.inf)
    return np.min(np.linalg.norm(colour[:, None, :] - centres[None], axis=2), axis=1)


class _Patches:
    """Touching patches of the surface and the borders between them; ``join`` merges two."""

    def __init__(self, patch: np.ndarray, rows: np.ndarray, cols: np.ndarray, area: np.ndarray,
                 step: np.ndarray | None = None, measured: np.ndarray | None = None):
        self.count = int(patch.max()) + 1
        self.owner = np.arange(self.count)
        self.area = np.bincount(patch, weights=area, minlength=self.count)
        cross = patch[rows] != patch[cols]
        a = np.minimum(patch[rows[cross]], patch[cols[cross]])
        b = np.maximum(patch[rows[cross]], patch[cols[cross]])
        key, inverse = np.unique(a * self.count + b, return_inverse=True)
        length = np.bincount(inverse).astype(float)
        known = np.bincount(inverse, weights=measured[cross]) if measured is not None else np.zeros(len(key))
        total = np.bincount(inverse, weights=step[cross] * measured[cross]) if step is not None else np.zeros(len(key))
        self.links: dict[int, dict[int, list[float]]] = {index: {} for index in range(self.count)}
        for pair, *stats in zip(key, length, known, total):
            i, j = int(pair // self.count), int(pair % self.count)
            self.links[i][j] = self.links[j][i] = list(stats)

    def step(self, i: int, j: int) -> float:
        """Mean colour step across the border of two patches (infinite if no painted cell touches it)."""
        _, known, total = self.links[i][j]
        return total / known if known > 0 else np.inf

    def join(self, gone: int, into: int) -> None:
        self.owner[self.owner == gone] = into
        self.area[into] += self.area[gone]
        for other, stats in self.links.pop(gone).items():
            self.links[other].pop(gone, None)
            if other == into:
                continue
            have = self.links[into].get(other)
            if have is None:
                self.links[into][other] = self.links[other][into] = list(stats)
            else:
                for index in range(3):
                    have[index] += stats[index]

    def weakest(self) -> tuple[float, int, int] | None:
        best = None
        for i, others in self.links.items():
            for j in others:
                if i < j:
                    step = self.step(i, j)
                    if best is None or step < best[0]:
                        best = (step, i, j)
        return best

    def absorb(self, loose) -> None:
        """Each patch that ``loose(index)`` accepts joins the neighbour with which it shares the longest border."""
        while True:
            found = [i for i in self.links if self.links[i] and loose(i)]
            if not found:
                return
            i = min(found, key=lambda index: self.area[index])
            self.join(i, max(self.links[i], key=lambda other: self.links[i][other][0]))


def shape_parts(mesh: Mesh, *, crease_radius: float, smooth_rounds: int = 12, min_share: float = 0.01
                ) -> tuple[np.ndarray, np.ndarray]:
    """Shape index of each mesh vertex (V,), and the concavity of the surface there (1/m, inward positive).

    A character's limbs, tail and ears meet the body in concave creases, and each of them
    is convex on the whole. The measure is taken on a smoothed copy of the mesh
    (``smooth_rounds`` neighbour averages), so that it finds the creases of the shape and
    not the valleys between tufts of fur.

    A crease is rarely a closed line, so the surface is not cut along a threshold. It is
    flooded from its most convex places upward (a watershed): the basins meet on the
    ridges of concavity. Two basins then join while the mean concavity along their border
    is below that of a circle of ``crease_radius``; what stays apart is separated by a
    crease. A shape smaller than ``min_share`` of the vertices joins the neighbour with
    the longest common border.
    """
    adjacency = mesh.adjacency()
    smooth = mesh.vertices.copy()
    for _ in range(smooth_rounds):
        smooth = 0.5 * smooth + 0.5 * (adjacency @ smooth)
    normal = Mesh(smooth, mesh.faces).vertex_normals()
    link = adjacency.tocoo()
    rows, cols = link.row, link.col
    step = smooth[cols] - smooth[rows]
    bend = -(step * (normal[cols] - normal[rows])).sum(axis=1) / np.maximum((step ** 2).sum(axis=1), 1e-18)
    count = len(smooth)
    concavity = np.bincount(rows, weights=bend, minlength=count) / np.maximum(np.bincount(rows, minlength=count), 1)
    for _ in range(2):
        concavity = 0.5 * concavity + 0.5 * (adjacency @ concavity)
    indptr, indices = adjacency.indptr, adjacency.indices
    basin = np.full(count, -1)
    basins = 0
    for vertex in np.argsort(concavity):          # flood from the most convex vertex upward
        around = indices[indptr[vertex]:indptr[vertex + 1]]
        flooded = around[basin[around] >= 0]
        if len(flooded):
            basin[vertex] = basin[flooded[np.argmin(concavity[flooded])]]
        else:
            basin[vertex] = basins
            basins += 1
    pieces = _Patches(basin, rows, cols, np.ones(count), step=np.maximum(concavity[rows], concavity[cols]),
                      measured=np.ones(len(rows)))
    while True:
        weakest = pieces.weakest()
        if weakest is None or weakest[0] >= 1.0 / crease_radius:
            break
        keep, gone = (weakest[1], weakest[2]) if pieces.area[weakest[1]] >= pieces.area[weakest[2]] else (weakest[2], weakest[1])
        pieces.join(gone, keep)
    pieces.absorb(lambda i: pieces.area[i] < min_share * count)
    return np.unique(pieces.owner[basin], return_inverse=True)[1], concavity


def recognise_parts(points: np.ndarray, normals: np.ndarray, colour: np.ndarray, seen: np.ndarray,
                    metres_per_px: float, *, cell_px: float = 4.0, classes: int = 10, contrast: float = 10.0,
                    min_share: float = 0.005, max_parts: int = 24, seed: int = 0, shape: np.ndarray | None = None,
                    bleed_share: float = 0.05, material_delta_e: float = 8.0
                    ) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Part index of each texel (N,), and a description of each part.

    ``colour`` (N, 3 uint8) is a first bake; ``seen`` marks the texels that a drawing
    painted (the others hold a fill colour and only inherit a part from their
    neighbours). The surface is cut into cells of ``cell_px`` aligned pixels, linked to
    the cells beside them on the surface (not through a thin flap, and not from a tail
    to the back that it hangs in front of).

    1. Each cell gets the colour class of most of its texels (``classes`` classes, k-means
       in CIE Lab). Connected cells of one class are the first patches.
    2. What makes two things different parts is a sharp border, not a different colour:
       fur changes from cream to brown without an edge, and a vest ends with one. So two
       touching patches join if the mean colour step across their border is below
       ``contrast`` (delta E, measured between the two sides about 12 pixels apart).
    3. A patch smaller than ``min_share`` of the surface (an eye, a button, a sliver of
       vest colour on the edge of the tail) joins the neighbour that it resembles most.
       A patch that no drawing painted joins the neighbour with the longest common border.
    4. If more than ``max_parts`` are left, the weakest borders go first.

    5. Geometry against colour. ``shape`` (N,) is the shape of each texel from
       ``shape_parts`` (tail, body, ...). The colours come from a bake, and the bake puts
       a part on the wrong shape where the mesh is wider than a drawing: the vest on the
       edge of the tail. A material that covers less than ``bleed_share`` of a shape,
       while most of that material lies on other shapes, is such a bleed. Its cells take
       the part of the cells beside them on the same shape. A real part that crosses a
       crease on a small area (the end of a sleeve) is treated the same: see the returned
       notes, and set ``bleed_share`` to 0 if that is wrong for a character.

    Returns the part of each texel, a description of each part, and the notes on bleed.
    Each description has ``id``, ``share``, ``rgb`` (mean colour), ``material`` (parts
    whose mean colours differ by less than ``material_delta_e`` are one material),
    ``main`` (its main colours) and ``palette`` (every colour it shows away from its
    borders), both in Lab with the lightness scaled by ``LIGHTNESS_WEIGHT``.
    """
    regions = build_regions(points, normals, sigma=3 * cell_px * metres_per_px, spacing=cell_px * metres_per_px)
    nodes = len(regions.texels)
    painted = regions.per_node(seen)
    node_rgb = np.stack([regions.per_node(colour[:, i] * seen) for i in range(3)], axis=1) / np.maximum(painted, 1)[:, None]
    known = painted > 0
    node_colour = lab(np.clip(node_rgb, 0, 255).astype(np.uint8))
    if known.sum() < classes:
        raise ValueError("Too little of the surface is painted to recognise parts")
    rng = np.random.default_rng(seed)
    sample = node_colour[known][rng.choice(int(known.sum()), size=min(int(known.sum()), 60_000), replace=False)]
    centres, _ = kmeans2(sample, classes, minit="++", seed=seed)
    centres = centres[np.isfinite(centres).all(axis=1)]
    # The class of a cell is the class of most of its texels, not the class of its mean colour:
    # a cell on a border would otherwise get a third class between the two parts.
    kind = np.full(nodes, -1)
    painted_texels = np.nonzero(seen)[0]
    tally = np.zeros((nodes, len(centres)))
    for start in range(0, len(painted_texels), 500_000):
        ids = painted_texels[start:start + 500_000]
        nearest = np.argmin(np.linalg.norm(lab(colour[ids])[:, None] - centres[None], axis=2), axis=1)
        np.add.at(tally, (regions.cell[ids], nearest), 1.0)
    kind[known] = np.argmax(tally[known], axis=1)
    near = regions.near[:, 1:SEAM_EDGES + 1]
    rows, cols = np.repeat(np.arange(nodes), near.shape[1]), near.ravel()
    # First patches: connected cells of one class. Unpainted cells form patches of their own (class -1).
    same = kind[rows] == kind[cols]
    _, patch = csgraph.connected_components(
        sparse.coo_matrix((np.ones(same.sum()), (rows[same], cols[same])), shape=(nodes, nodes)), directed=False)
    # Specks first: a few cells of another class on a border are mixed texels, not a part.
    specks = _Patches(patch, rows, cols, regions.texels)
    cells = np.bincount(patch, minlength=specks.count)
    for index in range(specks.count):
        specks.area[index] = cells[index]
    specks.absorb(lambda i: specks.area[i] < 12)
    patch = np.unique(specks.owner[patch], return_inverse=True)[1]
    # The colour on each side of a border: the mean over the cells of the same patch around a cell.
    wide, wide_gauss = regions.near, regions.gauss * regions.texels[regions.near]
    mine = (patch[wide] == patch[:, None]) & known[wide]
    weight = wide_gauss * mine
    beside = np.einsum("mk,mkc->mc", weight, node_colour[wide]) / np.maximum(weight.sum(axis=1), 1e-12)[:, None]
    has_side = weight.sum(axis=1) > 0
    patches = _Patches(patch, rows, cols, regions.texels,
                       step=np.linalg.norm(beside[rows] - beside[cols], axis=1),
                       measured=(has_side[rows] & has_side[cols]).astype(float))
    painted_area = np.bincount(patch, weights=regions.texels * known, minlength=patches.count)
    whole = patches.area.sum()

    def join(i: int, j: int) -> None:
        keep, gone = (i, j) if patches.area[i] >= patches.area[j] else (j, i)
        painted_area[keep] += painted_area[gone]
        patches.join(gone, keep)

    while True:   # soft borders
        weakest = patches.weakest()
        if weakest is None or weakest[0] >= contrast:
            break
        join(weakest[1], weakest[2])
    while True:   # unpainted and small patches, then the limit on the number of parts
        loose = [i for i in patches.links if patches.links[i]
                 and (painted_area[i] <= 0 or patches.area[i] < min_share * whole)]
        if loose:
            # The neighbour it resembles most; for a patch that no drawing painted, the longest border.
            i = min(loose, key=lambda index: patches.area[index])
            alike = min(patches.links[i], key=lambda other: patches.step(i, other))
            join(i, alike if np.isfinite(patches.step(i, alike))
                 else max(patches.links[i], key=lambda other: patches.links[i][other][0]))
            continue
        weakest = patches.weakest()
        if len(patches.links) <= max_parts or weakest is None or not np.isfinite(weakest[0]):
            break
        join(weakest[1], weakest[2])
    node_part = np.unique(patches.owner[patch], return_inverse=True)[1]

    def mean_colours(labels: np.ndarray) -> np.ndarray:
        weight = regions.texels * known
        total = np.bincount(labels, weights=weight, minlength=labels.max() + 1)
        return np.stack([np.bincount(labels, weights=weight * node_rgb[:, i], minlength=labels.max() + 1)
                         for i in range(3)], axis=1) / np.maximum(total, 1e-12)[:, None]

    notes, bleed, node_shape, was_material = [], None, None, None
    if shape is not None and bleed_share > 0:
        shapes = int(shape.max()) + 1
        tally_shape = np.zeros((nodes, shapes))
        np.add.at(tally_shape, (regions.cell, shape), 1.0)
        node_shape = np.argmax(tally_shape, axis=1)
        material_of = np.array(group_materials([{"rgb": c} for c in mean_colours(node_part)], material_delta_e))
        node_material = material_of[node_part]
        table = np.zeros((shapes, material_of.max() + 1))
        np.add.at(table, (node_shape, node_material), regions.texels)
        bleed = (table > 0) & (table < bleed_share * table.sum(axis=1, keepdims=True)) \
            & (table < 0.5 * table.sum(axis=0, keepdims=True))
        wrong = bleed[node_shape, node_material]
        if wrong.any():
            colours = mean_colours(node_material)
            for index, material in zip(*np.nonzero(bleed)):
                notes.append({"shape": int(index), "material_rgb": [int(round(v)) for v in colours[material]],
                              "share_of_shape": round(float(table[index, material] / table[index].sum()), 4),
                              "share_of_material": round(float(table[index, material] / table[:, material].sum()), 4)})
            settled = ~wrong
            for _ in range(400):   # from the cells beside them on the same shape
                give = ~settled[rows] & settled[cols] & (node_shape[rows] == node_shape[cols])
                if not give.any():
                    break
                node_part[rows[give]] = node_part[cols[give]]
                settled[rows[give]] = True
        was_material = material_of[node_part]      # per node, in the numbering of the ``bleed`` table
    _, node_part = np.unique(node_part, return_inverse=True)
    area = np.bincount(node_part, weights=regions.texels)
    order = np.argsort(-area)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    node_part = rank[node_part]
    area = area[order]
    parts_count = len(order)
    materials = group_materials([{"rgb": c} for c in mean_colours(node_part)], material_delta_e)
    # Colours of each part: the main ones from all of it, the full palette away from its borders.
    edge = np.zeros(nodes, dtype=bool)
    different = node_part[rows] != node_part[cols]
    edge[rows[different]] = True
    for _ in range(3):
        edge[rows[edge[cols]]] = True
    compare = centres * np.array([LIGHTNESS_WEIGHT, 1.0, 1.0])   # for comparison with other views
    descriptions = []
    for index in range(parts_count):
        inside = (node_part == index) & known
        weight = regions.texels * inside
        share_of = np.bincount(kind[inside], weights=weight[inside], minlength=len(centres)) / max(weight.sum(), 1e-12)
        core_cells = inside & ~edge
        core = np.bincount(kind[core_cells], weights=weight[core_cells], minlength=len(centres))
        core = core / core.sum() if core.sum() > 0.2 * weight.sum() else share_of
        mean = (node_rgb * weight[:, None]).sum(axis=0) / max(weight.sum(), 1e-12)
        height = regions.centre[node_part == index][:, 2]
        painted_part = weight.sum() > 0            # an island that no source painted has no colours to match
        descriptions.append({
            "id": f"part-{index + 1:02d}", "share": round(float(area[index] / whole), 4),
            "rgb": [int(round(v)) for v in mean], "material": materials[index],
            "main": compare[painted_part & (share_of >= min(DOMINANT_SHARE, share_of.max()))].round(2).tolist(),
            "palette": compare[painted_part & ((core >= 0.005) | (share_of >= DOMINANT_SHARE))].round(2).tolist(),
            "height_pct": [int(round(100 * v / max(regions.centre[:, 2].max(), 1e-9))) for v in np.percentile(height, [2, 98])]})
    # Texels near a border take the neighbouring part whose main colour they show, so the border follows the paint.
    label = node_part[regions.cell]
    candidate = np.zeros((nodes, parts_count), dtype=bool)
    candidate[np.arange(nodes), node_part] = True
    candidate[rows, node_part[cols]] = True
    if bleed is not None:       # not back into a material that is a bleed on this shape
        of_part = np.zeros(parts_count, dtype=np.int64)
        of_part[node_part] = was_material          # every node of a part has the same material
        candidate &= ~bleed[node_shape][:, of_part]
        candidate[np.arange(nodes), node_part] = True
    close = (candidate.sum(axis=1) > 1)[regions.cell] & seen
    if close.any():
        ids = np.nonzero(close)[0]
        shade = weighted_lab(colour[ids])
        own = np.full(len(ids), np.inf)
        best_other, chosen = np.full(len(ids), np.inf), label[ids].copy()
        for index in range(parts_count):
            distance = _distance_to(shade, np.asarray(descriptions[index]["main"]))
            mine = label[ids] == index
            own[mine] = distance[mine]
            better = candidate[regions.cell[ids], index] & ~mine & (distance < best_other)
            best_other[better], chosen[better] = distance[better], index
        move = best_other < own - 6.0
        label[ids[move]] = chosen[move]
    return label, descriptions, notes


def texel_shapes(mesh: Mesh, points: np.ndarray, normals: np.ndarray, metres_per_px: float, *,
                 crease_radius_px: float = 60.0) -> np.ndarray:
    """Shape index of each texel (see ``shape_parts``): the shape of the nearest vertex that faces the same way."""
    from scipy.spatial import cKDTree

    shape, _ = shape_parts(mesh, crease_radius=crease_radius_px * metres_per_px)
    scale = 12 * metres_per_px
    _, vertex = cKDTree(np.hstack([mesh.vertices / scale, mesh.vertex_normals()])).query(
        np.hstack([points / scale, normals]), workers=-1)
    return shape[vertex]


def group_materials(parts: list[dict], delta_e: float = 8.0) -> list[int]:
    """Material index of each part: parts whose mean colours differ by less than ``delta_e`` are one material.

    A strap cuts a vest into two parts, and both are vest. Parts of one material are
    baked together: a pixel of one may paint the other, and that does no harm.
    """
    colour = lab(np.clip(np.round(np.array([part["rgb"] for part in parts], dtype=np.float64)), 0, 255).astype(np.uint8))
    group = list(range(len(parts)))
    for i in range(len(parts)):
        for j in range(i):
            if np.linalg.norm(colour[i] - colour[j]) < delta_e:
                old, new = group[i], group[j]
                group = [new if value == old else value for value in group]
    order = {value: index for index, value in enumerate(dict.fromkeys(group))}
    return [order[value] for value in group]


def label_image(size: int, rows: np.ndarray, cols: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Part index + 1 in the UV layout (size x size, uint8); gutters take the nearest texel, 0 is no part."""
    picture = np.zeros((size, size), dtype=np.uint8)
    picture[rows, cols] = np.clip(label + 1, 0, 255)
    covered = np.zeros((size, size), dtype=bool)
    covered[rows, cols] = True
    _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    return picture[iy, ix]


def smooth_labels(labels: np.ndarray, window: int = 5) -> np.ndarray:
    """Each pixel takes the label that most of the pixels around it have (0 stays 0).

    Labels that are decided texel by texel come out speckled along a border, and each
    speck is a texel baked from the wrong material.
    """
    best = np.zeros(labels.shape, dtype=np.float32)
    result = labels.copy()
    for value in np.unique(labels[labels > 0]):
        share = ndimage.uniform_filter((labels == value).astype(np.float32), window)
        better = share > best
        best[better], result[better] = share[better], value
    result[labels == 0] = 0
    return result


def view_part_labels(mesh: Mesh, corner_uv: np.ndarray, labels_uv: np.ndarray, view: str, raw_map: dict,
                     rgba: np.ndarray, metres_per_px: float, parts: list[dict], *, slack_px: float = 60.0,
                     margin: float = 10.0, accept: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Part index of each pixel of a drawing (-1: none), and the plain projection of the mesh's parts.

    The projection says where the mesh puts each part in this view. A pixel moves to
    another part only on positive evidence: that part is projected within ``slack_px``
    (aligned pixels), the pixel shows one of its main colours (within ``accept``), and
    that fits better than any colour of the projected part by ``margin``. Pixels of the
    figure outside the mesh take the nearest projected part within the slack.
    """
    texture = np.repeat(labels_uv[..., None], 3, axis=2)
    rendered, mask = render_textured(mesh, corner_uv, texture, view, metres_per_px, 1.0)
    projected = np.where(mask, rendered[..., 0].astype(np.int16) - 1, -1)
    hole = ndimage.binary_closing(mask, iterations=2) & ~mask     # gaps between the splatted samples
    if hole.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
        projected[hole] = projected[iy[hole], ix[hole]]
    height, width = rgba.shape[:2]
    x, y = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    u = (x - raw_map["left"]) * raw_map["scale"] + raw_map["x0"] - raw_map["shift_px"]
    v = (y - raw_map["top"]) * raw_map["scale"] + raw_map["y0"]
    col, row = np.floor(u).astype(int), np.floor(v).astype(int)
    inside = (col >= 0) & (col < projected.shape[1]) & (row >= 0) & (row < projected.shape[0])
    plain = np.full((height, width), -1, dtype=np.int16)
    plain[inside] = projected[row[inside], col[inside]]
    figure = rgba[..., 3] > 200
    shade = weighted_lab(rgba[..., :3]).reshape(-1, 3)
    own = np.full(height * width, np.inf)
    for index, part in enumerate(parts):
        mine = (plain == index).ravel()
        if mine.any():
            own[mine] = _distance_to(shade[mine], np.asarray(part["palette"]))
    best, chosen = np.full(height * width, np.inf), plain.ravel().copy()
    nearest_part, nearest_distance = np.full(height * width, -1), np.full(height * width, np.inf)
    slack = slack_px / raw_map["scale"]
    for index, part in enumerate(parts):
        footprint = plain == index
        if not footprint.any():
            continue
        away = ndimage.distance_transform_edt(~footprint).ravel()
        nearer = away < nearest_distance
        nearest_part[nearer], nearest_distance[nearer] = index, away[nearer]
        reach = np.nonzero((away > 0) & (away <= slack) & figure.ravel())[0]
        distance = _distance_to(shade[reach], np.asarray(part["main"])) + 2.0 * away[reach] / slack
        better = (distance < accept) & (distance < own[reach] - margin) & (distance < best[reach])
        best[reach[better]], chosen[reach[better]] = distance[better], index
    outside = (plain.ravel() < 0) & (chosen < 0) & (nearest_distance <= slack)
    chosen[outside] = nearest_part[outside]
    chosen[~figure.ravel()] = -1
    return chosen.reshape(height, width).astype(np.int16), plain


def _owner_index(picture: np.ndarray) -> np.ndarray:
    """Source index of each pixel of an ``owners.png`` (-1: black, no view saw the texel)."""
    flat = picture.reshape(-1, 3).astype(np.int32)
    distance = np.abs(flat[:, None, :] - OWNER_COLOURS[None].astype(np.int32)).sum(axis=2)
    index = np.argmin(distance, axis=1)
    index[flat.max(axis=1) < 30] = -1
    return index.reshape(picture.shape[:2])


MIRROR_PAIRS = (("left", "right"), ("front_left", "front_right"), ("back_left", "back_right"))


def given_part_texels(mesh: Mesh, points: np.ndarray, normals: np.ndarray, masks: list[dict[str, np.ndarray]],
                      views: list[str], metres_per_px: float, *, scale: float = 0.5, mirror_missing: bool = False,
                      figures: dict[str, np.ndarray] | None = None, grow_px: float | list[float] = 0.0,
                      min_cosine: float = 0.35,
                      power: float = 6.0, smooth_px: float = 8.0,
                      region_px: float = 24.0, depth_tolerance_px: float = 4.0) -> np.ndarray:
    """For each given part, the texels that it covers: (len(masks), N) bool.

    ``masks[i]`` holds the silhouette of part i in the camera frame at ``scale``, for
    each view that shows it (``handoff.reference_masks`` of a part handoff). ``views``
    are the drawings of the whole character, and ``figures`` their silhouettes in the
    same frame. Each drawing that sees a texel votes: the texel lies inside the part's
    pixels there, or it does not. A drawing that does not show the part at all votes
    against it. A drawing with no figure at that place (a hole where another part was
    taken out) does not vote, and a drawing that sees the surface at a grazing angle
    (below ``min_cosine``) does not: its pixels there belong to what lies beside it. A vote counts by cosine ** ``power`` between the view and
    the surface without its bumps (as the bake takes it, over ``region_px``), so the
    drawing that faces the surface decides, and the border does not follow each fold of
    a knit. A texel belongs to the part with the largest share of its votes, if that is
    more than half. ``mirror_missing`` adds the missing side views as mirror images, as
    the bake does.

    The pixels of a part can count ``grow_px`` aligned pixels wider than they are (one
    value, or one for each part). Where two parts then claim a texel, the first one has
    it. Use it for a part that lies on top: a scarf is drawn a little smaller than the
    mesh has it, and the rim of the mesh's scarf would take the colour of the vest below.
    Do not use it for a part with an opening that must stay open (the front of a vest).

    Last, the border of each part is smoothed on the surface, over ``smooth_px`` aligned
    pixels: two drawings never put a border at the same place, and where the surface
    turns from one drawing to the other, the border would have teeth.

    A texel without a vote (the side of a vest behind an arm) takes the answer of the
    nearest texel that has one and faces the same way.
    """
    from scipy.spatial import cKDTree

    sources = [(name, name, False) for name in views]
    if mirror_missing:
        for a, b in MIRROR_PAIRS:
            for real, missing in ((a, b), (b, a)):
                if real in views and missing not in views:
                    sources.append((missing, real, True))
    count = len(points)
    grow = list(grow_px) if isinstance(grow_px, (list, tuple)) else [float(grow_px)] * len(masks)
    if len(grow) != len(masks):
        raise ValueError("Give one grow_px, or one for each part")
    masks = [{view: ndimage.binary_dilation(mask, iterations=max(1, round(wider * scale))) if wider > 0 else mask
              for view, mask in part.items()} for part, wider in zip(masks, grow)]
    regions = build_regions(points, normals, sigma=region_px * metres_per_px, spacing=region_px * metres_per_px / 3)
    facing = regions.normal[regions.cell]
    inside_weight, total = np.zeros((len(masks), count)), np.zeros(count)
    for name, real, mirrored in sources:
        look = camera(name)
        _, depth = rasterize(mesh, look, metres_per_px, scale)
        col, row, reach = look.project(points, metres_per_px, scale)
        c, r = np.floor(col).astype(int), np.floor(row).astype(int)
        on_frame = (c >= 0) & (c < depth.shape[1]) & (r >= 0) & (r < depth.shape[0])
        front = np.full(count, np.nan)
        front[on_frame] = depth[r[on_frame], c[on_frame]]
        cosine = normals @ look.toward
        slope = np.minimum(np.sqrt(np.maximum(1 - cosine ** 2, 0)) / np.maximum(cosine, 1e-3), 10)
        smooth = facing @ look.toward
        visible = on_frame & (cosine > 0) & (smooth >= min_cosine) \
            & (reach >= front - (depth_tolerance_px + 1.5 * slope) * metres_per_px / scale)
        if mirrored:                                   # the opposite camera sees the mirror image about the frame centre
            c = np.floor(2 * CX * scale - col).astype(int)
        drawn = on_frame.copy()
        if figures is not None and real in figures:
            figure = figures[real]
            ok = on_frame & (c >= 0) & (c < figure.shape[1]) & (r < figure.shape[0])
            drawn = np.zeros(count, dtype=bool)
            drawn[ok] = figure[r[ok], c[ok]]
        visible &= drawn
        weight = np.where(visible, np.clip(smooth, 0, 1) ** power, 0.0)
        total += weight
        for index, part in enumerate(masks):
            mask = part.get(real)
            if mask is None:
                continue
            ok = on_frame & (c >= 0) & (c < mask.shape[1]) & (r < mask.shape[0])
            hit = np.zeros(count, dtype=bool)
            hit[ok] = mask[r[ok], c[ok]]
            inside_weight[index] += weight * hit
    seen = total > 0
    share = inside_weight / np.maximum(total, 1e-12)
    member = (share > 0.5) & seen
    member &= np.cumsum(member, axis=0) == 1            # more than one part: the first
    if seen.any() and not seen.all():
        # Position and facing together: a texel on the outside of a sleeve does not ask the vest under it.
        where = np.hstack([points, facing * 20 * metres_per_px])
        _, nearest = cKDTree(where[seen]).query(where[~seen], workers=-1)
        member[:, ~seen] = member[:, seen][:, nearest]
    if smooth_px > 0 and count > 1:
        # Each texel takes the mean answer of the surface around it, on its side of the surface. The
        # mean is taken over anchors, one for each cell of a quarter of the radius, so that it does
        # not depend on how many texels the texture has.
        radius = smooth_px * metres_per_px
        keys = np.floor(points / (radius / 4)).astype(np.int64)
        _, cell, texels = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        cell = cell.ravel()
        anchor = np.stack([np.bincount(cell, weights=points[:, axis]) for axis in range(3)], axis=1) / texels[:, None]
        towards = np.stack([np.bincount(cell, weights=facing[:, axis]) for axis in range(3)], axis=1)
        answer = np.stack([np.bincount(cell, weights=row.astype(np.float64)) / texels for row in member])
        distance, near = cKDTree(anchor).query(points, k=min(32, len(anchor)), distance_upper_bound=radius, workers=-1)
        if near.ndim == 1:
            distance, near = distance[:, None], near[:, None]
        valid = np.isfinite(distance)
        near = np.where(valid, near, 0)
        valid &= (towards[near] * facing[:, None, :]).sum(axis=2) > 0        # not through a thin flap
        number = np.maximum(valid.sum(axis=1), 1)
        member = np.stack([(row[near] * valid).sum(axis=1) / number for row in answer]) > 0.5
        member &= np.cumsum(member, axis=0) == 1
    return member


def add_given_parts(label: np.ndarray, parts: list[dict], member: np.ndarray, names: list[str],
                    pixels: list[np.ndarray], points: np.ndarray, *, dissolve_share: float = 0.5
                    ) -> tuple[np.ndarray, list[dict], list[dict]]:
    """The labels and the part list with the given parts added; and a report row for each.

    ``member`` is from ``given_part_texels``, ``pixels[i]`` the colours (K, 3 uint8) of
    part i in the drawings. Each given part is a material of its own. A recognised part
    that loses more than ``dissolve_share`` of its texels to given parts was the same
    thing, found by colour, with a ragged border: it is dissolved. What is left of it
    (the scarf's colour on the vest, where a drawing put it on a mesh that is wider)
    takes the recognised part of the nearest other texel. A dissolved part stays in the
    list with no texels, so that the indices of the others do not change.
    """
    from scipy.spatial import cKDTree

    label, parts = label.copy(), [dict(part) for part in parts]
    recognised = len(parts)
    taken = member.any(axis=0)
    before = np.bincount(label[label >= 0], minlength=recognised)
    lost = np.bincount(label[(label >= 0) & taken], minlength=recognised)
    dissolved = np.nonzero((before > 0) & (lost > dissolve_share * before))[0]
    remains = np.isin(label, dissolved) & ~taken
    kept = (label >= 0) & ~np.isin(label, dissolved) & ~taken
    if remains.any() and kept.any():
        _, nearest = cKDTree(points[kept]).query(points[remains], workers=-1)
        label[remains] = label[kept][nearest]
    material = max([part.get("material", index) for index, part in enumerate(parts)], default=-1) + 1
    rows, top = [], max(float(points[:, 2].max()), 1e-9)
    for index, name in enumerate(names):
        mine = member[index]
        label[mine] = recognised + index
        colours = np.asarray(pixels[index], dtype=np.uint8).reshape(-1, 3)
        if len(colours) == 0:
            raise ValueError(f"The given part {name!r} has no pixels in the drawings")
        shade = weighted_lab(colours[:: max(1, len(colours) // 20000)])
        distinct = np.unique(shade.round(1), axis=0)
        if len(distinct) <= 4:                         # a flat colour: its colours are its main colours
            centres = distinct
        else:
            centres, used = kmeans2(shade, 4, seed=0, minit="++")
            centres = centres[np.unique(used)]
        height = points[mine][:, 2] if mine.any() else np.zeros(1)
        parts.append({"id": name, "share": round(float(mine.mean()), 4),
                      "rgb": [int(round(v)) for v in colours.mean(axis=0)], "material": material + index,
                      "main": centres.round(2).tolist(), "palette": centres.round(2).tolist(),
                      "height_pct": [int(round(100 * v / top)) for v in np.percentile(height, [2, 98])], "given": True})
        rows.append({"id": name, "texels": int(mine.sum()), "share": round(float(mine.mean()), 4)})
    for index in dissolved:
        parts[index]["share"], parts[index]["dissolved"] = 0.0, True
    for row in rows:
        row["dissolved"] = [parts[index]["id"] for index in dissolved]
    return label, parts, rows


def find_parts(handoff: Path, mesh: Mesh, corner_uv: np.ndarray, out: Path, *, size: int = 1024,
               views: list[str] | None = None, slack_px: float = 60.0, classes: int = 10, contrast: float = 10.0,
               min_share: float = 0.005, max_parts: int = 24, material_delta_e: float = 8.0,
               crease_radius_px: float = 60.0, bleed_share: float = 0.05,
               mirror_missing: bool = False, extra: list[dict] | None = None,
               given: dict[str, Path] | None = None, given_grow_px: dict[str, float] | None = None, **bake) -> dict:
    """Recognise the parts and write them to ``out``; returns the report (also ``out/parts.json``).

    ``given`` names part handoffs of the same drawings (``split_views``, ``box_part``): the
    pixels that each holds are that part, on the mesh and in each drawing, and the
    recognition does the rest of the surface (``given_part_texels``, ``add_given_parts``).
    ``given_grow_px`` widens the pixels of a given part on the mesh, for a part that lies
    on top of others.

    Use the same sources as the bake that follows: ``mirror_missing`` for a symmetric
    character with one side drawn, ``extra`` for repaints of a side that no drawing shows.
    A side without any source has no colours to recognise, and its coat would be taken
    for the feathers next to it. Files: ``uv_labels.png`` (part index + 1 in the UV
    layout), ``uv/<part>.png``, ``views/<view>/<part>.png`` (masks in the pixel frame of
    ``handoff/raw/<view>.png``), ``views/<view>.png`` (index + 1), ``uv_shapes.png``
    (shape index + 1), ``semantic.json`` (the same masks as a ``resolve-conflicts``
    semantic policy) and ``turnaround.png`` (the first bake, the parts and the shapes,
    from eight sides).
    """
    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    px = spec["metres_per_aligned_px"]
    # The CLI accepts a pre-existing *empty* directory but no produced mask
    # revision may silently overwrite human-reviewed UV or view labels.
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Part-revision output already contains files: {out}")
    source_signature = _source_signature(
        handoff, mesh, corner_uv, views=views, extra=extra,
        mirror_missing=mirror_missing)
    out.mkdir(parents=True, exist_ok=True)
    bake = {k: v for k, v in bake.items() if k != "diagnostics"}
    with tempfile.TemporaryDirectory() as scratch:
        first, _ = bake_texture(handoff, mesh, corner_uv, size=size, views=views, mirror_missing=mirror_missing,
                                extra=extra, diagnostics=Path(scratch), **bake)
        owners = _owner_index(np.asarray(Image.open(Path(scratch) / "owners.png").convert("RGB")))
    rows, cols, points, normals = texel_samples(mesh, corner_uv, size)
    shape = texel_shapes(mesh, points, normals, px, crease_radius_px=crease_radius_px)
    label, parts, bleed = recognise_parts(points, normals, first[rows, cols], owners[rows, cols] >= 0, px,
                                          classes=classes, contrast=contrast, min_share=min_share,
                                          max_parts=max_parts, shape=shape, bleed_share=bleed_share,
                                          material_delta_e=material_delta_e)
    names = [n for n in spec["raw"] if (views is None or n in views) and (handoff / "raw" / f"{n}.png").is_file()]
    given_rows, given_alpha = [], {}                    # part name -> {view: pixels of the part in that drawing}
    if given:
        from .handoff import reference_masks

        ids = {part["id"] for part in parts}
        masks, pixels = [], []
        for name, folder in given.items():
            if name in ids or not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"Not a usable name for a given part: {name!r}")
            part_spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
            shown, colours = {}, []
            for view in names:
                path = folder / "raw" / f"{view}.png"
                if not path.is_file():
                    continue
                if part_spec["raw"].get(view) != spec["raw"][view]:
                    raise ValueError(f"{folder} has another frame for the {view} drawing than {handoff}")
                with Image.open(path) as picture:
                    rgba = np.asarray(picture.convert("RGBA"))
                shown[view] = rgba[..., 3] > 127
                colours.append(rgba[..., :3][shown[view]])
            if not shown:
                raise ValueError(f"{folder} shows the part {name!r} in none of the drawings {names}")
            given_alpha[name] = shown
            masks.append({view: mask for view, mask in reference_masks(folder, 0.5).items() if view in shown})
            pixels.append(np.concatenate(colours))
        figures = {view: mask for view, mask in reference_masks(handoff, 0.5).items() if view in names}
        unknown = sorted(set(given_grow_px or {}) - set(given))
        if unknown:
            raise ValueError(f"No given part has the name {unknown}")
        member = given_part_texels(mesh, points, normals, masks, names, px, mirror_missing=mirror_missing,
                                   figures=figures, grow_px=[float((given_grow_px or {}).get(name, 0.0)) for name in given])
        label, parts, given_rows = add_given_parts(label, parts, member, list(given), pixels, points)
    labels_uv = smooth_labels(label_image(size, rows, cols, label))
    shapes_uv = label_image(size, rows, cols, shape)
    Image.fromarray(shapes_uv).save(out / "uv_shapes.png")
    Image.fromarray(labels_uv).save(out / "uv_labels.png")
    (out / "uv").mkdir(exist_ok=True)
    for index, part in enumerate(parts):
        Image.fromarray(np.where(labels_uv == index + 1, 255, 0).astype(np.uint8)).save(out / "uv" / f"{part['id']}.png")
    agreement, semantic = [], {part["id"]: {} for part in parts}
    index_of = {part["id"]: index for index, part in enumerate(parts)}
    for name in names:
        rgba = np.asarray(Image.open(handoff / "raw" / f"{name}.png").convert("RGBA"))
        chosen, plain = view_part_labels(mesh, corner_uv, labels_uv, name, spec["raw"][name], rgba, px, parts,
                                         slack_px=slack_px)
        if given_alpha:
            # In a drawing, a given part is its own pixels and no others. A pixel that the mesh
            # puts on it, and that is not one of them, takes the part of the nearest other pixel.
            given_ids = [index_of[part] for part in given_alpha]
            marked = np.zeros(chosen.shape, dtype=bool)
            for part, shown in given_alpha.items():
                if name in shown and shown[name].shape == chosen.shape:
                    marked |= shown[name]
            stray = np.isin(chosen, given_ids) & ~marked
            usable = (chosen >= 0) & ~np.isin(chosen, given_ids)
            if stray.any() and usable.any():
                _, (iy, ix) = ndimage.distance_transform_edt(~usable, return_indices=True)
                chosen[stray] = chosen[iy[stray], ix[stray]]
            elif stray.any():
                chosen[stray] = -1
            for part, shown in given_alpha.items():
                if name in shown and shown[name].shape == chosen.shape:
                    chosen[shown[name] & (rgba[..., 3] > 200)] = index_of[part]
        (out / "views" / name).mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(chosen + 1, 0, 255).astype(np.uint8)).save(out / "views" / f"{name}.png")
        figure = rgba[..., 3] > 200
        for index, part in enumerate(parts):
            shown, placed = chosen == index, (plain == index) & figure
            if shown.any():
                Image.fromarray(np.where(shown, 255, 0).astype(np.uint8)).save(out / "views" / name / f"{part['id']}.png")
                semantic[part["id"]][name] = f"views/{name}/{part['id']}.png"
            if shown.any() or placed.any():
                agreement.append({"part": part["id"], "view": name,
                                  "iou": round(float((shown & placed).sum() / max(1, (shown | placed).sum())), 4),
                                  "mesh_only_px": int((placed & ~shown).sum()), "drawing_only_px": int((shown & ~placed).sum()),
                                  "drawn_px": int(shown.sum())})
    policy = {"parts": [{"id": part["id"], "uv_mask": f"uv/{part['id']}.png", "view_masks": semantic[part["id"]]}
                        for part in parts if semantic[part["id"]]]}
    (out / "semantic.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    colours = np.vstack([[(40, 40, 40)], PART_COLOURS[np.arange(len(parts)) % len(PART_COLOURS)]]).astype(np.uint8)
    turnaround(mesh, corner_uv, [first, colours[labels_uv], colours[np.clip(shapes_uv, 0, len(colours) - 1)]], px
               ).save(out / "turnaround.png")
    if _source_signature(handoff, mesh, corner_uv, views=views, extra=extra,
                         mirror_missing=mirror_missing) != source_signature:
        raise ValueError("Part recognition sources changed while processing; regenerate the revision")
    report = {"schema_version": 1, "source_confidence": "estimated", "review": "required", "size": size,
              "source_signature_sha256": source_signature,
              "views": names, "slack_px": slack_px, "mirror_missing": mirror_missing,
              "extra": [item["name"] for item in extra or []],
              "parts": [{**part, "colour_in_pictures": [int(c) for c in PART_COLOURS[i % len(PART_COLOURS)]],
                         "views": sorted(semantic[part["id"]])} for i, part in enumerate(parts)],
              "given": given_rows,
              "shapes": int(shape.max()) + 1, "crease_radius_px": crease_radius_px, "bleed": bleed,
              "agreement": sorted(agreement, key=lambda row: row["iou"]),
              "advice": part_advice(parts, agreement)}
    (out / "parts.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def part_advice(parts: list[dict], agreement: list[dict], *, notable_px: int = 1500, low_iou: float = 0.8) -> list[str]:
    """Where the shape and a drawing disagree about a part, in plain sentences."""
    advice = []
    for row in sorted(agreement, key=lambda item: -(item["mesh_only_px"] + item["drawing_only_px"])):
        if row["iou"] >= low_iou or row["mesh_only_px"] + row["drawing_only_px"] < notable_px:
            continue
        if row["mesh_only_px"] > 2 * row["drawing_only_px"]:
            what = (f"the mesh shows {row['part']} on {row['mesh_only_px']} pixels where the drawing shows another "
                    "part: the mesh part is larger than drawn there, or it hides a part that the drawing shows")
        elif row["drawing_only_px"] > 2 * row["mesh_only_px"]:
            what = (f"the drawing shows {row['part']} on {row['drawing_only_px']} pixels where the mesh has another "
                    "part: the mesh part is smaller than drawn there")
        else:
            what = f"{row['part']} is in a different place on the mesh than in the drawing"
        advice.append(f"{row['view']}: {what} (IoU {row['iou']}). The bake keeps each part to its own pixels; "
                      "to remove the cause, correct the shape of that part.")
        if len(advice) == 6:
            break
    return advice or ["Each part lies where the drawings show it (no part below IoU "
                      f"{low_iou} with a notable area)."]


def bake_parts(handoff: Path, mesh: Mesh, corner_uv: np.ndarray, parts_dir: Path, *, size: int = 2048,
               views: list[str] | None = None, extra: list[dict] | None = None, polish: bool = True,
               polish_reach_px: float = 12.0, tone_delta_e: float = 18.0, part_edge_px: float = 3.0, rim_px: float = 5.0,
               diagnostics: Path | None = None, **bake) -> tuple[np.ndarray, dict]:
    """Texture in which each recognised part is baked only from the pixels that show that part.

    ``parts_dir`` is the output of ``find_parts``. Parts of one material are baked
    together (``group_materials``). For each material, every source image is masked to
    its parts (``views/<view>.png``; a repaint in ``extra`` is labelled here in the same
    way) and ``bake_texture`` runs on the masked set, so a fur texel can take only fur
    pixels and an unseen texel is filled only from its own material. A mirrored
    side (``mirror_missing``) then cannot receive a one-sided accessory: the bag's
    pixels belong to the bag. Texels without a part, and parts that no source shows, come
    from a plain bake. The leveling closes differences between two sources up to about
    twice ``tone_delta_e``. A higher value also evens out a repaint that is much darker
    than the drawing beside it, but at 45 it mixed the pupil of one view into the fur of
    another on Pip (features that two drawings place apart). ``polish`` then fills the unseen texels smoothly
    (``polish.smooth_fill``) and removes the step that is left at the seams
    (``polish.polish_seams``, over ``polish_reach_px`` aligned pixels).
    ``diagnostics`` gets ``owners.png``, ``parts.png`` and ``turnaround.png``.
    """
    from .polish import polish_seams, seam_contrast, smooth_fill, stitch_islands

    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    px = spec["metres_per_aligned_px"]
    found = json.loads((parts_dir / "parts.json").read_text(encoding="utf-8"))
    original = found.get("source_signature_sha256")
    if not isinstance(original, str) or len(original) != 64:
        raise ValueError("Part labels lack source provenance; regenerate the parts revision")
    current = _source_signature(
        handoff, mesh, corner_uv, views=views, extra=extra,
        mirror_missing=bool(bake.get("mirror_missing")))
    if current != original:
        raise ValueError("Part labels are stale: source drawing, mesh, UV or camera inputs changed; regenerate parts")
    parts = found["parts"]
    if bool(bake.get("mirror_missing")) != bool(found.get("mirror_missing")):
        raise ValueError("The parts were recognised with a different --assume-symmetric setting than this bake. "
                         "Run `parts` again with the same sources.")
    rows, cols, points, normals = texel_samples(mesh, corner_uv, size)
    stored = Image.open(parts_dir / "uv_labels.png").convert("L")
    labels_uv = np.asarray(stored.resize((size, size), Image.NEAREST))
    if size > stored.width:    # round the blocks of the enlarged labels
        labels_uv = smooth_labels(labels_uv, 2 * round(size / stored.width) + 1)
    label = labels_uv[rows, cols].astype(np.int16) - 1
    names = [n for n in spec["raw"] if (views is None or n in views) and (handoff / "raw" / f"{n}.png").is_file()]
    sources = []   # (name, path, RGBA, part index of each pixel, extra item or None)
    for name in names:
        path = handoff / "raw" / f"{name}.png"
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        stored_view = parts_dir / "views" / f"{name}.png"
        chosen = None
        if stored_view.is_file():
            with Image.open(stored_view) as picture:
                if picture.size == (rgba.shape[1], rgba.shape[0]):
                    chosen = np.asarray(picture.convert("L")).astype(np.int16) - 1
        if chosen is None:      # a drawing that the parts were not recognised from
            chosen, _ = view_part_labels(mesh, corner_uv, np.asarray(Image.open(parts_dir / "uv_labels.png").convert("L")),
                                         name, spec["raw"][name], rgba, px, parts, slack_px=found.get("slack_px", 60.0))
        sources.append((name, path, rgba, chosen, None))
    for item in extra or []:
        rgba = np.asarray(Image.open(item["image"]).convert("RGBA"))
        chosen, _ = view_part_labels(mesh, corner_uv, np.asarray(Image.open(parts_dir / "uv_labels.png").convert("L")),
                                     item["name"], item["map"], rgba, px, parts, slack_px=found.get("slack_px", 60.0))
        sources.append((item["name"], Path(item["image"]), rgba, chosen, item))
    settings = {k: v for k, v in bake.items() if k not in ("edge_px", "seam_robust_delta_e")}
    plain, plain_report = bake_texture(handoff, mesh, corner_uv, size=size, views=views, extra=extra, **settings)
    colour = plain[rows, cols].astype(np.float64)
    owner = np.full(len(rows), -1)
    painted_by_part, rows_of_report = np.zeros(len(rows), dtype=bool), []
    material_of = np.array([part.get("material", index) for index, part in enumerate(parts)] + [-1])
    material = material_of[label]            # label -1 reads the last entry: no material
    for index in np.unique(material_of[:-1]):
        mine = material == index
        members = [part["id"] for part, value in zip(parts, material_of) if value == index]
        row = {"material": int(index), "parts": members, "texels": int(mine.sum()), "sources": [], "baked": False}
        if mine.any():
            with tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                (root / "raw").mkdir()
                (root / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
                masked_extra = []
                for name, path, rgba, chosen, item in sources:
                    figure = rgba[..., 3] > 200
                    keep = (material_of[chosen] == index) & (ndimage.distance_transform_edt(figure) >= rim_px)
                    if keep.any():
                        row["sources"].append(name)
                    cut = rgba.copy()
                    cut[..., 3] = np.where(keep, 255, 0)
                    if item is None:
                        Image.fromarray(cut).save(root / "raw" / f"{name}.png")
                    else:
                        Image.fromarray(cut).save(root / f"{name}.png")
                        masked_extra.append({**item, "image": str(root / f"{name}.png")})
                if row["sources"]:
                    try:
                        image, _ = bake_texture(root, mesh, corner_uv, size=size, views=views, extra=masked_extra or None,
                                                edge_px=part_edge_px, seam_robust_delta_e=tone_delta_e,
                                                diagnostics=root / "d", **settings)
                    except ValueError:      # no source sees this part of the mesh
                        image = None
                    if image is not None:
                        owners = _owner_index(np.asarray(Image.open(root / "d" / "owners.png").convert("RGB")))
                        colour[mine] = image[rows, cols][mine]
                        owner[mine] = owners[rows, cols][mine]
                        painted_by_part |= mine
                        row["baked"] = True
                        row["seen"] = round(float((owners[rows, cols][mine] >= 0).mean()), 4)
        rows_of_report.append(row)
    covered = np.zeros((size, size), dtype=bool)
    covered[rows, cols] = True
    _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)

    def layout(values: np.ndarray, fill: float = 0) -> np.ndarray:
        picture = np.full((size, size) + values.shape[1:], fill, dtype=values.dtype)
        picture[rows, cols] = values
        return picture

    owner_uv = layout(owner, -1)
    part_uv = layout(np.where(painted_by_part, material, -1), -1)
    stitches = stitch_islands(rows, cols, points, normals, covered)
    report = {"size": size, "materials": rows_of_report, "plain": {k: plain_report[k] for k in ("seen_fraction", "owned")},
              "texels_from_parts": round(float(painted_by_part.mean()), 4),
              "seam_contrast_before": seam_contrast(layout(colour), owner_uv, part_uv, covered, stitches=stitches)}
    if polish:
        image, filled = smooth_fill(layout(colour), owner_uv, part_uv, covered, stitches=stitches)
        report.update(filled)
        corners = mesh.vertices[mesh.faces]
        surface_px = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                                          axis=1).sum() / px ** 2
        texels_per_px = float(np.sqrt(covered.sum() / max(surface_px, 1e-9)))
        image, polished = polish_seams(image, owner_uv, part_uv, covered, stitches=stitches,
                                       reach=int(np.clip(round(polish_reach_px * texels_per_px), 2, 64)))
        report.update(polished)
        report["seam_contrast_after"] = seam_contrast(image, owner_uv, part_uv, covered, stitches=stitches)
    else:
        image = layout(colour)
    image = np.clip(np.round(image[iy, ix]), 0, 255).astype(np.uint8)
    if diagnostics is not None:
        diagnostics.mkdir(parents=True, exist_ok=True)
        owners_picture = np.zeros((size, size, 3), dtype=np.uint8)
        owners_picture[owner_uv >= 0] = OWNER_COLOURS[owner_uv[owner_uv >= 0] % len(OWNER_COLOURS)]
        parts_picture = np.vstack([[(40, 40, 40)], PART_COLOURS[np.arange(len(parts)) % len(PART_COLOURS)]]
                                  ).astype(np.uint8)[labels_uv]
        Image.fromarray(owners_picture[iy, ix]).save(diagnostics / "owners.png")
        Image.fromarray(parts_picture).save(diagnostics / "parts.png")
        turnaround(mesh, corner_uv, [image, parts_picture, owners_picture[iy, ix]], px).save(diagnostics / "turnaround.png")
    return image, report
