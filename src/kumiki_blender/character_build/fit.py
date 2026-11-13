"""Deform a mesh until its orthographic silhouettes match the reference views.

Use it on any mesh that has the right parts but the wrong proportions: an
image-to-3D result, an AI-built ``bpy`` model, or a smoothed blockout.

Each iteration, per view:

- a vertex that projects outside the reference silhouette moves inward;
- a rim vertex (normal perpendicular to the view, on the model outline) that
  sits inside the reference silhouette moves outward to the reference outline.

The move follows the gradient of the reference signed distance field, blurred
by ``outline_blur_px`` so fur and feather fringes do not groove the surface. Moves
from all views are averaged per vertex, with the front and primary side
weighted ``anchor_weight`` times (they fix the frame; the other AI views agree
with them only roughly). Only the displacement field is smoothed, so regions
that need no move keep their detail. The best iteration by weighted IoU is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .mesh import Mesh, rasterize, surface_samples
from .views import AXIS_VIEWS, FIGURE_PX, camera


@dataclass
class FitResult:
    mesh: Mesh
    history: list[dict]
    best_iteration: int
    transform: dict
    folds: int = 0  # edges whose two triangles face nearly opposite ways (> 154 degrees)


def fold_faces(mesh: Mesh, cosine: float = -0.9) -> np.ndarray:
    """Indices of faces on an edge whose two triangles meet at a crease sharper than acos(cosine).

    -0.9 (154 degrees) finds folds. 0.0 (90 degrees) also finds the crumples that outline
    pushes leave near the rim.
    """
    normals, _ = mesh.face_normals()
    f = mesh.faces
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(f)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    pair = (edges[1:] == edges[:-1]).all(axis=1)
    a, b = owner[:-1][pair], owner[1:][pair]
    bad = np.einsum("ij,ij->i", normals[a], normals[b]) < cosine
    return np.unique(np.concatenate([a[bad], b[bad]]))


def untangle(mesh: Mesh, rounds: int = 20, rings: int = 2, cosine: float = 0.0) -> Mesh:
    """Smooth only the vertices around folds and crumples until none remain. The rest stays."""
    adjacency = mesh.adjacency()
    vertices = mesh.vertices.copy()
    for _ in range(rounds):
        bad = fold_faces(Mesh(vertices, mesh.faces), cosine)
        if not len(bad):
            break
        region = np.zeros(len(vertices), dtype=bool)
        region[np.unique(mesh.faces[bad])] = True
        for _ in range(rings):
            region |= (adjacency @ region.astype(float)) > 0
        vertices[region] = (adjacency @ vertices)[region]
    return Mesh(vertices, mesh.faces, mesh.colours)


def count_folds(mesh: Mesh) -> int:
    normals, _ = mesh.face_normals()
    f = mesh.faces
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(f)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    pair = (edges[1:] == edges[:-1]).all(axis=1)
    return int((np.einsum("ij,ij->i", normals[owner[:-1][pair]], normals[owner[1:][pair]]) < -0.9).sum())


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Positive outside the silhouette, negative inside, in grid pixels."""
    return ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)


def _sample(field: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(field, [row - 0.5, col - 0.5], order=1, mode="nearest")


def _place(points: np.ndarray, yaw_deg: float, height_m: float
           ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Rotate about Z, scale to the height, stand on Z = 0, centre on the bounding box.

    Returns the placed points, the scale, the rotation and the offset, so other
    point sets (surface samples) can get the same transform.
    """
    a = np.radians(yaw_deg)
    rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    turned = points @ rotation.T
    low, high = turned.min(axis=0), turned.max(axis=0)
    factor = height_m / (high[2] - low[2])
    offset = -np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]]) * factor
    return turned * factor + offset, factor, rotation, offset


def normalise(mesh: Mesh, masks: dict[str, np.ndarray], height_m: float, px: float, scale: float,
              try_yaw: bool = True, coarse_deg: float = 15.0, fine_deg: float = 3.0) -> tuple[Mesh, dict]:
    """Scale to the reference height, stand on Z = 0, centre like the references, choose the facing.

    Generators export with any facing, and a mesh made from a 3/4 view comes out
    turned about 45 degrees. The search tries every ``coarse_deg`` step, then
    refines around the best in ``fine_deg`` steps. It scores the mean IoU over
    all views: the side views tell the front from the back (the face and the
    feet point one way, a tail the other), which the front and back cannot.
    """
    samples = surface_samples(mesh, 0.5 * px / scale)[0]
    samples = samples[::max(1, len(samples) // 300_000)]  # ranking angles needs no dense cover

    def score(yaw: float) -> float:
        _placed, factor, rotation, offset = _place(mesh.vertices, yaw, height_m)
        points = (samples @ rotation.T) * factor + offset
        values = []
        for name, target in masks.items():
            cam = camera(name)
            col, row, _ = cam.project(points, px, scale)
            model = np.zeros_like(target)
            c, r = np.floor(col).astype(int), np.floor(row).astype(int)
            keep = (c >= 0) & (c < target.shape[1]) & (r >= 0) & (r < target.shape[0])
            model[r[keep], c[keep]] = True
            if name not in AXIS_VIEWS:  # a 3/4 view has no shared horizontal centre
                values.append(max(_iou(np.roll(model, sh, axis=1), target) for sh in range(-20, 21, 4)))
            else:
                values.append(_iou(model, target))
        return float(np.mean(values))

    if try_yaw:
        coarse = {yaw: score(yaw) for yaw in np.arange(0, 360, coarse_deg)}
        start = max(coarse, key=coarse.get)
        fine = {yaw % 360: score(yaw) for yaw in np.arange(start - coarse_deg, start + coarse_deg + 1e-6, fine_deg)}
        yaw = max(fine, key=fine.get)
        best = fine[yaw]
    else:
        yaw, best = 0.0, score(0.0)
    placed, factor, _rotation, offset = _place(mesh.vertices, yaw, height_m)
    # p_placed = scale * Rz(yaw) @ p + offset: enough to put the source file (and its texture) in place.
    return Mesh(placed, mesh.faces, mesh.colours), {"yaw_deg": float(yaw), "scale": float(factor),
                                                    "offset_m": [float(v) for v in offset],
                                                    "silhouette_iou": round(best, 4)}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


class Lattice:
    """Trilinear deformation lattice over a bounding box (like Blender's Lattice modifier)."""

    def __init__(self, points: np.ndarray, cells: tuple[int, int, int], pad: float = 0.05):
        low, high = points.min(axis=0), points.max(axis=0)
        margin = pad * (high - low)
        self.low, self.size = low - margin, (high - low) + 2 * margin
        self.cells = np.array(cells)
        local = (points - self.low) / self.size * self.cells
        base = np.clip(np.floor(local).astype(int), 0, self.cells - 1)
        frac = local - base
        corners, weights = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    corner = base + [dx, dy, dz]
                    corners.append(np.ravel_multi_index(corner.T, tuple(self.cells + 1)))
                    weights.append(np.prod(np.where([dx, dy, dz], frac, 1 - frac), axis=1))
        self.corners, self.weights = np.stack(corners, axis=1), np.stack(weights, axis=1)  # (N, 8)
        self.nodes = int(np.prod(self.cells + 1))

    def pool(self, move: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Node moves: the weighted mean of the desired moves of the vertices around each node."""
        w = self.weights * weight[:, None]
        total = np.zeros((self.nodes, 3))
        count = np.zeros(self.nodes)
        for k in range(8):
            np.add.at(total, self.corners[:, k], move * w[:, k:k + 1])
            np.add.at(count, self.corners[:, k], w[:, k])
        node = np.zeros((self.nodes, 3))
        has = count > 1e-9
        node[has] = total[has] / count[has, None]
        return node

    def apply(self, node: np.ndarray) -> np.ndarray:
        return np.einsum("nk,nkd->nd", self.weights, node[self.corners])


def fit_silhouettes(mesh: Mesh, masks: dict[str, np.ndarray], height_m: float, *, scale: float = 0.25,
                    iterations: int = 40, step: float = 0.6, rim_cos: float = 0.35, smooth_radius_px: float = 4.0,
                    outline_blur_px: float = 12.0,
                    max_move_px: float = 3.0, anchor_weight: float = 2.0, normalise_first: bool = True,
                    lattice_cells: tuple[int, int, int] | None = None) -> FitResult:
    """Fit ``mesh`` to reference ``masks`` (camera frame at ``scale`` of the aligned frame).

    ``lattice_cells`` (e.g. 8, 8, 16) deforms the mesh only through a coarse trilinear
    lattice. Use it for a detailed generator mesh: per-vertex moves on fur tufts act like
    noise and crumple the surface, a lattice changes proportions and keeps the detail.
    """
    px = height_m / FIGURE_PX
    transform = {}
    if normalise_first:
        mesh, transform = normalise(mesh, masks, height_m, px, scale)
    vertices = mesh.vertices.copy()
    adjacency = Mesh(vertices, mesh.faces).adjacency()
    edges = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]]])
    median_edge = float(np.median(np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)))
    # Averaging over neighbours spreads a move about one edge per sqrt(round).
    smooth_rounds = int(np.clip((smooth_radius_px * height_m / FIGURE_PX / scale / max(median_edge, 1e-9)) ** 2, 4, 4000))
    # Target the low-passed outline, as the blockout does. The raw outline of a fur or feather
    # fringe changes on every row, and chasing it grooves the surface row by row.
    sigma = outline_blur_px * scale
    fields = {n: ndimage.gaussian_filter(_signed_distance(m), sigma) if sigma > 0 else _signed_distance(m)
              for n, m in masks.items()}
    gradients = {n: np.gradient(f) for n, f in fields.items()}  # (d/drow, d/dcol)
    grid_px = px / scale  # metres per grid pixel
    anchors = {"front", "right" if "right" in masks else "left"}
    weights = {n: anchor_weight if n in anchors else 1.0 for n in masks}
    history, best = [], None
    lattice = Lattice(vertices, lattice_cells) if lattice_cells else None
    faces = mesh.faces
    start_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    for iteration in range(iterations + 1):
        current = Mesh(vertices, mesh.faces)
        samples = surface_samples(current, 0.5 * grid_px)
        scores, move_sum, votes = {}, np.zeros_like(vertices), np.zeros(len(vertices))
        normals = current.vertex_normals()
        for name, target in masks.items():
            cam = camera(name)
            model, _ = rasterize(current, cam, px, scale, samples)
            scores[name] = float((model & target).sum() / max(1, (model | target).sum()))
            if iteration == iterations:
                continue
            col, row, _ = cam.project(vertices, px, scale)
            distance = _sample(fields[name], col, row)
            g_row, g_col = (_sample(g, col, row) for g in gradients[name])
            norm = np.maximum(np.hypot(g_row, g_col), 1e-6)
            on_outline = _sample(ndimage.distance_transform_edt(model).astype(np.float32), col, row) <= 1.5
            rim = (np.abs(normals @ cam.toward) < rim_cos) & on_outline
            act = (distance > 0.5) | (rim & (distance < -0.5))
            amount = np.clip(distance, -max_move_px, max_move_px) * step * grid_px
            screen = (g_col / norm)[:, None] * cam.right[None] + (g_row / norm)[:, None] * np.array([0, 0, -1.0])
            move = -amount[:, None] * screen
            move_sum[act] += weights[name] * move[act]
            votes[act] += weights[name]
        mean_iou = sum(weights[n] * v for n, v in scores.items()) / sum(weights.values())
        history.append({"iteration": iteration, "weighted_iou": round(mean_iou, 4),
                        "views": {k: round(v, 4) for k, v in scores.items()}})
        if best is None or mean_iou > best[0]:
            best = (mean_iou, vertices.copy(), iteration)
        if iteration == iterations:
            break
        field = np.zeros_like(vertices)
        moved = votes > 0
        field[moved] = move_sum[moved] / votes[moved, None]
        if lattice is not None:
            # Vertices that need no move vote for zero, so a node moves only as far as its region agrees.
            field = lattice.apply(lattice.pool(field, np.ones(len(field))))
        else:
            # Smooth the displacement, not the shape: neighbours follow a move, unmoved detail stays.
            # The radius is physical (grid pixels), so a dense generator mesh does not crumple.
            for _ in range(smooth_rounds):
                field = 0.5 * field + 0.5 * (adjacency @ field)
        # Flip guard: a move that turns a triangle over folds the surface (QuadriFlow then refuses
        # the mesh). Compare with the starting orientation, not the last iteration: a triangle can
        # also turn over a little at a time. Cancel the moves at the corners of each triangle that
        # would face more than 90 degrees away from its start, a few times. A stricter limit
        # (60 degrees) leaves fewer folds but blocks real moves: -0.01 IoU on two real characters.
        for _ in range(4):
            moved_points = vertices + field
            after = np.cross(moved_points[faces[:, 1]] - moved_points[faces[:, 0]],
                             moved_points[faces[:, 2]] - moved_points[faces[:, 0]])
            cosine = np.einsum("ij,ij->i", start_normals, after) / np.maximum(
                np.linalg.norm(start_normals, axis=1) * np.linalg.norm(after, axis=1), 1e-30)
            flipped = cosine <= 0.0
            if not flipped.any():
                break
            field[np.unique(faces[flipped])] = 0.0
        # No height rescale here: the silhouettes fix the top and the soles. A rescale to the
        # mesh's own extremes stretches the whole body when a thin tip (a tuft) is short.
        vertices += field
    # Folds and rim crumples that the flip guard let through shade as specks and spoil the bake.
    result = untangle(Mesh(best[1], mesh.faces))
    return FitResult(result, history, best[2], transform, count_folds(result))


def edge_distortion(before: Mesh, after: Mesh, ratio: float = 1.5) -> float:
    """Share of the edges whose length changed by more than ``ratio`` (longer or shorter).

    A fit that only corrects proportions leaves about 1% of the edges of a generator mesh
    this far off. A fit that bends the mesh toward views that disagree with each other
    flattens whole parts: Pip's tail became a slab at 12%, with a higher IoU.
    """
    if before.faces.shape != after.faces.shape or len(before.vertices) != len(after.vertices):
        raise ValueError("edge_distortion compares two states of one mesh")
    edges = np.concatenate([before.faces[:, [0, 1]], before.faces[:, [1, 2]], before.faces[:, [2, 0]]])
    start = np.linalg.norm(before.vertices[edges[:, 0]] - before.vertices[edges[:, 1]], axis=1)
    end = np.linalg.norm(after.vertices[edges[:, 0]] - after.vertices[edges[:, 1]], axis=1)
    change = end / np.maximum(start, 1e-12)
    return float(((change > ratio) | (change < 1 / ratio)).mean())


def fit_ladder(mesh: Mesh, masks: dict[str, np.ndarray], height_m: float, *,
               ladder: tuple[tuple[int, int, int], ...] = ((4, 4, 8), (8, 8, 16)), max_new_folds: int = 0,
               max_distortion: float = 0.03, normalise_first: bool = True, **options) -> tuple[FitResult, list[dict]]:
    """Fit through a coarse lattice, then finer ones, and keep a stage only if it does no harm.

    A finer lattice always raises the IoU. With views that disagree it does so by bending
    the mesh between them. A stage is kept only if, against the mesh before any fit, it
    adds at most ``max_new_folds`` folds and leaves at most ``max_distortion`` of the edges
    stretched or squeezed by more than 1.5 (``edge_distortion``). The ladder stops at the
    first stage that fails. Returns the result of the last kept stage (the unfitted mesh
    if none is kept) and one row per stage tried; the first row is the unfitted mesh.
    ``options`` go to ``fit_silhouettes``.
    """
    options.pop("lattice_cells", None)
    placed = fit_silhouettes(mesh, masks, height_m, iterations=0, normalise_first=normalise_first,
                             **{k: v for k, v in options.items() if k != "iterations"})
    start_folds = count_folds(placed.mesh)
    kept = placed
    stages = [{"lattice": None, "weighted_iou": placed.history[0]["weighted_iou"], "views": placed.history[0]["views"],
               "new_folds": 0, "distorted_edges": 0.0, "kept": True}]
    for cells in ladder:
        result = fit_silhouettes(kept.mesh, masks, height_m, normalise_first=False, lattice_cells=cells, **options)
        distortion = edge_distortion(placed.mesh, result.mesh)
        row = {"lattice": list(cells), "weighted_iou": result.history[result.best_iteration]["weighted_iou"],
               "views": result.history[result.best_iteration]["views"], "new_folds": result.folds - start_folds,
               "distorted_edges": round(distortion, 4),
               "kept": result.folds - start_folds <= max_new_folds and distortion <= max_distortion}
        stages.append(row)
        if not row["kept"]:
            break
        kept = FitResult(result.mesh, result.history, result.best_iteration, placed.transform, result.folds)
    return kept, stages

