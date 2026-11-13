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


def fit_silhouettes(mesh: Mesh, masks: dict[str, np.ndarray], height_m: float, *, scale: float = 0.25,
                    iterations: int = 40, step: float = 0.6, rim_cos: float = 0.35, smooth_radius_px: float = 4.0,
                    outline_blur_px: float = 12.0,
                    max_move_px: float = 3.0, anchor_weight: float = 2.0, normalise_first: bool = True
                    ) -> FitResult:
    """Fit ``mesh`` to reference ``masks`` (camera frame at ``scale`` of the aligned frame)."""
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
