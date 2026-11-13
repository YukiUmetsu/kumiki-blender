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


def _ious(mesh: Mesh, masks: dict[str, np.ndarray], px: float, scale: float) -> dict[str, float]:
    samples = surface_samples(mesh, 0.5 * px / scale)
    out = {}
    for name, target in masks.items():
        model = rasterize(mesh, camera(name), px, scale, samples)[0]
        out[name] = float((model & target).sum() / max(1, (model | target).sum()))
    return out


def normalise(mesh: Mesh, masks: dict[str, np.ndarray], height_m: float, px: float, scale: float,
              try_yaw: bool = True) -> tuple[Mesh, dict]:
    """Scale to the reference height, stand on Z = 0, centre like the references, choose the facing.

    Generators export with any facing. With ``try_yaw`` the four quarter turns
    are tried and the best mean IoU over the axis views wins.
    """
    best = None
    for yaw in ((0, 90, 180, 270) if try_yaw else (0,)):
        a = np.radians(yaw)
        rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        points = mesh.vertices @ rotation.T
        low, high = points.min(axis=0), points.max(axis=0)
        factor = height_m / (high[2] - low[2])
        points = (points - [0, 0, low[2]]) * factor
        # Centre each axis on its silhouette bounding box, as the aligned views do.
        points[:, 0] -= (points[:, 0].min() + points[:, 0].max()) / 2
        points[:, 1] -= (points[:, 1].min() + points[:, 1].max()) / 2
        candidate = Mesh(points, mesh.faces, mesh.colours)
        axis = {n: m for n, m in masks.items() if n in AXIS_VIEWS} or masks
        score = float(np.mean(list(_ious(candidate, axis, px, scale).values())))
        if best is None or score > best[0]:
            best = (score, candidate, {"yaw_deg": yaw, "scale": float(factor), "axis_iou": round(score, 4)})
    return best[1], best[2]


def fit_silhouettes(mesh: Mesh, masks: dict[str, np.ndarray], height_m: float, *, scale: float = 0.25,
                    iterations: int = 40, step: float = 0.6, rim_cos: float = 0.35, smooth_rounds: int = 4,
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
