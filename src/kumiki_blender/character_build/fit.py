"""Deform a mesh until its orthographic silhouettes match the reference views.

Use it on any mesh that has the right parts but the wrong proportions: an
image-to-3D result, an AI-built ``bpy`` model, or a smoothed blockout.

Each iteration, per view:

- a vertex that projects outside the reference silhouette moves inward;
- a rim vertex (normal perpendicular to the view, on the model outline) that
  sits inside the reference silhouette moves outward to the reference outline.

The move follows the gradient of the reference signed distance field. Moves
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
                    max_move_px: float = 3.0, anchor_weight: float = 2.0, normalise_first: bool = True
                    ) -> FitResult:
    """Fit ``mesh`` to reference ``masks`` (camera frame at ``scale`` of the aligned frame)."""
    px = height_m / FIGURE_PX
    transform = {}
    if normalise_first:
        mesh, transform = normalise(mesh, masks, height_m, px, scale)
    vertices = mesh.vertices.copy()
    adjacency = Mesh(vertices, mesh.faces).adjacency()
    fields = {n: _signed_distance(m) for n, m in masks.items()}
    gradients = {n: np.gradient(f) for n, f in fields.items()}  # (d/drow, d/dcol)
    grid_px = px / scale  # metres per grid pixel
    anchors = {"front", "right" if "right" in masks else "left"}
    weights = {n: anchor_weight if n in anchors else 1.0 for n in masks}
    history, best = [], None
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
        # No height rescale here: the silhouettes fix the top and the soles. A rescale to the
        # mesh's own extremes stretches the whole body when a thin tip (a tuft) is short.
        vertices += field
    return FitResult(Mesh(best[1], mesh.faces), history, best[2], transform)
