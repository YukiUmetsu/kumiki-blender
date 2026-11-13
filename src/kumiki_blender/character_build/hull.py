"""Visual hull: the largest shape whose orthographic silhouettes fit every view.

The front and side views fix the frame. A diagonal view has no known
horizontal centre, so its shift is searched before it carves. The front and the primary side carve
exactly. The other masks are dilated by a tolerance first: AI views never
agree exactly, and a strict carve lets one bad view cut true parts away. ``carve`` reports, for
each view, the fraction of its silhouette that the final hull cannot explain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .mesh import Mesh
from .views import AXIS_VIEWS, BASELINE, CANVAS_H, CANVAS_W, CX, DIAGONAL_VIEWS, View, camera, downsample


@dataclass
class Hull:
    occupied: np.ndarray        # bool [row, y, x]; row 0 is the top of the aligned frame
    factor: int                 # aligned pixels per voxel
    x_start: int                # grid column of x index 0
    y_start: int
    shifts: dict[str, int]      # diagonal view shift in grid columns
    missing: dict[str, float]   # silhouette fraction of each view that the hull does not cover
    used: list[str]

    def centres(self, metres_per_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World coordinates of voxel centres along x, y and z (by row)."""
        voxel = self.factor * metres_per_px
        gcx = CX / self.factor
        rows, ys, xs = self.occupied.shape
        x = (np.arange(xs) + self.x_start + 0.5 - gcx) * voxel
        y = (np.arange(ys) + self.y_start + 0.5 - gcx) * voxel
        z = (BASELINE / self.factor - np.arange(rows) - 0.5) * voxel
        return x, y, z


def _project_columns(cam_name: str, x: np.ndarray, y: np.ndarray, gcx: float) -> np.ndarray:
    """Grid column of each (y, x) voxel centre, in voxel units, for one camera."""
    cam = camera(cam_name)
    return np.floor(x[None, :] * cam.right[0] + y[:, None] * cam.right[1] + gcx).astype(np.int64)


def silhouette(occupied: np.ndarray, cam_name: str, x: np.ndarray, y: np.ndarray, width: int, gcx: float
               ) -> np.ndarray:
    """Orthographic silhouette of voxels on the coarse grid (rows match the hull rows).

    A voxel covers |rx| + |ry| grid columns in projection, so diagonal views mark
    its whole footprint, not only its centre.
    """
    cam = camera(cam_name)
    footprint = abs(cam.right[0]) + abs(cam.right[1])
    u = x[None, :] * cam.right[0] + y[:, None] * cam.right[1] + gcx
    out = np.zeros((occupied.shape[0], width), dtype=bool)
    for offset in np.linspace(-footprint / 2 + 0.01, footprint / 2 - 0.01, 3):
        cols = np.clip(np.floor(u + offset).astype(np.int64), 0, width - 1)
        for row in range(occupied.shape[0]):
            out[row, cols[occupied[row]]] = True
    return out


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Start and end (exclusive) of each run of True values."""
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))


def round_sections(occupied: np.ndarray, exponent: float, vertical_sigma: float = 1.5) -> None:
    """Replace each box cross-section with an inscribed superellipse, in place.

    After the axis views, a row of the hull is a set of rectangles (front runs x side
    runs). A superellipse |dx/a|^p + |dy/b|^p <= 1 touches all four sides of its
    rectangle, so the front and side silhouettes do not change, but the body is
    round instead of boxy. p = 2 is an ellipse; a larger p is closer to the box.

    Jagged outlines (feather tips) change a section's size from one row to the next,
    which makes lathe-like rings. The rounded shape is blurred over ``vertical_sigma``
    rows before it cuts, so neighbouring rows agree. The silhouettes still cut exactly.
    """
    keep = np.zeros_like(occupied)
    for row in range(occupied.shape[0]):
        section = occupied[row]
        if not section.any():
            continue
        keep_row = keep[row]
        for y0, y1 in _runs(section.any(axis=1)):
            for x0, x1 in _runs(section.any(axis=0)):
                cy, cx, b, a = (y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2, (y1 - y0) / 2, (x1 - x0) / 2
                yy, xx = np.mgrid[y0:y1, x0:x1]
                keep_row[y0:y1, x0:x1] |= (np.abs((xx - cx) / a) ** exponent
                                           + np.abs((yy - cy) / b) ** exponent) <= 1
    if vertical_sigma > 0:
        keep = ndimage.gaussian_filter(keep.astype(np.float32), (vertical_sigma, 0, 0)) >= 0.5
    occupied &= keep


def carve(views: dict[str, View], factor: int = 5, tolerance_px: float = 10.0, rounding: float | None = None) -> Hull:
    """Carve a voxel hull from every available orthographic full-body view.

    ``tolerance_px`` (aligned pixels) dilates each secondary mask before carving.
    ``rounding`` (a superellipse exponent, e.g. 2.5) rounds the cross-sections
    before the diagonal views carve; ``None`` keeps the assumption-free box hull,
    which the consistency checks use. Needs ``front`` and ``right`` or ``left``.
    """
    if "front" not in views or not ({"right", "left"} & set(views)):
        raise ValueError("The hull needs front and right or left")
    width, height = CANVAS_W // factor, CANVAS_H // factor
    gcx = CX / factor
    masks = {n: downsample(v.mask, factor) for n, v in views.items() if n in AXIS_VIEWS + DIAGONAL_VIEWS}
    grow = max(0, int(round(tolerance_px / factor)))
    # The front and the primary side fix the frame exactly. Only the secondary views get the tolerance.
    anchors = {"front", "right" if "right" in masks else "left"}
    loose = {n: m if n in anchors or not grow else ndimage.binary_dilation(m, iterations=grow)
             for n, m in masks.items()}
    # Bounds from the axis views: x from front/back, y from the sides.
    front_cols = np.nonzero(loose["front"].any(axis=0))[0]
    side = loose.get("right", loose.get("left"))
    side_cols = np.nonzero(side.any(axis=0))[0]
    rows_used = np.nonzero(loose["front"].any(axis=1))[0]
    x_start = int(front_cols.min())
    x_cols = np.arange(front_cols.min(), front_cols.max() + 1)
    y_extent = max(abs(side_cols.min() - gcx), abs(side_cols.max() + 1 - gcx))
    y_start = int(np.floor(gcx - y_extent))
    y_cols = np.arange(y_start, int(np.ceil(gcx + y_extent)))
    x = x_cols + 0.5 - gcx
    y = y_cols + 0.5 - gcx
    occupied = np.zeros((height, len(y_cols), len(x_cols)), dtype=bool)
    occupied[rows_used.min():rows_used.max() + 1] = True

    def apply(name: str, shift: int = 0) -> None:
        cols = _project_columns(name, x, y, gcx) - shift
        valid = (cols >= 0) & (cols < width)
        cols = np.clip(cols, 0, width - 1)
        for row in range(rows_used.min(), rows_used.max() + 1):
            occupied[row] &= loose[name][row][cols] & valid

    used = [n for n in AXIS_VIEWS if n in masks]
    for name in used:
        apply(name)
    if rounding is not None:
        round_sections(occupied, rounding)
    shifts = {}
    for name in DIAGONAL_VIEWS:
        if name not in masks:
            continue
        projection = silhouette(occupied, name, x, y, width, gcx)
        figure = masks[name]
        total = max(1, figure.sum())
        best = min(range(-width // 8, width // 8 + 1),
                   key=lambda s: ((figure & ~np.roll(projection, s, axis=1)).sum() / total, abs(s)))
        shifts[name] = best
        apply(name, -best)
        used.append(name)
    missing = {}
    for name in used:
        projection = silhouette(occupied, name, x, y, width, gcx)
        projection = np.roll(projection, shifts.get(name, 0), axis=1)
        figure = masks[name]
        missing[name] = float((figure & ~projection).sum() / max(1, figure.sum()))
    return Hull(occupied, factor, x_start, y_start, shifts, missing, used)


def taubin(mesh: Mesh, rounds: int, lam: float = 0.5, mu: float = -0.53) -> Mesh:
    """Taubin smoothing: removes voxel terraces without the shrinkage of plain Laplacian smoothing."""
    adjacency, points = mesh.adjacency(), mesh.vertices.copy()
    for _ in range(rounds):
        for weight in (lam, mu):
            points += weight * (adjacency @ points - points)
    return Mesh(points, mesh.faces, mesh.colours)


def hull_mesh(hull: Hull, metres_per_px: float, smooth_sigma: float = 0.5, terrace_rounds: int = 15) -> Mesh:
    """Smooth triangle mesh of the hull: marching cubes on a lightly blurred field, then Taubin smoothing.

    A stronger voxel blur would also remove the terraces, but it shortens thin tips (tufts, ears).
    """
    from skimage.measure import marching_cubes

    field = np.pad(hull.occupied.astype(np.float32), 2)
    if smooth_sigma > 0:
        field = ndimage.gaussian_filter(field, smooth_sigma)
    verts, faces, _normals, _values = marching_cubes(field, level=0.5)
    voxel = hull.factor * metres_per_px
    gcx = CX / hull.factor
    row, yi, xi = verts[:, 0] - 2, verts[:, 1] - 2, verts[:, 2] - 2
    points = np.column_stack([(xi + hull.x_start + 0.5 - gcx) * voxel,
                              (yi + hull.y_start + 0.5 - gcx) * voxel,
                              (BASELINE / hull.factor - row - 0.5) * voxel])
    # skimage winds these faces inward for an occupancy field. Reverse them so normals point out.
    mesh = Mesh(points, faces[:, ::-1].copy())
    return taubin(mesh, terrace_rounds) if terrace_rounds else mesh
