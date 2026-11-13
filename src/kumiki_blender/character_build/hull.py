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


def carve(views: dict[str, View], factor: int = 5, tolerance_px: float = 10.0) -> Hull:
    """Carve a voxel hull from every available orthographic full-body view.

    ``tolerance_px`` (aligned pixels) dilates each secondary mask before carving.
    This is the assumption-free box hull. The checks use it, and it measures
    the shifts of diagonal views. Needs ``front`` and ``right`` or ``left``.
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


def drop_fragments(mesh: Mesh, keep_area_fraction: float = 0.05) -> Mesh:
    """Keep the main body and pieces with 5% or more of its area; drop carving debris.

    Thin outline features that the carve cuts off leave small floating pieces.
    QuadriFlow rejects a mesh with them, and they are not sculpt geometry.
    """
    from ..reconstruction.mesh_fragments import FragmentLimits, choose_faces

    keep, _removed = choose_faces(mesh.vertices, mesh.faces, FragmentLimits(1e-9, keep_area_fraction))
    faces = mesh.faces[keep]
    used, remap = np.unique(faces, return_inverse=True)
    colours = mesh.colours[used] if mesh.colours is not None else None
    return Mesh(mesh.vertices[used], remap.reshape(-1, 3), colours)


def sdf_blockout(views: dict[str, View], hull: Hull, metres_per_px: float, *, tolerance_px: float = 10.0,
                 round_px: float = 6.0, outline_blur_px: float = 12.0) -> Mesh:
    """Smooth blockout from the silhouettes' signed distance fields.

    Each view's 2D signed distance field (aligned pixels, positive outside) is
    computed at full resolution and extruded along its camera direction. The
    extrusions are intersected with a smooth maximum: k * log(sum exp(d / k)),
    k = ``round_px``. The smooth maximum rounds the edges where two views meet,
    and the fields are continuous from row to row, so the surface has no voxel
    terraces. In an extrusion intersection, the front surface is the side outline
    extruded across the body, so each wobble of a fur or feather fringe becomes a
    groove. A Gaussian blur of each 2D field (``outline_blur_px``) low-passes the
    contour and removes them. ``hull`` gives the grid and the diagonal shifts.
    The front and the primary side are exact; other views get ``tolerance_px``.
    """
    from skimage.measure import marching_cubes

    factor, gcx = hull.factor, CX / hull.factor
    anchors = {"front", "right" if "right" in hull.used else "left"}
    rows, ys, xs = hull.occupied.shape
    margin = 4
    x = (np.arange(-margin, xs + margin) + hull.x_start + 0.5 - gcx)        # voxel units from the centre
    y = (np.arange(-margin, ys + margin) + hull.y_start + 0.5 - gcx)
    row_px = (np.arange(rows) + 0.5) * factor                               # aligned pixel row of each voxel row
    k = max(round_px, 1e-3)
    total = np.zeros((rows, len(y), len(x)))
    for name in hull.used:
        mask = views[name].mask
        sdf = (ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)).astype(np.float32)
        if outline_blur_px > 0:
            sdf = ndimage.gaussian_filter(sdf, outline_blur_px)
        if name not in anchors:
            sdf -= tolerance_px
        cam = camera(name)
        col_px = (x[None, :] * cam.right[0] + y[:, None] * cam.right[1] + gcx + hull.shifts.get(name, 0)) * factor
        for r in range(rows):
            values = ndimage.map_coordinates(sdf, [np.full(col_px.size, row_px[r] - 0.5), col_px.ravel() - 0.5],
                                             order=1, mode="nearest").reshape(col_px.shape)
            total[r] += np.exp(np.clip(values / k, -60, 60))
    field = k * np.log(total)                                               # smooth max, aligned pixels
    field = np.pad(field, ((2, 2), (0, 0), (0, 0)), constant_values=field.max())
    verts, faces, _n, _v = marching_cubes(field, level=0.0, allow_degenerate=False)  # no sliver triangles
    row, yi, xi = verts[:, 0] - 2, verts[:, 1] - margin, verts[:, 2] - margin
    voxel = factor * metres_per_px
    points = np.column_stack([(xi + hull.x_start + 0.5 - gcx) * voxel, (yi + hull.y_start + 0.5 - gcx) * voxel,
                              (BASELINE / factor - row - 0.5) * voxel])
    mesh = Mesh(points, faces)
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:                  # make the normals point out
        mesh = Mesh(mesh.vertices, mesh.faces[:, ::-1].copy())
    return drop_fragments(mesh)
