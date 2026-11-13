"""Bake a UV texture for a mesh from the full-resolution reference views.

Each texel of the mesh's UV layout gets a 3D point and normal. For every view
that sees the point (depth-buffer test against the mesh itself) the raw
reference image is sampled at full resolution. The view that sees the surface
most directly dominates: weights are (n . d)^power, and 3/4 views count less,
so two views do not blend into ghost features. Texels no view sees are filled
from their neighbours. This keeps the drawings' crisp clothes, buttons and fur
painting, which per-vertex colours blur.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .mesh import Mesh, rasterize, surface_samples
from .views import DIAGONAL_VIEWS, camera


def read_obj_uv(path: Path) -> tuple[Mesh, np.ndarray]:
    """OBJ with UVs. Returns the mesh (Blender axes) and UVs per face corner, shape (M, 3, 2)."""
    vertices, uvs, faces, corner_uv = [], [], [], []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(v) for v in line.split()[1:3]])
            elif line.startswith("f "):
                ids, tex = [], []
                for token in line.split()[1:]:
                    parts = token.split("/")
                    ids.append(int(parts[0]) - 1)
                    tex.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1)
                for i in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[i], ids[i + 1]])
                    corner_uv.append([tex[0], tex[i], tex[i + 1]])
    if not uvs or min(min(c) for c in corner_uv) < 0:
        raise ValueError(f"{path.name} has faces without UVs")
    points = np.array(vertices, dtype=np.float64)
    points = np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])  # OBJ Y up -> Z up
    return Mesh(points, np.array(faces)), np.array(uvs)[np.array(corner_uv)]


def texel_samples(mesh: Mesh, corner_uv: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Texel centres covered by each UV triangle: (row, col), 3D point, interpolated normal."""
    normals = mesh.vertex_normals()
    rows, cols, points, nrm = [], [], [], []
    pixel = corner_uv * size  # u right, v up
    for f, tri in enumerate(pixel):
        x, y = tri[:, 0], size - tri[:, 1]
        x0, x1 = int(np.floor(x.min())), int(np.ceil(x.max()))
        y0, y1 = int(np.floor(y.min())), int(np.ceil(y.max()))
        gx, gy = np.meshgrid(np.arange(max(x0, 0), min(x1 + 1, size)) + 0.5,
                             np.arange(max(y0, 0), min(y1 + 1, size)) + 0.5)
        if gx.size == 0:
            continue
        d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(d) < 1e-12:
            continue
        w0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
        w1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        w = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
        corners = mesh.faces[f]
        rows.append(np.floor(gy[inside]).astype(int))
        cols.append(np.floor(gx[inside]).astype(int))
        points.append(w @ mesh.vertices[corners])
        nrm.append(w @ normals[corners])
    nrm = np.concatenate(nrm)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(points), nrm


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x, y = np.clip(x - 0.5, 0, w - 1.001), np.clip(y - 0.5, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def bake_texture(handoff: Path, mesh: Mesh, corner_uv: np.ndarray, *, size: int = 2048, power: float = 6.0,
                 diagonal_weight: float = 0.15, depth_scale: float = 1.0, depth_tolerance_px: float = 1.5
                 ) -> tuple[np.ndarray, dict]:
    """RGB texture (size x size, uint8) and a report with the coverage of each view."""
    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    px = spec["metres_per_aligned_px"]
    rows, cols, points, normals = texel_samples(mesh, corner_uv, size)
    samples = surface_samples(mesh, 0.5 * px / depth_scale)
    total = np.zeros((len(points), 3))
    weight = np.zeros(len(points))
    coverage = {}
    for name, raw_map in spec["raw"].items():
        raw_path = handoff / "raw" / f"{name}.png"
        if not raw_path.is_file():
            continue
        rgba = np.asarray(Image.open(raw_path).convert("RGBA")).astype(np.float64)
        core = ndimage.binary_erosion(rgba[..., 3] > 200, iterations=3)  # edge pixels mix with background
        cam = camera(name)
        _, depth = rasterize(mesh, cam, px, depth_scale, samples)
        col, row, point_depth = cam.project(points, px, depth_scale)
        c, r = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = (c >= 0) & (c < depth.shape[1]) & (r >= 0) & (r < depth.shape[0])
        nearest = np.full(len(points), np.nan)
        nearest[inside] = depth[r[inside], c[inside]]
        cosine = np.clip(normals @ cam.toward, 0, 1)
        slope = np.minimum(np.sqrt(np.maximum(1 - cosine ** 2, 0)) / np.maximum(cosine, 1e-3), 10)
        visible = inside & (point_depth >= nearest - (depth_tolerance_px + 1.5 * slope) * px / depth_scale)
        u, v = col / depth_scale, row / depth_scale  # camera-frame aligned pixels
        raw_x = (u + raw_map["shift_px"] - raw_map["x0"]) / raw_map["scale"] + raw_map["left"]
        raw_y = (v - raw_map["y0"]) / raw_map["scale"] + raw_map["top"]
        h, w = core.shape
        rx, ry = np.floor(raw_x).astype(int), np.floor(raw_y).astype(int)
        on_image = (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h)
        use = visible & on_image & (cosine > 0.05)
        use[use] = core[ry[use], rx[use]]
        factor = cosine ** power * (diagonal_weight if name in DIAGONAL_VIEWS else 1.0)
        colour = _bilinear(rgba[..., :3], raw_x[use], raw_y[use])
        total[use] += colour * factor[use, None]
        weight[use] += factor[use]
        coverage[name] = round(float(use.mean()), 4)
    image = np.zeros((size, size, 3))
    filled = np.zeros((size, size), dtype=bool)
    seen = weight > 0
    image[rows[seen], cols[seen]] = total[seen] / weight[seen, None]
    filled[rows[seen], cols[seen]] = True
    covered = np.zeros((size, size), dtype=bool)
    covered[rows, cols] = True
    # Fill unseen texels and the UV gutters from the nearest seen texel.
    if filled.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~filled, return_indices=True)
        image = image[iy, ix]
    report = {"size": size, "texels": int(covered.sum()), "seen_fraction": round(float(seen.mean()), 4),
              "per_view": coverage}
    return np.clip(np.round(image), 0, 255).astype(np.uint8), report
