"""Project the aligned reference images onto a mesh from the reference cameras.

``project_colours`` gives each vertex a colour from the views that see it. A
vertex is visible in a view when no other surface is nearer the camera at its
pixel (depth buffer from the NumPy rasterizer). Each visible view is weighted
by (n . d)^2, so a view that sees the surface straight on wins over a grazing
view.

``relief`` adds stylized surface detail from the clay views: the high-pass of
the projected clay brightness moves vertices along their normals. This is a
look, not measured geometry. Feathers and fur clumps in a clay painting are
mostly shading, so the result is plausible relief, not a reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .mesh import Mesh, rasterize, surface_samples
from .views import CANVAS_H, CANVAS_W, camera


@dataclass
class Projection:
    values: np.ndarray        # (N, C) weighted mean of the sampled image values
    weight: np.ndarray        # (N,) total weight; 0 means no view saw the vertex
    per_view: dict[str, float]  # fraction of vertices that each view coloured


def subdivide(mesh: Mesh) -> Mesh:
    """Split each triangle into four. Shared edges get one midpoint, so the mesh stays closed."""
    f = mesh.faces
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    mids = (mesh.vertices[unique[:, 0]] + mesh.vertices[unique[:, 1]]) / 2
    base = len(mesh.vertices)
    m = inverse.reshape(3, -1).T + base  # midpoint ids of edges (0-1, 1-2, 2-0) for each face
    faces = np.concatenate([np.column_stack([f[:, 0], m[:, 0], m[:, 2]]), np.column_stack([f[:, 1], m[:, 1], m[:, 0]]),
                            np.column_stack([f[:, 2], m[:, 2], m[:, 1]]), m])
    colours = None
    if mesh.colours is not None:
        colours = np.concatenate([mesh.colours, ((mesh.colours[unique[:, 0]].astype(int)
                                                  + mesh.colours[unique[:, 1]]) // 2).astype(np.uint8)])
    return Mesh(np.concatenate([mesh.vertices, mids]), faces, colours)


def project_images(mesh: Mesh, images: dict[str, np.ndarray], metres_per_px: float, *, scale: float = 0.5,
                   depth_tolerance_px: float = 1.0, power: float = 2.0) -> Projection:
    """Project per-pixel values (H, W, C) in the camera frame onto the vertices.

    ``images`` hold float values with NaN where the image has no figure.
    """
    normals = mesh.vertex_normals()
    samples = surface_samples(mesh, 0.5 * metres_per_px / scale)
    channels = next(iter(images.values())).shape[2]
    total = np.zeros((len(mesh.vertices), channels))
    weight = np.zeros(len(mesh.vertices))
    per_view = {}
    for name, image in images.items():
        cam = camera(name)
        _, depth = rasterize(mesh, cam, metres_per_px, scale, samples)
        col, row, vertex_depth = cam.project(mesh.vertices, metres_per_px, scale)
        c, r = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = (c >= 0) & (c < depth.shape[1]) & (r >= 0) & (r < depth.shape[0])
        # The vertex's own pixel always contains the vertex (samples include the corners), so its
        # depth buffer value is the nearest surface there. A tilted surface spans more depth inside
        # one pixel, so the tolerance grows with the slope: tan(angle) = sqrt(1 - c^2) / c.
        nearest = np.full(len(c), np.nan)
        nearest[inside] = depth[r[inside], c[inside]]
        cosine = np.clip(normals @ cam.toward, 1e-3, 1)
        slope = np.minimum(np.sqrt(1 - cosine ** 2) / cosine, 10.0)
        grid_px = metres_per_px / scale
        visible = inside & (vertex_depth >= nearest - (depth_tolerance_px + 1.5 * slope) * grid_px)
        facing = np.clip(normals @ cam.toward, 0, 1) ** power
        values = np.full((len(c), channels), np.nan)
        full_c = np.clip((col / scale).astype(int), 0, image.shape[1] - 1)
        full_r = np.clip((row / scale).astype(int), 0, image.shape[0] - 1)
        values[inside] = image[full_r[inside], full_c[inside]]
        use = visible & (facing > 0) & np.isfinite(values).all(axis=1)
        total[use] += values[use] * facing[use, None]
        weight[use] += facing[use]
        per_view[name] = round(float(use.mean()), 4)
    with np.errstate(invalid="ignore"):
        result = total / weight[:, None]
    return Projection(result, weight, per_view)


def fill_unseen(mesh: Mesh, values: np.ndarray, seen: np.ndarray, rounds: int = 200) -> np.ndarray:
    """Spread values from seen vertices into unseen ones over the mesh graph."""
    adjacency = mesh.adjacency()
    values, known = values.copy(), seen.copy()
    values[~known] = 0
    for _ in range(rounds):
        if known.all():
            break
        count = adjacency @ known.astype(float)
        spread = adjacency @ (values * known[:, None])
        grow = ~known & (count > 0)
        values[grow] = spread[grow] / count[grow, None]
        known = known | grow
    return values


def load_camera_images(handoff: Path, names: list[str], suffix: str = "") -> dict[str, np.ndarray]:
    """Aligned RGBA images in the camera frame as float RGB with NaN outside the figure."""
    images = {}
    for name in names:
        path = handoff / "aligned" / f"{name}{suffix}.png"
        if not path.is_file():
            continue
        rgba = np.asarray(Image.open(path).convert("RGBA")).astype(float)
        if rgba.shape[:2] != (CANVAS_H, CANVAS_W):
            raise ValueError(f"{path.name} is not in the {CANVAS_W}x{CANVAS_H} camera frame")
        rgb = rgba[..., :3]
        # Shrink the figure by 2 px: edge pixels mix figure and background colours.
        core = ndimage.binary_erosion(rgba[..., 3] > 127, iterations=2)
        rgb[~core] = np.nan
        images[name] = rgb
    return images


X_MIRROR = {"left": "right", "right": "left", "front_left": "front_right", "front_right": "front_left",
            "back_left": "back_right", "back_right": "back_left"}


def mirror_missing(images: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Add each missing side view as the left-right flip of its opposite side.

    Valid only for a left-right symmetric character: the camera frame is centred
    on x = 0, so a flip about the frame centre is the view from the mirrored
    camera. A one-sided accessory (a satchel) lands on the wrong side.
    """
    out, added = dict(images), []
    for name, source in X_MIRROR.items():
        if name not in images and source in images:
            out[name] = images[source][:, ::-1].copy()
            added.append(name)
    return out, added


def project_colours(mesh: Mesh, handoff: Path, names: list[str], metres_per_px: float, *,
                    assume_symmetric: bool = False) -> tuple[Mesh, dict]:
    images = load_camera_images(handoff, names)
    added = []
    if assume_symmetric:
        images, added = mirror_missing(images)
    projection = project_images(mesh, images, metres_per_px)
    seen = projection.weight > 0
    rgb = fill_unseen(mesh, np.nan_to_num(projection.values), seen)
    coloured = Mesh(mesh.vertices, mesh.faces, np.clip(np.round(rgb), 0, 255).astype(np.uint8))
    return coloured, {"seen_fraction": round(float(seen.mean()), 4), "per_view": projection.per_view,
                      "mirrored_views": added}


def relief(mesh: Mesh, brightness: np.ndarray, seen: np.ndarray, amplitude_m: float, *, smooth_rounds: int = 8,
           full_scale: float = 0.15) -> tuple[Mesh, dict]:
    """Move vertices along normals by the high-pass of projected clay brightness (0..1).

    A brightness step of ``full_scale`` or more gives the full ``amplitude_m``.
    The scale is absolute, so flat clay gives no relief. Bright clay faces the
    light, so the sign is a convention: high-pass > 0 moves out. Unseen
    vertices do not move.
    """
    adjacency = mesh.adjacency()
    # Normalised smoothing: average only over seen vertices, so unseen ones do not count as dark.
    weight, low = seen.astype(float), np.where(seen, brightness, 0.0)
    for _ in range(smooth_rounds):
        low, weight = adjacency @ low, adjacency @ weight
    with np.errstate(invalid="ignore", divide="ignore"):
        low = low / weight
    detail = np.where(seen & (weight > 0), brightness - low, 0.0)
    offset = np.clip(detail / full_scale, -1, 1) * amplitude_m
    moved = Mesh(mesh.vertices + mesh.vertex_normals() * offset[:, None], mesh.faces, mesh.colours)
    return moved, {"moved_fraction": round(float((np.abs(offset) > 0.05 * amplitude_m).mean()), 4),
                   "mean_offset_m": round(float(np.abs(offset).mean()), 7), "amplitude_m": amplitude_m,
                   "provenance": "stylized relief from clay shading, not measured geometry"}
