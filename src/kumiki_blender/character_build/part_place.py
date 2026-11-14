"""Put a part that was modelled alone into the frame of the whole character, and join parts for the later steps.

An image-to-3D generator returns a part (a tail, a hat) with any size, position and
facing. ``place_part`` finds the turn about the vertical axis, the scale and the position
at which the part covers its masks in the character's views. The masks come from a part
handoff (``split_views``): they are in the frame of the whole character, so a placed part
needs no further alignment, and ``fit``, ``texture`` and ``compare-region`` work on it.

``join_parts`` makes one mesh with one texture from placed, textured parts. The parts
stay separate pieces of surface with separate regions of the texture, so no colour of
one can land on another. ``compare``, ``appearance`` and ``rig`` take the joined mesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .mesh import Mesh, surface_samples
from .views import AXIS_VIEWS, BASELINE, CX, camera


def _extent(values: np.ndarray, trim: float) -> tuple[float, float]:
    low, high = np.percentile(values, [trim, 100 - trim])
    return float(low), float(high)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def place_part(mesh: Mesh, masks: dict[str, np.ndarray], metres_per_px: float, *, scale: float = 0.25,
               coarse_deg: float = 15.0, fine_deg: float = 3.0, trim: float = 0.5, yaw_deg: float | None = None
               ) -> tuple[Mesh, dict]:
    """The part where it covers ``masks`` best, and the transform: placed = scale * Rz(yaw) @ p + offset.

    ``masks`` are the part's silhouettes in the camera frame of the character at ``scale``
    of the aligned frame (``handoff.reference_masks`` of a part handoff). For each turn
    about Z, the scale and the position follow from the extents of the part and of its
    masks in the axis views (front, back, left, right), by least squares; the extents
    leave out ``trim`` percent at each end, so a tuft of fur does not set the size. The
    turn with the best mean IoU wins (every ``coarse_deg``, then ``fine_deg`` around the
    best). Give ``yaw_deg`` if the facing is known.

    Two axis views that look along different axes fix the part in all three directions.
    With one view, or two opposite ones, the depth along the view stays at the part's own
    centre, and the report says so.
    """
    axis = [name for name in masks if name in AXIS_VIEWS and masks[name].any()]
    if not axis:
        raise ValueError("place_part needs the part's mask in a front, back, left or right view")
    samples = surface_samples(mesh, np.ptp(mesh.vertices, axis=0).max() / 400)[0]
    samples = samples[:: max(1, len(samples) // 200_000)]
    grid = metres_per_px / scale                       # metres per mask pixel
    targets = {}
    for name in axis:                                  # extents of each mask in metres, in the character's frame
        rows, cols = np.nonzero(masks[name])
        u = ((cols + 0.5) / scale - CX) * metres_per_px
        z = (BASELINE - (rows + 0.5) / scale) * metres_per_px
        targets[name] = (_extent(u, trim), _extent(z, trim))
    looks = {name: camera(name) for name in masks}

    def solve(yaw: float) -> tuple[float, np.ndarray, np.ndarray]:
        a = np.radians(yaw)
        rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        turned = samples @ rotation.T
        rows, values = [], []
        for name in axis:
            right = looks[name].right
            (u0, u1), (z0, z1) = targets[name]
            m0, m1 = _extent(turned @ right, trim)
            h0, h1 = _extent(turned[:, 2], trim)
            for model, target, direction in ((m0, u0, right), (m1, u1, right), (h0, z0, (0, 0, 1)), (h1, z1, (0, 0, 1))):
                rows.append([model, *direction])
                values.append(target)
        matrix = np.array(rows, dtype=np.float64)
        solution = np.linalg.lstsq(matrix, np.array(values), rcond=None)[0]
        factor, offset = float(solution[0]), solution[1:].copy()
        centre = (turned.min(axis=0) + turned.max(axis=0)) / 2
        for index in range(3):                         # a direction that no view measures: centred on that axis
            if np.abs(matrix[:, 1 + index]).max() < 1e-9:
                offset[index] = -centre[index] * factor
        return factor, rotation, offset

    def score(yaw: float) -> tuple[float, dict]:
        factor, rotation, offset = solve(yaw)
        if factor <= 0:
            return 0.0, {}
        points = (samples @ rotation.T) * factor + offset
        values = {}
        for name, target in masks.items():
            col, row, _ = looks[name].project(points, metres_per_px, scale)
            model = np.zeros_like(target)
            c, r = np.floor(col).astype(int), np.floor(row).astype(int)
            keep = (c >= 0) & (c < target.shape[1]) & (r >= 0) & (r < target.shape[0])
            model[r[keep], c[keep]] = True
            if name in AXIS_VIEWS:
                values[name] = _iou(model, target)
            else:                                      # a 3/4 view has no shared horizontal centre
                values[name] = max(_iou(np.roll(model, shift, axis=1), target) for shift in range(-20, 21, 4))
        return float(np.mean(list(values.values()))), values

    if yaw_deg is None:
        coarse = {yaw: score(yaw)[0] for yaw in np.arange(0, 360, coarse_deg)}
        start = max(coarse, key=coarse.get)
        fine = {yaw % 360: score(yaw)[0] for yaw in np.arange(start - coarse_deg, start + coarse_deg + 1e-6, fine_deg)}
        yaw_deg = float(max(fine, key=fine.get))
    factor, rotation, offset = solve(yaw_deg)
    mean, per_view = score(yaw_deg)
    measured = np.abs(np.array([looks[name].right for name in axis])).max(axis=0) > 0.5
    info = {"yaw_deg": float(yaw_deg), "scale": factor, "offset_m": [float(v) for v in offset],
            "mean_iou": round(mean, 4), "views": {k: round(v, 4) for k, v in per_view.items()},
            "position_measured": {"x": bool(measured[0]), "y": bool(measured[1]), "z": True},
            "mask_px_m": grid}
    return Mesh(mesh.vertices @ rotation.T * factor + offset, mesh.faces, mesh.colours), info


def join_parts(parts: list[tuple[Mesh, np.ndarray, np.ndarray]], *, size: int = 4096, min_share: float = 0.15
               ) -> tuple[Mesh, np.ndarray, np.ndarray, list[dict]]:
    """One mesh, its UVs per face corner, one texture, and where each part went.

    Each part is (mesh, UVs per face corner, texture). The parts get columns of the new
    texture, as wide as their share of the surface (at least ``min_share``). The meshes
    are not welded to each other: a part stays a separate piece of surface.
    """
    if not parts:
        raise ValueError("join_parts needs at least one part")
    area = np.array([part[0].area() for part in parts])
    width = np.maximum(area / area.sum(), min_share if len(parts) > 1 else 1.0)
    width /= width.sum()
    edges = np.round(np.concatenate([[0], np.cumsum(width)]) * size).astype(int)
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    vertices, faces, corner_uv, layout = [], [], [], []
    start = 0
    for index, (mesh, uv, image) in enumerate(parts):
        left, right = int(edges[index]), int(edges[index + 1])
        tile = Image.fromarray(np.asarray(image)[..., :3]).resize((right - left, size), Image.LANCZOS)
        texture[:, left:right] = np.asarray(tile)
        placed = uv.astype(np.float64).copy()
        placed[..., 0] = (left + np.clip(placed[..., 0], 0, 1) * (right - left)) / size
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + start)
        corner_uv.append(placed)
        layout.append({"part": index, "u_range": [left / size, right / size], "vertices": [start, start + len(mesh.vertices)],
                       "faces": [sum(len(f) for f in faces[:-1]), sum(len(f) for f in faces)]})
        start += len(mesh.vertices)
    return Mesh(np.concatenate(vertices), np.concatenate(faces)), np.concatenate(corner_uv), texture, layout


def write_obj_uv(mesh: Mesh, corner_uv: np.ndarray, path: Path, comment: str = "") -> None:
    """A triangle OBJ with UVs, Y up (the interchange convention that ``texture.read_obj_uv`` reads back)."""
    lines = [f"# {comment}"] if comment else []
    v = mesh.vertices
    lines += [f"v {x:.7g} {z:.7g} {-y:.7g}" for x, y, z in v]
    flat = corner_uv.reshape(-1, 2)
    lines += [f"vt {u:.6f} {w:.6f}" for u, w in flat]
    for index, (a, b, c) in enumerate(mesh.faces + 1):
        t = 3 * index + 1
        lines.append(f"f {a}/{t} {b}/{t + 1} {c}/{t + 2}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
