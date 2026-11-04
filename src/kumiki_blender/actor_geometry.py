"""Pure geometry for turning flat rig layers into lit, bendable meshes.

Numpy only, so the same code runs in tests and inside Blender's bundled Python.
Art coordinates are (x, y) pixels with y down, matching Kumiki's character
evaluator. Nothing here knows about a specific character.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math

import numpy as np


def _erosion_distance(mask: np.ndarray) -> np.ndarray:
    """Octagonal distance from the silhouette, by alternating 4- and 8-neighbour erosion."""
    distance = np.zeros(mask.shape, np.float32)
    current = np.pad(mask, 1)
    step = 0
    while current.any():
        distance += current[1:-1, 1:-1]
        core = current[1:-1, 1:-1] & current[:-2, 1:-1] & current[2:, 1:-1] & current[1:-1, :-2] & current[1:-1, 2:]
        if step % 2:
            core &= current[:-2, :-2] & current[:-2, 2:] & current[2:, :-2] & current[2:, 2:]
        current = np.pad(core, 1)
        step += 1
    return distance


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return values
    size = 2 * radius + 1
    out = values
    for axis in (0, 1):
        pad = [(radius, radius) if index == axis else (0, 0) for index in (0, 1)]
        padded = np.pad(out, pad, mode="edge")
        cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
        zero_shape = list(cumulative.shape)
        zero_shape[axis] = 1
        cumulative = np.concatenate((np.zeros(zero_shape), cumulative), axis=axis)
        upper = np.take(cumulative, range(size, cumulative.shape[axis]), axis=axis)
        lower = np.take(cumulative, range(0, cumulative.shape[axis] - size), axis=axis)
        out = (upper - lower) / size
    return out


def inflate_mask(mask: np.ndarray, *, plateau: float = .8, smooth: int = 2) -> np.ndarray:
    """Normalised dome height for a filled mask: a round cross-section, 0 outside."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("Cannot inflate an empty layer")
    distance = _erosion_distance(mask)
    reach = max(float(distance.max()) * plateau, 1.0)
    unit = np.clip(distance / reach, 0, 1)
    height = np.sqrt(1 - (1 - unit) ** 2)
    if smooth:
        height = _box_blur(height, smooth)
        height *= mask
    peak = float(height.max())
    return (height / peak).astype(np.float32) if peak > 0 else height.astype(np.float32)


def variant_union(masks: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[int, int]]:
    """One silhouette for every drawing of a part, so swapping a drawing never changes the shape.

    Empty drawings (a hidden state) are skipped. Drawn variants must share one size.
    """
    drawn = {key: np.asarray(mask, bool) for key, mask in masks.items() if np.asarray(mask).any()}
    if not drawn:
        raise ValueError("A part needs at least one drawn variant")
    sizes = {mask.shape for mask in drawn.values()}
    if len(sizes) != 1:
        raise ValueError(f"Variants of one part must be the same size, got {sorted(sizes)}")
    union = np.zeros(next(iter(sizes)), bool)
    for mask in drawn.values():
        union |= mask
    return union, union.shape


@dataclass(frozen=True)
class HeightMesh:
    """Grid vertices for both faces of a layer. `sides` is -1 front, +1 back."""
    pixels: np.ndarray    # (n, 2) art pixels
    heights: np.ndarray   # (n,) normalised dome height
    sides: np.ndarray     # (n,)
    faces: np.ndarray     # (m, 4) vertex indices


def heightfield_mesh(height: np.ndarray, step: int, footprint: np.ndarray | None = None) -> HeightMesh:
    """Two grids that meet at the silhouette, without cells fully outside it.

    `footprint` widens the kept cells beyond the dome, flat at height 0, so other
    drawings of the part have geometry under them without changing the dome.
    """
    rows, columns = height.shape
    y, x = np.mgrid[:rows, :columns]
    index = np.arange(rows * columns).reshape(rows, columns)
    if footprint is not None and footprint.shape != height.shape:
        raise ValueError("footprint must match the height field")
    cover = height if footprint is None else np.maximum(height, footprint.astype(height.dtype))
    corners = cover[:-1, :-1] + cover[:-1, 1:] + cover[1:, :-1] + cover[1:, 1:]
    keep = corners > 0
    a, b = index[:-1, :-1][keep], index[:-1, 1:][keep]
    c, d = index[1:, 1:][keep], index[1:, :-1][keep]
    front = np.stack((a, d, c, b), axis=1)  # faces -Y once mapped to Blender
    back = np.stack((a, b, c, d), axis=1) + rows * columns
    pixels = np.stack((x.ravel(), y.ravel()), axis=1).astype(np.float64) * step
    return HeightMesh(np.concatenate((pixels, pixels)), np.concatenate((height.ravel(), height.ravel())),
                      np.concatenate((np.full(height.size, -1), np.full(height.size, 1))),
                      np.concatenate((front, back)))


def _nearest_on_mask(mask: np.ndarray, point: tuple[float, float]) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot trace a chain on an empty layer")
    nearest = int(np.argmin((xs - point[0]) ** 2 + (ys - point[1]) ** 2))
    return int(xs[nearest]), int(ys[nearest])


def _farthest(mask: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    """Point on the art with the longest walk from `start`."""
    rows, columns = mask.shape
    seen = {start}
    queue = deque([start])
    last = start
    while queue:
        last = queue.popleft()
        x, y = last
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            point = (x + dx, y + dy)
            if 0 <= point[0] < columns and 0 <= point[1] < rows and mask[point[1], point[0]] and point not in seen:
                seen.add(point)
                queue.append(point)
    return last


def _medial_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Cheapest walk across the art, where steps near the outline cost more."""
    interior = _erosion_distance(mask)
    rows, columns = mask.shape
    costs = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    while queue:
        cost, point = heapq.heappop(queue)
        if point == end:
            break
        if cost > costs[point]:
            continue
        x, y = point
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < columns and 0 <= ny < rows and mask[ny, nx]):
                continue
            total = cost + math.hypot(dx, dy) * (1 + 6 / (1 + float(interior[ny, nx])))
            if total < costs.get((nx, ny), math.inf):
                costs[(nx, ny)] = total
                previous[(nx, ny)] = point
                heapq.heappush(queue, (total, (nx, ny)))
    if end not in costs:
        raise ValueError("Chain tip is not connected to the start on the art")
    path = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def trace_chain(mask: np.ndarray, start: tuple[float, float], *, joints: int = 3,
                tip: tuple[float, float] | None = None) -> list[tuple[float, float]]:
    """Follow a limb from `start` to its far end. Returns joints + 1 distinct points on the art.

    The path favours the middle of the limb, so bones sit inside the drawing even
    when the drawn limb is already bent.
    """
    if joints < 1:
        raise ValueError("A chain needs at least one bone")
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("Cannot trace a chain on an empty layer")
    step = max(1, math.ceil(max(mask.shape) / 160))
    small = mask[::step, ::step]
    origin = _nearest_on_mask(small, (start[0] / step, start[1] / step))
    end = _nearest_on_mask(small, (tip[0] / step, tip[1] / step)) if tip else _farthest(small, origin)
    path = _medial_path(small, origin, end)
    lengths = np.cumsum([0.0] + [math.dist(a, b) for a, b in zip(path, path[1:])])
    if lengths[-1] <= 0:
        raise ValueError("Chain start and tip are the same point")
    picks = [path[min(int(np.searchsorted(lengths, lengths[-1] * i / joints)), len(path) - 1)] for i in range(joints + 1)]
    return [(float(x * step), float(y * step)) for x, y in picks]


def chain_weights(vertices: np.ndarray, chain: list[tuple[float, float]], *, blend: float) -> np.ndarray:
    """Per-vertex bone weights with a smooth blend across each joint."""
    points = np.asarray(chain, float)
    starts, ends = points[:-1], points[1:]
    spans = ends - starts
    lengths = np.linalg.norm(spans, axis=1)
    if (lengths <= 0).any():
        raise ValueError("Chain has a zero-length bone")
    joints = np.concatenate(([0.0], np.cumsum(lengths)))
    relative = vertices[:, None, :] - starts[None, :, :]
    along = np.clip((relative * spans[None]).sum(axis=2) / lengths ** 2, 0, 1)
    nearest = starts[None] + along[..., None] * spans[None]
    bone = np.argmin(((vertices[:, None, :] - nearest) ** 2).sum(axis=2), axis=1)
    rows = np.arange(len(vertices))
    position = joints[bone] + along[rows, bone] * lengths[bone]
    weights = np.zeros((len(vertices), len(lengths)))
    for index in range(len(lengths)):
        lower = np.ones(len(vertices)) if index == 0 else _smoothstep((position - joints[index] + blend) / (2 * blend))
        upper = np.ones(len(vertices)) if index == len(lengths) - 1 else 1 - _smoothstep((position - joints[index + 1] + blend) / (2 * blend))
        weights[:, index] = lower * upper
    return weights / weights.sum(axis=1, keepdims=True)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    t = np.clip(value, 0, 1)
    return t * t * (3 - 2 * t)


def local_matrix(*, pivot: tuple[float, float], x: float, y: float, rotation: float,
                 scale_x: float, scale_y: float) -> np.ndarray:
    """Kumiki partLocalMatrix: u -> pivot + (x, y) + S·R(u − pivot), clockwise in y-down."""
    angle = math.radians(rotation)
    cos, sin = math.cos(angle), math.sin(angle)
    linear = np.array([[cos * scale_x, -sin * scale_y], [sin * scale_x, cos * scale_y]])
    origin = np.array(pivot, float)
    offset = origin + np.array((x, y), float) - linear @ origin
    return np.array([[linear[0, 0], linear[0, 1], offset[0]],
                     [linear[1, 0], linear[1, 1], offset[1]],
                     [0.0, 0.0, 1.0]])


def part_transform(*, pivot: tuple[float, float], x: float, y: float, rotation: float, scale_x: float, scale_y: float,
                   metres_per_px: float, yaw: float = 0, pitch: float = 0) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Blender location, rotation matrix and scale for a part, with a 3D turn about its pivot.

    Art faces -Y with X right and Z up. Positive yaw turns toward screen right. Positive pitch looks up.
    """
    flat = math.radians(rotation)
    cos, sin = math.cos(flat), math.sin(flat)
    about_y = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    turn_yaw, turn_pitch = math.radians(yaw), math.radians(-pitch)
    about_z = np.array([[math.cos(turn_yaw), -math.sin(turn_yaw), 0], [math.sin(turn_yaw), math.cos(turn_yaw), 0], [0, 0, 1]])
    about_x = np.array([[1, 0, 0], [0, math.cos(turn_pitch), -math.sin(turn_pitch)], [0, math.sin(turn_pitch), math.cos(turn_pitch)]])
    orientation = about_z @ about_x @ about_y
    # Depth scales with X, so a limb under a scaled parent reaches equally far in every direction.
    scale = np.array((scale_x, scale_x, scale_y))
    centre = np.array((pivot[0] * metres_per_px, 0.0, -pivot[1] * metres_per_px))
    placed = centre + np.array((x * metres_per_px, 0.0, -y * metres_per_px))
    return placed - orientation @ (scale * centre), orientation, (scale_x, scale_x, scale_y)
