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
from scipy import ndimage, sparse
from scipy.sparse import csgraph

from .mesh import Mesh, surface_samples
from .views import AXIS_VIEWS, AZIMUTH, BASELINE, CX, camera


def _extent(values: np.ndarray, trim: float) -> tuple[float, float]:
    low, high = np.percentile(values, [trim, 100 - trim])
    return float(low), float(high)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def place_part(mesh: Mesh, masks: dict[str, np.ndarray], metres_per_px: float, *, scale: float = 0.25,
               coarse_deg: float = 15.0, fine_deg: float = 3.0, trim: float = 0.5, yaw_deg: float | None = None,
               refine: bool = True, ignore_thinner_px: float = 0.0, stretch: float = 0.15,
               anchor: Mesh | None = None, tilt_range_deg: float = 0.0) -> tuple[Mesh, dict]:
    """The part where it covers ``masks`` best, and the transform:
    placed = scale * stretch * (tilt @ Rz(yaw) @ p) + offset, with one stretch factor for each axis.

    ``masks`` are the part's silhouettes in the camera frame of the character at ``scale``
    of the aligned frame (``handoff.reference_masks`` of a part handoff). For each turn
    about Z, the scale and the position follow from the extents of the part and of its
    masks in the axis views (front, back, left, right), by least squares. Both extents
    are taken over silhouette pixels and leave out ``trim`` percent at each end, so a
    tuft of fur does not set the size. The
    turn with the best mean IoU wins (every ``coarse_deg``, then ``fine_deg`` around the
    best). Give ``yaw_deg`` if the facing is known.

    Each direction then gets its own scale, at most ``stretch`` from the common one, so the
    part has the height and the width of its masks (0: one scale for all directions).
    ``refine`` moves the part in small steps while the mean IoU rises. ``ignore_thinner_px``
    (aligned pixels) drops the thin pieces of the masks first: Pip's bag masks hold the
    whole strap, the generated bag has a stub of it.

    With ``tilt_range_deg`` above 0, the part can also turn about the line of sight of the
    first axis view, at most that far (every ``fine_deg``). A generator stands a part as it
    likes: Pip's forearm, drawn at 40 degrees below level, came back at 25. The turn about
    Z cannot give that.

    Two axis views that look along different axes fix the part in all three directions.
    With one view, or two opposite ones, the depth along the view is not measured. The
    part then sits where ``anchor`` (the body mesh) is at that place, or, without an
    anchor, on the plane through the origin. The report says which.
    """
    if ignore_thinner_px > 0:
        # A thin piece that the mesh lacks (the strap of a bag across the chest) would pull the
        # part toward it. Keep the thick core of each mask.
        radius = max(1, round(ignore_thinner_px * scale / 2))
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disc = xx * xx + yy * yy <= radius * radius
        masks = {name: ndimage.binary_opening(mask, structure=disc) for name, mask in masks.items()}
        masks = {name: mask for name, mask in masks.items() if mask.any()}
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

    sight = np.asarray(looks[axis[0]].toward, dtype=np.float64)
    cross = np.array([[0, -sight[2], sight[1]], [sight[2], 0, -sight[0]], [-sight[1], sight[0], 0]])

    def solve(yaw: float, tilt: float = 0.0, own_scales: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
        a, t = np.radians(yaw), np.radians(tilt)
        rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        if tilt:
            rotation = (np.eye(3) + np.sin(t) * cross + (1 - np.cos(t)) * cross @ cross) @ rotation
        turned = samples @ rotation.T
        rows, values = [], []
        for name in axis:
            right = looks[name].right
            (u0, u1), (z0, z1) = targets[name]
            # Extents of the part's silhouette, as for the mask: over the cells that it covers, not
            # over its surface points. Flat soles hold many points and few pixels, and with points
            # Pip's body came out 6% short.
            u, z = turned @ right, turned[:, 2]
            cell = max(float(np.ptp(u)), float(np.ptp(z))) / 300
            covered = np.unique(np.column_stack([np.floor(u / cell), np.floor(z / cell)]).astype(np.int64), axis=0)
            m0, m1 = _extent((covered[:, 0] + 0.5) * cell, trim)
            h0, h1 = _extent((covered[:, 1] + 0.5) * cell, trim)
            for model, target, direction in ((m0, u0, right), (m1, u1, right), (h0, z0, (0, 0, 1)), (h1, z1, (0, 0, 1))):
                rows.append([model, *direction])
                values.append(target)
        matrix, wanted = np.array(rows, dtype=np.float64), np.array(values)
        factor = float(np.linalg.lstsq(matrix, wanted, rcond=None)[0][0])
        # Then each direction alone, within ``stretch`` of the common scale: a part that was generated
        # alone has its own proportions (Pip's body came out wider for its height than it is drawn,
        # and one scale for all directions left it 6% short).
        size, offset = np.full(3, factor), np.zeros(3)
        centre = (turned.min(axis=0) + turned.max(axis=0)) / 2
        for index in range(3):
            use = np.abs(matrix[:, 1 + index]) > 0.5
            if not use.any():                          # a direction that no view measures: centred on that axis
                offset[index] = -centre[index] * factor
                continue
            model, sign, target = matrix[use, 0], matrix[use, 1 + index], wanted[use]
            if own_scales and stretch > 0 and np.ptp(model) > 1e-12:
                own = float(np.linalg.lstsq(np.column_stack([model, sign]), target, rcond=None)[0][0])
                size[index] = float(np.clip(own, factor * (1 - stretch), factor * (1 + stretch)))
            offset[index] = float((sign * (target - size[index] * model)).mean())
        return size, rotation, offset

    def overlap(size: np.ndarray, rotation: np.ndarray, offset: np.ndarray) -> tuple[float, dict]:
        if size.min() <= 0:
            return 0.0, {}
        points = (samples @ rotation.T) * size + offset
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

    def score(yaw: float) -> tuple[float, dict]:
        return overlap(*solve(yaw))

    if yaw_deg is None:
        coarse = {yaw: score(yaw)[0] for yaw in np.arange(0, 360, coarse_deg)}
        start = max(coarse, key=coarse.get)
        fine = {yaw % 360: score(yaw)[0] for yaw in np.arange(start - coarse_deg, start + coarse_deg + 1e-6, fine_deg)}
        yaw_deg = float(max(fine, key=fine.get))
    tilt_deg = 0.0
    if tilt_range_deg > 0:
        # With one scale for all directions: a scale for each direction can make up for a wrong
        # tilt (a level ellipse, squashed, is much like a tilted one).
        tilts = {float(tilt): overlap(*solve(yaw_deg, float(tilt), own_scales=False))[0]
                 for tilt in np.arange(-tilt_range_deg, tilt_range_deg + 1e-6, fine_deg)}
        tilts.setdefault(0.0, overlap(*solve(yaw_deg, own_scales=False))[0])
        best = max(tilts, key=tilts.get)
        tilt_deg = best if tilts[best] > tilts[0.0] + 1e-4 else 0.0      # no gain: no tilt
    size, rotation, offset = solve(yaw_deg, tilt_deg)
    mean, per_view = overlap(size, rotation, offset)
    start = mean
    measured = np.abs(np.array([looks[name].right for name in axis])).max(axis=0) > 0.5
    if refine:
        # The extents can be off where a mask holds more than the mesh (the whole strap of a bag, of
        # which the generator made a stub). Move the part, in steps that halve, while the mean IoU
        # rises. The size stays: the overlap alone prefers a part that is too small.
        move_step = 12 * grid
        for _ in range(6):
            improved = True
            while improved:
                improved = False
                for index in range(3):
                    if not (measured[index] or index == 2):
                        continue
                    for sign in (1, -1):
                        trial = offset + np.eye(3)[index] * sign * move_step
                        value, views = overlap(size, rotation, trial)
                        if value > mean + 1e-4:
                            offset, mean, per_view, improved = trial, value, views, True
            move_step /= 2
    factor = float(np.median(size))
    anchored = False
    if anchor is not None and not measured.all():
        # A direction that no view measures (a hand drawn from the front and the back only): where
        # the anchor mesh (the body) is, at the place of the part in the measured directions.
        placed = (samples @ rotation.T) * size + offset
        low, high = placed.min(axis=0), placed.max(axis=0)
        near = np.ones(len(anchor.vertices), bool)
        for index in range(3):
            if measured[index] or index == 2:
                near &= (anchor.vertices[:, index] >= low[index]) & (anchor.vertices[:, index] <= high[index])
        if near.any():
            for index in range(2):
                if not measured[index]:
                    offset[index] += float(np.median(anchor.vertices[near, index])) - (low[index] + high[index]) / 2
            anchored = True
            mean, per_view = overlap(size, rotation, offset)
    info = {"yaw_deg": float(yaw_deg), "tilt_deg": float(tilt_deg), "tilt_about_sight_of": axis[0],
            "scale": factor, "stretch": [round(float(v / factor), 4) for v in size],
            "offset_m": [float(v) for v in offset],
            "mean_iou": round(mean, 4), "mean_iou_from_extents": round(start, 4),
            "views": {k: round(v, 4) for k, v in per_view.items()},
            "position_measured": {"x": bool(measured[0]), "y": bool(measured[1]), "z": True},
            "unmeasured_from_anchor": anchored,
            "mask_px_m": grid}
    return Mesh(mesh.vertices @ rotation.T * size + offset, mesh.faces, mesh.colours), info


def seat_part(part: Mesh, body: Mesh, metres_per_px: float, *, view: str | None = None, embed: float = 0.15,
              scale: float = 0.5, flatten_m: float | None = None) -> tuple[Mesh, dict]:
    """Move a part out of the body it hangs on, until only ``embed`` of its surface is still inside.

    The masks place a bag where the drawings show it. The body was modelled without the
    bag, and its surface there can lie further out than the bag, which then shows only a
    corner. ``view`` names the side from which the part sits on the body (``left`` for a
    bag on the left hip). The part moves toward that camera, as one rigid piece, by the
    depth at which ``embed`` of its surface points are still behind the body's surface.
    It never moves inward.

    Without ``view``, it is the level view, of the eight around the character, that needs
    the shortest move, among those that look at the part from its side of the body. A
    bag at the front of the left hip comes out to the front left: toward the left alone,
    Pip's bag had to move 30 mm, and toward the front left 13 mm.

    The part of the surface that stays inside can come out on the other side of a thin
    body: Pip's bag came through the back of the skin flap that it hangs in front of.
    With ``flatten_m``, each vertex that is still deeper than that behind the body's
    surface, seen from ``view``, moves up to that depth. The back of the part then lies
    just under the surface that it sits on. The faces and the UVs stay.
    """
    from .mesh import rasterize

    samples = surface_samples(part, np.ptp(part.vertices, axis=0).max() / 200)[0]

    def measure(name: str) -> tuple[float, np.ndarray]:
        look = camera(name)
        _, depth = rasterize(body, look, metres_per_px, scale)
        col, row, reach = look.project(samples, metres_per_px, scale)
        c, r = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = (c >= 0) & (c < depth.shape[1]) & (r >= 0) & (r < depth.shape[0])
        surface = np.full(len(samples), -np.inf)
        surface[inside] = np.nan_to_num(depth[r[inside], c[inside]], nan=-np.inf)
        burial = surface - reach                      # positive: behind the body's surface, seen from the camera
        finite = np.isfinite(burial)
        return (max(0.0, float(np.percentile(burial[finite], 100 * (1 - embed)))) if finite.any() else 0.0), burial

    if view is None:
        outward = part.vertices.mean(axis=0) - body.vertices.mean(axis=0)
        outward[2] = 0.0
        facing = sorted(AZIMUTH, key=lambda name: -float(camera(name).toward @ outward))   # the most direct first
        facing = [name for name in facing if float(camera(name).toward @ outward) > 0] or facing[:1]
        moves = {name: measure(name)[0] for name in facing}
        view = min(facing, key=lambda name: (round(moves[name] / 0.0005), facing.index(name)))
    move, burial = measure(view)
    look = camera(view)
    vertices = part.vertices + move * look.toward
    info = {"view": view, "moved_m": round(move, 5), "buried_before": round(float((burial > 0).mean()), 4),
            "buried_after": round(float((burial - move > 0).mean()), 4)}
    if flatten_m is not None:
        _, depth = rasterize(body, look, metres_per_px, scale)
        col, row, reach = look.project(vertices, metres_per_px, scale)
        c, r = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = (c >= 0) & (c < depth.shape[1]) & (r >= 0) & (r < depth.shape[0])
        surface = np.full(len(vertices), -np.inf)
        surface[inside] = np.nan_to_num(depth[r[inside], c[inside]], nan=-np.inf)
        lift = np.where(np.isfinite(surface), np.maximum(surface - reach - flatten_m, 0.0), 0.0)
        vertices = vertices + lift[:, None] * look.toward
        info["flattened_vertices"] = int((lift > 0).sum())
        info["flatten_max_m"] = round(float(lift.max()), 5) if len(lift) else 0.0
    return Mesh(vertices, part.faces, part.colours), info


def cut_under(mesh: Mesh, corner_uv: np.ndarray | None, masks: dict[str, np.ndarray], metres_per_px: float, *,
              scale: float = 0.25, grow_px: float = 6.0, loose_faces: int = 200,
              join: list[Mesh] | None = None, join_px: float = 14.0, smooth: int = 2, take: bool = False
              ) -> tuple[Mesh, np.ndarray | None, int]:
    """The mesh without the faces that a separate part replaces; its UVs; the number of faces removed.

    The body was modelled with its hands. When the hands become a part of their own, the
    body's hands stay under them and show through. A face goes if its centre lies inside
    the part's masks (the camera frame at ``scale``, as ``handoff.reference_masks`` gives
    it), grown by ``grow_px`` aligned pixels, along every direction that the views look.
    Opposite views (front and back) look along one direction and draw the same outline, so
    one of them is enough for a face: drawings do not agree to the pixel, and what lies
    in only one of the two outlines is still the part. Two views along different
    directions bound the part in space; with one direction, everything behind the part
    goes too. The cut leaves an open edge, which the new part covers. A piece of fewer
    than ``loose_faces`` faces that the cut separated from the mesh (a fingertip outside
    the outline) goes too.

    With ``take``, the result is the other piece: only the faces that would go (and the
    number is that of the faces taken). The piece keeps the UVs of the mesh, so it can use
    the texture of the mesh. Mirrored (``mirror_part``), it replaces the same limb on the
    side where the generator made it badly.

    Whole faces go, so the open edge has teeth of the size of a face, and between it and
    the new part a gap can show. With ``join`` (the meshes of the new parts, in place),
    the edge is smoothed ``smooth`` times along itself, and each of its vertices that is
    within ``join_px`` aligned pixels of a part moves onto the surface of that part. The
    body then ends on the part: a flap of skin reaches the arm that it hangs from.
    """
    centre = mesh.vertices[mesh.faces].mean(axis=1)
    gone = _under_masks(centre, masks, metres_per_px, scale, grow_px)      # no view shows the part: no cut
    if gone.any() and loose_faces > 0:
        kept = mesh.faces[~gone]
        graph = sparse.coo_matrix((np.ones(len(kept) * 2), (np.repeat(kept[:, 0], 2), kept[:, 1:].ravel())),
                                  shape=(len(mesh.vertices),) * 2)
        _, piece = csgraph.connected_components(graph, directed=False)
        faces_in = np.bincount(piece[kept[:, 0]], minlength=piece.max() + 1)
        at_cut = np.unique(piece[np.unique(mesh.faces[gone])])
        loose = at_cut[faces_in[at_cut] < loose_faces]
        gone[~gone] = np.isin(piece[kept[:, 0]], loose)
    if take:
        used, inverse = np.unique(mesh.faces[gone], return_inverse=True)
        piece = Mesh(mesh.vertices[used], inverse.reshape(-1, 3))
        return piece, (corner_uv[gone] if corner_uv is not None else None), int(gone.sum())
    used, inverse = np.unique(mesh.faces[~gone], return_inverse=True)
    cut = Mesh(mesh.vertices[used], inverse.reshape(-1, 3))
    if join and gone.any():
        cut = Mesh(_close_cut(cut, np.isin(used, np.unique(mesh.faces[gone])), join, join_px * metres_per_px, smooth),
                   cut.faces)
    return cut, (corner_uv[~gone] if corner_uv is not None else None), int(gone.sum())


def _close_cut(mesh: Mesh, at_cut: np.ndarray, parts: list[Mesh], reach_m: float, smooth: int) -> np.ndarray:
    """Vertex positions with the cut edge (``at_cut`` vertices on the open border) smoothed and set on ``parts``."""
    from scipy.spatial import cKDTree

    edges = np.sort(np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]), axis=1)
    edges, count = np.unique(edges, axis=0, return_counts=True)
    border = edges[count == 1]
    border = border[at_cut[border].any(axis=1)]
    moving = np.zeros(len(mesh.vertices), bool)
    moving[border.ravel()] = True
    moving &= at_cut
    vertices = mesh.vertices.copy()
    for _ in range(max(0, smooth)):
        total, number = np.zeros_like(vertices), np.zeros(len(vertices))
        for a, b in ((0, 1), (1, 0)):
            np.add.at(total, border[:, a], vertices[border[:, b]])
            np.add.at(number, border[:, a], 1)
        use = moving & (number >= 2)
        vertices[use] = 0.5 * vertices[use] + 0.5 * total[use] / number[use, None]
    surface = np.concatenate([surface_samples(part, reach_m / 8)[0] for part in parts])
    distance, nearest = cKDTree(surface).query(vertices[moving])
    near = distance <= reach_m
    index = np.nonzero(moving)[0][near]
    vertices[index] = surface[nearest[near]]
    return vertices


def _under_masks(points: np.ndarray, masks: dict[str, np.ndarray], metres_per_px: float, scale: float,
                 grow_px: float) -> np.ndarray:
    """The points that lie inside the masks along every line of sight (opposite views are one line)."""
    along: dict[tuple[int, ...], np.ndarray] = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        view = camera(name)
        wide = ndimage.binary_dilation(mask, iterations=max(1, round(grow_px * scale))) if grow_px > 0 else mask
        col, row, _ = view.project(points, metres_per_px, scale)
        c, r = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = (c >= 0) & (c < wide.shape[1]) & (r >= 0) & (r < wide.shape[0])
        hit = np.zeros(len(points), bool)
        hit[inside] = wide[r[inside], c[inside]]
        toward = np.round(np.asarray(view.toward, float), 3)
        if tuple(toward) < tuple(-toward):
            toward = -toward
        key = tuple(int(v) for v in np.round(toward * 1000))
        along[key] = along.get(key, np.zeros(len(points), bool)) | hit
    return np.logical_and.reduce(list(along.values())) if along else np.zeros(len(points), bool)


def shows_under(mesh: Mesh, masks: dict[str, np.ndarray], view: str, metres_per_px: float, *, scale: float = 0.25,
                grow_px: float = 6.0, depth_tolerance_m: float = 0.0015) -> np.ndarray:
    """Where the mesh under a part's masks shows in ``view``: a mask in the aligned frame (full size).

    A drawing can show a limb at another place than the model has it. Seen from the side,
    the skin flap of Pip's model lies in front of the vest of the side drawing, and the bake
    paints it green. The pixels of that drawing where the model's limb is the nearest
    surface must not paint: cut them out with ``split_views.box_part(aligned_masks=...)``.
    The faces are those under ``masks`` (as for ``cut_under``); the result is grown by
    ``grow_px`` aligned pixels.
    """
    from .mesh import rasterize

    centre = mesh.vertices[mesh.faces].mean(axis=1)
    under = _under_masks(centre, masks, metres_per_px, scale, 0.0)
    look = camera(view)
    whole, nearest = rasterize(mesh, look, metres_per_px, 1.0)
    if not under.any():
        return np.zeros(whole.shape, dtype=bool)
    part, depth = rasterize(Mesh(mesh.vertices, mesh.faces[under]), look, metres_per_px, 1.0)
    front = part & (np.nan_to_num(depth, nan=-np.inf) >= np.nan_to_num(nearest, nan=np.inf) - depth_tolerance_m)
    front = ndimage.binary_closing(front, iterations=3)
    return ndimage.binary_dilation(front, iterations=max(1, round(grow_px))) if grow_px > 0 else front


def smooth_under(mesh: Mesh, masks: dict[str, np.ndarray], metres_per_px: float, *, scale: float = 0.25,
                 grow_px: float = 0.0, rounds: int = 30, strength: float = 0.5, pass_band: float = 0.1,
                 fade: int = 4, shrink: bool = False, wrap: Mesh | None = None) -> tuple[Mesh, dict]:
    """The mesh with the surface under a part's masks smoothed; the faces, and so the UVs, stay.

    A generator makes a thin sheet (the skin between an arm and a body) with a crumpled
    surface and a ragged edge. The vertices that lie inside ``masks`` (as for
    ``cut_under``) are smoothed ``rounds`` times. Each round is one step toward the mean
    of the neighbours and one step back (Taubin), so that the sheet keeps its size.
    ``pass_band`` sets what stays: at 0.1 only fine wrinkles go, and at 0.01, with more
    rounds, waves of several faces go too. With ``shrink`` there is no step back: a fin
    or a deep fold that the generator made (the end of a scarf as a sheet that stands off
    a shoulder) sinks into the surface, and the region gets smaller. ``wrap`` gives the
    volume back: after the smoothing, each vertex of the region moves to the nearest
    point of that mesh. Use the mirror image of the mesh (``mirror_part``) where the other
    side came out clean. The effect fades over ``fade`` rings of vertices at the border of the region, and a vertex
    on an open edge of the mesh does not move.
    """
    count = len(mesh.vertices)
    edges = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    both = np.sort(edges, axis=1)
    unique, times = np.unique(both, axis=0, return_counts=True)
    open_edge = np.zeros(count, bool)
    open_edge[unique[times == 1].ravel()] = True
    link = sparse.coo_matrix((np.ones(len(unique) * 2), (np.concatenate([unique[:, 0], unique[:, 1]]),
                                                         np.concatenate([unique[:, 1], unique[:, 0]]))),
                             shape=(count, count)).tocsr()
    degree = np.maximum(np.asarray(link.sum(axis=1)).ravel(), 1)
    mean = sparse.diags(1.0 / degree) @ link
    inside = _under_masks(mesh.vertices, masks, metres_per_px, scale, grow_px)
    weight = inside.astype(np.float64)
    for _ in range(max(0, fade)):                      # 1 well inside the region, 0 outside it
        weight = np.minimum(weight, mean @ weight)
    weight[open_edge] = 0.0
    vertices = mesh.vertices.copy()
    steps = (strength,) if shrink else (strength, -strength / (1 - pass_band * strength))   # Taubin: 1/a + 1/b = pass band
    for _ in range(max(0, rounds)):
        for step in steps:
            vertices += (step * weight)[:, None] * (mean @ vertices - vertices)
    if wrap is not None and (weight > 0).any():
        from scipy.spatial import cKDTree

        length = float(np.median(np.linalg.norm(mesh.vertices[unique[:, 0]] - mesh.vertices[unique[:, 1]], axis=1)))
        target = cKDTree(surface_samples(wrap, length / 4)[0])
        ids = np.nonzero(weight > 0)[0]
        for _ in range(3):                             # onto the target, and spread along it
            _, nearest = target.query(vertices[ids], workers=-1)
            vertices[ids] += weight[ids, None] * (target.data[nearest] - vertices[ids])
            vertices += (0.5 * weight)[:, None] * (mean @ vertices - vertices)
        _, nearest = target.query(vertices[ids], workers=-1)
        vertices[ids] += weight[ids, None] * (target.data[nearest] - vertices[ids])
    moved = np.linalg.norm(vertices - mesh.vertices, axis=1)
    return Mesh(vertices, mesh.faces, mesh.colours), {
        "vertices_in_region": int(inside.sum()), "vertices_moved": int((moved > 1e-9).sum()),
        "mean_move_mm": round(float(moved[inside].mean() * 1000), 3) if inside.any() else 0.0,
        "max_move_mm": round(float(moved.max() * 1000), 3)}


def shift_under(mesh: Mesh, masks: dict[str, np.ndarray], metres_per_px: float, move_m: tuple[float, float, float],
                *, scale: float = 0.25, grow_px: float = 0.0, fade: int = 8, relax: int = 30) -> tuple[Mesh, dict]:
    """The mesh with what lies under a part's masks moved by ``move_m``; the faces, and so the UVs, stay.

    A generated body can hold a limb at another depth than a drawing shows it. Pip's left
    arm was 16 mm further forward than the side drawing has it, and than the right arm.
    The side drawing then painted the vest on the sleeve. The front and back drawings
    cannot see such a fault: they look along the depth.

    The vertices inside ``masks`` (as for ``cut_under``) move by the whole vector. Around
    them the move fades to nothing over ``fade`` rings of vertices. Mark the whole limb, from
    where it leaves the body: the fade then lies on the body, where the limb slides along
    the surface, and the limb keeps its shape. Behind the limb the faces of the fade are
    pushed together and fold. ``relax`` rounds of smoothing (as ``smooth_under``) on the
    vertices of the fade spread them again.
    """
    count = len(mesh.vertices)
    edges = np.unique(np.sort(np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]),
                              axis=1), axis=0)
    link = sparse.coo_matrix((np.ones(len(edges) * 2), (np.concatenate([edges[:, 0], edges[:, 1]]),
                                                        np.concatenate([edges[:, 1], edges[:, 0]]))),
                             shape=(count, count)).tocsr()
    inside = _under_masks(mesh.vertices, masks, metres_per_px, scale, grow_px)
    rings = np.where(inside, 0.0, np.inf)               # rings of vertices away from the region
    reached = inside.copy()
    for ring in range(1, max(0, fade) + 1):
        new = ((link @ reached.astype(np.float64)) > 0) & ~reached
        rings[new] = ring
        reached |= new
    t = np.clip(1.0 - rings / (max(0, fade) + 1), 0.0, 1.0)
    weight = t * t * (3 - 2 * t)                        # 1 in the region, 0 beyond the last ring, smooth between
    vertices = mesh.vertices + weight[:, None] * np.asarray(move_m, dtype=np.float64)
    between = ((weight > 0) & (weight < 1)).astype(np.float64)
    if relax > 0 and between.any():
        degree = np.maximum(np.asarray(link.sum(axis=1)).ravel(), 1)
        mean = sparse.diags(1.0 / degree) @ link
        for _ in range(relax):
            for step in (0.5, -0.5 / (1 - 0.02 * 0.5)):          # Taubin, as smooth_under with a pass band of 0.02
                vertices += (step * between)[:, None] * (mean @ vertices - vertices)
    return Mesh(vertices, mesh.faces, mesh.colours), {
        "vertices_in_region": int(inside.sum()), "vertices_moved": int((weight > 0).sum()),
        "move_mm": [round(float(v) * 1000, 3) for v in move_m]}


def plate_part(mask: np.ndarray, metres_per_px: float, *, view: str = "front", scale: float = 0.25,
               thickness_m: float = 0.003, anchor: Mesh | None = None, outside: Mesh | None = None,
               overlap_px: float = 12.0, smooth: int = 3, drape: bool = False, lift_m: float = 0.001,
               both_sides: bool = False) -> tuple[Mesh, np.ndarray, dict]:
    """A thin plate with the outline of ``mask``, its UVs for each face corner, and a short report.

    A generator makes a thin sheet badly: the skin between Pip's arm and body came out a
    third as wide as it is drawn. A sheet that one drawing shows face-on needs no
    generator. ``mask`` is its silhouette in the camera frame of ``view`` at ``scale``
    (``handoff.reference_masks`` of a part handoff). Each mask cell becomes a square of the
    plate, ``thickness_m`` thick, and the outline is smoothed ``smooth`` times.

    The plate lies in the plane that fits ``anchor`` (the body and the parts beside it)
    best where the mask is, so it meets the arm and the hip that it hangs between.
    Without an anchor it lies in the plane through the origin. With ``outside`` (the
    body), only the cells outside that mesh's silhouette are kept, and ``overlap_px``
    aligned pixels inside it: the plate then adds what the body lacks. Where the two
    outlines differ by a pixel or two, nothing is added.

    With ``drape``, the plate is no plane: it lies on the surface of ``anchor`` that
    ``view`` sees, ``lift_m`` above it, as the end of a scarf lies on a shoulder. Where
    the mask reaches beyond the anchor (a fringe that hangs free), it continues at the
    depth of the nearest surface. The depth is smoothed, so the plate does not follow
    each fold. What hangs free shows from the opposite side too, as it would on the
    character.

    UVs: the side toward the camera has the left half of the texture, the other side the
    right half. With ``both_sides``, the other side uses the left half too: a part that
    only one drawing shows (the end of a scarf, seen from behind) has that drawing on both
    of its sides, and no side is a fill colour. ``texture`` bakes both from the drawings of the part handoff. The rim takes
    the colour just inside the outline: the outline itself is pale, mixed with the
    background of the drawing.
    """
    from .mesh import rasterize

    look = camera(view)
    if abs(look.toward[2]) > 1e-9:
        raise ValueError("plate_part needs a level view")
    keep = np.asarray(mask, dtype=bool).copy()
    if outside is not None:
        body, _ = rasterize(outside, look, metres_per_px, scale)
        # What the body lacks: the mask outside its silhouette, without the slivers where two
        # outlines differ by a pixel or two, and without specks.
        lacking = ndimage.binary_opening(keep & ~body, iterations=max(1, round(4 * scale)))
        labels, count = ndimage.label(lacking)
        if count > 1:
            sizes = ndimage.sum_labels(lacking, labels, range(1, count + 1))
            lacking = np.isin(labels, 1 + np.nonzero(sizes >= 0.1 * sizes.max())[0])
        band = max(1, round(overlap_px * scale))
        reach = ndimage.binary_dilation(lacking, iterations=band + max(1, round(4 * scale))) if lacking.any() else lacking
        inner = ndimage.binary_erosion(body, iterations=band) if overlap_px > 0 else body
        keep &= reach & ~inner
    if not keep.any():
        raise ValueError("No cell of the mask is left for the plate")
    rows, cols = np.nonzero(keep)
    grid = np.full((keep.shape[0] + 1, keep.shape[1] + 1), -1)
    corners = np.unique(np.concatenate([np.column_stack([rows + dr, cols + dc]) for dr in (0, 1) for dc in (0, 1)]), axis=0)
    grid[corners[:, 0], corners[:, 1]] = np.arange(len(corners))
    a, b, c, d = grid[rows, cols], grid[rows, cols + 1], grid[rows + 1, cols + 1], grid[rows + 1, cols]
    quads = np.column_stack([a, b, c, d])
    flat = corners.astype(np.float64)                   # (row, col) of each corner, in mask cells
    edges = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
    ordered, times = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    rim = ordered[times == 1]
    on_rim = np.zeros(len(corners), bool)
    on_rim[rim.ravel()] = True
    for _ in range(max(0, smooth)):                     # the staircase of the cells, rounded off
        total, number = np.zeros_like(flat), np.zeros(len(flat))
        for first, second in ((0, 1), (1, 0)):
            np.add.at(total, rim[:, first], flat[rim[:, second]])
            np.add.at(number, rim[:, first], 1)
        move = on_rim & (number >= 2)
        flat[move] = 0.5 * flat[move] + 0.5 * total[move] / number[move, None]
        shift = flat - corners                          # at most a third of a cell, so that no face turns over
        length = np.linalg.norm(shift, axis=1)
        far = length > 0.33
        flat[far] = corners[far] + shift[far] * (0.33 / length[far])[:, None]
    u = (flat[:, 1] / scale - CX) * metres_per_px
    z = (BASELINE - flat[:, 0] / scale) * metres_per_px
    depth, fitted = np.zeros(len(flat)), False
    if anchor is not None and drape:
        seen, nearest = rasterize(anchor, look, metres_per_px, scale)
        if seen.any():
            _, (iy, ix) = ndimage.distance_transform_edt(~seen, return_indices=True)
            surface = np.nan_to_num(nearest, nan=0.0)[iy, ix]              # beyond the anchor: the nearest surface
            surface = ndimage.gaussian_filter(surface, max(1.0, 6 * scale))
            ri = np.clip(np.round(flat[:, 0]).astype(int), 0, keep.shape[0] - 1)
            ci = np.clip(np.round(flat[:, 1]).astype(int), 0, keep.shape[1] - 1)
            depth, fitted = surface[ri, ci] + lift_m + 0.5 * thickness_m, True
    elif anchor is not None:
        col, row, reach = look.project(anchor.vertices, metres_per_px, scale)
        near_mask = ndimage.binary_dilation(keep, iterations=max(1, round(8 * scale)))
        ci, ri = np.floor(col).astype(int), np.floor(row).astype(int)
        ok = (ci >= 0) & (ci < keep.shape[1]) & (ri >= 0) & (ri < keep.shape[0])
        ok[ok] = near_mask[ri[ok], ci[ok]]
        if ok.sum() >= 3:
            au, az = anchor.vertices[ok] @ look.right, anchor.vertices[ok][:, 2]
            design = np.column_stack([au, az, np.ones(ok.sum())])
            plane = np.linalg.lstsq(design, reach[ok], rcond=None)[0]
            slope = np.clip(plane[:2], -0.6, 0.6)       # near to face-on: the sheet is drawn in this view
            offset = float(np.median(reach[ok] - design[:, :2] @ slope))
            depth, fitted = u * slope[0] + z * slope[1] + offset, True
    centre = np.outer(u, look.right) + np.outer(depth, look.toward)
    centre[:, 2] += z
    half = 0.5 * thickness_m * look.toward
    count = len(corners)
    vertices = np.concatenate([centre + half, centre - half])
    front = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    normal = np.cross(vertices[front[0, 1]] - vertices[front[0, 0]], vertices[front[0, 2]] - vertices[front[0, 0]])
    if normal @ look.toward < 0:
        front = front[:, [0, 2, 1]]
    back = front[:, [0, 2, 1]] + count
    # The rim: two triangles for each open edge. Of the two ways to wind them, the right one gives
    # the solid its volume (area times thickness).
    code = np.sort(edges, axis=1) @ np.array([count, 1])
    directed = edges[np.isin(code, rim @ np.array([count, 1]))]
    wall = np.concatenate([np.column_stack([directed[:, 1], directed[:, 0], directed[:, 0] + count]),
                           np.column_stack([directed[:, 1], directed[:, 0] + count, directed[:, 1] + count])])
    local = vertices - vertices.mean(axis=0)

    def volume(triangles: np.ndarray) -> float:
        return float(np.einsum("ij,ij->i", local[triangles[:, 0]],
                               np.cross(local[triangles[:, 1]], local[triangles[:, 2]])).sum() / 6)

    sides = volume(np.concatenate([front, back]))
    cell = (metres_per_px / scale) ** 2
    wanted = float(keep.sum()) * cell * thickness_m
    if abs(sides + volume(wall[:, [0, 2, 1]]) - wanted) < abs(sides + volume(wall) - wanted):
        wall = wall[:, [0, 2, 1]]
    faces = np.concatenate([front, back, wall])
    span = np.maximum(corners.max(axis=0) - corners.min(axis=0), 1).astype(np.float64)
    uv_front = np.column_stack([0.02 + 0.46 * (flat[:, 1] - corners[:, 1].min()) / span[1],
                                0.98 - 0.96 * (flat[:, 0] - corners[:, 0].min()) / span[0]])
    uv = np.concatenate([uv_front, uv_front if both_sides else uv_front + [0.5, 0.0]])
    corner_uv = uv[faces]
    # The rim takes the colour just inside the outline, on the side toward the camera (the fourth
    # nearest inner corner). With its own corners in both halves of the texture, each rim triangle
    # would cross the whole texture.
    inward = np.arange(count)
    interior = np.nonzero(~on_rim)[0]
    if len(interior):
        from scipy.spatial import cKDTree

        inward[on_rim] = interior[cKDTree(flat[interior]).query(flat[on_rim], k=min(4, len(interior)))[1].reshape(
            int(on_rim.sum()), -1)[:, -1]]
    corner_uv[len(front) + len(back):] = uv_front[inward[wall % count]]
    mesh = Mesh(vertices, faces)
    low, high = corners.min(axis=0) / scale, corners.max(axis=0) / scale      # (row, column) in aligned pixels
    return mesh, corner_uv, {"cells": int(keep.sum()), "faces": int(len(faces)), "depth_from_anchor": fitted,
                             "area_cm2": round(float(keep.sum()) * (metres_per_px / scale) ** 2 * 1e4, 2),
                             "box_aligned_px": [float(low[1]), float(low[0]), float(high[1]), float(high[0])]}


def plate_texture(box_aligned_px: list[float], near: np.ndarray, far: np.ndarray | None = None, *, size: int = 1024
                  ) -> np.ndarray:
    """The texture of a plate from ``plate_part``: the drawings' own pixels, with no bake.

    ``box_aligned_px`` is from the report of ``plate_part``. ``near`` is the aligned
    drawing (RGBA, 1600 x 1200) of the plate's view, masked to the part. It fills the
    left half of the texture, where the side toward the camera has its UVs. ``far`` is the
    aligned drawing of the opposite view. It is mirrored, so that it lines up with the
    near view, and fills the right half. Without it, the right half is the near drawing
    again. Outside the part, each texel takes the nearest pixel of the part.

    The bake does not trust pixels near an outline, and a fringe is all outline: baked,
    two thirds of the end of Pip's scarf were a fill colour.
    """
    x0, y0, x1, y1 = box_aligned_px
    width, height = round(0.46 * size), round(0.96 * size)

    def half(rgba: np.ndarray) -> np.ndarray:
        picture = Image.fromarray(rgba).transform((width, height), Image.AFFINE,
                                                  ((x1 - x0) / width, 0, x0, 0, (y1 - y0) / height, y0), Image.BILINEAR)
        tile = np.asarray(picture)
        shown = tile[..., 3] > 127
        if not shown.any():
            return np.zeros((height, width, 3), dtype=np.uint8)
        _, (iy, ix) = ndimage.distance_transform_edt(~shown, return_indices=True)
        return tile[..., :3][iy, ix]

    texture = np.zeros((size, size, 3), dtype=np.uint8)
    left = half(near)
    right = half(far[:, ::-1]) if far is not None else left
    if not right.any():
        right = left
    top = round(0.02 * size)
    for start, tile in ((round(0.02 * size), left), (round(0.52 * size), right)):
        texture[top:top + height, start:start + width] = tile
    covered = np.zeros((size, size), dtype=bool)
    covered[top:top + height, round(0.02 * size):round(0.02 * size) + width] = True
    covered[top:top + height, round(0.52 * size):round(0.52 * size) + width] = True
    _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    return texture[iy, ix]


def mirror_part(mesh: Mesh, corner_uv: np.ndarray | None, mid_x: float = 0.0, *, anchor: Mesh | None = None
                ) -> tuple[Mesh, np.ndarray | None]:
    """The part on the other side of the character (x -> 2 mid_x - x), with the same texture.

    A generated body is not exactly symmetric: Pip's right sleeve ends 16 mm further
    back than the left one, and the mirror image of the left hand floated in front of
    it. With ``anchor`` (the body), the mirrored part moves front or back (along y) to
    where the body is at its place: the median depth of the body's vertices inside the
    part's extent in x and z, as ``place_part`` sets a depth that no view measures. Only
    a translation: the part keeps its shape and its texture.
    """
    vertices = mesh.vertices.copy()
    vertices[:, 0] = 2 * mid_x - vertices[:, 0]
    if anchor is not None:
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        near = np.ones(len(anchor.vertices), bool)
        for index in (0, 2):
            near &= (anchor.vertices[:, index] >= low[index]) & (anchor.vertices[:, index] <= high[index])
        if near.any():
            vertices[:, 1] += float(np.median(anchor.vertices[near, 1])) - (low[1] + high[1]) / 2
    faces = mesh.faces[:, [0, 2, 1]]                   # keep the outside out
    return Mesh(vertices, faces, mesh.colours), (corner_uv[:, [0, 2, 1]] if corner_uv is not None else None)


def join_parts(parts: list[tuple[Mesh, np.ndarray, np.ndarray]], *, size: int = 4096, min_share: float = 0.15,
               share: list[int] | None = None) -> tuple[Mesh, np.ndarray, np.ndarray, list[dict]]:
    """One mesh, its UVs per face corner, one texture, and where each part went.

    Each part is (mesh, UVs per face corner, texture). The parts get columns of the new
    texture, as wide as their share of the surface (at least ``min_share``). The meshes
    are not welded to each other: a part stays a separate piece of surface.

    ``share`` gives a group number for each part. Parts of one group use one texture (that
    of the first) and one column: a part and its mirror image (``mirror_part``), or a
    piece that was cut out of a body and keeps the body's UVs. The texture is then stored
    once, and the other parts keep more of the resolution.
    """
    if not parts:
        raise ValueError("join_parts needs at least one part")
    if share is None:
        share = list(range(len(parts)))
    if len(share) != len(parts):
        raise ValueError("share needs one group number for each part")
    groups = list(dict.fromkeys(share))                 # in the order of their first part
    column_of = np.array([groups.index(value) for value in share])
    area = np.bincount(column_of, weights=[part[0].area() for part in parts], minlength=len(groups))
    width = np.maximum(area / area.sum(), min_share if len(groups) > 1 else 1.0)
    width /= width.sum()
    edges = np.round(np.concatenate([[0], np.cumsum(width)]) * size).astype(int)
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    vertices, faces, corner_uv, layout = [], [], [], []
    start, filled = 0, set()
    for index, (mesh, uv, image) in enumerate(parts):
        column = int(column_of[index])
        left, right = int(edges[column]), int(edges[column + 1])
        if column not in filled:
            tile = Image.fromarray(np.asarray(image)[..., :3]).resize((right - left, size), Image.LANCZOS)
            texture[:, left:right] = np.asarray(tile)
            filled.add(column)
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
