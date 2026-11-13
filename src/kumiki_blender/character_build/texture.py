"""Bake a UV texture for a mesh from the full-resolution reference views.

Each texel of the mesh's UV layout gets a 3D point and normal. For every view
that sees the point (depth-buffer test against the mesh itself) the raw
reference image is sampled at full resolution. Each texel takes its colour from
one view (see ``bake_texture``), so two views never blend into ghost features.
This keeps the drawings' crisp clothes, buttons and fur painting, which
per-vertex colours blur.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .mesh import Mesh, rasterize, surface_samples
from .views import CX, DIAGONAL_VIEWS, camera


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


def bake_texture(handoff: Path, mesh: Mesh, corner_uv: np.ndarray, *, size: int = 2048, power: float = 1.0,
                 diagonal_weight: float = 1.0, depth_scale: float = 1.0, depth_tolerance_px: float = 1.5,
                 edge_px: float = 12.0, views: list[str] | None = None, min_cosine: float = 0.35,
                 vote_samples: int = 150_000, vote_neighbours: int = 24, vote_rounds: int = 2,
                 agree_delta_e: float = 15.0, mirror_missing: bool = False, mirror_weight: float = 0.3,
                 extra: list[dict] | None = None) -> tuple[np.ndarray, dict]:
    """RGB texture (size x size, uint8) and a report with the coverage of each view.

    Each texel takes its colour from one view, not a blend: AI views of one character
    disagree in places, and a blend shows both (ghost eyes). The view with the highest
    score (cos^power, 3/4 views scaled by ``diagonal_weight``) wins. Then a weighted majority
    vote among the ``vote_neighbours`` nearest texels on the surface flips thin minority patches,
    such as a side view winning a strip of a curved tail where the side drawing shows a
    different part. ``min_cosine`` drops grazing views, which sample a drawing's rim-lit
    outline. Texels no view sees take the colour of the nearest seen point on the surface.

    Before the choice, each view's score is scaled by exp(-d / ``agree_delta_e``), with d the
    mean colour difference to the other views that see the same point. A drawing that shows
    a different part there (the views disagree about the 3D shape) then rarely wins.

    ``mirror_missing`` adds each missing side view (right, front_right, back_right or the
    left ones) as the mirror image of its opposite. Use it only for a symmetric character:
    a one-sided accessory then appears on both sides. Mirrored views score ``mirror_weight``
    times a real view, so they paint mainly where no real drawing sees the surface.

    ``extra`` adds images that are not drawings, such as views repainted from renders of
    this mesh (``repaint.py``): dicts with ``name`` (a camera name, e.g. az30_el40),
    ``image`` (RGBA path; alpha marks the figure), ``map`` (the camera-frame to image
    mapping, same keys as ``spec["raw"]``) and ``weight`` (score factor).
    """
    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    px = spec["metres_per_aligned_px"]
    rows, cols, points, normals = texel_samples(mesh, corner_uv, size)
    samples = surface_samples(mesh, 0.5 * px / depth_scale)
    # One source per painting camera: (camera name, image path, camera-frame -> image map,
    # mirrored about the frame centre, score factor).
    sources = [(n, handoff / "raw" / f"{n}.png", spec["raw"][n], False, 1.0) for n in spec["raw"]
               if (views is None or n in views) and (handoff / "raw" / f"{n}.png").is_file()]
    if mirror_missing:
        have = {source[0] for source in sources}
        for a, b in (("left", "right"), ("front_left", "front_right"), ("back_left", "back_right")):
            for real, missing in ((a, b), (b, a)):
                if real in have and missing not in have:
                    sources.append((missing, handoff / "raw" / f"{real}.png", spec["raw"][real], True, mirror_weight))
    drawn = len(sources)
    for item in extra or []:
        sources.append((item["name"], Path(item["image"]), item["map"], False, float(item["weight"])))
    if not sources:
        raise ValueError("No reference view to bake from")
    names = [source[0] for source in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"Two sources paint from the same camera: {sorted(names)}")
    count = len(points)
    score = np.full((len(names), count), -1.0, dtype=np.float32)  # -1: the view does not paint the texel
    colours = np.zeros((len(names), count, 3), dtype=np.uint8)
    edge_trust = np.zeros((len(names), count), dtype=np.float32)
    coverage = {}
    for index, (name, image_path, raw_map, mirrored, factor) in enumerate(sources):
        rgba = np.asarray(Image.open(image_path).convert("RGBA")).astype(np.float64)
        # Wispy fur at a silhouette edge is blended with the background colour, inside the cut-out
        # mask. Trust a pixel more the further it is from the edge: full trust edge_px inside.
        inner = ndimage.distance_transform_edt(rgba[..., 3] > 200)
        core, trust = inner >= 3, np.clip(inner / edge_px, 0, 1)
        cam = camera(name)
        _, depth = rasterize(mesh, cam, px, depth_scale, samples)
        col, row, point_depth = cam.project(points, px, depth_scale)
        c, r = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = (c >= 0) & (c < depth.shape[1]) & (r >= 0) & (r < depth.shape[0])
        nearest = np.full(count, np.nan)
        nearest[inside] = depth[r[inside], c[inside]]
        cosine = np.clip(normals @ cam.toward, 0, 1)
        slope = np.minimum(np.sqrt(np.maximum(1 - cosine ** 2, 0)) / np.maximum(cosine, 1e-3), 10)
        visible = inside & (point_depth >= nearest - (depth_tolerance_px + 1.5 * slope) * px / depth_scale)
        u, v = col / depth_scale, row / depth_scale  # camera-frame aligned pixels
        if mirrored:  # the opposite camera sees the mirror image about the frame centre
            u = 2 * CX - u
        raw_x = (u + raw_map["shift_px"] - raw_map["x0"]) / raw_map["scale"] + raw_map["left"]
        raw_y = (v - raw_map["y0"]) / raw_map["scale"] + raw_map["top"]
        h, w = core.shape
        rx, ry = np.floor(raw_x).astype(int), np.floor(raw_y).astype(int)
        use = visible & (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h) & (cosine > min_cosine)
        use[use] = core[ry[use], rx[use]]
        score[index, use] = (cosine[use] ** power * factor
                             * (diagonal_weight if name in DIAGONAL_VIEWS else 1.0))
        edge_trust[index, use] = trust[ry[use], rx[use]]
        colours[index, use] = np.clip(np.round(_bilinear(rgba[..., :3], raw_x[use], raw_y[use])), 0, 255)
        coverage[name] = round(float(use.mean()), 4)
    every = np.arange(count)
    if agree_delta_e and len(names) > 1:
        # Agreement: a view whose colour differs from what the drawings show at the same point
        # (a side drawing showing body fur where the back drawings show tail) loses score. Only
        # drawings are witnesses: extra images repainted from this mesh share one source, so three
        # of them would outvote a drawing (Pip: right view delta E 4.5 -> 7.9).
        from .checks import lab

        lab_colours = np.stack([lab(colours[i]) for i in range(len(names))]).astype(np.float32)  # (V, N, 3)
        witness = np.array([index < drawn for index in range(len(names))], dtype=np.float32)
        weights = np.clip(score, 0, None) * witness[:, None]
        disagreement = np.zeros_like(score)
        for i in range(len(names)):
            others = weights.copy()
            others[i] = 0
            distance = np.linalg.norm(lab_colours - lab_colours[i][None], axis=2)  # (V, N)
            total = others.sum(axis=0)
            disagreement[i] = np.where(total > 0, (distance * others).sum(axis=0) / np.maximum(total, 1e-9), 0)
        score = np.where(score > 0, score * np.exp(-disagreement / agree_delta_e), score)
    label = np.argmax(score, axis=0)
    # Drawings are rim-lit: fur at a silhouette edge is painted near white. A texel whose best
    # view samples an edge pixel counts as unseen.
    seen = (score[label, every] > 0) & (edge_trust[label, every] >= 0.5)
    if not seen.any():
        raise ValueError("No view sees the mesh: check that the mesh stands in the handoff frame")
    flipped = 0
    if vote_rounds and len(names) > 1:
        from scipy.spatial import cKDTree

        # Vote on a subsample (the decision is coarse anyway), then each texel takes the label of
        # its nearest subsample point, if that view paints the texel.
        ids = np.nonzero(seen)[0]
        rng = np.random.default_rng(0)
        sub = np.sort(rng.choice(ids, size=min(len(ids), vote_samples), replace=False))
        tree = cKDTree(points[sub])
        _, near = tree.query(points[sub], k=min(vote_neighbours, len(sub)), workers=-1)
        sub_label = label[sub].copy()
        sub_score = score[sub_label, sub]
        allowed = (score[:, sub] > 0).T
        for _ in range(vote_rounds):
            votes = np.zeros((len(sub), len(names)))
            np.add.at(votes, (np.repeat(np.arange(len(sub)), near.shape[1]), sub_label[near].ravel()),
                      sub_score[near].ravel())
            votes[~allowed] = -1
            new = np.argmax(votes, axis=1)
            flipped = int((new != sub_label).sum())
            sub_label = new
            sub_score = score[sub_label, sub]
        _, nearest = tree.query(points[ids], workers=-1)
        proposal = sub_label[nearest]
        ok = score[proposal, ids] > 0
        label[ids[ok]] = proposal[ok]
    colour = np.zeros((count, 3))
    colour[seen] = colours[label[seen], every[seen]]
    if (~seen).any():
        from scipy.spatial import cKDTree

        _, nearest = cKDTree(points[seen]).query(points[~seen])
        colour[~seen] = colour[seen][nearest]
    image = np.zeros((size, size, 3))
    image[rows, cols] = colour
    covered = np.zeros((size, size), dtype=bool)
    covered[rows, cols] = True
    # The UV gutters between islands take the nearest covered texel (for filtering and mipmaps).
    _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    image = image[iy, ix]
    winners = np.bincount(label[seen], minlength=len(names)) / max(1, seen.sum())
    report = {"size": size, "texels": int(covered.sum()), "seen_fraction": round(float(seen.mean()), 4),
              "per_view": coverage, "owned": {n: round(float(w), 4) for n, w in zip(names, winners)},
              "mirrored_views": sorted(source[0] for source in sources if source[3]),
              "vote_flipped_last_round": flipped}
    return np.clip(np.round(image), 0, 255).astype(np.uint8), report


def _uv_samples(mesh: Mesh, corner_uv: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Surface points (spacing <= spacing_m) and their UVs, from a barycentric grid per triangle."""
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    longest = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                                 np.linalg.norm(a - c, axis=1)])
    steps = np.clip(np.ceil(longest / spacing_m).astype(int), 1, 64)
    points, uvs = [], []
    for n in np.unique(steps):
        ids = np.nonzero(steps == n)[0]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        wb, wc = i[keep] / n, j[keep] / n
        wa = 1 - wb - wc
        weights = np.stack([wa, wb, wc], axis=1)                       # (k, 3)
        points.append(np.einsum("kc,fcd->fkd", weights, mesh.vertices[mesh.faces[ids]]).reshape(-1, 3))
        uvs.append(np.einsum("kc,fcd->fkd", weights, corner_uv[ids]).reshape(-1, 2))
    return np.concatenate(points), np.concatenate(uvs)


def render_textured(mesh: Mesh, corner_uv: np.ndarray, texture: np.ndarray, view: str, metres_per_px: float,
                    scale: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic, unlit render of the textured mesh in the camera frame (scaled). RGB and mask."""
    from .views import CANVAS_H, CANVAS_W

    cam = camera(view)
    points, uvs = _uv_samples(mesh, corner_uv, 0.5 * metres_per_px / scale)
    col, row, depth = cam.project(points, metres_per_px, scale)
    height, width = int(CANVAS_H * scale), int(CANVAS_W * scale)
    c, r = np.floor(col).astype(int), np.floor(row).astype(int)
    inside = (c >= 0) & (c < width) & (r >= 0) & (r < height)
    flat = r[inside] * width + c[inside]
    order = np.argsort(depth[inside])            # far first; the nearest sample is written last and wins
    image = np.zeros((height * width, 3))
    size = texture.shape[0]
    tx = np.clip((uvs[inside, 0] * size).astype(int), 0, size - 1)
    ty = np.clip(((1 - uvs[inside, 1]) * size).astype(int), 0, size - 1)
    image[flat[order]] = texture[ty[order], tx[order], :3]
    mask = np.zeros(height * width, dtype=bool)
    mask[flat] = True
    return image.reshape(height, width, 3), mask.reshape(height, width)


def appearance_report(handoff: Path, mesh: Mesh, corner_uv: np.ndarray, texture: np.ndarray, *,
                      scale: float = 0.5, out: Path | None = None) -> dict:
    """Colour agreement between the textured mesh and each aligned reference view.

    For pixels inside both silhouettes (shrunk by 2 px against edge blending): median
    CIE76 delta E, and the fraction within delta E 10 (a just-visible difference for a
    stylized character). ``out`` gets side-by-side images: reference, render, difference.
    """
    from .checks import lab

    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    px = spec["metres_per_aligned_px"]
    rows = []
    for view in spec["views"]:
        path = handoff / "aligned" / f"{view}.png"
        if not path.is_file():
            continue
        rendered, model = render_textured(mesh, corner_uv, texture, view, px, scale)
        size = model.shape[::-1]
        ref = np.asarray(Image.open(path).convert("RGBA").resize(size, Image.BOX)).astype(float)
        both = ndimage.binary_erosion(model & (ref[..., 3] > 200), iterations=2)
        if not both.any():
            continue
        delta = np.linalg.norm(lab(ref[..., :3][both].astype(np.uint8)) - lab(rendered[both].astype(np.uint8)),
                               axis=1)
        rows.append({"view": view, "median_delta_e": round(float(np.median(delta)), 2),
                     "within_10": round(float((delta <= 10).mean()), 4)})
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            heat = np.zeros_like(rendered)
            heat[both] = np.clip(delta, 0, 40)[:, None] / 40 * np.array([255, 60, 60])
            panel = np.concatenate([np.where(model[..., None] | (ref[..., 3:] > 0), ref[..., :3], 60),
                                    np.where(model[..., None], rendered, 60), heat], axis=1)
            Image.fromarray(panel.astype(np.uint8)).save(out / f"{view}.png")
    return {"views": rows,
            "mean_median_delta_e": round(float(np.mean([r["median_delta_e"] for r in rows])), 2) if rows else None}
