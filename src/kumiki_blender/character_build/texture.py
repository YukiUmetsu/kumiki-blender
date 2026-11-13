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
from .views import CANVAS_H, CANVAS_W, CX, DIAGONAL_VIEWS, camera


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


TONE_SEAM_DELTA_E = 6.0   # a seam with a larger tone difference (before leveling) is reported
CONFLICT_COLOURS = {"misplaced_part": (230, 60, 60), "contested_feature": (240, 190, 40),
                    "tone_seam": (60, 120, 230), "unseen": (90, 90, 90)}
OWNER_COLOURS = np.array([(230, 60, 60), (60, 120, 230), (60, 190, 90), (240, 190, 40), (170, 80, 200),
                          (40, 200, 200), (240, 130, 40), (150, 150, 150), (120, 70, 30), (250, 130, 190)], dtype=np.uint8)


def _occlusion_distance(depth: np.ndarray, jump: float) -> np.ndarray:
    """Distance in pixels to the nearest depth discontinuity of a depth buffer (NaN is empty)."""
    filled = np.where(np.isnan(depth), -np.inf, depth)
    edge = np.zeros(depth.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        vertical = np.abs(filled[1:] - filled[:-1]) > jump
        horizontal = np.abs(filled[:, 1:] - filled[:, :-1]) > jump
    edge[1:] |= vertical
    edge[:-1] |= vertical
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    return ndimage.distance_transform_edt(~edge)


def bake_texture(handoff: Path, mesh: Mesh, corner_uv: np.ndarray, *, size: int = 2048, power: float = 4.0,
                 diagonal_weight: float = 1.0, depth_scale: float = 1.0, depth_tolerance_px: float = 1.5,
                 edge_px: float = 12.0, views: list[str] | None = None, min_cosine: float = 0.35,
                 region_px: float = 24.0, min_region_px: float = 40.0, occlusion_px: float = 30.0,
                 occlusion_jump_px: float = 15.0,
                 level: bool = True, level_reach_px: float = 150.0, seam_robust_delta_e: float = 18.0,
                 agree_delta_e: float = 15.0, slack_px: float = 60.0, contest_delta_e: float = 20.0,
                 mirror_missing: bool = False, mirror_weight: float = 0.3,
                 extra: list[dict] | None = None, diagnostics: Path | None = None) -> tuple[np.ndarray, dict]:
    """RGB texture (size x size, uint8) and a report with the coverage of each view.

    Each texel takes its colour from one view, not a blend: AI views of one character
    disagree in places, and a blend shows both (ghost eyes). Which view paints is decided
    per region, not per texel (``seams.py``): the view that sees the neighbourhood of a point
    best (radius ``region_px`` aligned pixels) paints it. Per texel, the better-facing view
    changes with every bump of a furry surface and the drawings mix in streaks. A texel
    that the view of its region does not see takes the nearest painted colour.

    The quality of a view at a texel is cos^``power`` (3/4 views scaled by ``diagonal_weight``)
    times two trust factors. ``edge_px``: drawings are rim-lit, so a pixel near the
    drawing's outline is trusted less. ``occlusion_px``: where one part hides another in
    this view (and at the mesh outline), the drawing's contour rarely matches the mesh
    exactly, so the scarf lands on the tail behind it; texels near such a depth edge are
    trusted less. ``min_cosine`` drops grazing views.

    A view also loses score where its colour (averaged over the neighbourhood) differs from
    what the other drawings show there by about ``agree_delta_e``: it then shows a different
    part (the views disagree about the 3D shape).

    ``level`` adds a smooth colour offset to each region so that the two sides of every
    seam meet; the offset fades over ``level_reach_px``. A difference above about twice
    ``seam_robust_delta_e`` is taken for two different parts and left alone. The drawings differ in tone (Pip's
    tail is light in the side view and dark in the back view), and without it each seam shows.

    ``mirror_missing`` adds each missing side view (right, front_right, back_right or the
    left ones) as the mirror image of its opposite. Use it only for a symmetric character:
    a one-sided accessory then appears on both sides. Mirrored views score ``mirror_weight``
    times a real view, so they paint mainly where no real drawing sees the surface.

    ``extra`` adds images that are not drawings, such as views repainted from renders of
    this mesh (``repaint.py``): dicts with ``name`` (a camera name, e.g. az30_el40),
    ``image`` (RGBA path; alpha marks the figure), ``map`` (the camera-frame to image
    mapping, same keys as ``spec["raw"]``) and ``weight`` (score factor).

    The report lists the conflicts between the sources under ``conflicts``: the share of
    the surface with a misplaced part, a contested feature, a tone seam and no view at
    all; each pair of sources with its overlap; and cameras that face the faults (for a
    new drawing or ``repaint``). ``diagnostics`` (a folder) also gets pictures:
    ``owners.png`` and ``conflicts.png`` in the UV layout, and ``turnaround.png`` with the
    texture, the owners and the conflicts from eight sides (colours in the report).
    """
    from .checks import lab
    from .seams import (LIGHTNESS_WEIGHT, NORMAL_WEIGHT, Testimony, build_regions, choose_sources, level_seams,
                        pair_table, per_texel, suggest_views)

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
    regions = build_regions(points, normals, sigma=region_px * px, spacing=region_px * px / 3)
    # The facing of the surface without its bumps: on modelled fur the true normal swings by
    # tens of degrees from texel to texel, and a view that faces the tail squarely would score low.
    facing = regions.normal[regions.cell]
    quality = np.zeros((len(names), count), dtype=np.float32)  # 0: the view does not paint the texel
    colours = np.zeros((len(names), count, 3), dtype=np.uint8)
    inner_enough = np.zeros((len(names), count), dtype=bool)
    coverage = {}
    testimony = []
    step = region_px / 3   # aligned px per cell of the coarse pictures that the views testify with
    grid_u, grid_v = np.meshgrid(np.arange(0, CANVAS_W, step) + step / 2, np.arange(0, CANVAS_H, step) + step / 2)
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
        true_cosine = np.clip(normals @ cam.toward, 0, 1)
        cosine = np.clip(facing @ cam.toward, 0, 1)
        slope = np.minimum(np.sqrt(np.maximum(1 - true_cosine ** 2, 0)) / np.maximum(true_cosine, 1e-3), 10)
        visible = inside & (point_depth >= nearest - (depth_tolerance_px + 1.5 * slope) * px / depth_scale)
        u, v = col / depth_scale, row / depth_scale  # camera-frame aligned pixels
        if mirrored:  # the opposite camera sees the mirror image about the frame centre
            u = 2 * CX - u
        raw_x = (u + raw_map["shift_px"] - raw_map["x0"]) / raw_map["scale"] + raw_map["left"]
        raw_y = (v - raw_map["y0"]) / raw_map["scale"] + raw_map["top"]
        h, w = core.shape
        rx, ry = np.floor(raw_x).astype(int), np.floor(raw_y).astype(int)
        use = visible & (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h) & (cosine > min_cosine) & (true_cosine > 0.1)
        use[use] = core[ry[use], rx[use]]
        clear = np.ones(int(use.sum()))
        if occlusion_px > 0:
            apart = _occlusion_distance(depth, occlusion_jump_px * px)[r[use], c[use]] / depth_scale
            clear = 0.1 + 0.9 * np.clip(apart / occlusion_px, 0, 1)
        quality[index, use] = (cosine[use] ** power * factor * trust[ry[use], rx[use]] * clear
                               * (diagonal_weight if name in DIAGONAL_VIEWS else 1.0))
        # Drawings are rim-lit: fur at a silhouette edge is painted near white. Such a pixel never paints.
        inner_enough[index, use] = trust[ry[use], rx[use]] >= 0.5
        colours[index, use] = np.clip(np.round(_bilinear(rgba[..., :3], raw_x[use], raw_y[use])), 0, 255)
        coverage[name] = round(float(use.mean()), 4)
        # Testimony: a low-pass picture of this source in its camera frame, and how well it sees each node.
        blur = region_px / raw_map["scale"]   # the same low-pass as the node colours
        alpha = ndimage.gaussian_filter((rgba[..., 3] > 200).astype(np.float64), blur)
        soft = np.dstack([ndimage.gaussian_filter(rgba[..., i] * (rgba[..., 3] > 200), blur) for i in range(3)])
        soft /= np.maximum(alpha, 1e-6)[..., None]
        gx = ((2 * CX - grid_u if mirrored else grid_u) + raw_map["shift_px"] - raw_map["x0"]) / raw_map["scale"] + raw_map["left"]
        gy = (grid_v - raw_map["y0"]) / raw_map["scale"] + raw_map["top"]
        in_image = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        at_x, at_y = np.clip(gx, 0, w - 1).astype(int), np.clip(gy, 0, h - 1).astype(int)
        picture = lab(np.clip(soft[at_y, at_x], 0, 255).astype(np.uint8).reshape(-1, 3)).reshape(grid_u.shape + (3,))
        picture[..., 0] *= LIGHTNESS_WEIGHT
        node_col, node_row, _ = cam.project(regions.centre, px, 1.0)
        can_see = regions.per_node(visible & (true_cosine > 0.05)) / regions.texels
        # A view is no witness of a surface it sees at a grazing angle, and its rim-lit outline shows nothing.
        square = np.clip((regions.normal @ cam.toward - 0.1) / 0.3, 0, 1)
        testimony.append(Testimony(picture, in_image & (alpha[at_y, at_x] > 0.85),
                                   np.floor(node_col / step).astype(int), np.floor(node_row / step).astype(int),
                                   can_see * square))
    if not (quality > 0).any():
        raise ValueError("No view sees the mesh: check that the mesh stands in the handoff frame")
    regions.add_sources(quality, colours)
    # Only drawings are witnesses of agreement: extra images repainted from this mesh share one
    # source, so three of them would outvote a drawing (Pip: right view delta E 4.5 -> 7.9).
    chosen = choose_sources(regions, np.arange(len(names)) < drawn, testimony, agree_delta_e=agree_delta_e,
                            slack_cells=max(1, round(slack_px / step)), contest_delta_e=contest_delta_e,
                            min_region_nodes=max(1, round((min_region_px / step) ** 2)))
    offset, seams = (level_seams(regions, reach=level_reach_px * px, robust_delta_e=seam_robust_delta_e)
                     if len(names) > 1 else (None, {}))
    if not level:   # the seams are still measured for the report
        offset = None
        seams.pop("seam_delta_e_after", None)
    label, shift = per_texel(regions, points, chosen["score"], offset)
    every = np.arange(count)
    held = np.clip(label, 0, None)
    seen = (label >= 0) & (quality[held, every] > 0) & inner_enough[held, every]
    if not seen.any():
        raise ValueError("No view sees the mesh: check that the mesh stands in the handoff frame")
    colour = np.zeros((count, 3))
    colour[seen] = np.clip(colours[label[seen], every[seen]] + shift[seen], 0, 255)
    if (~seen).any():
        from scipy.spatial import cKDTree

        # Nearest painted texel on the same side of the surface: position and facing together,
        # so the back of the neck is not filled from the tail that hangs 2 cm behind it.
        where = np.hstack([points / regions.sigma, facing * NORMAL_WEIGHT])
        _, nearest = cKDTree(where[seen]).query(where[~seen], workers=-1)
        colour[~seen] = colour[seen][nearest]
    image = np.zeros((size, size, 3))
    image[rows, cols] = colour
    covered = np.zeros((size, size), dtype=bool)
    covered[rows, cols] = True
    # The UV gutters between islands take the nearest covered texel (for filtering and mipmaps).
    _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    image = image[iy, ix]
    # Conflicts, per node and as shares of the surface.
    node_seen = regions.per_node(seen) / regions.texels
    tone = seams.pop("node_jump", np.zeros(len(regions.texels))) > TONE_SEAM_DELTA_E
    kinds = {"misplaced_part": chosen["misplaced"], "contested_feature": chosen["contested"] & ~chosen["misplaced"],
             "unseen": node_seen < 0.5}
    kinds["tone_seam"] = tone & ~kinds["misplaced_part"] & ~kinds["contested_feature"] & ~kinds["unseen"]
    area = regions.texels / regions.texels.sum()
    conflicts = {"surface": {kind: round(float(area[mask].sum()), 4) for kind, mask in kinds.items()},
                 "colours": {kind: list(CONFLICT_COLOURS[kind]) for kind in kinds},
                 "pairs": pair_table(regions, names, contest_delta_e),
                 "views_that_face": {kind: suggest_views(regions, kinds[kind])
                                     for kind in ("contested_feature", "misplaced_part", "unseen")}}
    if diagnostics is not None:
        diagnostics.mkdir(parents=True, exist_ok=True)

        def in_layout(values: np.ndarray) -> np.ndarray:
            picture = np.zeros((size, size, 3), dtype=np.uint8)
            picture[rows, cols] = values
            return picture[iy, ix]

        owners = np.zeros((count, 3), dtype=np.uint8)
        owners[seen] = OWNER_COLOURS[label[seen] % len(OWNER_COLOURS)]
        fault = np.full((len(regions.texels), 3), 235, dtype=np.uint8)
        for kind in ("tone_seam", "unseen", "contested_feature", "misplaced_part"):   # the last one shows on top
            fault[kinds[kind]] = CONFLICT_COLOURS[kind]
        pictures = [np.clip(np.round(image), 0, 255).astype(np.uint8), in_layout(owners), in_layout(fault[regions.cell])]
        Image.fromarray(pictures[1]).save(diagnostics / "owners.png")
        Image.fromarray(pictures[2]).save(diagnostics / "conflicts.png")
        turnaround(mesh, corner_uv, pictures, px).save(diagnostics / "turnaround.png")
    winners = np.bincount(label[seen], minlength=len(names)) / max(1, seen.sum())
    report = {"size": size, "texels": int(covered.sum()), "seen_fraction": round(float(seen.mean()), 4),
              "per_view": coverage, "owned": {n: round(float(w), 4) for n, w in zip(names, winners)},
              "mirrored_views": sorted(source[0] for source in sources if source[3]),
              "owner_colours": {n: [int(c) for c in OWNER_COLOURS[i % len(OWNER_COLOURS)]] for i, n in enumerate(names)},
              **{k: chosen[k] for k in ("regions", "rejected_claims", "contested_nodes")}, **seams,
              "conflicts": conflicts}
    return np.clip(np.round(image), 0, 255).astype(np.uint8), report


def conflict_advice(report: dict, *, notable: float = 0.005) -> list[str]:
    """What to do about each kind of conflict in a ``bake_texture`` report, in plain sentences."""
    found = report["conflicts"]
    surface, faces = found["surface"], found["views_that_face"]

    def cameras(kind: str) -> str:
        return ", ".join(item["camera"] for item in faces[kind][:3])

    advice = []
    if surface["misplaced_part"] > notable:
        advice.append(f"Misplaced part on {surface['misplaced_part']:.1%} of the surface (red): a drawing shows a part "
                      "where the mesh has a different surface, so the mesh is wider or narrower there than the "
                      "drawing. The bake does not paint that part and uses the colour around it. To remove the "
                      f"cause, correct the shape. Look from: {cameras('misplaced_part')}.")
    if surface["contested_feature"] > notable:
        advice.append(f"Contested feature on {surface['contested_feature']:.1%} of the surface (yellow): two drawings "
                      "show different things at the same place. One drawing paints each patch whole. If a feature "
                      "still shows twice, add a picture that faces it (a drawing, or `repaint --views`): "
                      f"{cameras('contested_feature')}.")
    if surface["tone_seam"] > notable and "seam_delta_e_after" in report:
        advice.append(f"Tone seams on {surface['tone_seam']:.1%} of the surface (blue): the drawings differ in tone "
                      f"where they meet, by {report['seam_delta_e_before']} delta E on average. Leveling reduced "
                      f"that to {report['seam_delta_e_after']}.")
    elif surface["tone_seam"] > notable:
        advice.append(f"Tone seams on {surface['tone_seam']:.1%} of the surface (blue), "
                      f"{report['seam_delta_e_before']} delta E on average. Leveling is off: remove --no-level.")
    if surface["unseen"] > notable:
        advice.append(f"No view sees {surface['unseen']:.1%} of the surface (grey); it takes the nearest painted "
                      f"colour. For real detail, add a picture from: {cameras('unseen')}.")
    worst = [pair for pair in found["pairs"] if pair["different"] > 0.25 and pair["overlap"] > 0.01]
    for pair in worst[:3]:
        advice.append(f"{pair['views'][0]} and {pair['views'][1]} show different things on {pair['different']:.0%} of "
                      "the surface that both see. Check that the two drawings show the same design.")
    return advice or ["No notable conflict between the views."]


def turnaround(mesh: Mesh, corner_uv: np.ndarray, textures: list[np.ndarray], metres_per_px: float, *,
               azimuths: tuple[float, ...] = (0, 45, 90, 135, 180, -135, -90, -45), elevation: float = 0.0,
               tile: int = 360) -> Image.Image:
    """One row per texture, one column per camera: the mesh from all sides, unlit, on grey."""
    from .views import CANVAS_H, CANVAS_W

    scale = tile / CANVAS_H
    width = int(CANVAS_W * scale * 0.6)   # the figure stands in the middle of the frame
    sheet = Image.new("RGB", (width * len(azimuths), tile * len(textures)), (225, 225, 225))
    for row, texture in enumerate(textures):
        for column, azimuth in enumerate(azimuths):
            picture, mask = render_textured(mesh, corner_uv, texture, f"az{azimuth:g}_el{elevation:g}", metres_per_px, scale)
            frame = np.full(picture.shape, 225, dtype=np.uint8)
            frame[mask] = np.clip(picture[mask], 0, 255)
            left = (frame.shape[1] - width) // 2
            sheet.paste(Image.fromarray(frame[:, left:left + width]), (column * width, row * tile))
    return sheet


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
                     "mean_delta_e": round(float(delta.mean()), 2),
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
