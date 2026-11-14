"""Make the eyes of a character separate eyeballs, and take the painted eyes off the head.

Two drawings never put an eye at the same place on the head: Pip's left eye sat 60
pixels further round the head in the side drawing than in the front drawing. A texture
from both shows two eyelid lines, and no rule on the drawings removes that. An eyeball
that is its own piece of surface, placed from one drawing, cannot be painted twice.

``split_eyes`` finds the iris and pupil of each eye in the drawing that looks at the face
(``find_eyes``), places one lens for each on the head mesh (``eye_globes``), and writes two
handoffs in the frame of the whole character, like ``split_views``:

- ``<out>/eyes/``: the drawing's own eye pixels. Bake each eyeball from it with
  ``texture``: the iris, the pupil and the highlights stay as they are drawn.
- ``<out>/rest/``: the face drawing without its iris discs, and every other drawing
  without the surroundings of the eyes (the white and the lids), so that only the face
  drawing paints there.

It also writes ``eyes/<id>.obj`` (the eyeballs, with UVs) and ``anatomy.json`` with the
same eyes as ``ocular_globe`` parts for ``plan-anatomy`` / ``build-anatomy``. The eyes
are found by colour (a dark pupil, a saturated iris around it): review ``eyes.png``
before use. Closed eyes and eyes without a dark pupil are not found.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .checks import lab
from .mesh import Mesh, rasterize
from .part_place import write_obj_uv
from .split_views import write_handoff
from .views import BASELINE, CX, camera

PUPIL_L = 24.0   # CIE L* below which a pixel is pupil-dark


def _disc(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return xx * xx + yy * yy <= radius * radius


def find_eyes(rgba: np.ndarray, *, head_share: float = 0.55, max_eyes: int = 2, min_pupil_share: float = 0.0008
              ) -> tuple[list[dict], np.ndarray]:
    """The eyes of a drawing (RGBA, alpha marks the figure), and the mask of their iris discs.

    1. A pupil is a thick, very dark blob in the upper ``head_share`` of the figure (lash
       and lid lines are thin and drop out), at least ``min_pupil_share`` of the figure.
    2. The iris is what surrounds the pupil in one colour: the most saturated colour
       next to the pupil, if it is clearly more saturated than the fur further out. An
       eye without a coloured iris counts its pupil as iris.

    The white of the eye is not found: it has the colour of light fur, and a lid line
    does not always close it (tried on Pip: the region leaked between the eyes). Each
    eye has ``centre_px`` and ``radii_px`` of its iris disc, ``pupil_px``,
    ``pupil_radius_px``, ``iris_radius_px`` and ``iris_lab``.
    """
    figure = rgba[..., 3] > 200
    rows = np.nonzero(figure.any(axis=1))[0]
    if not len(rows):
        return [], np.zeros(figure.shape, bool)
    height = rows[-1] - rows[0] + 1
    colour = lab(rgba[..., :3].reshape(-1, 3)).reshape(figure.shape + (3,))
    light = colour[..., 0]
    head = figure.copy()
    head[int(rows[0] + head_share * height):] = False
    dark = ndimage.binary_opening(head & (light < PUPIL_L), structure=_disc(max(3, round(0.008 * height))))
    labels, _ = ndimage.label(dark)
    found = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        blob = labels[box] == index
        area, (tall, wide) = int(blob.sum()), blob.shape
        if area < min_pupil_share * figure.sum() or area < 0.5 * tall * wide or not 0.4 < tall / wide < 2.5:
            continue                                   # too small, or not a disc: a brow, a nostril, a shadow
        cy, cx = ndimage.center_of_mass(blob)
        found.append((area, cx + box[1].start, cy + box[0].start))
    eyes, mask = [], np.zeros(figure.shape, bool)
    yy, xx = np.mgrid[:figure.shape[0], :figure.shape[1]]
    for area, cx, cy in sorted(found, reverse=True)[:max_eyes]:
        pupil = float(np.sqrt(area / np.pi))
        distance = np.hypot(xx - cx, yy - cy)
        # The iris can be a thin band. Its colour is the most saturated one next to the pupil,
        # if that is clearly more saturated than the fur further out.
        chroma = np.hypot(colour[..., 1], colour[..., 2])
        ring = figure & (distance > pupil) & (distance < 2.0 * pupil) & (light > PUPIL_L)
        around = figure & (distance > 2.5 * pupil) & (distance < 3.5 * pupil)
        iris_colour, own, iris_area = colour[int(cy), int(cx)], 0, 0
        if ring.sum() > 50:
            vivid = ring & (chroma >= np.percentile(chroma[ring], 80))
            iris_colour = np.median(colour[vivid], axis=0)
            if not around.any() or np.hypot(*iris_colour[1:]) > 1.5 * np.median(chroma[around]) + 5:
                like = np.linalg.norm((colour - iris_colour) * (0.4, 1.0, 1.0), axis=-1) < 24
                disc = ndimage.binary_closing(figure & (distance < 3.5 * pupil) & (like | (light < PUPIL_L)), iterations=3)
                disc = ndimage.binary_opening(disc, structure=_disc(max(2, round(0.25 * pupil))))
                grown, _ = ndimage.label(disc)
                own = grown[int(round(cy)), int(round(cx))]
                iris_area = int(ndimage.binary_fill_holes(grown == own).sum()) if own else 0
        if own and area < iris_area < 12 * area:
            iy, ix = ndimage.center_of_mass(ndimage.binary_fill_holes(grown == own))
            iris = float(np.sqrt(iris_area / np.pi))
        else:                                          # no coloured iris: a bead eye
            ix, iy, iris = cx, cy, pupil
        grow = 1.06 * iris                             # with the dark ring around the iris
        region = figure & (np.hypot(xx - ix, yy - iy) <= grow)
        eyes.append({"centre_px": [float(ix), float(iy)], "radii_px": [grow, grow],
                     "pupil_px": [float(cx), float(cy)], "pupil_radius_px": pupil, "iris_radius_px": iris,
                     "iris_lab": [float(v) for v in iris_colour]})
        mask |= region
    return sorted(eyes, key=lambda eye: eye["centre_px"][0]), mask


def uv_ellipsoid(centre: np.ndarray, radii: np.ndarray, *, segments: int = 48, rings: int = 24,
                 seam_toward: np.ndarray = (0.0, 1.0, 0.0)) -> tuple[Mesh, np.ndarray]:
    """An ellipsoid with latitude/longitude UVs; the poles are on Z and the UV seam faces ``seam_toward``."""
    start = np.arctan2(seam_toward[1], seam_toward[0])
    lon = start + np.linspace(0, 2 * np.pi, segments + 1)
    lat = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    unit = np.stack([np.cos(grid_lat) * np.cos(grid_lon), np.cos(grid_lat) * np.sin(grid_lon), np.sin(grid_lat)], axis=-1)
    vertices = (unit * radii + centre).reshape(-1, 3)
    uv = np.stack([(grid_lon - start) / (2 * np.pi), (grid_lat + np.pi / 2) / np.pi], axis=-1).reshape(-1, 2)
    faces = []
    for r in range(rings):
        for s in range(segments):
            a, b = r * (segments + 1) + s, r * (segments + 1) + s + 1
            c, d = a + segments + 1, b + segments + 1
            if r > 0:
                faces.append((a, b, d))
            if r < rings - 1:
                faces.append((a, d, c))
    faces = np.array(faces)
    return Mesh(vertices, faces), uv[faces]


def eye_globes(eyes: list[dict], raw_map: dict, head: Mesh, metres_per_px: float, *, view: str = "front",
               depth_ratio: float = 0.6, protrude: float = 0.5, grow: float = 1.03) -> list[dict]:
    """Each eye as an ellipsoid on the head: ``id``, ``center_m``, ``radii_m`` (x, y, z) and ``forward_m``.

    The centre and the two radii across the view come from the drawing. Along the view,
    the radius is ``depth_ratio`` of their mean, and the eyeball sits in the head so that
    ``protrude`` of that radius stands out of the surface at the eye's centre. ``grow``
    makes the eyeball a little larger than drawn, so its rim is inside the head.
    """
    look = camera(view)
    _, depth = rasterize(head, look, metres_per_px, 1.0)
    midline = float(np.median(head.vertices[:, 0]))
    globes, counts = [], {}
    for eye in eyes:
        u = (eye["centre_px"][0] - raw_map["left"]) * raw_map["scale"] + raw_map["x0"] - raw_map.get("shift_px", 0)
        v = (eye["centre_px"][1] - raw_map["top"]) * raw_map["scale"] + raw_map["y0"]
        across = np.array(eye["radii_px"]) * raw_map["scale"] * metres_per_px * grow
        window = depth[max(0, int(v) - 3):int(v) + 4, max(0, int(u) - 3):int(u) + 4]
        if not np.isfinite(window).any():
            raise ValueError("An eye of the drawing lies outside the head mesh in this view")
        deep = depth_ratio * float(across.mean())
        surface = float(np.nanmedian(window))
        centre = look.right * (u - CX) * metres_per_px + np.array([0, 0, (BASELINE - v) * metres_per_px]) \
            + look.toward * (surface - deep * (1 - protrude))
        radii = np.abs(look.right) * across[0] + np.abs(look.toward) * deep + np.array([0, 0, across[1]])
        side = "L" if centre[0] > midline else "R"
        counts[side] = counts.get(side, 0) + 1
        globes.append({"id": f"eye.{side}" + (f".{counts[side]}" if counts[side] > 1 else ""),
                       "center_m": [float(x) for x in centre], "radii_m": [float(x) for x in radii],
                       "forward_m": [float(x) for x in look.toward], "iris_lab": eye["iris_lab"],
                       "iris_fraction": round(float(np.clip(eye["iris_radius_px"] / max(eye["radii_px"]), 0.05, 0.94)), 3),
                       "pupil_fraction": round(float(np.clip(0.9 * eye["pupil_radius_px"] / max(eye["radii_px"]), 0.025, 0.8)), 3)})
    return globes


def eye_zone_in_view(shape: tuple[int, int], raw_map: dict, view: str, zones: list[tuple[np.ndarray, float]],
                     head: Mesh, metres_per_px: float, *, face_view: str = "front", margin: float = 0.6) -> np.ndarray:
    """Pixels of another drawing that show the surroundings of an eye (the socket with its lid lines).

    ``zones`` are (centre in the aligned frame of ``face_view``, radius in aligned pixels).
    The head surface that ``face_view`` sees inside a zone is projected into ``view``; what
    this view sees of it, grown by ``margin`` of the radius, is the result. The margin
    allows for the distance at which two drawings put the same eye.
    """
    from .mesh import surface_samples

    face, look = camera(face_view), camera(view)
    samples = surface_samples(head, metres_per_px)[0]
    _, face_depth = rasterize(head, face, metres_per_px, 1.0, (samples, None))
    _, depth = rasterize(head, look, metres_per_px, 1.0, (samples, None))
    col, row, reach = face.project(samples, metres_per_px, 1.0)
    c, r = np.clip(col.astype(int), 0, face_depth.shape[1] - 1), np.clip(row.astype(int), 0, face_depth.shape[0] - 1)
    front_side = reach >= face_depth[r, c] - 6 * metres_per_px
    found = np.zeros(shape, bool)
    for centre, radius in zones:
        chosen = samples[front_side & (np.hypot(col - centre[0], row - centre[1]) <= radius)]
        if not len(chosen):
            continue
        u, v, near = look.project(chosen, metres_per_px, 1.0)
        cu, cv = np.clip(u.astype(int), 0, depth.shape[1] - 1), np.clip(v.astype(int), 0, depth.shape[0] - 1)
        seen = near >= depth[cv, cu] - 4 * metres_per_px
        x = ((u[seen] + raw_map.get("shift_px", 0) - raw_map["x0"]) / raw_map["scale"] + raw_map["left"]).astype(int)
        y = ((v[seen] - raw_map["y0"]) / raw_map["scale"] + raw_map["top"]).astype(int)
        inside = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
        zone = np.zeros(shape, bool)
        zone[y[inside], x[inside]] = True
        if zone.any():
            grow = max(1, round(margin * radius / raw_map["scale"]))
            reach = ndimage.distance_transform_edt(~ndimage.binary_closing(zone, iterations=3))
            found |= reach <= grow
    return found


def _write_eyeballs(globes: list[dict], out: Path, folder: str, view: str, note: str) -> None:
    """One OBJ with UVs for each eyeball in ``out/folder``, and ``out/anatomy.json`` for ``plan-anatomy``."""
    (out / folder).mkdir(parents=True, exist_ok=True)
    for globe in globes:
        mesh, corner_uv = uv_ellipsoid(np.array(globe["center_m"]), np.array(globe["radii_m"]),
                                       seam_toward=-np.array(globe["forward_m"]))
        write_obj_uv(mesh, corner_uv, out / folder / f"{globe['id']}.obj", comment=f"eyeball {globe['id']}")
        globe["mesh"] = f"{globe['id']}.obj" if folder == "." else f"{folder}/{globe['id']}.obj"
    anatomy = {"schema_version": 1, "character_id": "from-split-eyes", "parts": [
        {"id": globe["id"], "kind": "ocular_globe",
         "evidence": {"status": "inferred", "source_views": [view], "notes": note + "; review before use."},
         "center_m": globe["center_m"], "forward_m": globe["forward_m"], "up_m": [0, 0, 1], "radii_m": globe["radii_m"],
         "iris_fraction": globe["iris_fraction"], "pupil_fraction": globe["pupil_fraction"]} for globe in globes]}
    (out / "anatomy.json").write_text(json.dumps(anatomy, indent=2), encoding="utf-8")


def bake_eyeball(part_handoff: Path, mesh: Mesh, corner_uv: np.ndarray, *, view: str = "front", size: int = 1024,
                 white_percentile: float = 80.0) -> tuple[np.ndarray, dict]:
    """The texture of one eyeball from the eye pixels of ``view``; what the view cannot see is eye-white.

    ``texture`` fills a texel that no view sees with the nearest painted colour. On an
    eyeball that is the colour of its rim in the drawing: a dark lid line, or a strip of
    cheek fur. From the side, Pip's eyeball then was a tan disc. Here the unseen texels
    take the white of this eye: the median of its lightest, least coloured painted texels.
    """
    import tempfile

    from .texture import bake_texture

    with tempfile.TemporaryDirectory() as scratch:
        image, report = bake_texture(part_handoff, mesh, corner_uv, size=size, views=[view], min_cosine=0.15,
                                     occlusion_px=0, diagnostics=Path(scratch))
        with Image.open(Path(scratch) / "owners.png") as picture:
            unseen = np.asarray(picture.convert("RGB")).max(axis=2) < 30
    painted = lab(image[~unseen].reshape(-1, 3))
    chroma = np.hypot(painted[:, 1], painted[:, 2])
    lightest = painted[:, 0] >= np.percentile(painted[:, 0], white_percentile)
    lightest &= chroma <= np.median(chroma[lightest])      # the white, not light fur or skin at the rim
    white = np.median(image[~unseen].reshape(-1, 3)[lightest], axis=0)
    blend = ndimage.gaussian_filter(unseen.astype(np.float64), max(1.0, size / 256))[..., None]
    blend = np.where(unseen[..., None], 1.0, blend)
    result = np.clip(np.round(image * (1 - blend) + white * blend), 0, 255).astype(np.uint8)
    return result, {"seen_fraction": report["seen_fraction"], "white_rgb": [int(round(v)) for v in white]}


def eyeballs_from_part(part_handoff: Path, head: Mesh, out: Path, *, view: str = "front", min_share: float = 0.2,
                       depth_ratio: float = 0.7, protrude: float = 0.45, grow: float = 1.02,
                       source_handoff: Path | None = None, zone: float = 1.25) -> dict:
    """Eyeballs for the eyes that ``split-views`` took out of the drawings, the white of the eye included.

    ``find_eyes`` cannot find the white: it has the colour of light fur. ``split-views``
    with ``--part "two eyes, each with its iris, its pupil and the white"`` can: the
    image editor removes the eyes, and the part is where the drawing changed. Each piece
    of that part in ``view`` (at least ``min_share`` of the largest piece) becomes one
    ellipsoid on the head, as wide and as high as the piece, with its texture
    (``bake_eyeball``). Writes ``out/<id>.obj``, ``out/<id>.png``, ``out/anatomy.json``,
    ``out/eyes.json`` and ``out/eyes.png``.

    With ``source_handoff`` (the handoff that the eyes were split from), it also writes
    ``out/within/<name>.png`` for each other drawing: where an eye can show there, ``zone``
    times the eye's size around its place on the head. Give that folder to
    ``split-views --within`` for the other views, so that a change that the editor made
    elsewhere does not count as an eye.
    """
    spec = json.loads((part_handoff / "spec.json").read_text(encoding="utf-8"))
    with Image.open(part_handoff / "raw" / f"{view}.png") as picture:
        rgba = np.asarray(picture.convert("RGBA"))
    labels, count = ndimage.label(rgba[..., 3] > 200)
    if not count:
        raise ValueError(f"The part handoff shows nothing in the {view} view")
    sizes = ndimage.sum_labels(labels > 0, labels, range(1, count + 1))
    eyes = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if sizes[index - 1] < min_share * sizes.max():
            continue
        alone = rgba.copy()
        alone[..., 3] = np.where(labels == index, 255, 0)
        inner, _ = find_eyes(alone, head_share=1.0, max_eyes=1, min_pupil_share=0.01)
        high, wide = box[0].stop - box[0].start, box[1].stop - box[1].start
        pupil = inner[0]["pupil_radius_px"] if inner else 0.3 * min(high, wide) / 2
        eyes.append({"centre_px": [(box[1].start + box[1].stop) / 2, (box[0].start + box[0].stop) / 2],
                     "radii_px": [wide / 2, high / 2], "pupil_radius_px": float(pupil),
                     "iris_radius_px": float(inner[0]["iris_radius_px"] if inner else 1.5 * pupil),
                     "pupil_px": inner[0]["pupil_px"] if inner else [(box[1].start + box[1].stop) / 2, (box[0].start + box[0].stop) / 2],
                     "iris_lab": inner[0]["iris_lab"] if inner else [0.0, 0.0, 0.0]})
    eyes.sort(key=lambda eye: eye["centre_px"][0])
    globes = eye_globes(eyes, spec["raw"][view], head, spec["metres_per_aligned_px"], view=view,
                        depth_ratio=depth_ratio, protrude=protrude, grow=grow)
    out.mkdir(parents=True, exist_ok=True)
    _write_eyeballs(globes, out, ".", view, "The eye region is from split-views (an image edit); placed on the mesh by split-eyes")
    from .texture import read_obj_uv

    for globe in globes:
        mesh, corner_uv = read_obj_uv(out / globe["mesh"])
        texture, baked = bake_eyeball(part_handoff, mesh, corner_uv, view=view)
        Image.fromarray(texture).save(out / f"{globe['id']}.png")
        globe.update(texture=f"{globe['id']}.png", **baked)
    shown = rgba[..., :3].copy()
    shown[rgba[..., 3] <= 200] = (120, 150, 200)
    Image.fromarray(shown).save(out / "eyes.png")
    if source_handoff is not None:
        source = json.loads((source_handoff / "spec.json").read_text(encoding="utf-8"))
        raw_map, px = spec["raw"][view], spec["metres_per_aligned_px"]
        zones = [(np.array([(eye["centre_px"][0] - raw_map["left"]) * raw_map["scale"] + raw_map["x0"] - raw_map.get("shift_px", 0),
                            (eye["centre_px"][1] - raw_map["top"]) * raw_map["scale"] + raw_map["y0"]]),
                  zone * max(eye["radii_px"]) * raw_map["scale"]) for eye in eyes]
        (out / "within").mkdir(exist_ok=True)
        for name in source["views"]:
            path = source_handoff / "raw" / f"{name}.png"
            if name == view or not path.is_file():
                continue
            with Image.open(path) as picture:
                shape = (picture.height, picture.width)
            # A wide margin: this mask only has to leave out changes far from the eyes, and another
            # drawing can put the eye a whole eye-width from its place on the mesh.
            allowed = eye_zone_in_view(shape, source["raw"][name], name, zones, head, px, face_view=view, margin=1.5)
            Image.fromarray(np.where(allowed, 255, 0).astype(np.uint8)).save(out / "within" / f"{name}.png")
    report = {"schema_version": 1, "view": view, "white_included": True,
              "eyes": [{**eye, **{k: globe[k] for k in ("id", "center_m", "radii_m", "mesh", "texture", "seen_fraction",
                                                        "white_rgb")}} for eye, globe in zip(eyes, globes)],
              "source_confidence": "estimated", "review": "required"}
    (out / "eyes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def split_eyes(handoff: Path, head: Mesh, out: Path, *, view: str = "front", zone: float = 1.9, grow_px: int = 3,
               **globe_options) -> dict:
    """Write ``out/eyes/``, ``out/rest/``, the eyeballs and ``anatomy.json`` (see the module text).

    ``zone`` times the iris radius is the surroundings of an eye (the white and the lids).
    Only the ``view`` drawing paints there: the other drawings lose those pixels.
    """
    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    px = spec["metres_per_aligned_px"]
    names = [n for n in spec["views"] if (handoff / "raw" / f"{n}.png").is_file()]
    if view not in names:
        raise ValueError(f"The handoff has no drawing for the view {view!r}")
    images = {}
    for name in names:
        with Image.open(handoff / "raw" / f"{name}.png") as picture:
            images[name] = np.asarray(picture.convert("RGBA"))
    eyes, mask = find_eyes(images[view])
    if not eyes:
        raise ValueError(f"No eye found in the {view} drawing (a dark, round pupil in the upper part of the figure)")
    raw_map = spec["raw"][view]
    globes = eye_globes(eyes, raw_map, head, px, view=view, **globe_options)
    zones = [(np.array([(eye["centre_px"][0] - raw_map["left"]) * raw_map["scale"] + raw_map["x0"] - raw_map.get("shift_px", 0),
                        (eye["centre_px"][1] - raw_map["top"]) * raw_map["scale"] + raw_map["y0"]]),
              zone * eye["iris_radius_px"] * raw_map["scale"]) for eye in eyes]
    out.mkdir(parents=True, exist_ok=True)
    only = images[view].copy()
    only[..., 3] = np.where(mask, 255, 0)
    write_handoff(out / "eyes", spec, {view: only}, "eyes only: the drawing's own pixels")
    rest, rows = {}, []
    for name in names:
        gone = (ndimage.binary_dilation(mask, iterations=grow_px) if name == view else
                eye_zone_in_view(images[name].shape[:2], spec["raw"][name], name, zones, head, px, face_view=view))
        cut = images[name].copy()
        cut[..., 3] = np.where(gone, 0, cut[..., 3])
        rest[name] = cut
        rows.append({"view": name, "removed_px": int((gone & (images[name][..., 3] > 200)).sum())})
    write_handoff(out / "rest", spec, rest, "without the eyes: only the face view paints around them")
    _write_eyeballs(globes, out, "eyes", view, "Found by colour in the drawing and placed on the mesh by split-eyes")
    # A picture to review: the drawing with the found eyes outlined.
    shown = images[view][..., :3].copy()
    edge = mask ^ ndimage.binary_erosion(mask, iterations=3)
    shown[edge] = (0, 255, 0)
    Image.fromarray(shown).save(out / "eyes.png")
    report = {"schema_version": 1, "view": view, "eyes": [{**eye, **{k: globe[k] for k in ("id", "center_m", "radii_m", "mesh")}}
                                                          for eye, globe in zip(eyes, globes)],
              "removed": rows, "zone_px": [round(float(z[1]), 1) for z in zones], "source_confidence": "estimated", "review": "required"}
    (out / "eyes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
