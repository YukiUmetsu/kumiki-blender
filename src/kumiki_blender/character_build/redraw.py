"""Change one drawing of a character so that it agrees with the other drawings.

When some views are replaced, the views that stay can show an older design: Pip's front
view had bare arms, and the new side and back views have knit sleeves. A model from such
a set has one design on its front and the other on its back.

``borrow_view`` needs no editor. A front and a back drawing show one outline, mirrored. Where
they do not (Pip's arms were a hand's width higher in the front drawing), a region of one
drawing is taken from the mirror image of the other. Fur, cloth and a hand look much the
same from both sides.

``redraw_view`` has a local image editor (FLUX.2 [klein] in ComfyUI, loopback only) change
the drawing by an instruction, with a second drawing as the example of the design. The
result keeps the drawing's own pixels wherever the editor changed nothing: the editor
works at a lower resolution, and its copy of an untouched face is softer than the drawing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .repaint import klein_workflow, run_comfy
from .split_views import from_square, region_mask, removed_region, to_square, write_handoff
from .views import CANVAS_W, camera, figure_mask


def redraw_prompt(change: str, subject: str) -> str:
    return (f"Image 1 is a drawing of {subject}. Image 2 shows the same character from another side, in its "
            f"current design. Change image 1 to that design: {change}. Keep everything else in image 1 exactly "
            "the same: the same face, pose, size, position, camera angle, colours and soft 3D-rendered materials. "
            "Plain flat light grey background.")


def redraw_view(drawing: Path, example: Path, out: Path, client, *, change: str, subject: str = "the character",
                size: int = 768, steps: int = 20, seed: int = 7, change_delta_e: float = 7.0, band_px: int = 14,
                feather_px: float = 4.0) -> dict:
    """Write the changed drawing to ``out`` (RGBA, the pixel frame of ``drawing``); returns a report.

    ``change`` says what to change, in plain words ("the arms wear cream knit sleeves").
    ``example`` is a drawing that shows the wanted design. The edit is kept next to
    ``out`` (``<out stem>.edit.png`` with its settings) and reused if nothing changed.
    Where the edit differs from the drawing (see ``split_views.removed_region``), and in a
    band of ``band_px`` around that, the result takes the edit; elsewhere the drawing.
    """
    with Image.open(drawing) as picture:
        rgba = picture.convert("RGBA")
    original = np.asarray(rgba)
    if (original[..., 3] < 250).mean() < 0.01:          # a flat background: cut the figure out
        mask, _ = figure_mask(rgba)
        original = np.dstack([original[..., :3], np.where(ndimage.binary_fill_holes(mask), 255, 0).astype(np.uint8)])
        rgba = Image.fromarray(original)
    with Image.open(example) as picture:
        style, _ = to_square(picture.convert("RGBA"), size)
    square, back = to_square(rgba, size)
    prompt = redraw_prompt(change, subject)
    fingerprint = {"drawing_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                   "example_sha256": hashlib.sha256(np.asarray(style).tobytes()).hexdigest(),
                   "prompt": prompt, "size": size, "steps": steps, "seed": seed}
    out.parent.mkdir(parents=True, exist_ok=True)
    edit_path, record = out.with_suffix(".edit.png"), out.with_suffix(".edit.json")
    reused = edit_path.is_file() and record.is_file() and json.loads(record.read_text(encoding="utf-8")) == fingerprint
    if not reused:
        source, style_path = out.with_suffix(".input.png"), out.with_suffix(".example.png")
        square.save(source)
        style.save(style_path)
        data = run_comfy(client, klein_workflow(client.upload_image(source), client.upload_image(style_path), prompt,
                                                size=size, steps=steps, seed=seed))
        partial = edit_path.with_suffix(".part")
        partial.write_bytes(data)
        partial.replace(edit_path)
        record.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    with Image.open(edit_path) as picture:
        edited = np.asarray(from_square(picture, back, rgba.size))
    figure = original[..., 3] > 200
    region, edited_figure = removed_region(original, edited, change_delta_e=change_delta_e)
    # What the edit added outside the old outline (a sleeve is thicker than a bare arm) is a change too.
    added = ndimage.binary_opening(edited_figure & ~ndimage.binary_dilation(figure, iterations=3), iterations=2)
    region |= added
    wide = ndimage.binary_dilation(region, iterations=band_px)
    blend = np.clip(ndimage.gaussian_filter(wide.astype(np.float64), feather_px) * 1.5, 0, 1)[..., None]
    colour = original[..., :3] * (1 - blend) + edited * blend
    near = ndimage.binary_dilation(wide, iterations=int(2 * feather_px) + 2)
    alpha = np.where(near, edited_figure, figure)
    result = np.dstack([np.clip(np.round(colour), 0, 255).astype(np.uint8), np.where(alpha, 255, 0).astype(np.uint8)])
    Image.fromarray(result).save(out)
    return {"drawing": str(drawing), "example": str(example), "prompt": prompt, "reused": reused,
            "changed_share_of_figure": round(float((wide & figure).sum() / max(1, figure.sum())), 4),
            "outline_iou": round(float((alpha & figure).sum() / max(1, (alpha | figure).sum())), 4),
            "output": str(out), "review": "required"}


def borrow_view(handoff: Path, out: Path, *, view: str, source: str,
                boxes: list[tuple[int, int, int, int]] | None = None,
                polygons: list[list[tuple[int, int]]] | None = None, feather_px: float = 3.0) -> dict:
    """Write ``out`` as the handoff with a region of drawing ``view`` taken from the opposite drawing ``source``.

    ``boxes`` and ``polygons`` give the region in the pixels of ``handoff/raw/<view>.png``
    (as for ``split_views.box_part``). Each of its pixels takes the pixel of ``source``
    at the mirrored place in the character's frame, background included, with an edge
    that is ``feather_px`` soft. The two views must look along one line from opposite
    sides. With ``source`` the same as ``view`` (front or back), the region takes the
    other side of the same drawing: an arm from the other arm. The report has the overlap of the two outlines in the region before the change:
    a low value says that the drawings did not agree there.
    """
    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    for name in (view, source):
        if name not in spec["views"]:
            raise ValueError(f"The handoff has no drawing for {name!r}")
    if source != view and not np.allclose(camera(view).toward, -camera(source).toward, atol=1e-6):
        raise ValueError(f"{source} is not the view opposite to {view}")
    if source == view and view not in ("front", "back"):
        raise ValueError("Only a front or a back drawing has two sides that mirror each other")
    images = {}
    for name in spec["views"]:
        with Image.open(handoff / "raw" / f"{name}.png") as picture:
            images[name] = np.asarray(picture.convert("RGBA"))
    target, other = images[view], images[source].astype(np.float64)
    region = region_mask(target.shape[:2], boxes or [], polygons or [])
    if not region.any():
        raise ValueError("Give at least one box or polygon inside the drawing")
    here, there = spec["raw"][view], spec["raw"][source]
    rows, cols = np.mgrid[:target.shape[0], :target.shape[1]].astype(np.float64)
    u = (cols + 0.5 - here["left"]) * here["scale"] + here["x0"] - here.get("shift_px", 0)     # the character's frame
    v = (rows + 0.5 - here["top"]) * here["scale"] + here["y0"]
    x = (CANVAS_W - u + there.get("shift_px", 0) - there["x0"]) / there["scale"] + there["left"] - 0.5
    y = (v - there["y0"]) / there["scale"] + there["top"] - 0.5
    weight = other[..., 3:] / 255.0
    layers = np.dstack([other[..., :3] * weight, weight])                 # premultiplied, so the outline has no dark rim
    sampled = np.dstack([ndimage.map_coordinates(layers[..., c], [y, x], order=1, mode="constant", cval=0.0)
                         for c in range(4)])
    blend = np.clip(ndimage.gaussian_filter(region.astype(np.float64), feather_px), 0, 1)[..., None] if feather_px > 0 \
        else region[..., None].astype(np.float64)
    own = target[..., 3:].astype(np.float64) / 255.0
    mixed = np.dstack([target[..., :3] * own, own]) * (1 - blend) + sampled * blend
    figure = mixed[..., 3] > 0.5
    colour = np.where(figure[..., None], mixed[..., :3] / np.maximum(mixed[..., 3:], 1e-6), target[..., :3])
    result = np.dstack([np.clip(np.round(colour), 0, 255).astype(np.uint8), np.where(figure, 255, 0).astype(np.uint8)])
    before, mirrored = target[..., 3] > 127, sampled[..., 3] > 0.5
    agreed = float((before & mirrored & region).sum() / max(1, ((before | mirrored) & region).sum()))
    images[view] = result
    write_handoff(out, spec, images, spec.get("split", "the whole character") + f"; part of {view} from {source}")
    report = {"schema_version": 1, "view": view, "source": source, "region_px": int(region.sum()),
              "outline_iou_in_region_before": round(agreed, 4),
              "figure_px_before": int((before & region).sum()), "figure_px_after": int((figure & region).sum()),
              "output": str(out), "review": "required"}
    (out / "borrow.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
