"""Change one drawing of a character so that it agrees with the other drawings.

When some views are replaced, the views that stay can show an older design: Pip's front
view had bare arms, and the new side and back views have knit sleeves. A model from such
a set has one design on its front and the other on its back.

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
from .split_views import from_square, removed_region, to_square
from .views import figure_mask


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
