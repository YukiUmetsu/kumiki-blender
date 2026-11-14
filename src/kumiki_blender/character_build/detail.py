"""Give a small part of a character more pixels than the drawings have for it.

A hand is 60 pixels wide in a drawing of the whole character. The shape generator and
the texture bake can use only those pixels, so the hand comes out as a blob. Split off
as a part (``split_views``), it can have its own, larger drawings:

``detail_views`` cuts each drawing of a part to the part, enlarges it, and has a local
image editor (FLUX.2 [klein] in ComfyUI, loopback only) redraw it sharp, with the same
shape. The result is a part handoff in the frame of the whole character, like its input:
only the scale of its drawings differs, so ``place-part``, ``fit`` and ``texture`` use
the larger drawings without a change. The detail is the editor's invention inside the
outline of the drawing. Review the pictures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .repaint import BACKGROUND, run_comfy
from .split_views import edit_workflow, write_handoff
from .views import figure_mask


def detail_prompt(part: str, subject: str, detail: str) -> str:
    wanted = f" Show {detail}." if detail else ""
    return (f"Image 1 is an enlarged, blurry picture of {part} of {subject}. Redraw it sharp and detailed, as a "
            f"soft 3D render.{wanted} Keep exactly the same shape, pose, size, position, outline and colours as "
            "image 1. Plain flat light grey background.")


def match_colours(redrawn: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The redrawn picture with the Lab mean and spread of the original inside ``mask``.

    Asked to keep the colours, the editor still chose its own: Pip's pale hand came back
    bright pink. The detail is in the redrawn picture, and the colour is the drawing's.
    """
    from skimage.color import lab2rgb, rgb2lab

    if mask.sum() < 50:
        return redrawn
    source, target = rgb2lab(redrawn), rgb2lab(original)
    spread = np.clip(target[mask].std(axis=0) / np.maximum(source[mask].std(axis=0), 1e-6), 0.5, 2.0)
    matched = (source - source[mask].mean(axis=0)) * spread + target[mask].mean(axis=0)
    return np.clip(np.round(lab2rgb(matched) * 255), 0, 255).astype(np.uint8)


def detail_views(part_handoff: Path, out: Path, client, *, part: str, subject: str = "the character",
                 detail: str = "", size: int = 768, margin: float = 0.2, steps: int = 20, seed: int = 7,
                 min_iou: float = 0.6) -> dict:
    """Write ``out`` as a part handoff with enlarged, redrawn drawings; returns the report (``out/detail.json``).

    Each drawing of ``part_handoff`` must show the part as one piece (the largest piece
    is used). The piece is cut out with ``margin`` of its size around it and scaled to
    ``size`` pixels. If the outline of the redrawn picture overlaps the enlarged original
    by less than ``min_iou``, the editor changed the shape: the enlarged original is
    used for that view, and the report says so. The redrawn picture takes the colours of
    the original (``match_colours``). Finished edits are reused.
    """
    spec = json.loads((part_handoff / "spec.json").read_text(encoding="utf-8"))
    prompt = detail_prompt(part, subject, detail)
    (out / "edits").mkdir(parents=True, exist_ok=True)
    images, maps, rows = {}, {}, []
    for view in spec["views"]:
        path = part_handoff / "raw" / f"{view}.png"
        if not path.is_file():
            continue
        with Image.open(path) as picture:
            rgba = np.asarray(picture.convert("RGBA"))
        labels, count = ndimage.label(rgba[..., 3] > 200)
        if not count:
            continue
        piece = labels == 1 + int(np.argmax(ndimage.sum_labels(labels > 0, labels, range(1, count + 1))))
        ys, xs = np.nonzero(piece)
        side = int(round((1 + 2 * margin) * max(xs.max() - xs.min(), ys.max() - ys.min()) + 1))
        left, top = int(round((xs.min() + xs.max()) / 2 - side / 2)), int(round((ys.min() + ys.max()) / 2 - side / 2))
        canvas = np.full((side, side, 3), BACKGROUND, dtype=np.uint8)
        inside = np.zeros((side, side), bool)
        y0, x0, y1, x1 = max(top, 0), max(left, 0), min(top + side, rgba.shape[0]), min(left + side, rgba.shape[1])
        region = piece[y0:y1, x0:x1]
        canvas[y0 - top:y1 - top, x0 - left:x1 - left][region] = rgba[y0:y1, x0:x1, :3][region]
        inside[y0 - top:y1 - top, x0 - left:x1 - left] = region
        large = Image.fromarray(canvas).resize((size, size), Image.LANCZOS)
        large_mask = np.asarray(Image.fromarray(inside.astype(np.uint8) * 255).resize((size, size), Image.BILINEAR)) >= 128
        fingerprint = {"crop_sha256": hashlib.sha256(np.asarray(large).tobytes()).hexdigest(), "prompt": prompt,
                       "size": size, "steps": steps, "seed": seed}
        edit_path, record = out / "edits" / f"{view}.png", out / "edits" / f"{view}.json"
        reused = edit_path.is_file() and record.is_file() and json.loads(record.read_text(encoding="utf-8")) == fingerprint
        if not reused:
            source = out / "edits" / f"{view}_input.png"
            large.save(source)
            data = run_comfy(client, edit_workflow(client.upload_image(source), prompt, size=size, steps=steps, seed=seed))
            partial = edit_path.with_suffix(".part")
            partial.write_bytes(data)
            partial.replace(edit_path)
            record.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
        with Image.open(edit_path) as picture:
            edited = picture.convert("RGB").resize((size, size), Image.LANCZOS)
        figure, _ = figure_mask(edited, tolerance=14)
        figure = ndimage.binary_fill_holes(ndimage.binary_opening(figure, iterations=2))
        iou = float((figure & large_mask).sum() / max(1, (figure | large_mask).sum()))
        kept = iou >= min_iou
        coloured = match_colours(np.asarray(edited), np.asarray(large), figure & large_mask)
        result = (np.dstack([coloured, np.where(figure, 255, 0).astype(np.uint8)]) if kept else
                  np.dstack([np.asarray(large), np.where(large_mask, 255, 0).astype(np.uint8)]))
        images[view] = result
        old, enlarge = spec["raw"][view], side / size       # drawing pixels for each new pixel
        maps[view] = {"left": 0, "top": 0, "scale": old["scale"] * enlarge,
                      "x0": old["x0"] + (left - old["left"]) * old["scale"],
                      "y0": old["y0"] + (top - old["top"]) * old["scale"], "shift_px": old.get("shift_px", 0)}
        rows.append({"view": view, "reused": reused, "enlarged": round(1 / enlarge, 2), "outline_iou": round(iou, 4),
                     "redrawn": kept})
    if not images:
        raise ValueError("The part handoff has no drawing with a figure")
    write_handoff(out, {**spec, "raw": {**spec["raw"], **maps}}, images, f"{part}: enlarged and redrawn")
    report = {"schema_version": 1, "part": part, "prompt": prompt, "views": rows, "source_confidence": "estimated",
              "review": "required"}
    (out / "detail.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
