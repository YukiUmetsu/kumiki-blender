"""Split the drawings of a character into one part and the rest, so each can be modelled alone.

A model made in one piece from all drawings fuses its parts: tail, scarf and body are one
skin, the tail has the width that the generator guessed, and the texture of one part
lands on the other. Modelled alone, a part has its own outline to fit and its own
pixels to take colour from.

``split_views`` makes the two reference sets from a handoff:

1. A local image editor (FLUX.2 [klein] in ComfyUI, loopback only) removes the part
   from each drawing that shows it and paints what was behind it.
2. The part is where the drawing and the edit differ: the outline that went away, and
   the area that changed colour. That needs no segmentation model.
3. ``<out>/<part>/`` gets the drawing's own pixels of the part. ``<out>/rest/`` gets the
   drawing with the edit only where the part was, so every pixel that the drawing
   shows stays the drawing's.

Both folders are handoffs in the frame of the whole character (the same ``spec.json``,
``raw/``, ``masks/`` and ``aligned/``), so ``fit``, ``texture``, ``compare`` and
``place-part`` work on them, and the masks are full-frame region masks for a
CharacterSpec v2. What the editor paints behind the part is a guess; in the finished
model the part hides most of it.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .checks import lab
from .repaint import BACKGROUND, run_comfy
from .views import CANVAS_H, CANVAS_W, figure_mask


def edit_workflow(image_name: str, prompt: str, *, size: int, steps: int, seed: int, cfg: float = 4.0,
                  unet: str = "flux-2-klein-base-4b.safetensors", text_encoder: str = "qwen_3_4b.safetensors",
                  vae: str = "flux2-vae.safetensors") -> dict:
    """ComfyUI API workflow: FLUX.2 [klein] edit of one image, which is also the only reference."""
    prefix = "kumiki_split_" + uuid.uuid4().hex[:12]
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "11": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["9", 0], "latent": ["6", 0]}},
        "13": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["10", 0], "latent": ["6", 0]}},
        "15": {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["11", 0], "negative": ["13", 0], "cfg": cfg}},
        "16": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": size, "height": size}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "20": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["18", 0], "guider": ["15", 0], "sampler": ["17", 0], "sigmas": ["16", 0],
                          "latent_image": ["6", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["3", 0]}},
        "22": {"class_type": "SaveImage", "inputs": {"images": ["21", 0], "filename_prefix": prefix}},
    }


def removal_prompt(part: str, subject: str, behind: str, keep: str = "") -> str:
    """The instruction for the editor. ``keep`` names what it must not take away with the part: asked
    to remove Pip's bag, it also removed the fringe of the scarf and the skin flaps under the arms."""
    kept = f" Do not remove or change {keep}." if keep else ""
    return (f"Remove only the {part} of {subject}. Where the {part} was, show what is behind it: {behind}, "
            f"or the plain background.{kept} Keep everything else exactly the same: the same pose, size, "
            "position, colours, materials and camera angle. Plain flat light grey background.")


def to_square(rgba: Image.Image, size: int) -> tuple[Image.Image, dict]:
    """The figure on the plain background, padded to a square and scaled to ``size``; and the way back."""
    side = max(rgba.size)
    left, top = (side - rgba.width) // 2, (side - rgba.height) // 2
    canvas = Image.new("RGB", (side, side), BACKGROUND)
    canvas.paste(rgba, (left, top), rgba.getchannel("A"))
    return canvas.resize((size, size), Image.LANCZOS), {"side": side, "left": left, "top": top}


def from_square(picture: Image.Image, back: dict, size: tuple[int, int]) -> Image.Image:
    """An edited square picture in the pixel frame of the drawing."""
    full = picture.convert("RGB").resize((back["side"], back["side"]), Image.LANCZOS)
    return full.crop((back["left"], back["top"], back["left"] + size[0], back["top"] + size[1]))


def removed_region(original: np.ndarray, edited: np.ndarray, *, change_delta_e: float = 7.0, blur_px: float = 5.0,
                   shift_px: int = 4, close_px: int = 24, open_px: int = 6, min_share: float = 0.01
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Where the edit took something away from the drawing, and the figure of the edit.

    ``original`` is RGBA, ``edited`` RGB on the plain background, both in the drawing's
    pixel frame. The region is the outline that went away plus the area inside the
    figure whose colour changed by more than ``change_delta_e``.

    The editor never returns the rest pixel for pixel, and a tail in front of fur of
    nearly the same colour changes little. So the colours are compared after a blur of
    ``blur_px`` and at the best of nine small shifts (``shift_px``): that allows a low
    threshold without taking a moved outline for a change. Gaps up to ``close_px`` and
    holes are closed (a stripe of the tail that matches the fur behind it; on Pip, 10
    left the tail in two pieces in the back view and 24 gave it whole), pieces
    thinner than ``open_px`` or smaller than ``min_share`` of the figure are dropped.
    """
    figure = original[..., 3] > 200
    edited_figure, _ = figure_mask(Image.fromarray(edited))
    edited_figure = ndimage.binary_fill_holes(edited_figure)
    gone = figure & ~ndimage.binary_dilation(edited_figure, iterations=3)
    flat = np.where(figure[..., None], original[..., :3], BACKGROUND).astype(np.float64)

    def soft_lab(image: np.ndarray) -> np.ndarray:
        blurred = np.dstack([ndimage.gaussian_filter(image[..., c].astype(np.float64), blur_px) for c in range(3)])
        return lab(np.clip(blurred, 0, 255).astype(np.uint8).reshape(-1, 3)).reshape(figure.shape + (3,))

    before, after = soft_lab(flat), soft_lab(edited)
    change = np.full(figure.shape, np.inf)
    for dy in (-shift_px, 0, shift_px):
        for dx in (-shift_px, 0, shift_px):
            change = np.minimum(change, np.linalg.norm(before - np.roll(after, (dy, dx), axis=(0, 1)), axis=2))
    region = gone | (figure & (change > change_delta_e))

    def disc(radius: int) -> np.ndarray:
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        return xx * xx + yy * yy <= radius * radius

    region = ndimage.binary_opening(region, structure=disc(max(1, open_px // 2)))     # specks first
    pad = close_px + 1                                # closing at the image border must not erode the region
    closed = ndimage.binary_closing(np.pad(region, pad), structure=disc(max(1, close_px)))[pad:-pad, pad:-pad]
    region = ndimage.binary_opening(ndimage.binary_fill_holes(closed), structure=disc(max(1, open_px)))
    # The blur spreads a colour change about one blur width past its edge. Take that back, but
    # not at the outline that went away, which is exact.
    halo = max(1, round(0.8 * blur_px))
    region = ndimage.binary_erosion(region, structure=disc(halo)) | (region & ndimage.binary_dilation(gone, iterations=halo + 1))
    labels, count = ndimage.label(region)
    if count:
        sizes = ndimage.sum_labels(region, labels, range(1, count + 1))
        region = np.isin(labels, 1 + np.nonzero(sizes >= min_share * figure.sum())[0])
    return region & figure, edited_figure


def to_aligned(rgba: np.ndarray, raw_map: dict) -> Image.Image:
    """A drawing-frame RGBA image in the aligned frame of the handoff (1600 x 1200)."""
    scale = raw_map["scale"]
    shift = raw_map.get("shift_px", 0)
    coefficients = (1 / scale, 0, raw_map["left"] + (shift - raw_map["x0"]) / scale,
                    0, 1 / scale, raw_map["top"] - raw_map["y0"] / scale)
    return Image.fromarray(rgba).transform((CANVAS_W, CANVAS_H), Image.AFFINE, coefficients, Image.BILINEAR)


def write_handoff(root: Path, spec: dict, images: dict[str, np.ndarray], note: str) -> None:
    """A handoff in the character's frame from drawing-frame RGBA images (alpha marks the figure)."""
    for folder in ("raw", "masks", "aligned"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for name, rgba in images.items():
        Image.fromarray(rgba).save(root / "raw" / f"{name}.png")
        aligned = to_aligned(rgba, spec["raw"][name])
        aligned.save(root / "aligned" / f"{name}.png")
        Image.fromarray(np.where(np.asarray(aligned)[..., 3] > 127, 255, 0).astype(np.uint8)).save(
            root / "masks" / f"{name}.png")
    part_spec = {**spec, "views": [n for n in spec["views"] if n in images],
                 "raw": {n: spec["raw"][n] for n in images}, "split": note}
    part_spec.pop("blockout", None)   # the blockout is of the whole character
    (root / "spec.json").write_text(json.dumps(part_spec, indent=2), encoding="utf-8")


def split_views(handoff: Path, out: Path, client, *, part: str, name: str | None = None,
                subject: str = "the character", behind: str = "the body and the clothes", keep: str = "",
                views: list[str] | None = None, size: int = 768, steps: int = 20, seed: int = 7,
                change_delta_e: float = 7.0, feather_px: float = 4.0, band_px: int = 14,
                min_share: float = 0.01, within: Path | None = None, fill: bool = True) -> dict:
    """Write ``out/<name>/`` and ``out/rest/`` (see the module text); returns the report (``out/split.json``).

    ``within`` is a folder with ``<view>.png`` masks in the pixel frame of the drawings
    (white: where the part can be). The editor sometimes changes something else too: asked
    to remove Pip's eyes, it also repainted the inside of an ear in the side view. A change
    outside the mask is dropped, and the drawing stays as it is there. A view without a
    mask file is not limited.

    With ``fill`` off, ``rest`` gets a hole where the part was, not the editor's picture of
    what is behind it. Use that where the guess is bad: for Pip's eye in the side view the
    editor painted a dark socket. The bake then fills the hole from the texture around it.

    ``part`` says what to take out, in plain words for the editor ("brown leather bag and
    its strap"), ``behind`` what it hides, and ``keep`` what must stay. ``name`` is the
    folder of the part (default: ``part`` with dashes).

    ``views`` are the drawings that show the part; the other drawings go to ``rest``
    unchanged. A finished edit (``out/edits/<view>.png`` with the same drawing, prompt
    and settings) is reused, so a run that was cut short can be repeated.
    """
    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    names = [n for n in spec["views"] if (handoff / "raw" / f"{n}.png").is_file()]
    chosen = [n for n in names if views is None or n in views]
    unknown = sorted(set(views or []) - set(names))
    if unknown:
        raise ValueError(f"The handoff has no drawing for {unknown}")
    name = name or "-".join(part.lower().split())
    if not name.replace("-", "").replace("_", "").isalnum() or name in ("rest", "edits"):
        raise ValueError(f"Not a usable folder name for the part: {name!r}")
    folder = name
    (out / "edits").mkdir(parents=True, exist_ok=True)
    prompt = removal_prompt(part, subject, behind, keep)
    part_images, rest_images, rows = {}, {}, []
    for view in names:
        with Image.open(handoff / "raw" / f"{view}.png") as picture:
            rgba = picture.convert("RGBA")
        original = np.asarray(rgba)
        if view not in chosen:
            rest_images[view] = original
            continue
        square, back = to_square(rgba, size)
        fingerprint = {"drawing_sha256": hashlib.sha256(original.tobytes()).hexdigest(), "prompt": prompt,
                       "size": size, "steps": steps, "seed": seed}
        edit_path, record = out / "edits" / f"{view}.png", out / "edits" / f"{view}.json"
        reused = (edit_path.is_file() and record.is_file()
                  and json.loads(record.read_text(encoding="utf-8")) == fingerprint)
        if not reused:
            source = out / "edits" / f"{view}_input.png"
            square.save(source)
            data = run_comfy(client, edit_workflow(client.upload_image(source), prompt, size=size, steps=steps, seed=seed))
            partial = edit_path.with_suffix(".part")
            partial.write_bytes(data)
            partial.replace(edit_path)
            record.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
        with Image.open(edit_path) as picture:
            edited = np.asarray(from_square(picture, back, rgba.size))
        region, edited_figure = removed_region(original, edited, change_delta_e=change_delta_e, min_share=min_share)
        limited = False
        if within is not None and (within / f"{view}.png").is_file():
            with Image.open(within / f"{view}.png") as picture:
                allowed = np.asarray(picture.convert("L")) >= 128
            if allowed.shape != region.shape:
                raise ValueError(f"{within / (view + '.png')} must have the size of the {view} drawing")
            limited = bool((region & ~allowed).any())
            region &= allowed
        figure = original[..., 3] > 200
        share = float(region.sum() / max(1, figure.sum()))
        rows.append({"view": view, "reused": reused, "part_share_of_figure": round(share, 4),
                     "part_found": bool(region.any()), "changes_outside_dropped": limited})
        if not region.any():           # the editor changed nothing here: the drawing does not show the part
            rest_images[view] = original
            continue
        cut = original.copy()
        cut[..., 3] = np.where(region, 255, 0)
        part_images[view] = cut
        # The rest: the drawing's pixels, and the edit where the part was and in a band around it.
        # The found region ends a little inside the part (its fringe is thin and soft), and a ring
        # of the part's own pixels around the edit shows as a ragged outline on the model.
        wide = ndimage.binary_dilation(region, iterations=band_px)
        if not fill:
            hole = original.copy()
            hole[..., 3] = np.where(wide, 0, hole[..., 3])
            rest_images[view] = hole
            continue
        blend = np.clip(ndimage.gaussian_filter(wide.astype(np.float64), feather_px) * 1.5, 0, 1)[..., None]
        colour = original[..., :3] * (1 - blend) + edited * blend
        near = ndimage.binary_dilation(wide, iterations=int(2 * feather_px) + 2)
        # Away from the part, the drawing's figure; but not the fringe of the part that the edit no longer shows.
        alpha = np.where(near, edited_figure, figure & ndimage.binary_dilation(edited_figure, iterations=4))
        rest_images[view] = np.dstack([np.clip(np.round(colour), 0, 255).astype(np.uint8),
                                       np.where(alpha, 255, 0).astype(np.uint8)])
    if not part_images:
        raise ValueError(f"The editor removed nothing: no drawing shows a {part}, or the prompt did not work")
    write_handoff(out / folder, spec, part_images, f"{part} only: the drawings' own pixels")
    write_handoff(out / "rest", spec, rest_images, f"without the {part}: edited where the {part} was")
    report = {"schema_version": 1, "part": part, "name": folder, "prompt": prompt, "views": rows,
              "source_confidence": "estimated", "review": "required",
              "folders": {folder: str((out / folder).resolve()), "rest": str((out / "rest").resolve())}}
    (out / "split.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def region_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]],
                polygons: list[list[tuple[int, int]]]) -> np.ndarray:
    """The pixels of a (height, width) picture inside boxes (x0, y0, x1, y1) or polygons of (x, y)."""
    inside = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        inside[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    for outline in polygons:
        if len(outline) < 3:
            raise ValueError(f"A polygon needs three points or more, not {len(outline)}")
        sheet = Image.new("1", (shape[1], shape[0]), 0)
        ImageDraw.Draw(sheet).polygon([(int(x), int(y)) for x, y in outline], fill=1)
        inside |= np.asarray(sheet, bool)
    return inside


def box_part(handoff: Path, out: Path, *, name: str, boxes: dict[str, list[tuple[int, int, int, int]]] | None = None,
             polygons: dict[str, list[list[tuple[int, int]]]] | None = None,
             colours: list[tuple[int, int, int]] | None = None, delta_e: float = 25.0, close_px: int = 3,
             min_share: float = 0.05, open_px: int = 0, like: Path | None = None) -> dict:
    """Split off the part of each drawing that lies inside boxes or polygons; no image editor.

    ``boxes`` gives, for each view, one or more (x0, y0, x1, y1) in the pixels of
    ``handoff/raw/<view>.png``; ``polygons`` gives outlines as lists of (x, y) for a part
    that no box fits (an arm at an angle). The figure's pixels inside them are the part. Use it where
    the editor does not do what you ask: asked to remove Pip's hands, it left them in one
    view and drew new ones in the other. Writes ``out/<name>/`` with the views that have
    a box, and ``out/rest/`` with a hole where the part was (as ``split_views`` without fill).

    ``colours`` (sRGB) limits the part to the pixels within ``delta_e`` (CIE76) of one of
    them: a red scarf is the red inside a box around the neck, and no outline has to be
    drawn. Gaps of up to ``close_px`` pixels (the shadow between two strands of wool) and
    holes are closed, and a piece smaller than ``min_share`` of the largest piece is
    dropped (brown fur has some red in it). ``open_px`` removes what is thinner than twice
    that many pixels: tufts of dark fur that hang on the edge of a green vest. It also
    removes a fringe, so leave it at 0 for a scarf. With colours and no box for a view,
    the whole drawing is searched.

    ``like`` is a part handoff of the same drawings (from ``split_views`` or this function).
    Its pixels are the part: this takes a part out of another handoff of the character,
    one that an earlier step changed.
    """
    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    if not name.replace("-", "").replace("_", "").isalnum() or name in ("rest", "edits"):
        raise ValueError(f"Not a usable folder name for the part: {name!r}")
    boxes, polygons = boxes or {}, polygons or {}
    unknown = sorted((set(boxes) | set(polygons)) - set(spec["views"]))
    if unknown:
        raise ValueError(f"The handoff has no drawing for {unknown}")
    wanted = lab(np.asarray(colours, dtype=np.uint8).reshape(-1, 3)) if colours else None
    like_spec = json.loads((like / "spec.json").read_text(encoding="utf-8")) if like else None
    part_images, rest_images, rows = {}, {}, []
    for view in spec["views"]:
        path = handoff / "raw" / f"{view}.png"
        if not path.is_file():
            continue
        with Image.open(path) as picture:
            original = np.asarray(picture.convert("RGBA"))
        outlined = view in boxes or view in polygons
        other = like / "raw" / f"{view}.png" if like else None
        if not outlined and not (other and other.is_file()) and (wanted is None or boxes or polygons):
            rest_images[view] = original
            continue
        inside = region_mask(original.shape[:2], boxes.get(view, []), polygons.get(view, [])) if outlined \
            else np.ones(original.shape[:2], bool)
        if other and other.is_file():
            if like_spec["raw"][view] != spec["raw"][view]:
                raise ValueError(f"{like} has another frame for the {view} drawing than {handoff}")
            with Image.open(other) as picture:
                shown = np.asarray(picture.convert("RGBA"))[..., 3] > 127
            if shown.shape != inside.shape:
                raise ValueError(f"{other} has another size than the {view} drawing")
            inside &= shown
        elif like and not outlined:
            rest_images[view] = original
            continue
        region = inside & (original[..., 3] > 200)
        if wanted is not None and region.any():
            pixels = lab(original[..., :3][region])
            near = (np.linalg.norm(pixels[:, None, :] - wanted[None, :, :], axis=2) <= delta_e).any(axis=1)
            coloured = np.zeros_like(region)
            coloured[region] = near
            if close_px > 0:
                coloured = ndimage.binary_closing(np.pad(coloured, close_px), iterations=close_px)[close_px:-close_px,
                                                                                                    close_px:-close_px]
            coloured = ndimage.binary_fill_holes(coloured)
            if open_px > 0:
                coloured = ndimage.binary_opening(coloured, iterations=open_px)
            region &= coloured
            labels, count = ndimage.label(region)
            if count > 1 and min_share > 0:
                sizes = ndimage.sum_labels(region, labels, range(1, count + 1))
                region = np.isin(labels, 1 + np.nonzero(sizes >= min_share * sizes.max())[0])
        if not region.any():
            if wanted is not None and not outlined:
                rest_images[view] = original           # the colour is not in this drawing
                continue
            raise ValueError(f"No part of the figure lies inside the boxes and polygons of the {view} drawing")
        cut, hole = original.copy(), original.copy()
        cut[..., 3] = np.where(region, 255, 0)
        hole[..., 3] = np.where(region, 0, hole[..., 3])
        part_images[view], rest_images[view] = cut, hole
        rows.append({"view": view, "part_px": int(region.sum()),
                     "boxes": [list(map(int, box)) for box in boxes.get(view, [])],
                     "polygons": [[[int(x), int(y)] for x, y in outline] for outline in polygons.get(view, [])]})
    if not part_images:
        raise ValueError("Give at least one box or polygon, colours, or a part handoff")
    write_handoff(out / name, spec, part_images, f"{name}: the figure inside boxes and polygons")
    write_handoff(out / "rest", spec, rest_images, f"without {name}: a hole where it was")
    report = {"schema_version": 1, "name": name, "views": rows,
              "folders": {name: str((out / name).resolve()), "rest": str((out / "rest").resolve())}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "box.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

