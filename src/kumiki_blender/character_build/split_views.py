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
from PIL import Image
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
                min_share: float = 0.01) -> dict:
    """Write ``out/<name>/`` and ``out/rest/`` (see the module text); returns the report (``out/split.json``).

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
        figure = original[..., 3] > 200
        share = float(region.sum() / max(1, figure.sum()))
        rows.append({"view": view, "reused": reused, "part_share_of_figure": round(share, 4),
                     "part_found": bool(region.any())})
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
