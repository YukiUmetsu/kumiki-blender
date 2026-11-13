"""Repaint views of the model that no drawing covers well, so every texel has a consistent source.

AI drawings of one character disagree at angles between the drawn views: a 3/4 view
textured from the front and side drawings shows the eyes twice. A new drawing of that
angle would be one more independent guess. Here the angle comes from the model instead:

1. Render the textured mesh from the camera (``texture.render_textured``).
2. A local image editor (FLUX.2 [klein] in ComfyUI) repaints the render in the style of
   the nearest reference drawing, keeping the render's outline and layout.
3. The repainted image is checked against the render silhouette and baked back with
   ``bake_texture(extra=...)``. Its colours agree with the shape by construction.

Only a local ComfyUI (loopback) is used. The repaint needs about 13 minutes per view at
768 px on an Apple M4 with 24 GB.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .mesh import Mesh
from .texture import render_textured
from .views import _FREE_VIEW, AZIMUTH, CANVAS_H, CANVAS_W, camera, figure_mask

BACKGROUND = (217, 217, 217)


def describe(view: str) -> str:
    """Plain-language camera direction for the prompt."""
    cam = camera(view)
    a = (cam.azimuth + 180) % 360 - 180
    sides = {0: "the front", 45: "the front-left", 90: "the left side", 135: "the back-left", 180: "the back",
             -180: "the back", -135: "the back-right", -90: "the right side", -45: "the front-right"}
    nearest = min(sides, key=lambda k: abs(k - a))
    text = f"seen from {sides[nearest]} of the character"
    if cam.elevation >= 15:
        text += f", from {round(cam.elevation)} degrees above"
    return text


def nearest_drawing(view: str, available: list[str]) -> str:
    """The reference drawing whose camera is closest in direction (style and local details)."""
    toward = camera(view).toward
    return max(available, key=lambda name: float(camera(name).toward @ toward))


def render_condition(mesh: Mesh, corner_uv: np.ndarray, texture: np.ndarray, view: str, metres_per_px: float,
                     size: int = 768, margin: float = 0.08) -> tuple[Image.Image, np.ndarray, dict]:
    """Render the textured mesh, crop a square around it and scale to ``size``.

    Returns the RGB image (figure on the plain background), the figure mask at ``size`` and
    the camera-frame -> image map in the keys that ``bake_texture`` reads.
    """
    image, mask = render_textured(mesh, corner_uv, texture, view, metres_per_px, 1.0)
    mask = ndimage.binary_closing(mask, iterations=2)
    if not mask.any():
        raise ValueError(f"The mesh is outside the frame of view {view}")
    ys, xs = np.nonzero(mask)
    side = (1 + 2 * margin) * max(ys.max() - ys.min(), xs.max() - xs.min())
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    side = int(round(side))
    canvas = np.full((side, side, 3), BACKGROUND, dtype=np.uint8)
    keep = np.zeros((side, side), dtype=bool)
    # Copy the overlap of the crop and the frame (the crop can extend past the frame).
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + side, CANVAS_H), min(left + side, CANVAS_W)
    region = mask[y0:y1, x0:x1]
    canvas[y0 - top:y1 - top, x0 - left:x1 - left][region] = image[y0:y1, x0:x1][region]
    keep[y0 - top:y1 - top, x0 - left:x1 - left] = region
    picture = Image.fromarray(canvas).resize((size, size), Image.LANCZOS)
    small_mask = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize((size, size), Image.BOX)) >= 128
    mapping = {"left": 0.0, "top": 0.0, "scale": side / size, "x0": float(left), "y0": float(top), "shift_px": 0.0}
    return picture, small_mask, mapping


def klein_workflow(render_name: str, style_name: str, prompt: str, *, size: int, steps: int, seed: int,
                   cfg: float = 4.0, denoise: float = 1.0, unet: str = "flux-2-klein-base-4b.safetensors",
                   text_encoder: str = "qwen_3_4b.safetensors", vae: str = "flux2-vae.safetensors") -> dict:
    """ComfyUI API workflow: FLUX.2 [klein] edit with the render and a style drawing as references.

    Sampling starts from the render's latent and runs the last ``denoise`` of the schedule.
    1.0 repaints fully: the render, as a reference, still fixes the outline and layout. At
    0.7 the doubled eye of the render survived. The prompt asks for 3D-rendered fur and no
    outlines: with "illustration" the editor drew a flat outlined cartoon.
    """
    prefix = "kumiki_repaint_" + uuid.uuid4().hex[:12]
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "LoadImage", "inputs": {"image": render_name}},
        "5": {"class_type": "LoadImage", "inputs": {"image": style_name}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "7": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["5", 0], "upscale_method": "lanczos", "megapixels": 0.5, "resolution_steps": 16}},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "11": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["9", 0], "latent": ["6", 0]}},
        "12": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["11", 0], "latent": ["8", 0]}},
        "13": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["10", 0], "latent": ["6", 0]}},
        "14": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["13", 0], "latent": ["8", 0]}},
        "15": {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["12", 0], "negative": ["14", 0], "cfg": cfg}},
        "16": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": size, "height": size}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "19": {"class_type": "SplitSigmasDenoise", "inputs": {"sigmas": ["16", 0], "denoise": denoise}},
        "20": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["18", 0], "guider": ["15", 0], "sampler": ["17", 0], "sigmas": ["19", 1],
                          "latent_image": ["6", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["3", 0]}},
        "22": {"class_type": "SaveImage", "inputs": {"images": ["21", 0], "filename_prefix": prefix}},
    }


def prompt_for(view: str, subject: str) -> str:
    return (f"Image 1 is a 3D model of {subject} from image 2, {describe(view)}, with patchy, misaligned "
            "texture. Clean up image 1: fix seams, smears and doubled features so each eye, ear, button and strap "
            "appears once, where image 1 shows it. Match the soft, detailed 3D-rendered fur, materials and colours "
            "of image 2. No outlines, not a flat cartoon. Keep the exact outline, pose, size, position and camera "
            "angle of image 1. Plain flat light grey background.")


def run_comfy(client, workflow: dict, timeout_s: float = 3600) -> bytes:
    """Queue a workflow, wait for its SaveImage output and return the PNG bytes."""
    prompt_id = client.queue(workflow)
    started = time.time()
    while time.time() - started < timeout_s:
        record = client.history(prompt_id).get(prompt_id)
        if record and record.get("status", {}).get("status_str") == "error":
            raise RuntimeError(f"ComfyUI failed: {json.dumps(record['status'])[:1000]}")
        if record and record.get("outputs"):
            image = next(iter(record["outputs"].values()))["images"][0]
            return client.view_image(image)
        time.sleep(3)
    raise TimeoutError(f"ComfyUI did not finish within {timeout_s:.0f} s")


def alignment(repainted: Image.Image, render_mask: np.ndarray, margin_px: int = 16) -> tuple[float, np.ndarray]:
    """IoU of the repainted figure with the render silhouette, and the pixels that can be trusted.

    Where the editor moved the outline (it often redraws a tail), the picture near that
    place shows a different shape, and its colours would land on the wrong surface. The
    rest of the picture is still good: a face is not worse because the tail behind it
    changed. Trusted pixels are inside both outlines and at least ``margin_px`` from any
    place where the outlines differ.
    """
    figure, _ = figure_mask(repainted.convert("RGB").resize(render_mask.shape[::-1], Image.LANCZOS))
    figure = ndimage.binary_fill_holes(ndimage.binary_opening(figure, iterations=2))
    differ = ndimage.binary_opening(figure ^ render_mask, iterations=3)   # not the 1-2 px jitter of an outline
    trusted = figure & render_mask & ~ndimage.binary_dilation(differ, iterations=margin_px)
    return float((figure & render_mask).sum() / max(1, (figure | render_mask).sum())), trusted


def match_tone(repainted: np.ndarray, render: np.ndarray, drawn: np.ndarray, *, sigma_px: float | None = None,
               max_shift: float = 40.0, erode_px: int = 4) -> np.ndarray:
    """Give the repaint the tone of the render, smoothly, around the pixels that a drawing painted.

    The editor adds its own shading: Pip's repaints were 5 to 10 L* darker than their
    renders, and baked dark patches into the texture. The render carries the drawings'
    colours, but only on ``drawn`` pixels; elsewhere it is a fill colour. So the correction
    is the low-pass difference between render and repaint over the drawn pixels (radius
    ``sigma_px``, default a tenth of the picture: wider than an eye, so a feature that the
    repaint moved or removed does not come back as a smudge). It fades where less than a
    quarter of the neighbourhood is drawn, and the repaint keeps its own tone there. One
    shift for the whole picture made a face pale when the drawn pixels were all on the tail.
    """
    sigma = sigma_px if sigma_px is not None else max(repainted.shape[:2]) / 10
    core = ndimage.binary_erosion(drawn, iterations=erode_px)   # the outline band holds background-blended fur
    if not core.any():
        return repainted
    weight = ndimage.gaussian_filter(core.astype(np.float64), sigma)
    difference = (render.astype(np.float64) - repainted.astype(np.float64)) * core[..., None]
    shift = np.dstack([ndimage.gaussian_filter(difference[..., i], sigma) for i in range(3)])
    shift = np.clip(shift / np.maximum(weight, 1e-6)[..., None], -max_shift, max_shift)
    shift *= np.clip(weight / 0.25, 0, 1)[..., None]
    return np.clip(np.round(repainted + shift), 0, 255).astype(np.uint8)


def _fingerprint(render: Image.Image, style: Image.Image, prompt: str, **settings) -> dict:
    return {"render_sha256": hashlib.sha256(np.asarray(render).tobytes()).hexdigest(),
            "style_sha256": hashlib.sha256(np.asarray(style).tobytes()).hexdigest(),
            "prompt": prompt, **settings}


def repaint_views(handoff: Path, mesh: Mesh, corner_uv: np.ndarray, texture: np.ndarray, views: list[str],
                  out: Path, client, *, subject: str = "the character", size: int = 768, steps: int = 20,
                  denoise: float = 1.0, seed: int = 7, min_iou: float = 0.75, min_usable: float = 0.5,
                  weight: float = 0.8, owners: np.ndarray | None = None, overwrite: bool = False) -> dict:
    """Repaint each view and write ``out/manifest.json`` for ``texture --extra``.

    A repaint counts only where its outline agrees with the render (``alignment``): the
    part around a redrawn tail is masked out, the face next to it is kept. The whole
    repaint is rejected only if its IoU with the render is below ``min_iou`` (the editor
    moved the figure) or less than ``min_usable`` of the figure is left. Accepted
    repaints score ``weight`` times a drawing in the bake.

    Each accepted repaint gets the tone of its render around the pixels that a drawing
    painted (``match_tone``). ``owners`` is the ``owners.png`` of the bake that made
    ``texture`` (black: no view saw the texel, the colour is a fill). Without it, every
    pixel of the render counts as drawn; pass it when a side has no drawing.

    A finished repaint (``<view>_repainted.png`` with ``<view>_repainted.json`` recording
    the same render, style image and settings) is reused unless ``overwrite`` is set, so
    a batch cut short can be rerun into the same folder.
    """
    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    px = spec["metres_per_aligned_px"]
    drawings = [n for n in spec["raw"] if n in AZIMUTH]
    out.mkdir(parents=True, exist_ok=True)
    accepted, rows = [], []
    for view in views:
        render, mask, mapping = render_condition(mesh, corner_uv, texture, view, px, size)
        render_path = out / f"{view}_render.png"
        render.save(render_path)
        style = nearest_drawing(view, drawings)
        rgba = Image.open(handoff / "raw" / f"{style}.png").convert("RGBA")
        flat = Image.new("RGB", rgba.size, BACKGROUND)
        flat.paste(rgba, mask=rgba.getchannel("A"))
        style_path = out / f"{view}_style.png"
        flat.save(style_path)
        started = time.time()
        prompt = prompt_for(view, subject)
        # A repaint takes minutes; reusing finished ones lets an interrupted batch resume. The
        # fingerprint ties a repaint to its inputs, so a changed mesh, texture or prompt repaints again.
        fingerprint = _fingerprint(render, flat, prompt, size=size, steps=steps, seed=seed, denoise=denoise)
        repainted_path = out / f"{view}_repainted.png"
        inputs_path = out / f"{view}_repainted.json"
        reused = (not overwrite and repainted_path.is_file() and inputs_path.is_file()
                  and json.loads(inputs_path.read_text(encoding="utf-8")) == fingerprint)
        if not reused:
            workflow = klein_workflow(client.upload_image(render_path), client.upload_image(style_path),
                                      prompt, size=size, steps=steps, seed=seed, denoise=denoise)
            data = run_comfy(client, workflow)
            partial = repainted_path.with_suffix(".part")
            partial.write_bytes(data)
            partial.replace(repainted_path)
            inputs_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
        repainted = Image.open(repainted_path).convert("RGB").resize((size, size), Image.LANCZOS)
        iou, trusted = alignment(repainted, mask)
        usable = float(trusted.sum() / max(1, mask.sum()))
        row = {"view": view, "style": style, "iou_with_render": round(iou, 4), "usable": round(usable, 4),
               "seconds": round(time.time() - started), "reused": reused,
               "accepted": iou >= min_iou and usable >= min_usable}
        rows.append(row)
        if row["accepted"]:
            # The repaint's colours count only where its figure and the mesh agree (see ``alignment``).
            drawn = trusted
            if owners is not None:
                shown, _, _ = render_condition(mesh, corner_uv, owners, view, px, size)
                drawn = trusted & (np.asarray(shown).max(axis=2) > 40)   # not the black of an unseen texel
            row["drawn"] = round(float(drawn.sum() / max(1, trusted.sum())), 4)
            toned = match_tone(np.asarray(repainted), np.asarray(render.convert("RGB")), drawn)
            image = np.dstack([toned, np.where(trusted, 255, 0).astype(np.uint8)])
            painted = out / f"{view}_paint.png"
            Image.fromarray(image).save(painted)
            accepted.append({"name": view, "image": str(painted.resolve()), "map": mapping, "weight": weight})
    manifest = {"schema_version": 1, "views": rows, "extra": accepted}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def mirror_twin(view: str) -> str | None:
    """The free camera seen in a mirror across the character's centre plane (az -> -az)."""
    match = _FREE_VIEW.match(view)
    if not match:
        return None
    azimuth, elevation = float(match.group(1)), match.group(2)
    if azimuth % 180 == 0:  # straight front or back: its own mirror image
        return view
    return f"az{-azimuth:g}_el{elevation}"


def screen_extras(handoff: Path, mesh: Mesh, corner_uv: np.ndarray, extra: list[dict], *,
                  max_added_delta_e: float = 0.5, size: int = 1024, **bake) -> tuple[list[dict], list[dict]]:
    """Keep the repaints that do not contradict the drawings.

    A repaint can keep the outline and still draw a different face: Quill's view from
    above the front lost the glasses, passed the IoU check and raised the front view
    median error from 1.5 to 2.6 delta E. Each repaint is baked alone on top of the
    drawings, and it is dropped if it raises the mean error of any drawn view by more
    than ``max_added_delta_e``. With ``mirror_missing`` one side has no drawing to measure, so
    a repaint is judged by the worse of itself and its mirror twin.
    """
    from .texture import appearance_report, bake_texture

    def errors(items):
        texture, _ = bake_texture(handoff, mesh, corner_uv, size=size, extra=items or None, **bake)
        # The mean, not the median: lost glasses cover a few percent of a view and leave the median as it was.
        return {r["view"]: r["mean_delta_e"] for r in appearance_report(handoff, mesh, corner_uv, texture)["views"]}

    drawn = bake.get("views")
    base = errors([])
    measured = [v for v in base if drawn is None or v in drawn]
    added = {}
    for item in extra:
        after = errors([item])
        added[item["name"]] = max((after[v] - base[v] for v in measured), default=0.0)
    rows, kept = [], []
    for item in extra:
        twin = mirror_twin(item["name"]) if bake.get("mirror_missing") else None
        twin = None if twin == item["name"] else twin
        cost = max(added[item["name"]], added.get(twin, 0.0))
        rows.append({"view": item["name"], "added_delta_e": round(added[item["name"]], 2),
                     "judged_with": twin if twin in added else None, "kept": cost <= max_added_delta_e})
        if rows[-1]["kept"]:
            kept.append(item)
    return kept, rows


def default_views(elevation: float = 35.0) -> list[str]:
    """The four 3/4 directions at eye level, and four from above (tops of the head and tail)."""
    return ([f"az{a}_el0" for a in (45, 135, -135, -45)]
            + [f"az{a}_el{round(elevation)}" for a in (0, 90, 180, -90)])

