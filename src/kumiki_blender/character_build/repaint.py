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
from .views import AZIMUTH, CANVAS_H, CANVAS_W, camera, figure_mask

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


def alignment(repainted: Image.Image, render_mask: np.ndarray) -> float:
    """IoU of the repainted figure with the render silhouette (the repaint must not move the figure)."""
    figure, _ = figure_mask(repainted.convert("RGB").resize(render_mask.shape[::-1], Image.LANCZOS))
    figure = ndimage.binary_fill_holes(ndimage.binary_opening(figure, iterations=2))
    return float((figure & render_mask).sum() / max(1, (figure | render_mask).sum()))


def repaint_views(handoff: Path, mesh: Mesh, corner_uv: np.ndarray, texture: np.ndarray, views: list[str],
                  out: Path, client, *, subject: str = "the character", size: int = 768, steps: int = 20,
                  denoise: float = 1.0, seed: int = 7, min_iou: float = 0.9, weight: float = 1.5,
                  overwrite: bool = False) -> dict:
    """Repaint each view and write ``out/manifest.json`` for ``texture --extra``.

    A repaint whose silhouette IoU with the render is below ``min_iou`` is rejected:
    the editor moved or reshaped the figure, so its colours would land on the wrong
    surface. Accepted repaints score ``weight`` times a drawing in the bake. A finished
    repaint (``<view>_repainted.png`` with ``<view>_repainted.json`` recording the same
    render, style image and settings) is reused unless ``overwrite`` is set, so a batch cut
    short can be rerun into the same folder. With
    ``texture --power 4`` (Pip) a drawing keeps the surface it faces and the repaints take
    the angles in between: front delta E 3.1 -> 4.1, no doubled eye in the 3/4 view.
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
        fingerprint = {"render_sha256": hashlib.sha256(np.asarray(render).tobytes()).hexdigest(),
                       "style_sha256": hashlib.sha256(np.asarray(flat).tobytes()).hexdigest(),
                       "prompt": prompt, "size": size, "steps": steps, "seed": seed, "denoise": denoise}
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
        iou = alignment(repainted, mask)
        row = {"view": view, "style": style, "iou_with_render": round(iou, 4),
               "seconds": round(time.time() - started), "reused": reused, "accepted": iou >= min_iou}
        rows.append(row)
        if row["accepted"]:
            # The render silhouette marks the figure: the repaint's colours only count where the mesh is.
            image = np.dstack([np.asarray(repainted), np.where(mask, 255, 0).astype(np.uint8)])
            painted = out / f"{view}_paint.png"
            Image.fromarray(image).save(painted)
            accepted.append({"name": view, "image": str(painted.resolve()), "map": mapping, "weight": weight})
    manifest = {"schema_version": 1, "views": rows, "extra": accepted}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def default_views(elevation: float = 35.0) -> list[str]:
    """The four 3/4 directions at eye level, and four from above (tops of the head and tail)."""
    return ([f"az{a}_el0" for a in (45, 135, -135, -45)]
            + [f"az{a}_el{round(elevation)}" for a in (0, 90, 180, -90)])

