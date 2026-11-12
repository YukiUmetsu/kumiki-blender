"""Prompt-guided character sheet part masks using a local Meta SAM 2.1 checkpoint.

No Hugging Face APIs or network calls. Predicted masks are proposals only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
}
PART_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
CATEGORIES = {"anatomy", "costume", "accessory", "prop"}


def regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing, non-symlink file: {path}")
    return path.resolve()


def load_prompts(path: Path, size: tuple[int, int]) -> dict:
    data = json.loads(regular_file(path, "Prompts").read_text(encoding="utf-8"))
    parts = data.get("parts") if isinstance(data, dict) else None
    if data.get("schema_version") != 1 or not isinstance(parts, list) or not 1 <= len(parts) <= 32:
        raise ValueError("Expected schema_version=1 and 1 to 32 parts")
    width, height = size
    used = set()
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("Part must be an object")
        name = part.get("id")
        if not isinstance(name, str) or not PART_ID.fullmatch(name) or name in used:
            raise ValueError("Part IDs must be unique and filename safe")
        used.add(name)
        if part.get("category") not in CATEGORIES:
            raise ValueError(f"Invalid category for {name}")
        pos, neg = part.get("positive_points", []), part.get("negative_points", [])
        if not isinstance(pos, list) or not isinstance(neg, list) or len(pos) + len(neg) > 64:
            raise ValueError("Invalid point prompts")
        for point in pos + neg:
            if (not isinstance(point, list) or len(point) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in point) or
                not 0 <= point[0] < width or not 0 <= point[1] < height):
                raise ValueError("Point must be a pixel inside the image")
        box = part.get("box")
        if box is not None and (
            not isinstance(box, list) or len(box) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or
            not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
        ):
            raise ValueError(f"Invalid xyxy box for {name}")
        if box is None and not pos:
            raise ValueError(f"Part {name} needs a foreground point or box")
    return data


def make_predictor(config: str, checkpoint: Path, device: str):
    import torch  # noqa: F401 -- installed only for optional model inference
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    # Important: never invoke SAM2ImagePredictor.from_pretrained().
    return SAM2ImagePredictor(build_sam2(config, str(checkpoint), device=device, mode="eval"))


def segment_sheet(image_path, prompt_path, checkpoint_path, output, *,
                  model="small", device="cpu", min_predicted_iou=0.80,
                  clip_to_alpha=True, predictor_factory=None) -> dict:
    if model not in CONFIGS or device not in {"cpu", "mps", "cuda"}:
        raise ValueError("Unsupported model or device")
    if not isinstance(min_predicted_iou, (int, float)) or not 0 <= min_predicted_iou <= 1:
        raise ValueError("Expected min_predicted_iou within [0,1]")
    image_path = regular_file(Path(image_path), "Image")
    prompt_path = regular_file(Path(prompt_path), "Prompts")
    checkpoint_path = regular_file(Path(checkpoint_path), "Checkpoint")
    out = Path(output)
    if out.exists() or out.is_symlink():
        raise ValueError("Output directory must be new")
    out = out.resolve()
    if out.is_relative_to(image_path.parent) or out.is_relative_to(checkpoint_path.parent):
        raise ValueError("Output must be outside source and checkpoint directories")
    with Image.open(image_path) as original:
        if original.width * original.height > 16_777_216:
            raise ValueError("Image exceeds sixteen megapixels")
        canvas = original.convert("RGB")
        alpha = np.asarray(original.getchannel("A")) >= 128 if "A" in original.getbands() else None
    prompts = load_prompts(prompt_path, canvas.size)
    factory = predictor_factory or make_predictor
    predictor = factory(CONFIGS[model], checkpoint_path, device)
    predictor.set_image(np.asarray(canvas))
    proposals = []
    results = []
    for part in prompts["parts"]:
        pos, neg = part.get("positive_points", []), part.get("negative_points", [])
        pts = pos + neg
        coords = np.asarray(pts, dtype=np.float32).reshape(-1, 2) if pts else None
        labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int32) if pts else None
        box = np.asarray(part["box"], dtype=np.float32) if "box" in part else None
        masks, scores, _ = predictor.predict(
            point_coords=coords, point_labels=labels, box=box, multimask_output=True)
        masks, scores = np.asarray(masks), np.asarray(scores)
        if (masks.ndim != 3 or masks.shape[1:] != (canvas.height, canvas.width)
                or len(masks) != len(scores) or not np.all(np.isfinite(scores))):
            raise ValueError(f"Unexpected SAM2 prediction for {part['id']}")
        index = int(np.argmax(scores))
        binary = masks[index].astype(bool)
        if clip_to_alpha and alpha is not None:
            binary &= alpha
        fraction = float(binary.mean())
        status = ("review-required" if float(scores[index]) < min_predicted_iou or
                  fraction < 1e-5 or fraction > 0.85 else "proposal-review-required")
        proposals.append((part["id"], binary))
        results.append({"part": part["id"], "category": part["category"],
                        "proposed_mask": f"masks/{part['id']}.png",
                        "predicted_iou": float(scores[index]),
                        "foreground_fraction": fraction,
                        "status": status, "approved": False})
    out.mkdir(parents=True, exist_ok=False)
    (out / "masks").mkdir()
    previews = []
    for i, (name, binary) in enumerate(proposals):
        Image.fromarray(binary.astype(np.uint8) * 255).save(out / "masks" / f"{name}.png")
        pane = canvas.copy()
        tint = Image.new("RGB", canvas.size, ((50 + 65 * i) % 255, 112, 240))
        pane.paste(tint, Image.fromarray(binary.astype(np.uint8) * 95))
        ImageDraw.Draw(pane).text((8, 8), name, fill="white", stroke_width=2, stroke_fill="black")
        previews.append(pane)
    preview = Image.new("RGB", (canvas.width * len(previews), canvas.height), "white")
    for i, pane in enumerate(previews):
        preview.paste(pane, (i * canvas.width, 0))
    preview.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
    preview.save(out / "review.jpg", quality=90)
    report = {"schema_version": 1, "provider": "sam2.1-local", "model": model,
              "device": device, "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
              "checkpoint": checkpoint_path.name, "clip_to_alpha": clip_to_alpha,
              "status": "unapproved-proposals / human-review-required", "parts": results,
              "limitations": ["Predicted IoU is not compared to artist-approved masks.",
                              "2D segmentation provides no verified depth or hidden geometry."]}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Segment artwork using local ungated SAM2.1")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=tuple(CONFIGS), default="small")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--min-predicted-iou", type=float, default=0.80)
    parser.add_argument("--no-alpha-clipping", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = segment_sheet(args.image, args.prompts, args.checkpoint, args.output,
                               model=args.model, device=args.device,
                               min_predicted_iou=args.min_predicted_iou,
                               clip_to_alpha=not args.no_alpha_clipping)
        print(f"{args.output}/report.json: {result['status']}")
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"character-fidelity/sam2: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
