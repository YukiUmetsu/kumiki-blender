"""MapAnything feed-forward inference, run as an isolated subprocess.

Torch and MapAnything are imported only here so the photo pipeline stays
importable without them. The parent process checks the saved arrays before
using any pose, depth or point. Device choice is ``cuda`` > ``mps`` > ``cpu``
when ``auto``; explicit requests fail rather than silently falling back.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def choose_device(requested: str, *, cuda_available: bool, mps_available: bool) -> str:
    """Pure device policy so it can be tested without torch."""
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {', '.join(DEVICE_CHOICES)}")
    if requested == "auto":
        if cuda_available:
            return "cuda"
        return "mps" if mps_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise ValueError("CUDA was requested but torch.cuda is not available")
    if requested == "mps" and not mps_available:
        raise ValueError("MPS was requested but torch.backends.mps is not available")
    return requested


def _to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def _run(input_path: Path, output: Path, meta_path: Path) -> int:
    # Some MapAnything ops are not implemented on MPS yet; fall back per-op.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import numpy as np
    import torch
    from PIL import Image

    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    request = json.loads(input_path.read_text(encoding="utf-8"))
    device = torch.device(choose_device(
        request["device"],
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available()))
    use_amp = device.type == "cuda"  # bf16 autocast is only trusted on CUDA here
    model = MapAnything.from_pretrained(request["model_id"]).to(device).eval()

    # MapAnything resizes and crops a batch to one shared aspect ratio, so every
    # photo is pasted onto the same square canvas at the offset the parent process
    # recorded. Predictions then stay on the canvas grid the parent expects.
    canvas_size = int(request["canvas"])
    views = []
    for entry in request["images"]:
        width, height = int(entry["width"]), int(entry["height"])
        x0, y0 = int(entry["x0"]), int(entry["y0"])
        with Image.open(entry["path"]) as image:
            rgb = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        canvas[y0:y0 + height, x0:x0 + width] = np.asarray(rgb)
        # MapAnything takes HWC uint8 in [0, 255] and normalizes internally.
        views.append({"img": torch.from_numpy(canvas)})
    processed = preprocess_inputs(views)
    with torch.inference_mode():
        predictions = model.infer(
            processed,
            memory_efficient_inference=True,
            use_amp=use_amp,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
    if len(predictions) != len(views):
        raise RuntimeError("MapAnything returned a different number of views")

    arrays = {}
    for index, prediction in enumerate(predictions):
        arrays[f"pose_{index}"] = _to_numpy(prediction["camera_poses"][0])
        arrays[f"K_{index}"] = _to_numpy(prediction["intrinsics"][0])
        arrays[f"pts_{index}"] = _to_numpy(prediction["pts3d"][0])
        arrays[f"depth_{index}"] = _to_numpy(prediction["depth_z"][0])[..., 0]
        arrays[f"mask_{index}"] = _to_numpy(prediction["mask"][0])[..., 0] > 0.5
    np.savez(output, **arrays)
    meta_path.write_text(json.dumps({
        "schema_version": 1,
        "model_id": request["model_id"],
        "device": device.type,
        "amp": "bf16" if use_amp else "fp32",
        "torch_version": torch.__version__,
        "views": len(predictions),
        "canvas": canvas_size,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mapanything_worker")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if arguments.output.suffix != ".npz":
        raise ValueError("--output must end with .npz")
    return _run(arguments.input, arguments.output, arguments.meta)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"MapAnything worker failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
