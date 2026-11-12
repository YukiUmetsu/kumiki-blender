"""Estimated relative depth from a local MoGe-2 checkpoint, never ground truth.

No automatic network access or Hugging Face account/approval. Requires the
upstream moge package and weights already downloaded to a local file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def relative_depth(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Normalize valid near-small depth robustly, preserving invalids as NaN."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("Depth map must have two dimensions")
    good = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        if np.asarray(valid).shape != depth.shape:
            raise ValueError("Depth and validity masks have different shapes")
        good &= np.asarray(valid, dtype=bool)
    if int(good.sum()) < 10:
        raise ValueError("Insufficient valid depth pixels")
    lo, hi = np.percentile(depth[good], [2, 98])
    if float(hi - lo) < 1e-6:
        raise ValueError("Insufficient depth variation")
    result = np.full(depth.shape, np.nan, dtype=np.float32)
    result[good] = np.clip((depth[good] - lo) / (hi - lo), 0, 1)
    return result


def _infer(rgb: np.ndarray, checkpoint: Path, device: str):
    import torch
    from moge.model.v2 import MoGeModel
    # A local filesystem path, not a registry name, is mandatory here.
    model = MoGeModel.from_pretrained(str(checkpoint)).to(device).eval()
    tensor = torch.as_tensor(rgb.copy(), device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
    with torch.inference_mode():
        return model.infer(tensor)


def estimate_depth_hint(image_path, checkpoint_path, output, *, device="cpu", infer=None) -> dict:
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("Unsupported device")
    image_path, checkpoint = Path(image_path), Path(checkpoint_path)
    if (image_path.is_symlink() or not image_path.is_file() or
            checkpoint.is_symlink() or not checkpoint.is_file()):
        raise ValueError("Image and locally downloaded checkpoint must be regular files")
    out = Path(output)
    if out.is_symlink() or out.exists():
        raise ValueError("Output directory must be new")
    out = out.resolve()
    if out.is_relative_to(image_path.resolve().parent) or out.is_relative_to(checkpoint.resolve().parent):
        raise ValueError("Output must be outside image and checkpoint directories")
    with Image.open(image_path) as source:
        if source.width * source.height > 16_777_216:
            raise ValueError("Source exceeds sixteen megapixels")
        rgb = np.asarray(source.convert("RGB"))
    result = (infer or _infer)(rgb, checkpoint.resolve(), device)

    def array(value):
        if hasattr(value, "detach"):
            return np.asarray(value.detach().cpu().numpy())
        return np.asarray(value)

    if "depth" not in result:
        raise ValueError("MoGe model did not return a depth map")
    predicted = array(result["depth"])
    if predicted.shape != rgb.shape[:2]:
        raise ValueError("Model returned depth with incorrect dimensions")
    valid = array(result["mask"]) if "mask" in result else None
    normalized = relative_depth(predicted, valid)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "depth_hint.npz", depth_relative=normalized)
    Image.fromarray(np.uint8(np.round(np.where(np.isfinite(normalized), normalized, 1) * 255))).save(
        out / "depth_hint.png")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {
        "schema_version": 1, "provider": "moge-2-local",
        "status": "estimated-only / review-required", "device": device,
        "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": digest.hexdigest(), "shape": list(normalized.shape),
        "manifest_reference_depth_hint": {"path": "depth_hint.npz",
                                          "provenance": "estimated", "near_is_small": True},
        "limitations": ["Single artwork does not establish true metric depth.",
                        "Only use as reference_depth_hint, never as measured reference_depth.",
                        "Comparison requires reviewed matching camera/pose registration."]
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Relative depth hints from local MoGe-2 weights")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        result = estimate_depth_hint(args.image, args.checkpoint, args.output, device=args.device)
        print(f"{args.output}/report.json: {result['status']}")
    except (OSError, ValueError, ImportError) as exc:
        print(f"character-fidelity/moge: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
