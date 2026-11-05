"""Isolated optional rembg inference, never imported by the SAM2/MPS worker.

ONNXRuntime and PyTorch can load incompatible OpenMP runtimes in one macOS
process, causing a fatal SIGABRT. Keep this small helper in its own process;
its exit code is advisory, never permission to discard original photo pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


def run(image: Path, checkpoint: Path, output: Path) -> None:
    from PIL import Image
    from .masking import _ISNET_SHA256, _rembg_mask, _verified_model_sha
    from .photo_preprocess_cache import _atomic_image

    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("IS-Net model checkpoint is missing or unsafe")
    with checkpoint.open("rb") as weights:
        checksum = hashlib.file_digest(weights, "sha256").hexdigest()
    if checksum != _ISNET_SHA256:
        raise ValueError("IS-Net model checkpoint is not the verified cached model")
    if (image.is_symlink() or not image.is_file() or
            output.is_symlink() or output.parent.is_symlink() or
            not output.parent.is_dir()):
        raise ValueError("Unsafe or missing source/target for auxiliary mask")

    # Use exactly these offline weights. This process must not import torch or
    # SAM2; it is independently terminated if ONNXRuntime hits a native error.
    os.environ["REMBG_HOME"] = str(checkpoint.parent)
    os.environ["U2NET_HOME"] = str(checkpoint.parent)
    from rembg import new_session  # type: ignore[import-not-found]
    session = new_session("isnet-general-use")
    if _verified_model_sha(session) != _ISNET_SHA256:
        raise ValueError("IS-Net model identity changed during auxiliary inference")
    with Image.open(image) as source:
        rgb = source.convert("RGB")
        mask = _rembg_mask(rgb, session)
    _atomic_image(Image.fromarray(mask, mode="L"), output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.image, args.checkpoint, args.output)
    except Exception as error:
        print(f"Isolated IS-Net inference failed: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
