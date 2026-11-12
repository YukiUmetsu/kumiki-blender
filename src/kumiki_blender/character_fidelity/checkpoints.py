"""Read-only local structural inspection of SAM 2.1 and MoGe-2 checkpoints.

Uses torch.load(weights_only=True); never invokes neural model code or a model
registry. Checkpoint metadata does not prove inference accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_checkpoint(path: str | Path) -> dict:
    import torch

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Checkpoint must be a local regular file: {source}")
    obj = torch.load(str(source), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(obj, dict) or not isinstance(obj.get("model"), dict):
        raise ValueError("Expected a tensor state dictionary under 'model'")
    state = obj["model"]
    if "model_config" in obj and "encoder.backbone.patch_embed.proj.weight" in state:
        config = obj["model_config"]
        if not isinstance(config, dict):
            raise ValueError("MoGe model configuration must be a dictionary")
        return {"status": "recognized", "family": "moge-2",
                "backbone": config.get("encoder", {}).get("backbone"),
                "normal_head": any(k.startswith("normal_head.") for k in state),
                "weights": len(state), "filename": source.name,
                "requires_huggingface_approval": False}
    key = "image_encoder.trunk.patch_embed.proj.weight"
    if key in state and "maskmem_tpos_enc" in state:
        dim = int(state[key].shape[0])
        variant = {112: "base-plus", 144: "large"}.get(dim,
            "tiny-or-small" if dim == 96 else "unknown")
        return {"status": "recognized", "family": "sam2.1", "variant": variant,
                "patch_embedding_channels": dim, "weights": len(state),
                "filename": source.name, "requires_huggingface_approval": False}
    raise ValueError("Checkpoint architecture is not recognized as SAM 2.1 or MoGe 2")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect local weights without remote access")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        result = [inspect_checkpoint(p) for p in args.paths]
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
