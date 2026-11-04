"""Generate a local ComfyUI API workflow from *installed* checkpoints.

This is conservative image-to-image artwork, not calibrated novel-view
synthesis. It never downloads a model or sends photos to a remote endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .completion_image_generation import ImageGenerationError, _LocalComfyClient


_ASSET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_CHECKPOINT = re.compile(r"[A-Za-z0-9_. -]{1,200}\.(?:safetensors|ckpt)\Z", re.I)


def _checkpoint_choices(info: Any) -> list[str]:
    """Extract the required ckpt_name combo from ComfyUI's actual node schema."""
    if not isinstance(info, dict):
        return []
    try:
        values = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    except (KeyError, TypeError, IndexError):
        return []
    if not isinstance(values, list):
        return []
    return sorted({item for item in values
                   if isinstance(item, str) and _CHECKPOINT.fullmatch(item)})


def _choose_checkpoint(choices: list[str]) -> str:
    if not choices:
        raise ImageGenerationError(
            "ComfyUI reports no compatible local .safetensors/.ckpt checkpoint")

    def score(name: str) -> tuple[int, int, str]:
        lowered = name.lower()
        # Prefer lower-memory SD 1.5 checkpoints for Apple Silicon previews.
        familiar = (0 if any(word in lowered for word in (
            "sd15", "sd_1.5", "sd-1.5", "v1-5", "realisticvision", "dreamshaper"))
            else 1 if any(word in lowered for word in ("sdxl", "sd_xl", "juggernautxl"))
            else 2)
        return familiar, 0 if lowered.endswith(".safetensors") else 1, lowered

    return min(choices, key=score)


def _local_model_directories() -> tuple[Path, ...]:
    configured = [Path(value) / "models" / "checkpoints" for key in (
        "KUMIKI_COMFYUI_DIR", "COMFYUI_DIR") if (value := os.environ.get(key))]
    return tuple(configured + [
        Path.home() / "ComfyUI" / "models" / "checkpoints",
        Path.home() / "Documents" / "ComfyUI" / "models" / "checkpoints",
    ])


def _scan_local_checkpoints(directories: tuple[Path, ...]) -> list[str]:
    """Bounded fallback when ComfyUI is installed but its server is stopped."""
    for directory in directories:
        if directory.is_symlink() or not directory.is_dir():
            continue
        choices = []
        try:
            for path in directory.iterdir():
                if (not path.is_symlink() and path.is_file()
                        and _CHECKPOINT.fullmatch(path.name)
                        and path.stat().st_size > 1024 * 1024):
                    choices.append(path.name)
        except OSError:
            continue
        if choices:
            return sorted(set(choices))
    return []


def _workflow(checkpoint: str, asset_id: str) -> dict[str, dict[str, Any]]:
    subject = asset_id.replace("_", " ").replace("-", " ")
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "3": {"class_type": "VAEEncode", "inputs": {
            "pixels": ["1", 0], "vae": ["2", 2]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["2", 1], "text": (
                f"realistic photograph of {subject}, the same ancient monument, "
                "accurate sandstone carving, natural surface erosion, photographic lighting, "
                "faithful to the supplied reference image")}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["2", 1], "text": (
                "modern replica, invented body parts, fantasy ornament, illustration, "
                "plastic, text, watermark, duplicate monument")}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["3", 0], "seed": 1435229, "steps": 20,
            "cfg": 5.0, "sampler_name": "euler", "scheduler": "normal",
            "denoise": 0.35}},
        "7": {"class_type": "VAEDecode", "inputs": {
            "samples": ["6", 0], "vae": ["2", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {
            "images": ["7", 0], "filename_prefix": "kumiki_bridge_preview"}},
    }


def create_auto_bridge_workflow(destination: Path, *, asset_id: str,
                                server: str = "http://127.0.0.1:8188",
                                client: Any | None = None,
                                model_directories: tuple[Path, ...] | None = None) -> dict[str, Any]:
    """Create a new executable workflow using a locally discovered checkpoint.

    The caller owns the parent directory. No file is created on discovery
    failure, avoiding a deceptively usable workflow with a fake model name.
    """
    if not isinstance(asset_id, str) or not _ASSET.fullmatch(asset_id):
        raise ImageGenerationError("Invalid asset ID for ComfyUI workflow")
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ImageGenerationError("Auto workflow needs an unused file in an existing directory")
    runtime = client or _LocalComfyClient(server)
    try:
        choices = _checkpoint_choices(runtime.checkpoint_node_info())
        selection = "local-comfyui-checkpoint-list / filename-heuristic"
    except ImageGenerationError:
        choices = _scan_local_checkpoints(
            model_directories if model_directories is not None else _local_model_directories())
        selection = "known-local-model-directory / filename-heuristic"
    checkpoint = _choose_checkpoint(choices)
    graph = _workflow(checkpoint, asset_id)
    serialized = json.dumps(graph, indent=2) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return {"path": str(destination), "checkpoint": checkpoint,
            "workflow_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "selection": selection,
            "limitations": "Simple image-to-image preview, not a reliable novel camera view."}
