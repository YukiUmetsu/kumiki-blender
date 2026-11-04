"""Ungated image-to-3D candidate generation and conservative photo-gap hints.

This does not promote a generated mesh to observational evidence. TripoSR is
run in-process, rather than through upstream run.py, because that script forces
CPU whenever CUDA is unavailable (including Apple Silicon MPS machines).
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Callable

from ..asset_import import AssetImportError, inspect_source_glb


class GenerationError(ValueError):
    """Invalid or failed inference, hardware, input, or output boundary."""


UNGATED_TRIPOSR = "stabilityai/TripoSR"
_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_REVISION = re.compile(r"(?:main|[0-9a-f]{40})\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_torch_device(torch_module: Any, requested: str = "auto") -> str:
    """Prefer Metal; never quietly fall back to CPU from a GPU request."""
    if requested not in ("auto", "mps", "cuda", "cpu"):
        raise GenerationError("device must be auto, mps, cuda, or cpu")
    mps = bool(getattr(getattr(torch_module.backends, "mps", None), "is_available", lambda: False)())
    cuda = bool(torch_module.cuda.is_available())
    if requested == "mps":
        if not mps:
            raise GenerationError("Apple Metal MPS device is not available")
        return "mps"
    if requested == "cuda":
        if not cuda:
            raise GenerationError("CUDA device is not available")
        return "cuda:0"
    if requested == "cpu":
        return "cpu"
    if mps:
        return "mps"
    if cuda:
        return "cuda:0"
    raise GenerationError("No accelerated Torch device available; use --device cpu explicitly")


def completion_runtime_doctor() -> dict[str, Any]:
    """Read-only local dependency and accelerator preflight; no model downloads."""
    import importlib.util

    modules = ("torch", "tsr", "numpy", "PIL", "omegaconf", "huggingface_hub",
               "skimage", "trimesh")
    available = {module: importlib.util.find_spec(module) is not None for module in modules}
    devices: list[str] = []
    if available["torch"]:
        try:
            import torch
            if torch.backends.mps.is_available():
                devices.append("mps")
            if torch.cuda.is_available():
                devices.append("cuda:0")
        except (OSError, RuntimeError, AttributeError):
            pass
    return {
        "model_id": UNGATED_TRIPOSR, "model_license": "MIT", "model_gated": False,
        "dependency_available": available, "accelerators": devices,
        "ready_for_accelerated_inference": all(available.values()) and bool(devices),
        "limitations": "Dependency readiness does not guarantee an upstream operator runs on MPS; no weights are downloaded by doctor.",
    }


def _prepare_image(path: Path) -> Any:
    """Require a segmented alpha input, then composite it on neutral gray."""
    try:
        import numpy as np
        from PIL import Image
        from tsr.utils import resize_foreground
    except ImportError as error:
        raise GenerationError(f"TripoSR image dependencies are missing: {error}") from error
    try:
        Image.MAX_IMAGE_PIXELS = 16_000_000
        with Image.open(path) as opened:
            if opened.width * opened.height > 16_000_000:
                raise GenerationError("Image exceeds the 16-megapixel limit")
            rgba = opened.convert("RGBA")
        extrema = rgba.getchannel("A").getextrema()
        if extrema is None or extrema[0] == 255 or extrema[1] == 0:
            raise GenerationError("Provide a segmented PNG/WebP with non-empty transparent background")
        framed = resize_foreground(rgba, 0.85)
        pixels = np.asarray(framed, dtype=np.float32) / 255.0
        rgb = pixels[:, :, :3] * pixels[:, :, 3:4] + (1 - pixels[:, :, 3:4]) * 0.5
        return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    except (OSError, ValueError) as error:
        if isinstance(error, GenerationError):
            raise
        raise GenerationError(f"Cannot decode segmented image: {error}") from error


def _cpu_marching_cubes(level: Any, threshold: float) -> tuple[Any, Any]:
    """Avoid native CUDA-only marching-cubes builds on Apple Silicon."""
    import numpy as np
    import torch
    from skimage.measure import marching_cubes

    verts, faces, _normals, _values = marching_cubes(
        level.detach().float().cpu().numpy(), level=threshold,
    )
    return torch.from_numpy(np.ascontiguousarray(verts)), torch.from_numpy(
        np.ascontiguousarray(faces.astype(np.int64))
    )


def _enable_mps_marching_cubes_compat() -> None:
    """Supply TripoSR's tiny torchmcubes import API when CUDA bindings are absent."""
    # A preinstalled CUDA torchmcubes can itself fail to import on macOS.
    # The CLI is an isolated process, so prefer the portable shim on MPS.
    shim = types.ModuleType("torchmcubes")
    shim.marching_cubes = _cpu_marching_cubes  # type: ignore[attr-defined]
    sys.modules["torchmcubes"] = shim


def _resolve_public_model(snapshot_download: Any, revision: str, offline: bool) -> Path:
    """Never forward ambient HF credentials or request gated models."""
    return Path(snapshot_download(
        repo_id=UNGATED_TRIPOSR, revision=revision,
        allow_patterns=["config.yaml", "model.ckpt"],
        token=False, local_files_only=offline,
    )).resolve()


def _load_weights_only(torch: Any, tsr_class: Any, omegaconf: Any,
                       config: Path, weights: Path) -> Any:
    """Bypass upstream unsafe default torch.load of remote .ckpt objects."""
    config_data = omegaconf.load(config)
    omegaconf.resolve(config_data)
    model = tsr_class(config_data)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def _infer_triposr(image: Path, output: Path, *, device: str, revision: str,
                   resolution: int, chunk_size: int, offline: bool) -> dict[str, str]:
    """Load public MIT weights without an HF token or license acceptance flow."""
    try:
        import torch
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf
    except ImportError as error:
        raise GenerationError(
            f"TripoSR runtime is not installed: {error}. Install the optional upstream TripoSR "
            "runtime and PyTorch with MPS support on Apple Silicon."
        ) from error
    actual_device = choose_torch_device(torch, device)
    try:
        if actual_device == "mps" and platform.system() == "Darwin":
            _enable_mps_marching_cubes_compat()
        from tsr.system import TSR
        model_dir = _resolve_public_model(snapshot_download, revision, offline)
        config = model_dir / "config.yaml"
        weights = model_dir / "model.ckpt"
        if not config.is_file() or not weights.is_file():
            raise GenerationError("Incomplete TripoSR checkpoint (config.yaml/model.ckpt)")
        # Upstream TSR.from_pretrained uses torch.load without weights_only.
        # Load state tensors explicitly so a remote pickle cannot execute code.
        model = _load_weights_only(torch, TSR, OmegaConf, config, weights)
        model.renderer.set_chunk_size(chunk_size)
        model.to(actual_device)
        prepared = _prepare_image(image)
        with torch.no_grad():
            codes = model([prepared], device=actual_device)
            if actual_device == "mps":
                # No native Metal isosurface kernel. Keep inference on MPS;
                # use scikit-image on CPU for only the surface extraction.
                model.set_marching_cubes_resolution(resolution)
                model.isosurface_helper.mc_func = _cpu_marching_cubes
            meshes = model.extract_mesh(codes, True, resolution=resolution)
        if len(meshes) != 1 or meshes[0].is_empty:
            raise GenerationError("TripoSR returned no usable surface")
        meshes[0].export(str(output))
    except GenerationError:
        raise
    except Exception as error:
        raise GenerationError(f"TripoSR inference failed on {actual_device}: {error}") from error
    return {
        "device": actual_device,
        "revision": model_dir.name,
        "config_sha256": _file_sha256(config),
        "checkpoint_sha256": _file_sha256(weights),
    }


def generate_triposr_candidate(
    image: str | Path, output: str | Path, *, model_id: str = UNGATED_TRIPOSR,
    requested_device: str = "auto", revision: str = "main", resolution: int = 128,
    chunk_size: int = 2048, offline: bool = False,
    inference: Callable[..., dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Create an exclusive GLB+provenance revision; never copy the input photo."""
    if model_id != UNGATED_TRIPOSR:
        raise GenerationError("Only ungated MIT stabilityai/TripoSR is allowed by this adapter")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise GenerationError("Model revision must be main or a 40-character commit SHA")
    if requested_device not in ("auto", "mps", "cuda", "cpu"):
        raise GenerationError("Unknown accelerator")
    if type(resolution) is not int or resolution not in (64, 128, 192, 256):
        raise GenerationError("Resolution must be 64, 128, 192 or 256")
    if type(chunk_size) is not int or not 128 <= chunk_size <= 8192:
        raise GenerationError("Chunk size must be 128–8192")
    source = Path(image).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in (".png", ".webp"):
        raise GenerationError("Reference must be a local PNG or WebP")
    if source.stat().st_size > 20 * 1024 * 1024:
        raise GenerationError("Input image exceeds 20 MiB")
    digest = _file_sha256(source)
    destination = Path(output).absolute()
    if destination.is_symlink() or destination.exists():
        raise GenerationError("Output revision already exists")
    if not destination.parent.is_dir():
        raise GenerationError("Output parent directory does not exist")
    destination.mkdir(exist_ok=False)
    try:
        candidate = destination / "candidate.glb"
        metadata = (inference or _infer_triposr)(
            source, candidate, device=requested_device, revision=revision,
            resolution=resolution, chunk_size=chunk_size, offline=offline,
        )
        if _file_sha256(source) != digest:
            raise GenerationError("Source image changed during inference; candidate is not reproducible")
        if not isinstance(metadata, dict) or metadata.get("device") not in ("mps", "cuda:0", "cpu"):
            raise GenerationError("Inference did not report its actual compute device")
        if not all(isinstance(metadata.get(k), str) and _SHA.fullmatch(metadata[k])
                   for k in ("checkpoint_sha256", "config_sha256")):
            raise GenerationError("Inference did not pin its model checkpoint and config")
        if not isinstance(metadata.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", metadata["revision"]):
            raise GenerationError("Inference did not pin its resolved model revision")
        try:
            result = inspect_source_glb(candidate)
        except (AssetImportError, OSError) as error:
            raise GenerationError(f"Invalid generated GLB: {error}") from error
        report: dict[str, Any] = {
            "schema_version": 1, "origin": "generated", "status": "artist-review-required",
            "generator": {"adapter": "triposr-local", "model_id": model_id, "license": "MIT",
                          "revision": metadata["revision"],
                          "checkpoint_sha256": metadata["checkpoint_sha256"],
                          "config_sha256": metadata["config_sha256"]},
            "device": metadata["device"], "requested_device": requested_device,
            "input_sha256": digest, "candidate_file": "candidate.glb",
            "candidate_sha256": result["input_sha256"],
            "resolution": resolution, "chunk_size": chunk_size,
            "historical_evidence": False,
            "limitations": "Single-image inferred geometry. No verified missing-part fit or photo fidelity.",
        }
        (destination / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    except BaseException:
        # The exact directory was created exclusively by this invocation.
        shutil.rmtree(destination)
        raise


def suggest_completion_regions(audit: str | Path, *, min_views: int = 2) -> dict[str, Any]:
    """Turn reference-audit coverage gaps into review tasks, not geometry claims."""
    if type(min_views) is not int or not 1 <= min_views <= 10:
        raise GenerationError("min_views must be 1–10")
    path = Path(audit)
    try:
        if path.stat().st_size > 128 * 1024:
            raise GenerationError("Reference audit exceeds 128 KiB")
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        if isinstance(error, GenerationError):
            raise
        raise GenerationError(f"Cannot read reference audit: {error}") from error
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise GenerationError("Expected version-one reference-audit JSON")
    asset_id = data.get("asset_id")
    counts = data.get("region_counts")
    missing = data.get("missing_regions")
    views = data.get("missing_views")
    if (not isinstance(asset_id, str) or not _SLUG.fullmatch(asset_id)
            or not isinstance(counts, dict) or len(counts) > 64
            or not isinstance(missing, list) or len(missing) > 64
            or not isinstance(views, list) or len(views) > 64):
        raise GenerationError("Reference audit has invalid coverage fields")
    if (any(not isinstance(k, str) or not _SLUG.fullmatch(k)
            or type(v) is not int or not 0 <= v <= 10000 for k, v in counts.items())
            or any(not isinstance(k, str) or not _SLUG.fullmatch(k) for k in missing + views)):
        raise GenerationError("Reference audit has invalid coverage values")
    candidates = [
        {"region": region, "reference_count": counts.get(region, 0),
         "reason": ("no annotated reference" if region in missing else
                    f"fewer than {min_views} independent annotated references")}
        for region in sorted(set(missing) | {k for k, n in counts.items() if n < min_views})
    ]
    return {
        "schema_version": 1, "asset_id": asset_id, "status": "coverage-suggestions-only",
        "geometry_missing_verified": False, "missing_views": sorted(set(views)),
        "candidates": candidates,
        "limitations": "Metadata coverage is not camera visibility or a missing-mesh measurement; review real photos and geometry.",
    }
