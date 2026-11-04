"""Local-only ComfyUI API adapter for masked completion-image candidates.

ComfyUI executes a user-owned API-format workflow with already-installed
checkpoints. No cloud inference, account, model-gating, or remote host is used.
The generated image is illustrative evidence only, never an observed photo.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class ImageGenerationError(ValueError):
    """Invalid ComfyUI transport, request, workflow or output."""


_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_PNG = b"\x89PNG\r\n\x1a\n"
_LOCAL_NODES = frozenset({
    "LoadImage", "SaveImage", "LoadImageMask", "CheckpointLoaderSimple",
    "CLIPTextEncode", "VAEEncode", "VAEEncodeForInpaint", "VAEDecode",
    "KSampler", "KSamplerAdvanced", "InpaintModelConditioning",
    "SetLatentNoiseMask", "EmptyLatentImage", "ImageScale", "ImageScaleBy",
    "ImageToMask", "MaskToImage", "InvertMask", "GrowMask",
    "ImageCompositeMasked", "LatentUpscale", "LatentUpscaleBy",
    "ControlNetLoader", "ControlNetApply", "ControlNetApplyAdvanced",
})


def validate_local_comfyui_url(server: str) -> str:
    """Network boundary: refuse non-loopback hosts and redirects to cloud."""
    if not isinstance(server, str):
        raise ImageGenerationError("ComfyUI URL must be a loopback HTTP URL")
    try:
        u = urlsplit(server)
        port = u.port
    except ValueError as error:
        raise ImageGenerationError("Invalid ComfyUI URL") from error
    if (u.scheme != "http" or u.hostname not in ("127.0.0.1", "localhost", "::1")
            or not port or not 1 <= port <= 65535 or u.username or u.password
            or u.path or u.query or u.fragment):
        raise ImageGenerationError("ComfyUI must be a local http://127.0.0.1:PORT server")
    return server


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        raise ImageGenerationError("ComfyUI HTTP redirects are forbidden")


class _LocalComfyClient:
    def __init__(self, server: str):
        self.server = validate_local_comfyui_url(server)
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _request(self, method: str, route: str, *, data: bytes | None = None,
                 content_type: str | None = None, max_bytes: int = 1024 * 1024) -> bytes:
        headers = {"Content-Type": content_type} if content_type else {}
        request = Request(self.server + route, data=data, method=method, headers=headers)
        try:
            with self.opener.open(request, timeout=20) as response:
                content = response.read(max_bytes + 1)
        except (HTTPError, URLError, TimeoutError) as error:
            raise ImageGenerationError(f"Local ComfyUI request failed: {error}") from error
        if len(content) > max_bytes:
            raise ImageGenerationError("ComfyUI response exceeds the size budget")
        return content

    def _json(self, method: str, route: str, data: bytes | None = None,
              content_type: str | None = None) -> Any:
        try:
            return json.loads(self._request(method, route, data=data, content_type=content_type))
        except (UnicodeError, ValueError) as error:
            raise ImageGenerationError("Invalid ComfyUI JSON response") from error

    def system_stats(self) -> Any:
        return self._json("GET", "/system_stats")

    def checkpoint_node_info(self) -> Any:
        """Read only the local checkpoint-loader schema, including installed names."""
        return self._json("GET", "/object_info/CheckpointLoaderSimple")

    def upload_image(self, path: Path) -> str:
        boundary = "kumiki" + uuid.uuid4().hex
        name = "kumiki-" + uuid.uuid4().hex + path.suffix.lower()
        content = path.read_bytes()
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            "Content-Type: image/png\r\n\r\n".encode("ascii"),
            content,
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n"
            f"--{boundary}--\r\n".encode("ascii"),
        ]
        result = self._json("POST", "/upload/image", b"".join(parts),
                            f"multipart/form-data; boundary={boundary}")
        received = result.get("name") if isinstance(result, dict) else None
        if received != name or result.get("subfolder") not in (None, ""):
            raise ImageGenerationError("ComfyUI returned an unexpected uploaded filename")
        return name

    def queue(self, workflow: dict[str, Any]) -> str:
        body = json.dumps({"prompt": workflow, "client_id": uuid.uuid4().hex}).encode("utf-8")
        result = self._json("POST", "/prompt", body, "application/json")
        prompt_id = result.get("prompt_id") if isinstance(result, dict) else None
        try:
            uuid.UUID(prompt_id)
        except (TypeError, ValueError, AttributeError) as error:
            raise ImageGenerationError("ComfyUI rejected the workflow or returned no prompt_id") from error
        return prompt_id

    def history(self, prompt_id: str) -> Any:
        return self._json("GET", "/history/" + prompt_id)

    def view_image(self, image: dict[str, str]) -> bytes:
        filename = image.get("filename")
        subfolder = image.get("subfolder", "")
        if (not isinstance(filename, str) or not filename.endswith(".png")
                or not _ID.fullmatch(filename[:-4]) or "/" in filename or "\\" in filename
                or not isinstance(subfolder, str) or subfolder not in ("", ".")):
            raise ImageGenerationError("ComfyUI output path must be a flat PNG filename")
        if image.get("type") != "output":
            raise ImageGenerationError("ComfyUI returned a non-output image")
        return self._request("GET", "/view?" + urlencode({"filename": filename,
                        "subfolder": "", "type": "output"}), max_bytes=32 * 1024 * 1024)


def _valid_workflow(path: Path, image_node: str, output_node: str) -> tuple[dict[str, Any], str]:
    if not _ID.fullmatch(image_node) or not _ID.fullmatch(output_node):
        raise ImageGenerationError("Workflow node IDs must be simple local identifiers")
    try:
        if path.stat().st_size > 256 * 1024:
            raise ImageGenerationError("Workflow exceeds the 256 KiB limit")
        raw = path.read_bytes()
        workflow = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as error:
        if isinstance(error, ImageGenerationError):
            raise
        raise ImageGenerationError(f"Cannot read ComfyUI API workflow: {error}") from error
    if not isinstance(workflow, dict) or not 2 <= len(workflow) <= 128:
        raise ImageGenerationError("Workflow must be an API-format node mapping")
    if (image_node not in workflow or output_node not in workflow
            or not all(isinstance(k, str) and _ID.fullmatch(k)
                       and isinstance(v, dict) and isinstance(v.get("class_type"), str)
                       and isinstance(v.get("inputs"), dict) for k, v in workflow.items())):
        raise ImageGenerationError("Invalid API-format workflow nodes")
    if workflow[image_node]["class_type"] != "LoadImage":
        raise ImageGenerationError("The image input node must be ComfyUI LoadImage")
    if workflow[output_node]["class_type"] != "SaveImage":
        raise ImageGenerationError("The output node must be ComfyUI SaveImage")
    unknown = sorted({node["class_type"] for node in workflow.values()}
                     - _LOCAL_NODES)
    if unknown:
        raise ImageGenerationError(
            f"Workflow uses unsupported or potentially network-capable nodes {unknown}; "
            "use built-in local image/inpainting nodes only"
        )
    return copy.deepcopy(workflow), hashlib.sha256(raw).hexdigest()


def _output_image(record: Any, output_node: str) -> dict[str, str]:
    try:
        if not isinstance(record, dict):
            raise KeyError("record")
        images = record["outputs"][output_node]["images"]
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
            raise KeyError("one image")
        return images[0]
    except (TypeError, KeyError) as error:
        raise ImageGenerationError("Expected exactly one image from the requested SaveImage node") from error


def _verify_png(content: bytes) -> None:
    if not content.startswith(_PNG) or len(content) > 32 * 1024 * 1024:
        raise ImageGenerationError("ComfyUI did not return a bounded PNG image")
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 16_000_000
        with Image.open(io.BytesIO(content)) as opened:
            if opened.format != "PNG" or opened.width * opened.height > 16_000_000:
                raise ImageGenerationError("ComfyUI PNG format or dimensions are invalid")
            opened.verify()
    except (ImportError, OSError, ValueError) as error:
        if isinstance(error, ImageGenerationError):
            raise
        raise ImageGenerationError(f"Cannot verify ComfyUI PNG: {error}") from error


def generate_comfyui_image(
    image: str | Path, workflow: str | Path, output: str | Path, *,
    image_node: str, output_node: str, server: str = "http://127.0.0.1:8188",
    require_mps: bool = False, timeout_seconds: float = 360,
    client: Any | None = None, sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run an explicitly supplied *local* ComfyUI workflow and pin its image."""
    validate_local_comfyui_url(server)
    if not 0 < timeout_seconds <= 3600:
        raise ImageGenerationError("Timeout must be between 0 and 3600 seconds")
    source = Path(image).resolve(strict=True)
    if (not source.is_file() or source.suffix.lower() != ".png"
            or source.stat().st_size > 20 * 1024 * 1024):
        raise ImageGenerationError("Input must be a private PNG no larger than 20 MiB")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    graph, graph_sha = _valid_workflow(Path(workflow), image_node, output_node)
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ImageGenerationError("Output revision already exists")
    if not destination.parent.is_dir():
        raise ImageGenerationError("Output parent directory does not exist")
    runtime = client or _LocalComfyClient(server)
    stats = runtime.system_stats()
    devices = stats.get("devices", []) if isinstance(stats, dict) else []
    mps = (isinstance(devices, list) and any(
        isinstance(item, dict) and "mps" in f"{item.get('name', '')} {item.get('type', '')}".lower()
        for item in devices))
    if require_mps and not mps:
        raise ImageGenerationError("ComfyUI did not confirm an Apple MPS device; refusing CPU execution")
    destination.mkdir(exist_ok=False)
    try:
        uploaded_name = runtime.upload_image(source)
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_digest:
            raise ImageGenerationError("Input image changed during upload")
        graph[image_node]["inputs"]["image"] = uploaded_name
        prompt_id = runtime.queue(graph)
        deadline = time.monotonic() + timeout_seconds
        while True:
            history = runtime.history(prompt_id)
            record = history.get(prompt_id) if isinstance(history, dict) else None
            if record is not None:
                status = record.get("status", {})
                if not isinstance(status, dict) or status.get("status_str") in ("error", "failed"):
                    raise ImageGenerationError("ComfyUI workflow failed; inspect local server logs")
                if status.get("completed") is True:
                    reference = _output_image(record, output_node)
                    content = runtime.view_image(reference)
                    _verify_png(content)
                    (destination / "candidate.png").write_bytes(content)
                    report = {
                        "schema_version": 1, "origin": "generated", "status": "artist-review-required",
                        "generator": {"adapter": "comfyui-local", "model_id": "workflow-dependent",
                                      "license": "review-checkpoint-license"},
                        "mps_verified": mps, "workflow_sha256": graph_sha,
                        "input_sha256": source_digest,
                        "output_file": "candidate.png",
                        "output_sha256": hashlib.sha256(content).hexdigest(),
                        "historical_evidence": False,
                        "limitations": "Inpainted/generated pixels are synthetic; never feed them to real-photo registration or photo-fidelity scoring.",
                    }
                    (destination / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    return report
            if time.monotonic() >= deadline:
                raise ImageGenerationError("ComfyUI workflow timed out; job may continue on the local server")
            sleep(min(1.0, max(0.01, deadline - time.monotonic())))
    except BaseException:
        shutil.rmtree(destination)
        raise
