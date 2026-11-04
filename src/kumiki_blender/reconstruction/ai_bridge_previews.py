"""Optional synthetic viewpoint *previews*, never camera-registration evidence.

Reuses PR #52's local-only ComfyUI image adapter. Its one-image workflow cannot
prove that an intermediate camera or true unseen Sphinx surface exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from .comfyui_auto_workflow import create_auto_bridge_workflow
from .completion_image_generation import generate_comfyui_image
from .references import ReferenceError


def _workflow_nodes(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ReferenceError("AI bridge workflow must be a local API-format JSON file")
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ReferenceError("Invalid ComfyUI workflow JSON") from error
    if not isinstance(graph, dict):
        raise ReferenceError("ComfyUI API workflow must be a node mapping")
    inputs = [key for key, node in graph.items()
              if isinstance(node, dict) and node.get("class_type") == "LoadImage"]
    outputs = [key for key, node in graph.items()
               if isinstance(node, dict) and node.get("class_type") == "SaveImage"]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ReferenceError("Bridge preview workflow needs exactly one LoadImage and one SaveImage node")
    return inputs[0], outputs[0]


def _verified_staged_photo(sparse: Path, item: dict[str, Any]) -> tuple[str, str] | None:
    filename = item.get("filename")
    digest = item.get("sha256")
    if (not isinstance(filename, str) or filename in (".", "..")
            or Path(filename).name != filename or not isinstance(digest, str)
            or len(digest) != 64):
        return None
    path = sparse / "images" / filename
    if path.is_symlink() or not path.is_file():
        return None
    with path.open("rb") as stream:
        calculated = hashlib.file_digest(stream, "sha256").hexdigest()
    if calculated != digest:
        raise ReferenceError(f"Staged photo changed after camera registration: {filename}")
    return filename, digest


def build_ai_bridge_previews(
    sparse: Path, coverage: dict[str, Any], output: Path, *,
    workflow: Path | None = None, server: str = "http://127.0.0.1:8188",
    require_mps: bool = True, limit: int = 3, asset_id: str = "unknown-asset",
    auto_workflow: bool = True,
    generator: Callable[..., dict[str, Any]] = generate_comfyui_image,
    workflow_factory: Callable[..., dict[str, Any]] = create_auto_bridge_workflow,
) -> dict[str, Any]:
    """Propose or generate private artist-only images from real source views."""
    if not 1 <= limit <= 10:
        raise ReferenceError("AI bridge preview limit must be between 1 and 10")
    if output.is_symlink() or output.exists() or output.resolve().is_relative_to(sparse.resolve()):
        raise ReferenceError("AI bridge previews require an unused output outside sparse data")
    index_path = sparse / "staged-reference-index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ReferenceError("Sparse photo index is missing or unsafe")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    items = index.get("images", [])
    if not isinstance(items, list):
        raise ReferenceError("Invalid source photo index")
    by_original = {entry.get("original_filename", entry.get("filename")): entry
                   for entry in items if isinstance(entry, dict)}
    suggested = []
    seen = set()
    for target in coverage.get("existing_photo_priorities", []):
        if not isinstance(target, dict) or target.get("kind") not in (
                "bridge-and-retry", "try-registration-first"):
            continue
        source = by_original.get(target.get("filename"))
        anchor = by_original.get(target.get("best_registered_bridge_photo"))
        if source is None or anchor is None:
            continue
        source_verified = _verified_staged_photo(sparse, source)
        anchor_verified = _verified_staged_photo(sparse, anchor)
        if source_verified is None or anchor_verified is None:
            continue
        filename, source_sha = source_verified
        anchor_filename, anchor_sha = anchor_verified
        if filename in seen or filename == anchor_filename:
            continue
        seen.add(filename)
        suggested.append({"target_photo": target["filename"],
                          "real_registered_anchor": target["best_registered_bridge_photo"],
                          "staged_photo": filename,
                          "registered_anchor_staged_photo": anchor_filename,
                          "registered_anchor_sha256": anchor_sha,
                          "source_photo_sha256": source_sha,
                          "verified_2d3d_candidates": target.get("existing_2d3d_candidates"),
                          "source_kind": "original-real-photo",
                          "limitations": "One-source AI image is a visual hypothesis, not calibrated intermediate camera evidence."})
        if len(suggested) >= limit:
            break
    saved_workflow = os.environ.get("KUMIKI_COMFYUI_BRIDGE_WORKFLOW")
    configured = workflow or (Path(saved_workflow) if saved_workflow else None)
    ignored_saved_workflow = None
    if workflow is None and configured is not None and (
            configured.is_symlink() or not configured.is_file()):
        ignored_saved_workflow = str(configured)
        configured = None
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "plan-only / no-workflow",
              "source_revision": str(sparse.resolve()),
              "origin": "synthetic / visual-reference-only",
              "distribution_allowed": False, "usable_as_sfm_input": False,
              "inputs_are_real_photographs": True, "requests": suggested,
              "generated": [], "workflow": str(configured) if configured else None,
              "limitations": "AI images may hallucinate geometry and cannot increase verified camera coverage."}
    if ignored_saved_workflow:
        report["warning"] = "Saved workflow is unavailable; trying local auto-discovery"
    if suggested and configured is None and auto_workflow:
        try:
            auto = workflow_factory(output / "auto-comfyui-workflow.json",
                                    asset_id=asset_id, server=server)
            configured = output / "auto-comfyui-workflow.json"
            report["workflow"] = str(configured)
            report["auto_workflow"] = auto
        except (OSError, ValueError) as error:
            report["status"] = "plan-only / automatic-workflow-unavailable"
            report["auto_workflow_error"] = str(error)[:250]

    def write_report():
        (output / "bridge-preview-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not suggested:
        report["status"] = "no-unregistered-photo-pairs"
    elif configured is not None:
        try:
            from PIL import Image, ImageOps
            image_node, output_node = _workflow_nodes(configured)
            for index, suggestion in enumerate(suggested, 1):
                filename = suggestion["staged_photo"]
                raw = sparse / "images" / filename
                with Image.open(raw) as source:
                    if source.width * source.height > 32_000_000:
                        raise ValueError("Synthetic preview source exceeds the 32-megapixel budget")
                    preview = ImageOps.exif_transpose(source).convert("RGB")
                    preview.thumbnail((960, 960))
                    # Conservative 64-pixel dimensions work with standard VAE
                    # downsampling and reduce peak MPS memory requirements.
                    width = max(64, preview.width // 64 * 64)
                    height = max(64, preview.height // 64 * 64)
                    preview = preview.resize((width, height), Image.Resampling.LANCZOS)
                    normalized = output / f"real-source-{index:02}.png"
                    preview.save(normalized, "PNG")
                result = generator(normalized, configured, output / f"synthetic-{index:02}",
                                   image_node=image_node, output_node=output_node,
                                   server=server, require_mps=require_mps)
                report["generated"].append({
                    "request": index, "provenance": result,
                    "candidate_path": f"synthetic-{index:02}/candidate.png",
                    "usable_as_sfm_input": False})
            report["status"] = "synthetic-previews-created / artist-review-required"
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            report["status"] = "synthetic-preview-generation-failed / plan-preserved"
            report["error"] = str(error)[:350]
    write_report()
    return report
