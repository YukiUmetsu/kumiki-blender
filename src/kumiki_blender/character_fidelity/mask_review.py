"""Conservative mask reconciliation before character-to-Blender fidelity checks.

Retain model proposals, mark ambiguous part ownership, and require explicit
source-layer masks or manually provided ownership masks to assign overlap.
Never manufacture approved semantic labels from SAM confidence alone.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .schema import CATEGORIES, asset


def _mask(path: Path, dimensions: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != dimensions:
            raise ValueError(f"Part mask must match full source canvas {dimensions}: {path}")
        if "A" in image.getbands():
            array = np.asarray(image.getchannel("A"))
        else:
            array = np.asarray(image.convert("L"))
        return array > 127


def _mapping(data: dict[str, Any], field: str, ids: set[str], root: Path,
             dimensions: tuple[int, int]) -> dict[str, np.ndarray]:
    specs = data.get(field, {})
    if not isinstance(specs, dict) or not set(specs).issubset(ids):
        raise ValueError(f"{field} must map known part IDs to file paths")
    return {name: _mask(asset(root, path), dimensions) for name, path in specs.items()}


def refine_masks(image: str | Path, segmentation: str | Path, output: str | Path,
                 *, rules: str | Path | None = None) -> dict[str, Any]:
    """Write a new SAM-compatible revision, plus unambiguous masks and audit.

    Rules JSON (schema_version 1):
      source_masks: {part_id: full-canvas source mask path}
      ownership_masks: {part_id: full-canvas hand-annotated *ownership* path}
    Source masks override SAM proposals for their part. Ownership masks only
    resolve multiply claimed pixels; unresolved pixels are omitted from ALL
    *analysis* masks and saved separately. Raw inputs are always preserved.
    """
    image, segmentation, output = Path(image), Path(segmentation), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be a new directory")
    if output.resolve().is_relative_to(segmentation.resolve()):
        raise ValueError("Output must be outside the original SAM revision")
    if image.is_symlink() or not image.is_file():
        raise ValueError("Source image must be a regular file")
    with Image.open(image) as source:
        if source.width * source.height > 16_777_216:
            raise ValueError("Image exceeds 16 megapixels")
        dimensions = source.size
        preview = source.convert("RGBA")
        background = Image.new("RGBA", dimensions, "white")
        background.alpha_composite(preview)
        preview = background.convert("RGB")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    original = json.loads(asset(segmentation, "report.json").read_text(encoding="utf-8"))
    if (original.get("provider") != "sam2.1-local" or original.get("schema_version") != 1 or
            original.get("image_sha256") != digest):
        raise ValueError("SAM report must match this exact source image SHA-256")
    entries = original.get("parts")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
        raise ValueError("SAM report must contain 1 to 32 part entries")
    ids, masks = [], []
    for part in entries:
        if (not isinstance(part, dict) or not isinstance(part.get("part"), str) or
                not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", part["part"]) or
                part.get("category") not in CATEGORIES or part.get("part") in ids):
            raise ValueError("Invalid or duplicate SAM part ID/category")
        ids.append(part["part"])
        masks.append(_mask(asset(segmentation, part.get("proposed_mask")), dimensions))
    spec = {}
    if rules is not None:
        rules = Path(rules)
        if rules.is_symlink() or not rules.is_file():
            raise ValueError("Mask rules must be a regular JSON file")
        spec = json.loads(rules.read_text(encoding="utf-8"))
        if (not isinstance(spec, dict) or spec.get("schema_version") != 1 or
                (spec.get("source_image_sha256") is not None and spec["source_image_sha256"] != digest)):
            raise ValueError("Mask rules schema/source digest mismatch")
        if set(spec) - {"schema_version", "source_image_sha256", "source_masks", "ownership_masks"}:
            raise ValueError("Unknown mask-review rule")
    overrides = _mapping(spec, "source_masks", set(ids), rules.parent if rules else image.parent, dimensions)
    owners = _mapping(spec, "ownership_masks", set(ids), rules.parent if rules else image.parent, dimensions)
    for index, name in enumerate(ids):
        if name in overrides:
            masks[index] = overrides[name]
    claimed = np.stack(masks).astype(bool)
    contested = claimed.sum(axis=0) > 1
    ownership = np.stack([owners.get(name, np.zeros(claimed.shape[1:], bool)) for name in ids])
    if np.any((ownership.sum(axis=0) > 1) & contested):
        raise ValueError("Ownership masks assign one ambiguous pixel to multiple parts")
    resolved = contested & (ownership.sum(axis=0) == 1)
    # Assign explicit ownership only for contested pixels, preserving regular
    # segmentation elsewhere. Manual ownership may correct an absent SAM claim.
    exclusive = claimed & ~contested[None, :, :]
    exclusive |= ownership & resolved[None, :, :]
    ambiguous = contested & ~resolved
    assert not np.any(exclusive.sum(axis=0) > 1)
    overlaps = []
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            shared = int(np.count_nonzero(claimed[i] & claimed[j]))
            if shared:
                overlaps.append({"first": a, "second": ids[j], "shared_pixels": shared,
                    "fraction_of_smaller_part": round(shared / max(1, min(int(claimed[i].sum()), int(claimed[j].sum()))), 6),
                    "status": "semantic-overlap-review-required"})
    destination_rows = []
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("masks", "raw_masks"):
        (output / folder).mkdir()
    for i, part in enumerate(entries):
        name = ids[i]
        Image.fromarray(np.uint8(claimed[i]) * 255).save(output / "raw_masks" / f"{name}.png")
        Image.fromarray(np.uint8(exclusive[i]) * 255).save(output / "masks" / f"{name}.png")
        lost = int(np.count_nonzero(claimed[i] & ambiguous))
        assigned = int(np.count_nonzero(exclusive[i] & resolved))
        destination_rows.append({**part, "proposed_mask": f"masks/{name}.png",
            "original_mask": f"raw_masks/{name}.png", "approved": False,
            "mask_review": {"provenance": "full-canvas-source-mask" if name in overrides else "sam-unapproved",
                            "unresolved_pixels_excluded": lost, "explicit_owner_pixels": assigned,
                            "incomplete_shape_evidence": lost >= max(10, int(np.count_nonzero(claimed[i]) * .01))}})
    Image.fromarray(np.uint8(ambiguous) * 255).save(output / "ambiguous_pixels.png")
    visual = preview.copy()
    # Magenta flags ambiguity; green denotes explicit pixel ownership. The
    # underlying painting remains visible for quick checking.
    overlay = np.asarray(visual).copy()
    overlay[resolved] = (38, 180, 107)
    overlay[ambiguous] = (224, 30, 154)
    Image.fromarray(overlay).save(output / "ownership_review.png")
    summary = {"schema_version": 1, "provider": "sam2.1-local",
        "image_sha256": digest, "model": original.get("model"),
        "status": "unapproved-partial-evidence / review-required", "parts": destination_rows,
        "semantic_overlap_warnings": overlaps,
        "mask_review": {"original_provider": original.get("provider"),
            "total_ambiguous_pixels": int(ambiguous.sum()), "explicitly_resolved_pixels": int(resolved.sum()),
            "raw_contested_pixels": int(contested.sum()), "source_overrides": sorted(overrides),
            "ownership_annotations": sorted(owners), "requires_review": True},
        "limitations": ["Unresolved overlaps are EXCLUDED from analysis masks and retained in raw masks.",
            "This can alter intrinsic silhouettes; do not make sculpt corrections from incomplete source evidence.",
            "Source-layer/hand annotations are not certified or automatically approved."]}
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "review.md").write_text(
        "# Semantic mask review\n\nMagenta: ambiguous ownership; green: explicit ownership. "
        "Raw proposals are preserved.\n\n" +
        f"- Contested pixels: {int(contested.sum())}\n" +
        f"- Explicitly owned: {int(resolved.sum())}\n" +
        f"- Remaining ambiguous (excluded): {int(ambiguous.sum())}\n\n" +
        "Review each mask before relying on any shape or placement metrics.\n", encoding="utf-8")
    return summary
