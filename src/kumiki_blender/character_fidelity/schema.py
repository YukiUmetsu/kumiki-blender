"""Validated, character-agnostic reference and mesh-part mapping contract."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

CATEGORIES = frozenset({"anatomy", "costume", "accessory", "prop"})
_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def asset(root: Path, name: str) -> Path:
    """Resolve only ordinary manifest-relative file paths, never traversal or links."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f"Expected relative asset path: {name!r}")
    relative = Path(name)
    if any(segment in ("..", "") for segment in relative.parts):
        raise ValueError(f"Unsafe asset path: {name!r}")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Missing, linked, or external asset: {name!r}")
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load the common contract used by the bpy exporter and image evaluator."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Manifest must be a regular JSON file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected character fidelity schema_version=1")
    if data.get("unassigned_behavior", "hide") not in ("hide", "occlude"):
        raise ValueError("unassigned_behavior must be hide or occlude")
    views, parts = data.get("views"), data.get("parts")
    if not isinstance(views, list) or not views or not isinstance(parts, list) or not parts:
        raise ValueError("Manifest must declare at least one view and one part")
    if len(views) > 12 or len(parts) > 24:
        raise ValueError("At most 12 views and 24 semantic parts per capture")
    view_ids, part_ids = set(), set()
    for view in views:
        if not isinstance(view, dict) or not _ID.fullmatch(str(view.get("id", ""))):
            raise ValueError("Invalid view id")
        vid = view["id"]
        if vid in view_ids:
            raise ValueError(f"Duplicate view: {vid}")
        view_ids.add(vid)
        if not isinstance(view.get("camera"), str) or not view["camera"].strip():
            raise ValueError(f"View {vid} requires a Blender camera name")
        if type(view.get("width")) is not int or type(view.get("height")) is not int or not (64 <= view["width"] <= 4096 and 64 <= view["height"] <= 4096):
            raise ValueError(f"Invalid image dimensions for {vid}")
        asset(path.parent, view.get("reference_image"))
        alignment = view.get("alignment")
        if not isinstance(alignment, dict) or type(alignment.get("verified")) is not bool:
            raise ValueError(f"View {vid} needs explicit alignment.verified")
        affine = alignment.get("source_to_render")
        if (not isinstance(affine, list) or len(affine) != 2 or
                any(not isinstance(row, list) or len(row) != 3 for row in affine) or
                any(not isinstance(number, (int, float)) or isinstance(number, bool) or
                    not math.isfinite(number) for row in affine for number in row) or
                abs(affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0]) < 1e-8):
            raise ValueError(f"View {vid} needs invertible 2x3 source_to_render affine")
        hint = view.get("reference_depth_hint")
        if hint is not None:
            if not isinstance(hint, dict) or hint.get("provenance") not in ("estimated", "artist-annotated") or type(hint.get("near_is_small")) is not bool:
                raise ValueError("Reference depth hint must declare estimated/artist-annotated provenance and near_is_small")
            asset(path.parent, hint.get("path"))
        depth = view.get("reference_depth")
        if depth is not None:
            if not isinstance(depth, dict) or depth.get("provenance") not in ("measured", "validated-multiview") or depth.get("units") != "meters":
                raise ValueError("Depth ground truth must be measured/validated and in meters")
            asset(path.parent, depth.get("path"))
    for part in parts:
        if not isinstance(part, dict) or not _ID.fullmatch(str(part.get("id", ""))):
            raise ValueError("Invalid part id")
        pid = part["id"]
        if pid in part_ids:
            raise ValueError(f"Duplicate part: {pid}")
        part_ids.add(pid)
        if part.get("category") not in CATEGORIES:
            raise ValueError(f"Part {pid} needs anatomy/costume/accessory/prop category")
        patterns = part.get("objects")
        if not isinstance(patterns, list) or not patterns or any(not isinstance(p, str) or not p.strip() for p in patterns):
            raise ValueError(f"Part {pid} needs Blender object name glob patterns")
        masks = part.get("reference_masks")
        if not isinstance(masks, dict) or not masks or not set(masks).issubset(view_ids):
            raise ValueError(f"Part {pid} needs source mask paths keyed by valid view ids")
        for name in masks.values():
            asset(path.parent, name)
        skipped = part.get("unmatched_pose_views", [])
        if not isinstance(skipped, list) or not set(skipped).issubset(view_ids):
            raise ValueError(f"Invalid unmatched_pose_views for {pid}")
    return data
