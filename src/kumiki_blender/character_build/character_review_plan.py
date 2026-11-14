"""Immutable multi-view character review plan; source art never auto-registered."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

VIEWS = ("front", "right", "back", "left", "front_right", "front_left",
         "back_right", "back_left", "top")
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}$")


def digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _check_images(value: dict, *, allowed_views: tuple[str, ...]) -> dict:
    if not isinstance(value, dict) or not set(value) <= set(allowed_views):
        raise ValueError("Reference view names must match approved orthographic review views")
    resolved = {}
    for view, filename in value.items():
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"{view}: reference path must be a nonempty string")
        path = Path(filename).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"{view}: reference must be a local PNG/JPEG/WEBP image")
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError(f"{view}: oversized reference image (>32MiB)")
        resolved[view] = {"path": str(path), "sha256": digest(path)}
    return resolved


def make_character_review_plan(
    blend: str | Path, *,
    frames: list[int], views: list[str],
    references: str | Path | None = None,
    armature: str = "CharacterRig",
    resolution: int = 720,
    margin: float = 1.18,
) -> dict:
    source = Path(blend).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".blend":
        raise ValueError("Character review requires an existing saved .blend")
    if not isinstance(armature, str) or not _SAFE_NAME.fullmatch(armature):
        raise ValueError("Invalid Blender armature object name")
    if (not isinstance(frames, list) or not 1 <= len(frames) <= 5 or
            any(type(f) is not int or not 0 <= f <= 10000 for f in frames) or
            len(set(frames)) != len(frames) or frames != sorted(frames)):
        raise ValueError("Specify 1-5 distinct, ascending Blender pose frames in [0,10000]")
    if (not isinstance(views, list) or not 1 <= len(views) <= 6 or
            any(not isinstance(v, str) or v not in VIEWS for v in views) or
            len(set(views)) != len(views)):
        raise ValueError("Specify 1-6 unique registered review camera directions")
    if type(resolution) is not int or not 256 <= resolution <= 2048:
        raise ValueError("Review resolution must be an integer in [256, 2048]")
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 1.05 <= margin <= 2:
        raise ValueError("Camera framing margin must be finite in [1.05,2]")
    source_views: dict = {}
    regions: list[dict] = []
    if references is not None:
        manifest = Path(references).expanduser().resolve()
        if not manifest.is_file():
            raise ValueError("Reference manifest file is missing")
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if (not isinstance(raw, dict) or set(raw) != {"schema_version", "views", "regions"}
                or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
            raise ValueError("Reference manifest requires version 1, views and regions only")
        source_views = _check_images(raw["views"], allowed_views=tuple(views))
        if source_views and "front" not in source_views:
            raise ValueError("Approved front reference must anchor any supplied full-body references")
        entries = raw["regions"]
        if not isinstance(entries, list) or len(entries) > 8:
            raise ValueError("Specify at most 8 named local anatomical review regions")
        used = set()
        for item in entries:
            if not isinstance(item, dict) or set(item) != {
                    "id", "object", "vertex_group", "views"}:
                raise ValueError("Region requires id, object, vertex_group and views only")
            identity, object_name, group = (
                item["id"], item["object"], item["vertex_group"])
            if (not isinstance(identity, str) or not _SAFE_NAME.fullmatch(identity)
                    or identity in used):
                raise ValueError("Region ids must be unique and filename-safe")
            used.add(identity)
            if not isinstance(object_name, str) or not _SAFE_NAME.fullmatch(object_name):
                raise ValueError("Region must name an exact Blender mesh object")
            if group is not None and (
                not isinstance(group, str) or not _SAFE_NAME.fullmatch(group)):
                raise ValueError("Region vertex_group must be a safe name or null")
            refs = _check_images(item["views"], allowed_views=tuple(views))
            regions.append({
                "id": identity, "object": object_name,
                "vertex_group": group, "references": refs,
            })
    plan = {
        "schema_version": 1,
        "source_blend": str(source), "source_sha256": digest(source),
        "armature": armature,
        "frames": frames, "views": views,
        "resolution": resolution, "margin": float(margin),
        "reference_views": source_views, "regions": regions,
        "reference_alignment": "unregistered-visual-review-only",
        "render_style": "workbench-neutral-clay",
    }
    validate_character_review_plan(plan)
    return plan


def validate_character_review_plan(plan: dict) -> None:
    required = {"schema_version", "source_blend", "source_sha256", "armature",
                "frames", "views", "resolution", "margin", "reference_views",
                "regions", "reference_alignment", "render_style"}
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("Review plan fields changed or were not specified")
    if plan["schema_version"] != 1 or type(plan["schema_version"]) is not int:
        raise ValueError("Unsupported review plan version")
    if (plan["reference_alignment"] != "unregistered-visual-review-only"
            or plan["render_style"] != "workbench-neutral-clay"):
        raise ValueError("Automatic source-art camera registration is not supported")
    path = Path(plan["source_blend"])
    if not path.is_absolute() or path.suffix.lower() != ".blend":
        raise ValueError("Source .blend path must be absolute")
    if not isinstance(plan["source_sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", plan["source_sha256"]):
        raise ValueError("Source SHA-256 malformed")
    if not isinstance(plan["armature"], str) or not _SAFE_NAME.fullmatch(plan["armature"]):
        raise ValueError("Armature name malformed")
    frames, views = plan["frames"], plan["views"]
    if (not isinstance(frames, list) or not 1 <= len(frames) <= 5 or
            any(type(v) is not int or not 0 <= v <= 10000 for v in frames) or
            frames != sorted(set(frames))):
        raise ValueError("Invalid ordered review frames")
    if (not isinstance(views, list) or not 1 <= len(views) <= 6 or
            any(v not in VIEWS for v in views) or len(views) != len(set(views))):
        raise ValueError("Invalid review views")
    if type(plan["resolution"]) is not int or not 256 <= plan["resolution"] <= 2048:
        raise ValueError("Review image resolution outside safe bounds")
    margin = plan["margin"]
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 1.05 <= margin <= 2:
        raise ValueError("Review framing margin invalid")
    if (not isinstance(plan["reference_views"], dict) or
            not set(plan["reference_views"]) <= set(views) or
            plan["reference_views"] and "front" not in plan["reference_views"]):
        raise ValueError("Review references need a verified front anchor")
    if not isinstance(plan["regions"], list) or len(plan["regions"]) > 8:
        raise ValueError("Invalid review region count")
    seen = set()
    for region in plan["regions"]:
        if not isinstance(region, dict) or set(region) != {
                "id", "object", "vertex_group", "references"}:
            raise ValueError("Region plan fields invalid")
        if (not isinstance(region["id"], str) or not _SAFE_NAME.fullmatch(region["id"])
                or region["id"] in seen):
            raise ValueError("Duplicate or unsafe review region")
        seen.add(region["id"])
        if (not isinstance(region["object"], str) or
                not _SAFE_NAME.fullmatch(region["object"])):
            raise ValueError("Region object name malformed")
        if (region["vertex_group"] is not None and
                (not isinstance(region["vertex_group"], str) or
                 not _SAFE_NAME.fullmatch(region["vertex_group"]))):
            raise ValueError("Region vertex group name malformed")
        if not isinstance(region["references"], dict) or not set(
                region["references"]) <= set(views):
            raise ValueError("Region reference view name invalid")
    items = [*plan["reference_views"].values(),
             *(ref for region in plan["regions"]
               for ref in region["references"].values())]
    for item in items:
        if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                or not isinstance(item["path"], str) or
                not Path(item["path"]).is_absolute() or
                not isinstance(item["sha256"], str) or not
                re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("Reference image origin lock malformed")


def confirm_review_sources(plan: dict) -> None:
    validate_character_review_plan(plan)
    for kind, item in (
        [("blend", {"path": plan["source_blend"], "sha256": plan["source_sha256"]})]
        + [(f"full:{name}", ref) for name, ref in plan["reference_views"].items()]
        + [(f"{region['id']}:{name}", ref)
           for region in plan["regions"]
           for name, ref in region["references"].items()]
    ):
        p = Path(item["path"])
        if not p.is_file() or digest(p) != item["sha256"]:
            raise ValueError(f"Source reference changed since review plan: {kind}")
