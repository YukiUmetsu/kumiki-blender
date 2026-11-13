"""Validated CharacterSpec v2 and immutable, dependency-aware local region plans.

This module deliberately has no Blender or third-party runtime dependency.
Reference masks are *registered full-frame masks*, not cropped body-part images.
No ML-inferred surface, symmetry, or camera alignment is automatically verified.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_VIEW = re.compile(r"^az-?\d+(?:\.\d+)?_el-?\d+(?:\.\d+)?$")
_VIEWS = frozenset({"front", "back", "left", "right", "front_left", "front_right",
                    "back_left", "back_right"})
_TYPES = frozenset({"body", "head", "eye", "eyelid", "muzzle", "mouth", "ear", "arm",
                    "wing", "feather", "hand", "leg", "foot", "tail", "clothing",
                    "hair", "accessory", "prop", "other"})
_ATTACHMENTS = frozenset({"separate", "bone-parent", "part-parent", "skinned"})
_CONFIDENCE = frozenset({"verified", "estimated", "unsupported"})
_GEOMETRY = frozenset({"separate", "join-guided", "weld-required"})
_TOP_KEYS = frozenset({"schema_version", "character_id", "height_m", "profile",
                       "reference_frame", "source_ref_manifest", "regions"})
_REGION_KEYS = frozenset({"id", "parent", "type", "reference_masks",
                          "verified_views", "source_confidence", "attachment",
                          "geometry_policy", "symmetry", "source_mesh",
                          "depends_on", "landmarks", "detail_critical"})


def _fail(message: str) -> None:
    raise ValueError(message)


def _safe_asset(root: Path, name: str) -> Path:
    """Only local regular files within the spec directory; reject linked paths."""
    if not isinstance(name, str) or not name or "\\" in name:
        _fail("Asset name must be a local POSIX-style relative path")
    relative = Path(name)
    if relative.is_absolute() or any(p in ("..", ".") for p in relative.parts):
        _fail(f"Unsafe asset path: {name!r}")
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root.resolve()):
        _fail(f"Asset escapes character source directory: {name!r}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _fail(f"Linked assets are not accepted: {name!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing character source: {name}")
    return candidate.resolve()


def _hash(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _valid_view(view: Any) -> bool:
    if not isinstance(view, str):
        return False
    if view in _VIEWS:
        return True
    if not _VIEW.fullmatch(view):
        return False
    try:
        return -89 <= float(view.split("_el", 1)[1]) <= 89
    except ValueError:
        return False


def _ident(name: Any, label: str) -> str:
    if not isinstance(name, str) or not _ID.fullmatch(name):
        _fail(f"{label}: expected a stable ID (a-z, 0-9, _, . or -)")
    return name


def _coords(point: Any, label: str) -> None:
    if (not isinstance(point, list) or len(point) != 3
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in point)):
        _fail(f"{label} must be three finite numeric coordinates")


def load_spec(path: str | Path) -> dict[str, Any]:
    """Validate a source spec, all local references, and the region dependency DAG."""
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - _TOP_KEYS:
        _fail("CharacterSpec must be an object containing only recognized fields")
    if raw.get("schema_version") != 2 or type(raw.get("schema_version")) is not int:
        _fail("CharacterSpec requires schema_version 2")
    _ident(raw.get("character_id"), "character_id")
    if (type(raw.get("height_m")) not in (float, int)
            or not math.isfinite(raw["height_m"]) or not 0.01 <= raw["height_m"] <= 100):
        _fail("height_m must be between 0.01 and 100")
    if (not isinstance(raw.get("profile"), str)
            or not raw["profile"].strip() or len(raw["profile"]) > 100):
        _fail("profile must be a nonempty rig profile identifier")
    frame = raw.get("reference_frame")
    if frame != {"width": 1600, "height": 1200, "figure_px": 1000}:
        _fail("reference_frame must match the existing registered 1600x1200, 1000px-height handoff")
    if "source_ref_manifest" in raw:
        _safe_asset(source.parent, raw["source_ref_manifest"])
    regions = raw.get("regions")
    if not isinstance(regions, list) or not regions or len(regions) > 256:
        _fail("regions must contain 1 to 256 parts")
    ids: set[str] = set()
    for region in regions:
        if not isinstance(region, dict) or set(region) - _REGION_KEYS:
            _fail("Region has unknown properties or is not an object")
        rid = _ident(region.get("id"), "region.id")
        if rid in ids:
            _fail(f"Duplicate region {rid}")
        ids.add(rid)
        if region.get("type") not in _TYPES:
            _fail(f"{rid}: unsupported anatomical type")
        if region.get("source_confidence") not in _CONFIDENCE:
            _fail(f"{rid}: source_confidence must be verified, estimated or unsupported")
        if region.get("geometry_policy", "separate") not in _GEOMETRY:
            _fail(f"{rid}: invalid geometry_policy")
        if type(region.get("detail_critical", False)) is not bool:
            _fail(f"{rid}: detail_critical must be boolean")
        masks = region.get("reference_masks", {})
        if not isinstance(masks, dict):
            _fail(f"{rid}: reference_masks must be a view-to-file mapping")
        for view, filename in masks.items():
            if not _valid_view(view) or not isinstance(filename, str) or not filename.lower().endswith(".png"):
                _fail(f"{rid}: invalid registered mask for {view}")
            _safe_asset(source.parent, filename)
        verified = region.get("verified_views", [])
        if (not isinstance(verified, list) or len(set(map(str, verified))) != len(verified)
                or any(v not in masks for v in verified)):
            _fail(f"{rid}: verified_views must be a unique subset of its reference_masks")
        if verified and region["source_confidence"] != "verified":
            _fail(f"{rid}: only verified evidence can supply registered-view measurements")
        if "source_mesh" in region:
            if not isinstance(region["source_mesh"], str) or not region["source_mesh"].lower().endswith((".glb", ".obj", ".ply")):
                _fail(f"{rid}: source_mesh must be OBJ, PLY or GLB")
            _safe_asset(source.parent, region["source_mesh"])
        attachment = region.get("attachment", {"mode": "separate"})
        if not isinstance(attachment, dict) or attachment.get("mode") not in _ATTACHMENTS:
            _fail(f"{rid}: invalid attachment")
        if set(attachment) - {"mode", "bone", "region"}:
            _fail(f"{rid}: invalid attachment field")
        if attachment["mode"] == "bone-parent" and not isinstance(attachment.get("bone"), str):
            _fail(f"{rid}: bone-parent needs a bone name")
        if attachment["mode"] == "part-parent" and not isinstance(attachment.get("region"), str):
            _fail(f"{rid}: part-parent needs another region id")
        if "parent" in region and region["parent"] is not None:
            _ident(region["parent"], f"{rid}.parent")
        deps = region.get("depends_on", [])
        if not isinstance(deps, list) or len(set(map(str, deps))) != len(deps):
            _fail(f"{rid}: depends_on must be a unique list")
        for dep in deps:
            _ident(dep, f"{rid}.depends_on")
        symmetry = region.get("symmetry")
        if symmetry is not None:
            if (not isinstance(symmetry, dict) or set(symmetry) != {"partner", "verified"}
                    or type(symmetry["verified"]) is not bool):
                _fail(f"{rid}: symmetry needs partner and explicit verified boolean")
            _ident(symmetry["partner"], f"{rid}.symmetry.partner")
        landmarks = region.get("landmarks", {})
        if not isinstance(landmarks, dict) or len(landmarks) > 128:
            _fail(f"{rid}: landmarks must be a bounded mapping")
        for label, coordinates in landmarks.items():
            _ident(label, f"{rid}.landmark")
            _coords(coordinates, f"{rid}.{label}")
    graph = region_graph(raw)
    for rid, links in graph.items():
        for dependency in links:
            if dependency not in ids or dependency == rid:
                _fail(f"{rid}: missing/self dependency {dependency}")
    by_id = {r["id"]: r for r in regions}
    for region in regions:
        partner = region.get("symmetry", {}).get("partner")
        if partner is not None and (partner not in by_id or partner == region["id"]):
            _fail(f"{region['id']}: invalid symmetry partner")
        if partner and region["symmetry"]["verified"]:
            other = by_id[partner].get("symmetry")
            if other != {"partner": region["id"], "verified": True}:
                _fail(f"{region['id']}: verified symmetry must be reciprocal")
    _topological(graph)
    return raw


def region_graph(spec: dict) -> dict[str, set[str]]:
    result = {}
    for region in spec["regions"]:
        rid = region["id"]
        deps = set(region.get("depends_on", []))
        if region.get("parent"):
            deps.add(region["parent"])
        attach = region.get("attachment", {})
        if attach.get("mode") == "part-parent":
            deps.add(attach["region"])
        result[rid] = deps
    return result


def _topological(graph: dict[str, set[str]]) -> list[str]:
    state: dict[str, int] = {}
    order: list[str] = []

    def visit(node: str) -> None:
        if state.get(node) == 1:
            _fail(f"Cycle detected at {node}")
        if state.get(node) == 2:
            return
        state[node] = 1
        for dep in sorted(graph[node]):
            visit(dep)
        state[node] = 2
        order.append(node)

    for rid in sorted(graph):
        visit(rid)
    return order


def affected_regions(spec: dict, changed_id: str) -> list[str]:
    """Return the changed region and all downstream dependents, topologically sorted."""
    graph = region_graph(spec)
    if changed_id not in graph:
        _fail(f"Unknown region {changed_id}")
    affected = {changed_id}
    while True:
        new = affected | {rid for rid, deps in graph.items() if deps & affected}
        if new == affected:
            break
        affected = new
    return [rid for rid in _topological(graph) if rid in affected]


def _sources(spec: dict, root: Path) -> dict[str, str]:
    paths = set()
    if spec.get("source_ref_manifest"):
        paths.add(spec["source_ref_manifest"])
    for region in spec["regions"]:
        paths.update(region.get("reference_masks", {}).values())
        if region.get("source_mesh"):
            paths.add(region["source_mesh"])
    return {name: _hash(_safe_asset(root, name)) for name in sorted(paths)}


def _write_new_folder(target: Path, name: str, payload: dict) -> None:
    if target.exists():
        raise FileExistsError(f"Output revision already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plan_character(spec_path: str | Path, output: str | Path) -> dict:
    source = Path(spec_path).resolve()
    spec = load_spec(source)
    paths = _sources(spec, source.parent)
    graph = region_graph(spec)
    plan = {
        "schema_version": 2,
        "character_id": spec["character_id"],
        "source_spec": str(source),
        "source_sha256": _hash(source),
        "source_assets_sha256": paths,
        "height_m": spec["height_m"],
        "profile": spec["profile"],
        "region_order": _topological(graph),
        "regions": {
            r["id"]: {
                "dependencies": sorted(graph[r["id"]]),
                "input_sha256": hashlib.sha256(json.dumps(
                    {"region": r, "assets": {name: paths[name]
                        for name in sorted(set(r.get("reference_masks", {}).values()) |
                                           ({r["source_mesh"]} if r.get("source_mesh") else set()))}},
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "verified_views": r.get("verified_views", []),
                "source_confidence": r["source_confidence"],
            } for r in spec["regions"]
        },
        "status": "planned-not-modeled",
    }
    _write_new_folder(Path(output), "character-plan.json", plan)
    return plan


def open_plan(folder: str | Path) -> tuple[dict, dict]:
    root = Path(folder)
    path = root / "character-plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing character plan: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 2:
        _fail("Only CharacterSpec v2 plans are supported")
    source = Path(plan["source_spec"])
    if _hash(source) != plan["source_sha256"]:
        _fail("Source spec changed: create a new character plan revision")
    spec = load_spec(source)
    if plan["source_assets_sha256"] != _sources(spec, source.parent):
        _fail("Reference/mesh inputs changed: create a new character plan revision")
    if (plan.get("character_id") != spec["character_id"]
            or plan.get("region_order") != _topological(region_graph(spec))):
        _fail("Character plan differs from its source spec")
    return plan, spec


def plan_region(character: str | Path, region_id: str, output: str | Path) -> dict:
    plan, spec = open_plan(character)
    if region_id not in plan["regions"]:
        _fail(f"Unknown region {region_id}")
    region = next(r for r in spec["regions"] if r["id"] == region_id)
    invalidated = affected_regions(spec, region_id)
    task = {
        "schema_version": 2,
        "character_plan": str((Path(character) / "character-plan.json").resolve()),
        "character_id": spec["character_id"],
        "region": region,
        "source_spec_sha256": plan["source_sha256"],
        "region_input_sha256": plan["regions"][region_id]["input_sha256"],
        "dependencies": plan["regions"][region_id]["dependencies"],
        "invalidate_on_change": invalidated,
        "preserve_unchanged": [rid for rid in plan["region_order"] if rid not in invalidated],
        "registered_views": region.get("verified_views", []),
        "status": "candidate-model-required",
    }
    _write_new_folder(Path(output), "region-task.json", task)
    return task
