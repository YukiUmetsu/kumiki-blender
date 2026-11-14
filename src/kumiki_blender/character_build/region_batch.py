"""M1 incremental, resumable character region planning for limited-memory Blender.

A deterministic DAG/receipt scheduler, not an arbitrary subprocess runner.
Source and dependency SHA receipts prevent resuming stale eye/lid/wing tasks.
Heavy Blender execution is deliberately left to the user's existing workers.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

from .character_spec_v2 import affected_regions, open_plan, region_graph

_WORKER_RESERVE_GIB = 14
_HOST_RESERVE_GIB = 4


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(data: dict) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_worker_budget(max_workers: int, memory_gib: int) -> None:
    if type(max_workers) is not int or not 1 <= max_workers <= 4:
        raise ValueError("max-workers must be an integer from 1 to 4")
    if type(memory_gib) is not int or not 8 <= memory_gib <= 256:
        raise ValueError("memory-gib must be an integer between 8 and 256")
    allowed = max(0, (memory_gib - _HOST_RESERVE_GIB) // _WORKER_RESERVE_GIB)
    if allowed < max_workers:
        raise ValueError(
            f"Memory budget {memory_gib} GiB only supports {allowed} heavy Blender worker(s) "
            f"at {_WORKER_RESERVE_GIB} GiB each plus {_HOST_RESERVE_GIB} GiB host reserve")


def make_region_batch(character: str | Path, changed_regions: list[str], *,
                      max_workers: int = 1, memory_gib: int = 24) -> dict:
    _validate_worker_budget(max_workers, memory_gib)
    plan_dir = Path(character).resolve()
    plan, spec = open_plan(plan_dir)
    if (not isinstance(changed_regions, list) or not changed_regions
            or len(changed_regions) > 128
            or any(not isinstance(r, str) or r not in plan["regions"] for r in changed_regions)
            or len(set(changed_regions)) != len(changed_regions)):
        raise ValueError("changed_regions must be distinct valid source regions")
    order = plan["region_order"]
    roots = [name for name in order if name in set(changed_regions)]
    affected = set()
    reasons = {}
    for rid in roots:
        for target in affected_regions(spec, rid):
            affected.add(target)
            reasons.setdefault(target, []).append(rid)
    changed = [rid for rid in order if rid in affected]
    preserved = [rid for rid in order if rid not in affected]
    graph = region_graph(spec)
    tasks = []
    for rid in changed:
        deps = sorted(graph[rid])
        tasks.append({
            "region": rid,
            "source_region_sha256": plan["regions"][rid]["input_sha256"],
            "change_roots": sorted(reasons[rid]),
            "needs_rebuild": True,
            "scheduled_dependencies": [d for d in deps if d in affected],
            "preserved_dependencies": [d for d in deps if d not in affected],
            "verified_source_views": plan["regions"][rid]["verified_views"],
        })
    pending = set(changed)
    waves = []
    while pending:
        ready = [t["region"] for t in tasks
                 if t["region"] in pending
                 and not set(t["scheduled_dependencies"]) & pending]
        if not ready:
            raise ValueError("Source dependency graph has no valid batch scheduling order")
        wave = ready[:max_workers]
        waves.append(wave)
        pending.difference_update(wave)
    payload = {
        "schema_version": 1,
        "character_plan": str(plan_dir),
        "character_id": spec["character_id"],
        "source_spec_sha256": plan["source_sha256"],
        "change_roots": roots,
        "region_input_sha256": {
            rid: plan["regions"][rid]["input_sha256"] for rid in order
        },
        "max_heavy_blender_workers": max_workers,
        "memory_budget_gib": memory_gib,
        "per_worker_reservation_gib": _WORKER_RESERVE_GIB,
        "host_memory_reservation_gib": _HOST_RESERVE_GIB,
        "scheduled": changed,
        "preserve_unchanged": preserved,
        "tasks": tasks,
        "theoretical_waves": waves,
        "status": "source-pinned-incremental-execution-plan",
        "limitations": [
            "This is a deterministic batch plan, NOT a Blender executor.",
            "A preserved region remains trusted only with an explicit artifact SHA receipt.",
            "The wave plan is a maximum permitted concurrency ceiling, not actual measured RAM usage.",
            "A completed region does not imply source likeness, correct topology, or rig deformation.",
        ],
    }
    return {**payload, "batch_sha256": _signature(payload)}


def save_region_batch(character: str | Path, changed_regions: list[str],
                      output: str | Path, *, max_workers: int = 1,
                      memory_gib: int = 24) -> dict:
    result = make_region_batch(character, changed_regions,
                               max_workers=max_workers, memory_gib=memory_gib)
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"Batch plan output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target/"region-batch-plan.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _load_batch(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError("Invalid or oversized batch-plan file")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or saved.get("schema_version") != 1:
        raise ValueError("Region batch plan must use schema_version 1")
    computed = make_region_batch(
        saved["character_plan"], saved["change_roots"],
        max_workers=saved["max_heavy_blender_workers"],
        memory_gib=saved["memory_budget_gib"])
    if computed != saved:
        raise ValueError("Batch plan or source character evidence changed: regenerate the revision")
    return saved


def _receipt_artifact(base: Path, obj: dict, label: str) -> tuple[str, str]:
    if not isinstance(obj, dict) or set(obj) != {"artifact", "sha256"}:
        raise ValueError(f"{label}: requires exact artifact and sha256")
    filename = obj["artifact"]
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{label}: invalid artifact path")
    path = (base / filename).resolve()
    if _sha(path) != obj["sha256"]:
        raise ValueError(f"{label}: saved artifact changed since its receipt")
    return str(path), obj["sha256"]


def resume_region_batch(batch_plan: str | Path, receipts: str | Path | None,
                        output: str | Path) -> dict:
    """Only verified upstream source/part hashes unblock downstream work."""
    batch_file = Path(batch_plan).resolve()
    batch = _load_batch(batch_file)
    folder = Path(output).resolve()
    if folder.exists():
        raise FileExistsError(f"Batch status output already exists: {folder}")
    evidence = {"schema_version": 1, "batch_sha256": batch["batch_sha256"],
                "preserved": {}, "completed": []}
    root = batch_file.parent
    if receipts is not None:
        receipt_file = Path(receipts).resolve()
        if not receipt_file.is_file() or receipt_file.stat().st_size > 2_000_000:
            raise ValueError("Invalid or oversized batch receipts")
        evidence = json.loads(receipt_file.read_text(encoding="utf-8"))
        root = receipt_file.parent
    if (not isinstance(evidence, dict) or evidence.get("schema_version") != 1
            or evidence.get("batch_sha256") != batch["batch_sha256"]
            or not isinstance(evidence.get("preserved"), dict)
            or not isinstance(evidence.get("completed"), list)):
        raise ValueError("Resume receipts do not belong to this source-locked region batch")
    tasks = {t["region"]: t for t in batch["tasks"]}
    preserved = {}
    for rid, record in evidence["preserved"].items():
        if rid not in batch["preserve_unchanged"]:
            raise ValueError(f"Cannot reuse unrelated or scheduled baseline {rid}")
        preserved[rid] = _receipt_artifact(root, record, f"preserved {rid}")[1]
    completed = {}
    # Resume receipts may be saved in any order by independent workers.
    # Validate them in stable source-DAG order, never input JSON order.
    if any(not isinstance(row, dict) for row in evidence["completed"]):
        raise ValueError("Completed region receipts must be objects")
    scheduled_order = {rid: i for i, rid in enumerate(batch["scheduled"])}
    ordered_records = sorted(
        evidence["completed"],
        key=lambda row: scheduled_order.get(row.get("region"), len(scheduled_order)))
    for record in ordered_records:
        if not isinstance(record, dict) or set(record) != {
                "region", "artifact", "sha256", "dependency_sha256"}:
            raise ValueError("A completed region requires source-bound dependency hashes")
        rid = record["region"]
        if rid not in tasks or rid in completed:
            raise ValueError(f"Duplicate or unknown completed region: {rid}")
        if not isinstance(record["dependency_sha256"], dict):
            raise ValueError(f"{rid}: dependency_sha256 must map upstream regions to artifacts")
        artifact_sha = _receipt_artifact(
            root, {"artifact": record["artifact"], "sha256": record["sha256"]},
            f"completed {rid}")[1]
        upstream = tasks[rid]["scheduled_dependencies"] + tasks[rid]["preserved_dependencies"]
        if set(record["dependency_sha256"]) != set(upstream):
            raise ValueError(f"{rid}: incomplete or unrecognized upstream dependency receipts")
        for dep in upstream:
            original = (completed.get(dep) if dep in tasks else preserved.get(dep))
            if original is None or record["dependency_sha256"][dep] != original:
                raise ValueError(f"{rid}: upstream dependency {dep} was changed or not completed")
        completed[rid] = artifact_sha
    pending = set(batch["scheduled"]) - set(completed)
    ready = []
    waiting = {}
    for rid in batch["scheduled"]:
        if rid not in pending:
            continue
        task = tasks[rid]
        missing = [
            d for d in task["scheduled_dependencies"]
            if d not in completed
        ] + [
            d for d in task["preserved_dependencies"]
            if d not in preserved
        ]
        if missing:
            waiting[rid] = missing
        else:
            ready.append(rid)
    dispatched = ready[:batch["max_heavy_blender_workers"]]
    result = {
        "schema_version": 1,
        "character_id": batch["character_id"],
        "batch_sha256": batch["batch_sha256"],
        "source_spec_sha256": batch["source_spec_sha256"],
        "status": ("complete-awaiting-source-and-animation-qa" if not pending
                   else "ready-for-external-execution" if dispatched
                   else "blocked-by-upstream-evidence"),
        "completed": list(completed),
        "preserved_receipts_verified": sorted(preserved),
        "pending": [rid for rid in batch["scheduled"] if rid in pending],
        "ready_for_blender": dispatched,
        "waiting_for_dependencies": waiting,
        "max_heavy_blender_workers": batch["max_heavy_blender_workers"],
        "heavy_blender_executions_started": 0,
        "artistic_and_deformation_review_required": True,
        "limitations": [
            "No subprocesses, Blender jobs or model changes are performed by this command.",
            "All completed/pre-existing region geometry must be checked by separate source and pose QA.",
            "A receipt SHA prevents stale reuse, not incorrect anatomy, UV layout or skin weights.",
        ],
    }
    # The plan may be valid while a source file changes during analysis:
    # always re-check its immutable source links before committing a status.
    if _load_batch(batch_file) != batch:
        raise ValueError("Batch sources changed during the resume audit")
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".character-region-batch-", dir=folder.parent))
    try:
        (staging/"batch-status.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        staging.rename(folder)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return result
