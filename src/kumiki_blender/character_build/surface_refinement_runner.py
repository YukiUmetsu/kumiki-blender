"""Non-destructive orchestration of Blender surface fairing and registered source QA.

Candidate generation and reference evaluation happen in a staging folder.
Only a completed report tree becomes the requested immutable output. A
numeric pass is NOT artistic approval and cannot silently replace a rig.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from .surface_fairing import load_surface_policy


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_surface_refinement(source: str | Path, config: str | Path,
                           output: str | Path, *, blender: str | None = None,
                           worker=None, source_gate=None) -> dict:
    """Create a separate candidate and optional per-view verified-silhouette QA."""
    from .blender import run_worker
    from .region_revision_gate import gate_region_revision

    plan = load_surface_policy(config)
    source = Path(source).resolve()
    config = Path(config).resolve()
    out = Path(output).resolve()
    if not source.is_file() or source.suffix.lower() != ".blend":
        raise ValueError("Expected a local source Blender .blend")
    if out.exists():
        raise FileExistsError(f"Surface revision already exists: {out}")
    if source.is_relative_to(out) or out == source.parent or out == config.parent:
        raise ValueError("Refuse to write a surface revision into its source/config directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    source_sha, config_sha = _sha(source), _sha(config)
    runner = worker or run_worker
    gate = source_gate or gate_region_revision
    with tempfile.TemporaryDirectory(prefix=".surface-fairing-", dir=out.parent) as stage_name:
        stage = Path(stage_name)
        model = stage / "model"
        generated = runner("surface-refine", {
            "plan": asdict(plan), "source_blend": str(source),
            "source_sha256": source_sha, "output": str(model),
        }, blender=blender, blend_file=source)
        if not isinstance(generated, dict) or generated.get("schema_version") != 1:
            raise RuntimeError("Blender worker produced an incompatible surface report")
        if _sha(source) != source_sha or _sha(config) != config_sha:
            raise ValueError("Source Blender/config changed during refinement; discard stale candidate")
        source_review = None
        if generated["status"] != "no-safe-improvement" and plan.reference_gate:
            opts = plan.reference_gate
            before = model / "baseline-source.obj"
            after = model / "surface-candidate-source.obj"
            if not before.is_file() or not after.is_file():
                raise RuntimeError("Blender did not export canonical Z-up baseline and candidate OBJ files")
            source_review = gate(
                opts["character_plan"], opts["region"], after, stage / "source-review",
                before=before, max_regression=opts["max_iou_regression"],
                min_iou=opts["min_iou"])
            if not isinstance(source_review, dict) or "numeric_gate_pass" not in source_review:
                raise RuntimeError("Registered source gate did not return verifiable results")
        if _sha(source) != source_sha or _sha(config) != config_sha:
            raise ValueError("Source Blender/config changed during reference QA")
        rejected = source_review is not None and not source_review["numeric_gate_pass"]
        status = ("reference-regression-rejected" if rejected
                  else "no-safe-improvement" if generated["status"] == "no-safe-improvement"
                  else "registered-candidate-art-review-required" if source_review
                  else "surface-candidate-art-review-required")
        report = {
            "schema_version": 1, "status": status,
            "numeric_reference_gate_pass": None if source_review is None else not rejected,
            "artistic_approval": False,
            "source_blend": str(source), "source_blend_sha256": source_sha,
            "policy_sha256": config_sha, "policy": asdict(plan),
            "worker_report": "model/report.json",
            "source_review": "source-review/report.json" if source_review is not None else None,
            "source_summary": ({
                "failed_views": source_review.get("failed_views", []),
                "worst_iou_delta": source_review.get("worst_iou_delta"),
                "verified_views": source_review.get("verified_views", []),
            } if source_review is not None else None),
            "limitations": [
                "Without verified CharacterSpec views no reference-fidelity claim is made.",
                "A passing registered silhouette gate cannot certify texture, anatomy, depth, or animation.",
                "This stage preserves topology and UVs; adaptive subdivision must be performed separately.",
                "The original source .blend remains unchanged.",
            ],
        }
        (stage / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if rejected:
            (stage / "SOURCE_REGRESSION_REJECTED.txt").write_text(
                "DO NOT promote this candidate. The verified source view comparison regressed.\n"
                "Inspect source-review/report.json and its per-view overlays.\n", encoding="utf-8")
        # Preserve complete diagnostics; never pretend that a rejected mesh is approved.
        shutil.move(str(stage), str(out))
        # The source-review tool records absolute artifact paths; relocate those
        # strings after moving, without altering any immutable mesh bytes/hashes.
        for report_path in out.rglob("*.json"):
            value = report_path.read_text(encoding="utf-8").replace(str(stage), str(out))
            report_path.write_text(value, encoding="utf-8")
        return report
