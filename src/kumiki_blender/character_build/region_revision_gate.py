"""Atomic before/after source-camera gate for a single CharacterSpec v2 region.

Uses the *existing* verified reference masks and registered rasterizer. A
structurally valid Blender edit cannot be called source-faithful just because
it has preserved topology, UVs or protected vertices: a decrease in the
registered silhouette of any view must be visible and can reject the edit.

The result is a *numeric change review*, never automatic artistic approval.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from .character_spec_v2 import _hash, open_plan
from .region_compare import compare_region

_DEFAULT_LIMIT = 0.01
_ALLOWED_EXTENSIONS = frozenset((".obj", ".ply", ".glb"))


def _canonical_mesh(path: str | Path, label: str) -> Path:
    file = Path(path).resolve()
    if not file.is_file() or file.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise ValueError(f"{label}: expected a local OBJ, PLY, or GLB source-frame part")
    if file.suffix.lower() == ".obj" and not (
        file.name.endswith("-source.obj") or file.name.endswith(".source.obj")
    ):
        raise ValueError(
            f"{label}: ambiguous Y-up OBJ convention; choose the explicitly "
            "Z-up *-source.obj / *.source.obj artifact, or export PLY")
    return file


def _number(value, name: str, lo: float, hi: float) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be finite between {lo} and {hi}")
    return float(value)


def _mosaic(baseline: Path, candidate: Path, destination: Path):
    """Two independent registered frames; no crops/registration adjustments."""
    with Image.open(baseline) as before_file, Image.open(candidate) as after_file:
        before = before_file.convert("RGB")
        after = after_file.convert("RGB")
    if before.size != (400, 300) or after.size != before.size:
        raise ValueError("Existing compare-region previews have unexpected dimensions")
    combined = Image.new("RGB", (800, 300))
    combined.paste(before, (0, 0))
    combined.paste(after, (400, 0))
    combined.save(destination)


def gate_region_revision(character: str | Path, region_id: str, after: str | Path,
                         output: str | Path, *, before: str | Path | None = None,
                         max_regression: float = _DEFAULT_LIMIT,
                         min_iou: float | None = None,
                         max_contour_regression: float = 0.03) -> dict:
    """Evaluate a changed part against unchanged full-frame source masks.

    The original source mesh is used as baseline when CharacterSpec provides
    source_mesh; otherwise an explicit pre-edit baseline mesh is required.

    Reports are *still emitted on numeric failure*, with side-by-side overlays
    and view-specific reasons. Invalid inputs never produce partial outputs.
    """
    _number(max_regression, "max_regression", 0, 1)
    _number(max_contour_regression, "max_contour_regression", 0, 1)
    if min_iou is not None:
        _number(min_iou, "min_iou", 0, 1)
    plan, spec = open_plan(character)
    if not isinstance(region_id, str):
        raise ValueError("Region ID must be an exact semantic string")
    region = next((row for row in spec["regions"] if row["id"] == region_id), None)
    if region is None:
        raise ValueError(f"Unknown source region: {region_id}")
    verified = region.get("verified_views", [])
    if not verified:
        raise ValueError(f"{region_id}: no verified views; numeric gate is not possible")
    source_root = Path(plan["source_spec"]).parent
    if before is None and "source_mesh" not in region:
        raise ValueError(f"{region_id}: no original source_mesh; pass an explicit --before")
    baseline_path = _canonical_mesh(
        before if before is not None else source_root / region["source_mesh"],
        "baseline")
    candidate_path = _canonical_mesh(after, "candidate")
    baseline_sha = _hash(baseline_path)
    candidate_sha = _hash(candidate_path)
    dest = Path(output).resolve()
    if dest.exists():
        raise FileExistsError(f"Revision gate output already exists: {dest}")
    if dest == baseline_path.parent or dest == candidate_path.parent:
        raise ValueError("Refuse to write revision reports over an input mesh folder")
    dest.parent.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix=".character-source-gate-", dir=dest.parent))
    try:
        before_report = compare_region(
            character, region_id, baseline_path, working/"baseline",
            min_iou=None)
        after_report = compare_region(
            character, region_id, candidate_path, working/"candidate",
            min_iou=min_iou, baseline=working/"baseline"/"report.json",
            max_regression=max_regression)
        before_rows = {v["view"]: v for v in before_report["views"]}
        after_rows = {v["view"]: v for v in after_report["views"]}
        if set(before_rows) != set(verified) or set(after_rows) != set(verified):
            raise RuntimeError("Registered camera set changed during before/after QA")
        statuses = []
        panels = working/"panels"
        panels.mkdir()
        for view in verified:
            old, new = before_rows[view], after_rows[view]
            if old["iou"] is None or new["iou"] is None:
                raise ValueError(f"Source region {region_id} has empty/undefined IoU in {view}")
            delta = round(new["iou"] - old["iou"], 6)
            previous_outline = old.get("boundary_f1_2px")
            revised_outline = new.get("boundary_f1_2px")
            if previous_outline is None:
                raise ValueError(f"{view}: baseline has no measurable registered silhouette contour")
            contour_delta = (round(revised_outline - previous_outline, 6)
                             if revised_outline is not None else None)
            reasons = []
            if revised_outline is None:
                reasons.append("candidate_contour_missing")
            elif contour_delta < -max_contour_regression - 1e-6:
                reasons.append("registered_contour_regression")
            if delta < -max_regression - 1e-6:
                reasons.append("registered_silhouette_regression")
            if min_iou is not None and new["iou"] < min_iou - 1e-6:
                reasons.append("below_absolute_iou_floor")
            _mosaic(
                working/"baseline"/"overlays"/f"{view}.png",
                working/"candidate"/"overlays"/f"{view}.png",
                panels/f"{view}.png")
            statuses.append({
                "view": view,
                "baseline_iou": old["iou"],
                "candidate_iou": new["iou"],
                "iou_delta": delta,
                "baseline_boundary_f1_2px": previous_outline,
                "candidate_boundary_f1_2px": revised_outline,
                "boundary_f1_2px_delta": contour_delta,
                "candidate_boundary_p95_distance_px": new["boundary_p95_distance_px"],
                "missing_pixels_after": new["missing_pixels"],
                "extra_pixels_after": new["extra_pixels"],
                "baseline_overlay": f"baseline/overlays/{view}.png",
                "candidate_overlay": f"candidate/overlays/{view}.png",
                "before_after_panel": f"panels/{view}.png",
                "rejection_reasons": reasons,
            })
        # Re-hash immutable source representations; don't emit a report
        # assembled from stale artifacts changed during the comparisons.
        checked_plan, _ = open_plan(character)
        if checked_plan != plan or _hash(baseline_path) != baseline_sha or _hash(candidate_path) != candidate_sha:
            raise ValueError("Source plan or mesh files changed during registered revision review")
        failures = [s["view"] for s in statuses if s["rejection_reasons"]]
        changed = baseline_sha != candidate_sha
        report = {
            "schema_version": 1,
            "character_id": spec["character_id"],
            "region_id": region_id,
            "status": "registered-silhouette-regression-rejected" if failures
                      else ("no-change-review-required" if not changed
                            else "no-registered-regression-artistic-review-required"),
            "numeric_gate_pass": not failures,
            "baseline_mesh": str(baseline_path),
            "baseline_sha256": baseline_sha,
            "candidate_mesh": str(candidate_path),
            "candidate_sha256": candidate_sha,
            "source_spec_sha256": plan["source_sha256"],
            "region_input_sha256": plan["regions"][region_id]["input_sha256"],
            "verified_views": list(verified),
            "max_iou_regression_per_view": float(max_regression),
            "max_boundary_f1_2px_regression_per_view": float(max_contour_regression),
            "min_iou_per_view": float(min_iou) if min_iou is not None else None,
            "failed_views": failures,
            "any_geometry_change": changed,
            "best_iou_delta": max(row["iou_delta"] for row in statuses),
            "worst_iou_delta": min(row["iou_delta"] for row in statuses),
            "views": statuses,
            "limitations": [
                "Source-registered boundary F1 measures 400x300 image contours; tiny details under about 4 original pixels remain unresolved.",
                "A source-registered silhouette check cannot prove likeness, colors, textures, depth or hidden anatomy.",
                "Only independently verified cameras listed in CharacterSpec v2 are evaluated.",
                "No automatic source registration, camera retargeting, pose matching or per-region recentering.",
                "The result is a numeric QA gate; real character artistic and animated-pose review remain mandatory.",
                "The baseline and candidate geometry are expected in the identical Z-up Blender world frame.",
            ],
        }
        (working/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        working.rename(dest)
        return report
    finally:
        shutil.rmtree(working, ignore_errors=True)
