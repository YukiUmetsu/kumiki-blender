"""Compare a separately modeled body part against registered full-frame masks.

Requires an already aligned Blender-space mesh. Never recenters, rescales, or
optimizes a camera to inflate overlap. Source art remains independent evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .character_spec_v2 import _hash, open_plan
from .mesh import rasterize, read_mesh
from .views import FIGURE_PX, camera

RESAMPLE = Image.Resampling.NEAREST
SCALE = 0.25


def _source_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (1600, 1200):
            raise ValueError(f"Reference mask must be registered 1600x1200 full frame: {path}")
        if image.mode in ("RGBA", "LA"):
            alpha = np.asarray(image.getchannel("A"))
            # An RGB image with an everywhere-opaque alpha channel is not a useful mask.
            layer = image.getchannel("A") if alpha.min() != alpha.max() else image.convert("L")
        else:
            layer = image.convert("L")
        return np.asarray(layer.resize((400, 300), RESAMPLE)) >= 128


def _bbox(mask: np.ndarray) -> list[int] | None:
    y, x = np.where(mask)
    return [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)] if x.size else None


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    intersection = int(np.logical_and(reference, candidate).sum())
    union = int(np.logical_or(reference, candidate).sum())
    missing = int(np.logical_and(reference, ~candidate).sum())
    extra = int(np.logical_and(candidate, ~reference).sum())
    return {
        "reference_pixels": int(reference.sum()),
        "candidate_pixels": int(candidate.sum()),
        "intersection_pixels": intersection,
        "missing_pixels": missing,
        "extra_pixels": extra,
        "iou": round(intersection / union, 6) if union else None,
        "bbox_reference": _bbox(reference),
        "bbox_candidate": _bbox(candidate),
        "pixel_scale_of_source": SCALE,
    }


def _overlay(reference: np.ndarray, candidate: np.ndarray, destination: Path) -> None:
    # Red: model missing source, blue: model extra, green: overlapping.
    rgb = np.full((*reference.shape, 3), 27, dtype=np.uint8)
    rgb[reference & ~candidate] = (240, 65, 67)
    rgb[~reference & candidate] = (67, 137, 240)
    rgb[reference & candidate] = (60, 186, 112)
    Image.fromarray(rgb, "RGB").save(destination)


def compare_region(character: str | Path, region_id: str, mesh: str | Path, output: str | Path,
                   *, min_iou: float | None = None, baseline: str | Path | None = None,
                   max_regression: float = 0.01) -> dict:
    """Create independent per-camera reports and reject explicit threshold regressions.

    No global artistic success is inferred. Unverified source views cannot vote.
    """
    if min_iou is not None and not 0 <= min_iou <= 1:
        raise ValueError("min_iou must be between 0 and 1")
    if not 0 <= max_regression <= 1:
        raise ValueError("max_regression must be between 0 and 1")
    plan, spec = open_plan(character)
    region = next((r for r in spec["regions"] if r["id"] == region_id), None)
    if region is None:
        raise ValueError(f"Unknown region {region_id}")
    source = Path(plan["source_spec"]).parent
    source_mesh = Path(mesh).resolve()
    if not source_mesh.is_file() or source_mesh.suffix.lower() not in (".ply", ".glb", ".obj"):
        raise ValueError("Candidate region must be a local OBJ, PLY or GLB")
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Comparison revision already exists: {out}")
    reference_views = region.get("verified_views", [])
    if not reference_views:
        raise ValueError(f"{region_id}: no verified views; review camera and masks first")
    # Validate baseline before writing any partial report.
    baseline_rows: dict[str, dict] = {}
    if baseline:
        raw = json.loads(Path(baseline).read_text(encoding="utf-8"))
        if raw.get("character_id") != spec["character_id"] or raw.get("region_id") != region_id:
            raise ValueError("Baseline report belongs to a different character/region")
        if raw.get("source_sha256") != plan["source_sha256"]:
            raise ValueError("Baseline source spec differs from this character plan")
        if raw.get("region_input_sha256") != plan["regions"][region_id]["input_sha256"]:
            raise ValueError("Baseline references changed; report is not comparable")
        baseline_rows = {v["view"]: v for v in raw.get("views", [])}
        if set(baseline_rows) != set(reference_views):
            raise ValueError("Baseline cameras differ from the verified region views")
    candidate = read_mesh(source_mesh)
    if not np.isfinite(candidate.vertices).all() or len(candidate.faces) == 0:
        raise ValueError("Candidate mesh must have finite vertices and triangles")
    source_hash = _hash(source_mesh)
    # The geometry is assumed to be in the canonical handoff frame. No automatic transform.
    rows = []
    previews = []
    for view in reference_views:
        mask_path = source / region["reference_masks"][view]
        reference = _source_mask(mask_path)
        measured, _depth = rasterize(candidate, camera(view), spec["height_m"] / FIGURE_PX, scale=SCALE)
        score = _metrics(reference, measured)
        score["view"] = view
        previous = baseline_rows.get(view, {}).get("iou")
        if previous is not None and score["iou"] is not None:
            score["iou_delta"] = round(score["iou"] - previous, 6)
        score["threshold_pass"] = score["iou"] is not None and (
            min_iou is None or score["iou"] >= min_iou)
        score["baseline_pass"] = ("iou_delta" not in score or
                                  score["iou_delta"] >= -max_regression)
        score["overlay"] = f"overlays/{view}.png"
        rows.append(score)
        previews.append((view, reference, measured))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    (out / "overlays").mkdir()
    for view, reference, measured in previews:
        _overlay(reference, measured, out / "overlays" / f"{view}.png")
    failures = [r["view"] for r in rows if not r["threshold_pass"] or not r["baseline_pass"]]
    report = {
        "schema_version": 2,
        "character_id": spec["character_id"],
        "region_id": region_id,
        "source_sha256": plan["source_sha256"],
        "region_input_sha256": plan["regions"][region_id]["input_sha256"],
        "candidate_mesh": str(source_mesh),
        "candidate_sha256": source_hash,
        "coordinate_frame": "Blender metres; +Z up; -Y forward; pre-aligned with CharacterSpec",
        "status": "geometric-review-failed" if failures else "artistic-review-required",
        "failed_views": failures,
        "threshold_min_iou": min_iou,
        "max_regression_iou": max_regression if baseline else None,
        "views": rows,
        "limitations": [
            "Registered orthographic silhouette only; cannot certify 3D internal volume or true anatomy.",
            "No inferred, unverified, or misregistered source view can establish fidelity.",
            "Source registration and human artistic review are required before accepting the mesh.",
            "Only region geometry is compared; surrounding parts and occlusion are not rendered here.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
