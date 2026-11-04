"""Apple-MPS MapAnything pose hypotheses for hard-to-register *real* photos.

These are explicit, unverified hypotheses. They must never be added to COLMAP
registered counts without real-image feature/track verification.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .feedforward_sparse import build_feedforward_sparse_reconstruction
from .references import ReferenceError
from .splat_input import load_photo_folder
from .sfm_cohort_recovery import _similarity


def select_learned_views(index: list[dict[str, Any]], registered: set[str], *,
                         limit: int = 8) -> list[str]:
    """Choose elevated missing views plus diverse real registered anchors."""
    if limit < 8 or limit > 16:
        raise ValueError("MapAnything MPS groups must contain 8–16 photographs")
    records = {str(item.get("filename")): item for item in index if isinstance(item, dict)}

    def priority(name: str) -> tuple[int, str]:
        original = str(records[name].get("original_filename") or name).lower()
        return (-sum(3 if term in original else 0 for term in
                     ("top", "aerial", "elevated", "above", "overhead"))
                - sum(term in original for term in ("rear", "back", "left", "right")), name)

    missing = sorted(set(records) - registered, key=priority)
    anchors = sorted(registered & set(records))
    # Cover the baseline rather than selecting near-duplicate early filenames.
    anchor_count = min(len(anchors), max(4, limit // 2))
    chosen_anchors = ([anchors[round(i * (len(anchors) - 1) / (anchor_count - 1))]
                       for i in range(anchor_count)] if anchor_count > 1 else anchors[:1])
    selected = list(dict.fromkeys(chosen_anchors + missing[:limit - len(chosen_anchors)]))
    if len(selected) < limit:
        selected += [name for name in anchors if name not in selected][:limit - len(selected)]
    return selected


def align_learned_camera_hypotheses(baseline, predicted, *, min_anchors: int = 4) -> dict[str, Any]:
    """Align predicted cameras using shared real-photo IDs; never label them verified."""
    import numpy as np

    before = {baseline.images[i].name: baseline.images[i] for i in baseline.reg_image_ids()}
    after = {predicted.images[i].name: predicted.images[i] for i in predicted.reg_image_ids()}
    shared = sorted(set(before) & set(after))
    unknown = sorted(set(after) - set(before))
    report: dict[str, Any] = {"status": "hypotheses-unverified", "shared_anchors": len(shared),
                              "predicted_unregistered": len(unknown), "pose_hypotheses": []}
    if len(shared) < min_anchors:
        report["reason"] = "Too few common verified cameras to align learned coordinate frames"
        return report
    try:
        match = _similarity(
            [after[name].projection_center() for name in shared],
            [before[name].projection_center() for name in shared],
            min_spread_ratio=0.04)
        deviations = []
        for name in shared:
            estimated = match["rotation"] @ np.asarray(after[name].viewing_direction(), dtype=float)
            reference = np.asarray(before[name].viewing_direction(), dtype=float)
            denom = float(np.linalg.norm(estimated) * np.linalg.norm(reference))
            deviations.append(float(np.degrees(np.arccos(np.clip(
                np.dot(estimated, reference) / denom, -1, 1)))) if denom > 1e-9 else math.inf)
        report.update(camera_rms_ratio=match["center_rms_ratio"],
                      max_camera_direction_error_degrees=max(deviations))
        if (match["center_max_ratio"] > 0.15 or max(deviations) > 15
                or not all(math.isfinite(x) for x in deviations)):
            report["reason"] = "Learned anchor poses disagree with registered cameras"
            return report
        for name in unknown:
            image = after[name]
            xyz = (match["scale"] * (match["rotation"] @
                   np.asarray(image.projection_center(), dtype=float)) + match["translation"])
            direction = match["rotation"] @ np.asarray(image.viewing_direction(), dtype=float)
            if np.isfinite(xyz).all() and np.isfinite(direction).all():
                report["pose_hypotheses"].append({
                    "filename": name, "position": [float(x) for x in xyz],
                    "viewing_direction": [float(x) for x in direction],
                    "provenance": "MapAnything-prediction / unverified-by-real-tracks"})
        report["status"] = "learned-camera-hypotheses / requires-pnp-verification"
    except (ValueError, FloatingPointError, KeyError, TypeError) as error:
        report["reason"] = str(error)[:240]
    return report


def build_learned_camera_hypotheses(source: Path, output: Path, *,
                                    device: str = "mps", timeout_seconds: int = 1800,
                                    builder: Callable[..., dict] = build_feedforward_sparse_reconstruction,
                                    model_loader=None) -> dict[str, Any]:
    """Run a bounded existing MapAnything worker and compare to real COLMAP poses."""
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(source.resolve()):
        raise ReferenceError("Learned camera output must be a fresh directory outside source")
    source_report = json.loads((source / "reconstruction-report.json").read_text())
    index = json.loads((source / "staged-reference-index.json").read_text())["images"]
    selected = select_learned_views(index, set(source_report.get("registered_filenames") or []))
    if len(selected) < 8 or all(name in source_report.get("registered_filenames", []) for name in selected):
        return {"status": "no-eligible-unregistered-image-group", "selected": selected}
    if importlib.util.find_spec("mapanything") is None or importlib.util.find_spec("torch") is None:
        return {"status": "unavailable / install-mapanything-extra", "selected": selected,
                "hint": "pip install -e '.[mapanything]'"}
    references = load_photo_folder(source / "images", source_report["asset_id"])
    subset = tuple(replace(item, image_id=item.path.stem)
                   for item in references.images if item.path.name in selected)
    manifest_hash = hashlib.sha256("|".join(sorted(item.sha256 for item in subset)).encode()).hexdigest()
    chosen = replace(references, images=subset, manifest_sha256=manifest_hash)
    result = builder(chosen, output, device=device, timeout_seconds=timeout_seconds)
    report: dict[str, Any] = {"status": "learned-inference-incomplete", "selected": selected,
                              "model_status": result.get("status"),
                              "model_report": str(output / "reconstruction-report.json"),
                              "limitations": "AI predictions are not verified camera measurements."}
    if not str(result.get("status", "")).startswith("candidate"):
        return report
    if model_loader is None:
        import pycolmap
        model_loader = pycolmap.Reconstruction
    best = str(source_report.get("source_sparse_model", "0"))
    baseline = model_loader(source / "sparse" / best)
    predicted = model_loader(output / "undistorted/text-model")
    report.update(align_learned_camera_hypotheses(baseline, predicted))
    return report
