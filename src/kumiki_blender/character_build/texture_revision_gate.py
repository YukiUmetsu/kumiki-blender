"""Source-anchored, fixed-geometry texture revision gate for character parts.

Compare two texture revisions on the *same* registered mesh/UV/cameras and
exactly the same visible reference pixels. This prevents making a candidate
look better by dropping difficult geometry or shrinking its visible area.

Photometric error is unlit CIE76 Lab. It is a useful numerical rejection
gate, not aesthetic certification or a PBR/shading/material review.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .checks import lab
from .texture import read_obj_uv, render_textured

_DEFAULT_SCALE = .25


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction(value: float, name: str, lo: float, hi: float) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be finite within [{lo}, {hi}]")
    return float(value)


def _image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as picture:
        if picture.width != picture.height or not 4 <= picture.width <= 8192:
            raise ValueError(f"{path.name}: texture must be a 4..8192 square pixel image")
        image = np.asarray(picture.convert("RGB")).copy()
    return image


def _scores(reference_rgb: np.ndarray, rendered_rgb: np.ndarray,
            pixels: np.ndarray) -> tuple[dict, np.ndarray]:
    if (reference_rgb.shape != rendered_rgb.shape or reference_rgb.ndim != 3
            or reference_rgb.shape[-1] != 3 or pixels.shape != reference_rgb.shape[:2]):
        raise ValueError("Source, render and sampling mask must have the same registered frame")
    if not pixels.any():
        raise ValueError("No source-visible pixels on the supplied mesh/view")
    original = lab(reference_rgb[pixels].astype(np.uint8))
    predicted = lab(np.clip(rendered_rgb[pixels], 0, 255).astype(np.uint8))
    delta = np.linalg.norm(original - predicted, axis=1)
    return ({
        "mean_delta_e": round(float(delta.mean()), 6),
        "median_delta_e": round(float(np.median(delta)), 6),
        "p90_delta_e": round(float(np.percentile(delta, 90)), 6),
        "within_10": round(float((delta <= 10).mean()), 6),
    }, delta)


def _panel(reference: np.ndarray, before: np.ndarray, after: np.ndarray,
           mask: np.ndarray, before_error: np.ndarray, after_error: np.ndarray,
           target: Path):
    """A fixed camera frame: original, old texture, revised texture and change heat."""
    original_rgb = np.clip(reference[..., :3], 0, 255).astype(np.uint8)
    images = [original_rgb, np.clip(before, 0, 255).astype(np.uint8),
              np.clip(after, 0, 255).astype(np.uint8)]
    shown = [np.where(mask[..., None], image, 45) for image in images]
    heat = np.zeros(mask.shape + (3,), dtype=np.uint8)
    # Red: error became worse; blue: improved; grey: unchanged.
    diff = np.clip(after_error - before_error, -40, 40)
    heat[:] = (42, 42, 42)
    worsening = np.clip(diff, 0, 40) / 40
    improving = np.clip(-diff, 0, 40) / 40
    heat[mask, 0] = (50 + 205 * worsening[mask]).astype(np.uint8)
    heat[mask, 1] = (50 - 30 * np.maximum(worsening[mask], improving[mask])).astype(np.uint8)
    heat[mask, 2] = (50 + 205 * improving[mask]).astype(np.uint8)
    combined = np.concatenate(shown + [heat], axis=1)
    Image.fromarray(combined).save(target)


def audit_texture_revision(handoff: str | Path, mesh_obj: str | Path,
                           baseline_texture: str | Path,
                           candidate_texture: str | Path, output: str | Path, *,
                           views: list[str] | None = None,
                           max_mean_increase: float = 1.5,
                           max_p90_increase: float = 3.0,
                           max_within_10_drop: float = .02,
                           min_pixels_per_view: int = 150,
                           scale: float = _DEFAULT_SCALE) -> dict:
    """Write immutable candidate evidence, including failures; return numeric QA.

    Registered source pixels are selected once, from *the one unchanged mesh*
    rendered in each source camera. Color comparisons use exactly the same
    sampling support for old and new textures. Source files are rehashed before
    reporting, so changed artwork or mesh/UV cannot silently pass.
    """
    _fraction(max_mean_increase, "max_mean_increase", 0, 50)
    _fraction(max_p90_increase, "max_p90_increase", 0, 80)
    _fraction(max_within_10_drop, "max_within_10_drop", 0, 1)
    _fraction(scale, "scale", .125, 1)
    if type(min_pixels_per_view) is not int or not 1 <= min_pixels_per_view <= 500_000:
        raise ValueError("min_pixels_per_view must be an integer between 1 and 500000")
    handoff = Path(handoff).resolve()
    mesh_obj = Path(mesh_obj).resolve()
    baseline_texture = Path(baseline_texture).resolve()
    candidate_texture = Path(candidate_texture).resolve()
    dest = Path(output).resolve()
    if dest.exists():
        raise FileExistsError(f"Texture QA revision already exists: {dest}")
    if not mesh_obj.is_file() or mesh_obj.suffix.lower() != ".obj":
        raise ValueError("Texture QA requires a local OBJ with actual UV corners")
    if dest in {handoff, mesh_obj.parent, baseline_texture.parent, candidate_texture.parent}:
        raise ValueError("Refuse to place QA output over a source directory")
    spec_file = handoff / "spec.json"
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    if type(spec.get("metres_per_aligned_px")) not in (int, float) or not math.isfinite(spec["metres_per_aligned_px"]) or spec["metres_per_aligned_px"] <= 0:
        raise ValueError("Source reference scale is missing or invalid")
    declared = spec.get("views")
    if not isinstance(declared, list) or not declared or any(not isinstance(name, str) for name in declared):
        raise ValueError("Source handoff contains no registered view names")
    if views is not None and (not isinstance(views, list) or not views
                              or len(views) != len(set(views)) or not set(views) <= set(declared)):
        raise ValueError("Selected views must be distinct registered handoff view names")
    selected = views if views is not None else declared
    source_images = [handoff / "aligned" / f"{name}.png" for name in selected]
    if not all(path.is_file() for path in source_images):
        raise ValueError("All selected source-camera aligned images must exist")
    before = _image(baseline_texture)
    after = _image(candidate_texture)
    mesh, corner_uv = read_obj_uv(mesh_obj)
    if (len(mesh.faces) > 500_000 or len(mesh.vertices) > 300_000
            or not np.isfinite(mesh.vertices).all()
            or not np.isfinite(corner_uv).all()):
        raise ValueError("Mesh/UVs are nonfinite or above source-QA budget")
    inputs = {
        "handoff_spec": spec_file,
        "mesh_obj": mesh_obj,
        "baseline_texture": baseline_texture,
        "candidate_texture": candidate_texture,
        **{f"aligned_{name}": path for name, path in zip(selected, source_images)},
    }
    hashes = {key: _hash(path) for key, path in inputs.items()}
    dest.parent.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix=".character-texture-qa-", dir=dest.parent))
    rows = []
    try:
        for name, src in zip(selected, source_images):
            # The unchanged mesh produces one fixed silhouette in this view;
            # source alpha is independent of either texture candidate.
            render_before, silhouette = render_textured(
                mesh, corner_uv, before, name, spec["metres_per_aligned_px"], scale)
            render_after, after_silhouette = render_textured(
                mesh, corner_uv, after, name, spec["metres_per_aligned_px"], scale)
            if not np.array_equal(silhouette, after_silhouette):
                raise RuntimeError("Render coverage changed despite using the same geometry and camera")
            with Image.open(src) as picture:
                reference = np.asarray(picture.convert("RGBA").resize(
                    silhouette.shape[::-1], Image.Resampling.BOX))
            mask = ndimage.binary_erosion(
                silhouette & (reference[..., 3] > 200),
                iterations=2)
            sampled = int(mask.sum())
            if sampled < min_pixels_per_view:
                raise ValueError(f"{name}: only {sampled} comparable original-art pixels; need {min_pixels_per_view}")
            ref_rgb = reference[..., :3]
            base_score, base_delta = _scores(ref_rgb, render_before, mask)
            new_score, new_delta = _scores(ref_rgb, render_after, mask)
            mean_increase = round(new_score["mean_delta_e"] - base_score["mean_delta_e"], 6)
            p90_increase = round(new_score["p90_delta_e"] - base_score["p90_delta_e"], 6)
            share_drop = round(base_score["within_10"] - new_score["within_10"], 6)
            failures = []
            if mean_increase > max_mean_increase + 1e-6:
                failures.append("mean_delta_e_regression")
            if p90_increase > max_p90_increase + 1e-6:
                failures.append("p90_delta_e_regression")
            if share_drop > max_within_10_drop + 1e-6:
                failures.append("within_10_fraction_regression")
            heat_before = np.zeros(mask.shape, dtype=float)
            heat_after = np.zeros(mask.shape, dtype=float)
            heat_before[mask], heat_after[mask] = base_delta, new_delta
            panel = working / f"{name}.png"
            _panel(reference, render_before, render_after,
                   mask, heat_before, heat_after, panel)
            rows.append({
                "view": name, "comparable_source_pixels": sampled,
                "baseline": base_score, "candidate": new_score,
                "mean_delta_e_increase": mean_increase,
                "p90_delta_e_increase": p90_increase,
                "within_10_fraction_drop": share_drop,
                "rejection_reasons": failures,
                "panel": f"{name}.png",
            })
        if any(_hash(path) != hashes[key] for key, path in inputs.items()):
            raise ValueError("Source art, mesh, UVs, or texture changed during evaluation")
        failed_views = [row["view"] for row in rows if row["rejection_reasons"]]
        unchanged = hashes["baseline_texture"] == hashes["candidate_texture"]
        result = {
            "schema_version": 1,
            "status": ("registered-texture-regression-rejected" if failed_views
                       else "no-change-artistic-review-required" if unchanged
                       else "no-registered-texture-regression-artistic-review-required"),
            "numeric_gate_pass": not failed_views,
            "failed_views": failed_views,
            "source_view_names": list(selected),
            "input_sha256": hashes,
            "thresholds": {
                "max_mean_delta_e_increase": float(max_mean_increase),
                "max_p90_delta_e_increase": float(max_p90_increase),
                "max_within_10_fraction_drop": float(max_within_10_drop),
                "min_source_pixels_per_view": min_pixels_per_view,
            },
            "mesh_and_uv_fixed": True,
            "same_camera_and_source_pixels_for_both_textures": True,
            "source_scale": float(scale),
            "views": rows,
            "limitations": [
                "CIE76 on unlit renders is not a material, shading, normal map or PBR fidelity assessment.",
                "No source camera registration or silhouette optimization is performed.",
                "Reference color differences may arise from illustration lighting or view inconsistency.",
                "A numeric pass never certifies correct character identity or rig performance.",
            ],
        }
        (working / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        working.rename(dest)
        return result
    finally:
        shutil.rmtree(working, ignore_errors=True)
