"""Deterministic, rollback-safe, configurable multi-view texture conflict repair.

No generative-model calls, destructive writes, unverified camera alignment, or
implicit geometry editing. Character-specific semantics are optional inputs.
Candidate mesh OBJ files may be supplied, but must share the exact UV topology.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Search parameters change painting, never the thresholds that define a conflict.
# Otherwise the optimizer could hide faults by changing their classification.
SEARCHABLE = frozenset({'power', 'diagonal_weight', 'min_cosine', 'region_px',
                        'min_region_px', 'occlusion_px', 'level_reach_px'})
DEFAULT_SETTINGS = {
    'power': 4.0, 'diagonal_weight': 1.0, 'min_cosine': 0.35,
    'region_px': 24.0, 'min_region_px': 40.0, 'occlusion_px': 30.0,
    'level_reach_px': 150.0,
}
DEFAULT_WEIGHTS = {'misplaced_part': 4.0, 'contested_feature': 3.0,
                   'unseen': 5.0, 'tone_seam': 0.3, 'appearance': 0.75, 'silhouette': 1.0}
_LIMITS = {'power': (0.5, 12.0), 'diagonal_weight': (0.05, 3.0),
           'min_cosine': (0.0, 0.95), 'region_px': (4.0, 120.0),
           'min_region_px': (4.0, 180.0), 'occlusion_px': (0.0, 200.0),
           'level_reach_px': (1.0, 400.0)}
_ALLOWED = frozenset({'schema_version', 'settings', 'search', 'guards', 'weights',
                      'semantic', 'candidate_meshes', 'final_size', 'evaluation_size',
                      'appearance_scale', 'views'})
_RETAINED = ('misplaced_part', 'contested_feature', 'unseen', 'tone_seam')


def _positive_number(value: Any, name: str, low: float, high: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be a finite number in [{low}, {high}]')
    return float(value)


def _relative_file(root: Path, raw: str) -> Path:
    if not isinstance(raw, str) or not raw or '\\' in raw:
        raise ValueError('Mesh path must be relative to the configuration')
    relative = Path(raw)
    if relative.is_absolute() or any(segment in ('..', '.') for segment in relative.parts):
        raise ValueError(f'Unsafe mesh path: {raw}')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or unsafe mesh path: {raw}')
    if path.suffix.lower() != '.obj':
        raise ValueError('Candidate meshes must be OBJ with UV coordinates')
    return path.resolve()


@dataclass(frozen=True)
class RepairPlan:
    root: Path
    settings: dict[str, float]
    steps: tuple[tuple[str, tuple[float, ...]], ...]
    rounds: int
    max_evaluations: int
    min_improvement: float
    max_view_regression: float
    max_unseen_increase: float
    max_view_silhouette_iou_drop: float
    weights: dict[str, float]
    evaluation_size: int
    final_size: int
    appearance_scale: float
    views: tuple[str, ...] | None
    semantic: dict | None
    candidate_meshes: tuple[Path, ...]


def read_repair_plan(path: str | Path) -> RepairPlan:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or raw.get('schema_version') != 1 or set(raw) - _ALLOWED:
        raise ValueError('Repair plan must have schema_version 1 and recognized keys only')
    settings = {**DEFAULT_SETTINGS}
    for key, value in raw.get('settings', {}).items():
        if key not in SEARCHABLE:
            raise ValueError(f'Unsupported bake setting: {key}')
        settings[key] = _positive_number(value, key, *_LIMITS[key])
    search = raw.get('search', {})
    if not isinstance(search, dict) or set(search) - {'steps', 'rounds', 'max_evaluations', 'min_improvement'}:
        raise ValueError('Invalid search options')
    rounds = int(_positive_number(search.get('rounds', 2), 'search.rounds', 0, 10))
    evaluations = int(_positive_number(search.get('max_evaluations', 12), 'max_evaluations', 1, 80))
    steps = []
    raw_steps = search.get('steps', [{'parameter': 'region_px', 'values': [16, 32, 48]},
                                     {'parameter': 'occlusion_px', 'values': [45, 60]},
                                     {'parameter': 'power', 'values': [2, 6]}])
    if not isinstance(raw_steps, list) or len(raw_steps) > 10:
        raise ValueError('search.steps must be a bounded array')
    seen = set()
    for step in raw_steps:
        if not isinstance(step, dict) or set(step) != {'parameter', 'values'}:
            raise ValueError('Each search step needs parameter and values')
        name, values = step['parameter'], step['values']
        if name not in SEARCHABLE or name in seen or not isinstance(values, list) or not 1 <= len(values) <= 12:
            raise ValueError('Duplicate or unsupported search parameter/values')
        seen.add(name)
        steps.append((name, tuple(_positive_number(x, name, *_LIMITS[name]) for x in values)))
    guards = raw.get('guards', {})
    if not isinstance(guards, dict) or set(guards) - {'max_view_mean_delta_e_increase', 'max_unseen_fraction_increase', 'max_view_silhouette_iou_drop'}:
        raise ValueError('Unrecognized guard')
    weights = {**DEFAULT_WEIGHTS}
    for name, value in raw.get('weights', {}).items():
        if name not in weights:
            raise ValueError(f'Unknown weight: {name}')
        weights[name] = _positive_number(value, f'weights.{name}', 0, 100)
    if all(value == 0 for value in weights.values()):
        raise ValueError('At least one score weight is required')
    raw_meshes = raw.get('candidate_meshes', [])
    if not isinstance(raw_meshes, list) or len(raw_meshes) > 8:
        raise ValueError('candidate_meshes must be a bounded array')
    meshes = tuple(_relative_file(source.parent, name) for name in raw_meshes)
    if len(set(meshes)) != len(meshes):
        raise ValueError('Duplicate candidate mesh path')
    views = raw.get('views')
    if views is not None and (not isinstance(views, list) or not views
                              or len(set(views)) != len(views) or not all(isinstance(x, str) and x for x in views)):
        raise ValueError('views must be a unique nonempty list of camera names')
    semantic = raw.get('semantic')
    if semantic is not None and not isinstance(semantic, dict):
        raise ValueError('semantic must be an object')
    return RepairPlan(
        root=source.parent, settings=settings, steps=tuple(steps), rounds=rounds,
        max_evaluations=evaluations,
        min_improvement=_positive_number(search.get('min_improvement', 0.0001), 'min_improvement', 0, 1),
        max_view_regression=_positive_number(guards.get('max_view_mean_delta_e_increase', 1.0),
                                             'max_view_mean_delta_e_increase', 0, 100),
        max_unseen_increase=_positive_number(guards.get('max_unseen_fraction_increase', 0.005),
                                             'max_unseen_fraction_increase', 0, 1),
        max_view_silhouette_iou_drop=_positive_number(guards.get('max_view_silhouette_iou_drop', 0.005),
                                                    'max_view_silhouette_iou_drop', 0, 1),
        weights=weights,
        evaluation_size=int(_positive_number(raw.get('evaluation_size', 512), 'evaluation_size', 128, 2048)),
        final_size=int(_positive_number(raw.get('final_size', 2048), 'final_size', 128, 8192)),
        appearance_scale=_positive_number(raw.get('appearance_scale', 0.20), 'appearance_scale', 0.05, 1),
        views=tuple(views) if views is not None else None,
        semantic=semantic, candidate_meshes=meshes,
    )


def _objective(conflicts: dict, appearance: dict, weights: dict[str, float], geometry: dict | None = None) -> float:
    surface = conflicts['surface']
    view_rows = appearance['views']
    if not view_rows:
        raise ValueError('Cannot optimize without registered appearance comparisons')
    if any(not math.isfinite(row['mean_delta_e']) for row in view_rows):
        raise ValueError('Nonfinite appearance error')
    measured = float(np.mean([row['mean_delta_e'] for row in view_rows])) / 100.0
    score = sum(weights[key] * surface[key] for key in _RETAINED) + weights['appearance'] * measured
    if geometry and geometry.get('views'):
        score += weights['silhouette'] * (1 - float(np.mean([v['iou'] for v in geometry['views']])))
    return float(score)


def _rejection(conflicts: dict, appearance: dict, baseline: dict, plan: RepairPlan, geometry: dict | None = None) -> list[str]:
    reasons = []
    if conflicts['surface']['unseen'] > baseline['conflicts']['surface']['unseen'] + plan.max_unseen_increase:
        reasons.append('unseen_surface_regression')
    original = {v['view']: v['mean_delta_e'] for v in baseline['appearance']['views']}
    current = {v['view']: v['mean_delta_e'] for v in appearance['views']}
    if set(original) != set(current):
        reasons.append('view_coverage_changed')
    else:
        reasons.extend(f'view_regression:{name}' for name, value in current.items()
                       if value > original[name] + plan.max_view_regression)
    if geometry is not None and baseline.get('geometry') is not None:
        previous = {v['view']: v['iou'] for v in baseline['geometry']['views']}
        current_iou = {v['view']: v['iou'] for v in geometry['views']}
        if set(previous) != set(current_iou):
            reasons.append('silhouette_view_coverage_changed')
        else:
            reasons.extend(f'silhouette_regression:{view}' for view, score in current_iou.items()
                           if previous[view] - score > plan.max_view_silhouette_iou_drop)
    return reasons


def _fingerprint(mesh: Path, settings: dict) -> str:
    return json.dumps({'mesh': str(mesh), 'settings': settings}, sort_keys=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_conflict_repair(
    handoff: str | Path, source_mesh: str | Path, config: str | Path, output: str | Path,
    *, evaluator=None,
) -> dict:
    """Bounded coordinate search, accepting only guarded improvements.

    evaluator may be injected by tests/other modelers. Signature:
    (mesh_path, settings, size, diagnostics_dir) -> (texture_array, conflicts, appearance).
    It must use fixed conflict thresholds and identical source camera comparisons.
    No model files are overwritten, even when no improvement.
    """
    plan = read_repair_plan(config)
    handoff, source_mesh, out = Path(handoff).resolve(), Path(source_mesh).resolve(), Path(output).resolve()
    if not (handoff / 'spec.json').is_file() or not source_mesh.is_file() or source_mesh.suffix.lower() != '.obj':
        raise ValueError('Need an existing handoff/spec.json and an OBJ with UVs')
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}')
    if out == handoff or out == source_mesh.parent:
        raise ValueError('Output cannot overwrite model or handoff directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh_files = (source_mesh,) + plan.candidate_meshes
    if evaluator is None:
        from .texture import appearance_report, bake_texture, read_obj_uv
        from .handoff import reference_masks
        from .mesh import rasterize, surface_samples
        from .views import camera
        px = json.loads((handoff / 'spec.json').read_text(encoding='utf-8'))['metres_per_aligned_px']
        silhouette_scale = 0.25
        masks = reference_masks(handoff, silhouette_scale)
        if not masks:
            raise ValueError('Need reference silhouette masks for geometry regression guards')
        baseline_mesh, baseline_uv = read_obj_uv(source_mesh)
        meshes = {}
        for name in mesh_files:
            mesh, uv = read_obj_uv(name)
            if (mesh.faces.shape != baseline_mesh.faces.shape
                    or not np.array_equal(mesh.faces, baseline_mesh.faces)
                    or uv.shape != baseline_uv.shape or not np.allclose(uv, baseline_uv, atol=1e-9)):
                raise ValueError(f'Candidate must share original face indices and UV layout: {name}')
            if not np.isfinite(mesh.vertices).all():
                raise ValueError(f'Candidate has non-finite geometry: {name}')
            meshes[name] = (mesh, uv)

        def evaluate(mesh_file, settings, size, diagnostics):
            mesh, uv = meshes[mesh_file]
            image, report = bake_texture(
                handoff, mesh, uv, size=size, views=list(plan.views) if plan.views else None,
                diagnostics=diagnostics, semantic=plan.semantic, semantic_root=plan.root, **settings)
            appearance = appearance_report(handoff, mesh, uv, image, scale=plan.appearance_scale)
            samples = surface_samples(mesh, 0.5 * px / silhouette_scale)
            geometry = {'views': []}
            for view, source_mask in masks.items():
                rendered_mask, _depth = rasterize(mesh, camera(view), px, silhouette_scale, samples)
                intersect = np.logical_and(source_mask, rendered_mask).sum()
                union = np.logical_or(source_mask, rendered_mask).sum()
                geometry['views'].append({'view': view, 'iou': float(intersect / max(1, union))})
            return image, report['conflicts'], appearance, geometry
    else:
        evaluate = evaluator
    attempts: list[dict] = []
    best = None
    visited = set()
    with tempfile.TemporaryDirectory(prefix='.character-conflicts-', dir=out.parent) as directory:
        tmp = Path(directory)

        def trial(mesh_file: Path, settings: dict) -> dict | None:
            nonlocal best
            fingerprint = _fingerprint(mesh_file, settings)
            if fingerprint in visited or len(attempts) >= plan.max_evaluations:
                return None
            visited.add(fingerprint)
            evaluated = evaluate(mesh_file, settings, plan.evaluation_size, None)
            image, conflicts, appearance = evaluated[:3]
            geometry = evaluated[3] if len(evaluated) > 3 else None
            score = _objective(conflicts, appearance, plan.weights, geometry)
            reference = baseline if attempts else None
            reasons = _rejection(conflicts, appearance, reference, plan, geometry) if reference is not None else []
            improved = best is None or (not reasons and score < best['score'] - plan.min_improvement)
            info = {'index': len(attempts), 'mesh': str(mesh_file), 'settings': settings.copy(),
                    'score': round(score, 8), 'conflicts': conflicts, 'appearance': appearance,
                    'geometry': geometry,
                    'rejected': reasons, 'accepted': improved}
            attempts.append(info)
            if improved:
                best = info
                Image.fromarray(image).save(tmp / 'preview.png')
            return info

        baseline = None
        first = trial(source_mesh, dict(plan.settings))
        if first is None:
            raise RuntimeError('Baseline evaluation failed')
        baseline = first
        # Optional pre-generated geometry revisions share UV and registered frame.
        for candidate in plan.candidate_meshes:
            trial(candidate, dict(plan.settings))
        for _ in range(plan.rounds):
            if len(attempts) >= plan.max_evaluations:
                break
            before = best['index']
            start_mesh, start_settings = Path(best['mesh']), best['settings'].copy()
            for parameter, values in plan.steps:
                for value in values:
                    candidate = {**start_settings, parameter: value}
                    trial(start_mesh, candidate)
                    if len(attempts) >= plan.max_evaluations:
                        break
                if len(attempts) >= plan.max_evaluations:
                    break
            if best['index'] == before:
                break
        winner = best
        final_result = evaluate(Path(winner['mesh']), winner['settings'], plan.final_size, tmp / 'diagnostics')
        final_texture, final_conflicts, final_appearance = final_result[:3]
        final_geometry = final_result[3] if len(final_result) > 3 else None
        # Recheck guards at output resolution, not just the coarser search resolution.
        if plan.final_size != plan.evaluation_size:
            final_baseline_result = evaluate(source_mesh, dict(plan.settings), plan.final_size, None)
            _, final_baseline_conflicts, final_baseline_appearance = final_baseline_result[:3]
            final_baseline_geometry = final_baseline_result[3] if len(final_baseline_result) > 3 else None
            final_reasons = _rejection(final_conflicts, final_appearance,
                                       {'conflicts': final_baseline_conflicts,
                                        'appearance': final_baseline_appearance,
                                        'geometry': final_baseline_geometry}, plan, final_geometry)
            final_baseline_score = _objective(final_baseline_conflicts, final_baseline_appearance, plan.weights, final_baseline_geometry)
            if final_reasons or _objective(final_conflicts, final_appearance, plan.weights, final_geometry) >= final_baseline_score:
                winner = baseline
                final_result = evaluate(source_mesh, dict(plan.settings), plan.final_size, tmp / 'diagnostics')
                final_texture, final_conflicts, final_appearance = final_result[:3]
                final_geometry = final_result[3] if len(final_result) > 3 else None
        Image.fromarray(final_texture).save(tmp / 'texture.png')
        if Path(winner['mesh']) != source_mesh:
            shutil.copy2(winner['mesh'], tmp / 'mesh.obj')
        report = {
            'schema_version': 1,
            'status': 'improved' if winner['index'] != 0 else 'baseline-retained',
            'baseline_mesh_sha256': _sha(source_mesh),
            'selected_mesh': str(winner['mesh']),
            'selected_mesh_sha256': _sha(Path(winner['mesh'])),
            'selected_settings': winner['settings'],
            'selected_attempt_index': winner['index'],
            'selected_score': round(_objective(final_conflicts, final_appearance, plan.weights, final_geometry), 8),
            'final_geometry': final_geometry,
            'final_conflicts': final_conflicts,
            'final_appearance': final_appearance,
            'attempts': attempts,
            'limitations': [
                'Only appearance and source-view conflict scores are optimized automatically.',
                'Different OBJ geometry must be supplied as a candidate with identical face/UV topology.',
                'No automatic mesh penetration fix or verified 3D anatomical fidelity is implied.',
                'Source camera calibration and semantic masks must be independently verified.',
            ],
        }
        (tmp / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        shutil.move(str(tmp), str(out))
        return report
