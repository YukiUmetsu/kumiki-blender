"""Guarded, isolated repair of a registered experimental COLMAP sparse model.

This handles sparse models not produced by prepare-roma-candidate-database,
including an already rectified COLMAP/OpenMVS workspace. It never edits or
promotes the supplied models; only a new sparse candidate and QA report result.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .registered_camera_recovery import validate_camera_recovery
from .roma_camera_audit import measure_target_observations


def _name_map(model):
    registered = set(model.reg_image_ids())
    names = {model.images[i].name: (i, model.images[i]) for i in registered}
    if len(names) != len(registered):
        raise ValueError('Registered model contains duplicate image names')
    return names


def _original_hashes(path: Path) -> dict[str, str]:
    hashes = {}
    for part in ('cameras', 'images', 'points3D'):
        file = path / f'{part}.bin'
        if file.is_symlink() or not file.is_file():
            raise ValueError(f'Expected regular COLMAP binary file: {file}')
        with file.open('rb') as stream:
            hashes[file.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return hashes


def repair_roma_sparse_model(baseline_path: Path, candidate_path: Path,
                             output_path: Path, *, target: str,
                             remove_indices: list[int], exclude_cameras: list[str],
                             outlier_px: float = 8.0,
                             allow_duplicate_repair: bool = False,
                             min_remaining_points: int = 30, model_loader=None) -> dict:
    """Keep one added target view, drop other new views, and prune approved errors.

    Explicit removal requires independently audited indices, a surviving native
    3D track, and, for a duplicate in the same image, proof that its other
    observation stays linked. This is still *not* independent pose validation.
    """
    baseline_requested, candidate_requested, output_requested = (
        Path(baseline_path), Path(candidate_path), Path(output_path))
    before, after, destination = (baseline_requested.resolve(),
                                  candidate_requested.resolve(), output_requested.resolve())
    if (baseline_requested.is_symlink() or candidate_requested.is_symlink() or
            output_requested.is_symlink() or not before.is_dir() or not after.is_dir() or
            before == after or destination.exists() or
            any(destination.is_relative_to(root) or root.is_relative_to(destination)
                for root in (before, after))):
        raise ValueError('Require distinct real input models and unused isolated output directory')
    if (not isinstance(target, str) or not target or
            not 1 <= len(remove_indices) <= 20 or
            any(type(i) is not int or i < 0 for i in remove_indices) or
            len(remove_indices) != len(set(remove_indices)) or
            not 1 <= len(exclude_cameras) <= 20 or
            len(exclude_cameras) != len(set(exclude_cameras)) or
            target in exclude_cameras or not all(isinstance(n, str) and n for n in exclude_cameras) or
            not math.isfinite(outlier_px) or not .5 <= outlier_px <= 100 or
            not 10 <= min_remaining_points <= 1000 or
            type(allow_duplicate_repair) is not bool):
        raise ValueError('Invalid explicit sparse camera repair request')
    source_hashes = {'baseline': _original_hashes(before), 'candidate': _original_hashes(after)}
    if model_loader is None:
        import pycolmap
        model_loader = pycolmap.Reconstruction
    baseline, candidate = model_loader(before), model_loader(after)
    base_names, candidate_names = _name_map(baseline), _name_map(candidate)
    newly_added = set(candidate_names) - set(base_names)
    if newly_added != {target, *exclude_cameras}:
        raise ValueError(f'New camera names differ from explicitly requested target/exclusions: {sorted(newly_added)}')
    before_gate = validate_camera_recovery(baseline, candidate)
    if (not before_gate['accepted'] or
            before_gate.get('max_original_center_displacement_ratio', math.inf) > 1e-9 or
            before_gate.get('max_original_direction_change_degrees', math.inf) > 1e-6):
        raise ValueError('Source baseline cameras are not unchanged in the candidate model')
    image_id, image = candidate_names[target]
    observations = measure_target_observations(candidate, target, None)
    by_index = {row['point2D_idx']: row for row in observations}
    if any(i not in by_index or by_index[i]['error_px'] <= outlier_px for i in remove_indices):
        raise ValueError('Every requested observation must be tracked and exceed the reprojection threshold')
    by_point = defaultdict(list)
    for row in observations:
        by_point[row['point3D_id']].append(row)
    duplicated = {point_id: rows for point_id, rows in by_point.items() if len(rows) > 1}
    selected = set(remove_indices)
    for point_id, rows in by_point.items():
        removing = [row for row in rows if row['point2D_idx'] in selected]
        if not removing:
            continue
        point = candidate.points3D[point_id]
        track = point.track
        if track.length() - len(removing) < 2:
            raise ValueError('Removing observation would delete its entire 3D point')
        target_track = {int(e.point2D_idx) for e in track.elements
                        if int(e.image_id) == int(image_id)}
        target_rows = {row['point2D_idx'] for row in rows}
        if target_rows != target_track:
            raise ValueError('3D track and target 2D observations disagree')
        if len(rows) > 1:
            remaining = [row for row in rows if row['point2D_idx'] not in selected]
            if (not allow_duplicate_repair or len(rows) != 2 or
                    len(removing) != 1 or len(remaining) != 1 or
                    removing[0]['error_px'] <= remaining[0]['error_px']):
                raise ValueError('Duplicate track repair requires an explicit opt-in and retaining the lower-error observation')
    # Never remove any original camera. On modern COLMAP one image belongs to
    # one frame; explicitly reject shared frames rather than de-registering a
    # multi-camera rig by accident.
    for name in exclude_cameras:
        excluded_image = candidate_names[name][1]
        count = sum(int(other.frame_id) == int(excluded_image.frame_id)
                    for _, other in candidate_names.values())
        if count != 1:
            raise ValueError('Cannot exclude a camera that shares a registered frame')
    # Mutations affect only this in-memory copy, never either source folder.
    for index in sorted(remove_indices, reverse=True):
        candidate.delete_observation(image_id, index)
    for name in sorted(exclude_cameras):
        candidate.deregister_frame(int(candidate_names[name][1].frame_id))
    after_names = _name_map(candidate)
    if set(after_names) != set(base_names) | {target}:
        raise ValueError('Sparse repair changed unexpected registered cameras')
    after_gate = validate_camera_recovery(baseline, candidate)
    if not after_gate['accepted'] or after_gate['new_filenames'] != [target]:
        raise ValueError('Repaired candidate does not pass preserved-baseline camera gate')
    remaining_rows = measure_target_observations(candidate, target, None)
    if len(remaining_rows) < min_remaining_points:
        raise ValueError('Repaired target has insufficient retained 3D observations')
    if any(c > 1 for c in Counter(row['point3D_id'] for row in remaining_rows).values()):
        raise ValueError('Repaired target still has duplicate 3D point associations')
    if any(row['error_px'] > outlier_px for row in remaining_rows):
        raise ValueError('Repaired target retains outliers above the chosen threshold')
    report = {'status': 'isolated-camera-candidate / independent-qa-required',
              'target': target, 'registered_images': len(after_names),
              'source_sha256': source_hashes,
              'before_camera_gate': before_gate, 'after_camera_gate': after_gate,
              'removed_observations': [by_index[i] for i in sorted(remove_indices)],
              'excluded_unverified_cameras': sorted(exclude_cameras),
              'duplicate_target_3d_points_before': sorted(duplicated),
              'remaining_target_3d_observations': len(remaining_rows),
              'remaining_target_max_reprojection_px': max(row['error_px'] for row in remaining_rows),
              'distribution_allowed': False,
              'warning': 'Registration, even with low residuals, is not independent object/pose evidence. The resulting sparse model is not yet a dense-ready rectified revision.'}
    destination.mkdir(parents=True, exist_ok=False)
    sparse_output = destination / 'sparse'
    sparse_output.mkdir()
    candidate.write(sparse_output)
    (destination / 'roma-sparse-repair-report.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report
