"""Audit experimental RoMa registrations and optionally prune explicit outliers.

Even low reprojection errors of predicted, synthetic keypoints are circular
evidence; the output is never a validated or production-ready camera revision.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

from .registered_camera_recovery import validate_camera_recovery
from .sfm_cohort_recovery import _source_paths


def measure_target_observations(model, name: str, original_keypoints: int) -> list[dict]:
    """Read calibrated native reprojection errors and observation provenance."""
    import numpy as np

    target = next(((image_id, image) for image_id, image in model.images.items()
                   if image.name == name and image_id in model.reg_image_ids()), None)
    if target is None:
        raise ValueError(f'Target camera is not registered: {name}')
    _, image = target
    camera = model.cameras[image.camera_id]
    pose = image.cam_from_world
    if callable(pose):
        pose = pose()
    report = []
    for point_index, observation in enumerate(image.points2D):
        if not observation.has_point3D():
            continue
        point3d = model.points3D.get(int(observation.point3D_id))
        if point3d is None:
            raise ValueError(f'COLMAP image references missing 3D point {observation.point3D_id}')
        xyz_camera = np.asarray(pose * point3d.xyz, dtype=float).reshape(3)
        xy = np.asarray(observation.xy, dtype=float).reshape(2)
        if not np.isfinite(xyz_camera).all() or xyz_camera[2] <= 0 or not np.isfinite(xy).all():
            raise ValueError('Camera observation has non-finite coordinates or negative depth')
        projected = np.asarray(camera.img_from_cam(xyz_camera), dtype=float).reshape(2)
        if not np.isfinite(projected).all():
            raise ValueError('Camera produced non-finite pixel coordinates')
        report.append({'point2D_idx': point_index, 'point3D_id': int(observation.point3D_id),
                       'source': 'existing-feature' if point_index < original_keypoints else
                                 'roma-predicted-keypoint',
                       'error_px': float(np.linalg.norm(projected - xy))})
    return report


def audit_roma_candidate_camera(revision: Path, candidate: Path, output: Path, *,
                                outlier_px: float = 8.0, remove_indices: list[int] | None = None,
                                model: int = 0, model_loader=None) -> dict:
    """Write immutable QA; remove only explicitly chosen high-error indices.

    The source model and experimental registration stay untouched. The selected
    indices must be real 3D observations above the supplied outlier threshold;
    the function rejects an accidental request to remove good measurements.
    """
    source = _source_paths(Path(revision), model)
    candidate_dir = Path(candidate).resolve()
    output_path = Path(output).resolve()
    removed = remove_indices or []
    if (Path(candidate).is_symlink() or not candidate_dir.is_dir() or
            Path(output).is_symlink() or output_path.exists() or
            output_path.is_relative_to(source['revision']) or
            output_path.is_relative_to(candidate_dir) or
            source['revision'].is_relative_to(output_path) or
            candidate_dir.is_relative_to(output_path)):
        raise ValueError('Audit output must be unused and outside both source and candidate')
    if (not math.isfinite(outlier_px) or not 0.5 <= outlier_px <= 100 or
            len(removed) > 20 or len(set(removed)) != len(removed) or
            any(type(index) is not int or index < 0 for index in removed)):
        raise ValueError('Invalid outlier threshold or removal indices')
    receipt = candidate_dir / 'candidate-report.json'
    model_path = candidate_dir / 'registered'
    if (receipt.is_symlink() or not receipt.is_file() or model_path.is_symlink() or
            not model_path.is_dir()):
        raise ValueError('Missing or unsafe isolated RoMa registration candidate')
    info = json.loads(receipt.read_text(encoding='utf-8'))
    target = info.get('target')
    if info.get('status') != 'unverified-candidate-matches' or not isinstance(target, str):
        raise ValueError('Not an unverified RoMa candidate receipt')
    with source['database'].open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != info.get('source_database_sha256'):
        raise ValueError('Source COLMAP database has changed since RoMa candidate creation')
    with sqlite3.connect(source['database'].resolve().as_uri() + '?mode=ro', uri=True) as db:
        original = db.execute('SELECT k.rows FROM images i JOIN keypoints k ON i.image_id=k.image_id WHERE i.name=?',
                              (target,)).fetchone()
    if original is None or type(original[0]) is not int or original[0] < 0:
        raise ValueError('Missing source target keypoint count')
    if model_loader is None:
        import pycolmap
        model_loader = pycolmap.Reconstruction
    baseline = model_loader(source['model'])
    registered = model_loader(model_path)
    native = validate_camera_recovery(baseline, registered)
    observations = measure_target_observations(registered, target, original[0])
    outliers = [row for row in observations if row['error_px'] > outlier_px]
    outlier_ids = {row['point2D_idx'] for row in outliers}
    if not set(removed) <= outlier_ids:
        raise ValueError('Only explicitly identified high-error 3D observations may be removed')
    from collections import Counter
    duplicate_points = {pid for pid, count in Counter(
        row['point3D_id'] for row in observations).items() if count > 1}
    if any(row['point2D_idx'] in removed and row['point3D_id'] in duplicate_points
           for row in observations):
        raise ValueError('Selected outlier shares a 3D point with another observation in the same image; inspect its native track before pruning')
    for row in observations:
        if row['point2D_idx'] not in removed:
            continue
        track = getattr(registered.points3D[row['point3D_id']], 'track', None)
        if track is not None and track.length() <= 2:
            raise ValueError('Removing this observation would delete the entire 3D point; inspect the track before pruning')
    # Creating output only after all read-only checks have passed avoids
    # leaving a misleading success report for bad or unsafe inputs.
    output_path.mkdir(parents=True, exist_ok=False)
    report = {'status': 'experimental-camera-needs-independent-validation',
              'target': target, 'outlier_threshold_px': outlier_px,
              'native_camera_geometry_gate': native,
              'measured_feature_observations': sum(row['source'] == 'existing-feature' for row in observations),
              'synthetic_feature_observations': sum(row['source'] == 'roma-predicted-keypoint' for row in observations),
              'high_error_observations': outliers, 'remove_requested': sorted(removed),
              'duplicate_target_point3D_ids': sorted(duplicate_points),
              'all_observations': observations, 'original_source_modified': False,
              'warning': 'RoMa-generated keypoints are predictions, not independent measured evidence. Inspect camera geometry and held-out real image features before dense reconstruction.'}
    if removed:
        for index in sorted(removed, reverse=True):
            registered.delete_observation(next(image_id for image_id, image in registered.images.items()
                                               if image.name == target), index)
        pruned = output_path / 'pruned-model'
        pruned.mkdir()
        registered.write(pruned)
        report['pruned_model'] = 'pruned-model'
        report['remaining_native_camera_geometry_gate'] = validate_camera_recovery(baseline, registered)
    (output_path / 'roma-camera-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report
