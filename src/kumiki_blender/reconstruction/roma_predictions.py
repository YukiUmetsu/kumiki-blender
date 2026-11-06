"""Generate *hypothetical* RoMa 2D-to-existing-3D-track correspondences.

Only already triangulated COLMAP tracks may be queried. Predictions are written
to a new directory, never into the original COLMAP database or sparse model.
The predicted pixel is not an independently measured keypoint or a validated
camera observation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from .roma_track_consensus import agree_track_predictions
from .sfm_cohort_recovery import _registered_names, _source_paths, _verify_staged_images
from .sfm_track_support import MAX_POINTS, MAX_TRACK_OBSERVATIONS


def _model_observations(paths: dict[str, Path], model: int, anchors: set[str]) -> dict[str, list[dict]]:
    """Read original 2D observations, linked to original 3D XYZ (never guesses)."""
    point_file = paths['sparse'] / 'text-models' / str(model) / 'points3D.txt'
    image_file = paths['text']
    if point_file.is_symlink() or not point_file.is_file():
        raise ValueError('Missing or linked COLMAP points3D.txt')
    xyz = {}
    with point_file.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8 or len(parts) % 2:
                raise ValueError('Invalid COLMAP 3D point record')
            pid = int(parts[0])
            coordinates = [float(v) for v in parts[1:4]]
            if pid in xyz or not all(map(math.isfinite, coordinates)):
                raise ValueError('Duplicate or non-finite COLMAP 3D point')
            xyz[pid] = coordinates
            if len(xyz) > MAX_POINTS:
                raise ValueError('COLMAP 3D points exceed memory limit')
    result = defaultdict(list)
    seen = set()
    observation_count = 0
    lines = iter(image_file.read_text(encoding='utf-8').splitlines())
    for line in lines:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        header = line.split()
        if len(header) < 10:
            raise ValueError('Malformed COLMAP registered image')
        name = header[9]
        image_id = int(header[0])
        observations = next(lines, '').split()
        if len(observations) % 3:
            raise ValueError(f'Malformed COLMAP image observations: {name}')
        if name not in anchors:
            continue
        if name in seen:
            raise ValueError('Duplicate registered anchor image')
        seen.add(name)
        for feature in range(0, len(observations), 3):
            px, py, pid = (float(observations[feature]), float(observations[feature + 1]),
                           int(observations[feature + 2]))
            if pid == -1:
                continue
            if pid not in xyz or not math.isfinite(px) or not math.isfinite(py):
                raise ValueError('Invalid tracked image observation')
            result[name].append({'track_id': pid, 'xyz': xyz[pid], 'pixel': [px, py],
                                 'source_image_id': image_id, 'feature_index': feature // 3})
            observation_count += 1
            if observation_count > MAX_TRACK_OBSERVATIONS:
                raise ValueError('Too many COLMAP image observations')
    if seen != anchors:
        raise ValueError(f'Unregistered or missing anchor: {sorted(anchors - seen)}')
    return result


def _qa_rasters(paths: dict[str, Path], names: set[str]):
    """Read optional tri-state source confidence rasters; hash-check first."""
    from PIL import Image
    index_file = paths['sparse'] / 'subject-feature-index.json'
    if not index_file.exists():
        return {}
    if index_file.is_symlink() or not index_file.is_file():
        raise ValueError('Unsafe subject-feature-index.json')
    index = json.loads(index_file.read_text(encoding='utf-8'))
    if index.get('schema') != 'subject-feature-index-v1' or not isinstance(index.get('images'), dict):
        raise ValueError('Unexpected subject-feature-index schema')
    images = {}
    for name in names:
        entry = index['images'].get(name)
        if entry is None:
            continue
        path = Path(entry['confidence_path'])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Unavailable or linked subject QA raster for {name}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != entry.get('confidence_sha256'):
            raise ValueError(f'Changed subject QA raster for {name}')
        with Image.open(path) as raster:
            if raster.mode != 'L':
                raise ValueError(f'Invalid subject QA raster mode for {name}')
            raster.load()
            images[name] = raster.copy()
    return images


def _not_background(raster, point) -> bool:
    if raster is None:
        return True  # Unknown != verified foreground; subject QA remains required.
    x, y = point
    if not (0 <= x < raster.width and 0 <= y < raster.height):
        return False
    return raster.getpixel((int(x), int(y))) != 0


def _fingerprint_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Unsafe or missing RoMa source file: {path}')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    """Write a complete receipt before atomically making it visible."""
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         prefix='.roma-', suffix='.tmp', dir=path.parent,
                                         delete=False) as stream:
            temp = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _check_resumable_predictions(destination: Path, report: dict, expected: dict,
                                 staged: dict[str, dict]) -> set[str]:
    """Skip only hash-verified target predictions from an identical run."""
    for key in ('source_revision', 'anchors', 'requested_targets', 'source_sha256',
                'model', 'min_confidence', 'max_tracks_per_anchor', 'device',
                'coarse_resolution', 'refine_resolution'):
        if report.get(key) != expected[key]:
            raise ValueError(f'RoMa resume mismatch for {key}; start a new output revision')
    if report.get('status') not in ('running-unverified-roma-predictions',
                                     'unverified-roma-predictions'):
        raise ValueError('RoMa resume receipt has an unexpected status')
    completed = report.get('targets')
    if not isinstance(completed, dict) or not set(completed) <= set(expected['requested_targets']):
        raise ValueError('RoMa resume receipt contains unexpected targets')
    for name in expected['requested_targets']:
        path = destination / f'{name}.roma.json'
        if name not in completed:
            if path.exists() or path.is_symlink():
                raise ValueError(f'Unrecorded RoMa output exists for {name}; do not overwrite it')
            continue
        record = completed[name]
        if (not isinstance(record, dict) or record.get('file') != path.name or
                not isinstance(record.get('sha256'), str) or
                type(record.get('predictions')) is not int or
                type(record.get('agreed_tracks')) is not int or
                _fingerprint_file(path) != record['sha256']):
            raise ValueError(f'Incomplete or changed RoMa receipt for {name}')
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError(f'Invalid resumed RoMa predictions for {name}') from error
        if (not isinstance(payload, dict) or payload.get('target') != name or
                payload.get('status') != 'predictions-not-measurements' or
                payload.get('source_photo_sha256') != staged[name]['sha256'] or
                not isinstance(payload.get('predictions'), list) or
                len(payload['predictions']) != record['predictions']):
            raise ValueError(f'RoMa resume payload does not match its receipt: {name}')
        try:
            agreed_count = len(agree_track_predictions(payload['predictions']))
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise ValueError(f'Invalid stored RoMa predictions for {name}') from error
        if agreed_count != record['agreed_tracks']:
            raise ValueError(f'RoMa resume agreement count changed for {name}')
    return set(completed)


class RoMaPredictor:
    """Optional MPS/CPU RoMa inference; Torch and romatch are loaded on demand."""

    def __init__(self, *, device='auto', coarse_resolution=420, refine_resolution=560):
        if not 128 <= coarse_resolution <= 1024 or not 128 <= refine_resolution <= 2048:
            raise ValueError('RoMa resolution outside safe bounds')
        try:
            import torch
            from romatch import roma_outdoor
        except ImportError as error:
            raise RuntimeError('RoMa requires torch and romatch; install optional dependencies in the active reconstruction environment') from error
        if device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        if device not in ('mps', 'cpu') or (device == 'mps' and not torch.backends.mps.is_available()):
            raise ValueError('Selected RoMa device is not available')
        self.torch = torch
        self.device = device
        self.model = roma_outdoor(device=device, coarse_res=coarse_resolution,
                                  upsample_res=(refine_resolution, refine_resolution))

    def predict(self, anchor: Path, target: Path, pixels: list[list[float]],
                anchor_size: tuple[int, int]) -> list[tuple[list[float], float]]:
        """Sample A-to-B dense warp exactly at the original anchor keypoints."""
        torch = self.torch
        import torch.nn.functional as functional
        width, height = anchor_size
        with torch.inference_mode():
            warp, certainty = self.model.match(str(anchor), str(target), device=self.device)
            # romatch returns one batch entry for a pair of image paths.
            if warp.ndim == 4 and certainty.ndim == 3 and len(warp) == 1 and len(certainty) == 1:
                warp, certainty = warp[0], certainty[0]
            # Symmetric RoMa concatenates A->B and B->A along width. The first
            # half is the only one whose source pixels refer to anchor A.
            out_h, out_w = warp.shape[:2]
            if warp.ndim != 3 or warp.shape[-1] != 4 or certainty.shape != (out_h, out_w):
                raise ValueError('Unsupported RoMa dense warp/certainty layout')
            ab_w = out_w // 2 if self.model.symmetric else out_w
            coords = torch.tensor([[2 * x / width - 1, 2 * y / height - 1]
                                   for x, y in pixels], dtype=torch.float32, device=self.device)
            if not len(coords):
                return []
            grid = coords[None, None]
            projected = functional.grid_sample(warp[:, :ab_w, 2:4].permute(2, 0, 1)[None],
                                               grid, mode='bilinear', align_corners=False)[0, :, 0].T
            confidence = functional.grid_sample(certainty[:, :ab_w][None, None],
                                                 grid, mode='bilinear', align_corners=False)[0, 0, 0]
            values = torch.cat([projected, confidence[:, None]], dim=1).float().cpu().tolist()
            return [([float(row[0]), float(row[1])], float(row[2])) for row in values]


def generate_roma_predictions(revision: Path, output: Path, *, targets: list[str],
                              anchors: list[str], model: int = 0, min_confidence: float = 0.05,
                              max_tracks_per_anchor: int = 4000, device: str = 'auto',
                              coarse_resolution: int = 420, refine_resolution: int = 560,
                              predictor=None, resume: bool = False) -> dict:
    """Generate isolated JSON candidates for one or more unregistered photos.

    No predicted correspondence is promoted to a registered camera or used to
    alter the source database. The importer will independently recheck source
    hashes, anchor tracks, and cross-view consistency.
    """
    paths = _source_paths(Path(revision), model)
    requested, destination = Path(output), Path(output).resolve()
    original = paths['revision'].resolve()
    if (requested.is_symlink() or (destination.exists() and not resume) or
            destination.is_relative_to(original) or original.is_relative_to(destination)):
        raise ValueError('RoMa output must be a new directory separate from the source')
    if resume and not destination.is_dir():
        raise ValueError('RoMa resume requires the existing output directory')
    if not (1 <= len(targets) <= 24 and 2 <= len(anchors) <= 12 and
            len(set(targets)) == len(targets) and len(set(anchors)) == len(anchors)):
        raise ValueError('Require 1–24 unique targets and 2–12 distinct anchors')
    if not math.isfinite(min_confidence) or not 0 < min_confidence <= 1:
        raise ValueError('RoMa minimum confidence must be in (0,1]')
    if not 10 <= max_tracks_per_anchor <= 20000:
        raise ValueError('RoMa anchor track limit must be 10–20000')
    if device not in ('auto', 'mps', 'cpu'):
        raise ValueError('RoMa device must be auto, mps or cpu')
    registered = _registered_names(paths['text'])
    if not set(anchors) <= registered or set(targets) & registered:
        raise ValueError('All anchors must be registered and all targets unregistered')
    import sqlite3
    with sqlite3.connect(paths['database'].resolve().as_uri() + '?mode=ro', uri=True) as db:
        staged = _verify_staged_images(paths, {name for (name,) in db.execute('SELECT name FROM images')})
        if not set(targets) <= staged.keys():
            raise ValueError('Unknown staged target photo')
    observations = _model_observations(paths, model, set(anchors))
    from PIL import Image
    dimensions = {}
    for name in set(targets + anchors):
        with Image.open(paths['images'] / name) as photo:
            dimensions[name] = photo.size
    rasters = _qa_rasters(paths, set(targets + anchors))
    for name, raster in rasters.items():
        if raster.size != dimensions[name]:
            raise ValueError(f'Subject QA raster size differs from original: {name}')
    model_points = paths['sparse'] / 'text-models' / str(model) / 'points3D.txt'
    db_wal = Path(str(paths['database']) + '-wal')
    if db_wal.is_symlink():
        raise ValueError('Unsafe linked RoMa source database WAL')
    source_sha256 = {
        'database': _fingerprint_file(paths['database']),
        'database_wal': _fingerprint_file(db_wal) if db_wal.exists() else None,
        'model_images': _fingerprint_file(paths['text']),
        'model_points': _fingerprint_file(model_points),
        'staged_index': _fingerprint_file(paths['index']),
        'subject_index': (_fingerprint_file(paths['sparse'] / 'subject-feature-index.json')
                          if (paths['sparse'] / 'subject-feature-index.json').exists() else None),
    }
    expected = {'source_revision': str(original), 'anchors': anchors,
                'requested_targets': targets, 'source_sha256': source_sha256,
                'model': model, 'min_confidence': min_confidence,
                'max_tracks_per_anchor': max_tracks_per_anchor,
                'device': device, 'coarse_resolution': coarse_resolution,
                'refine_resolution': refine_resolution}
    report_file = destination / 'roma-report.json'
    if resume:
        if report_file.is_symlink() or not report_file.is_file():
            raise ValueError('RoMa resume requires a regular roma-report.json')
        try:
            report = json.loads(report_file.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError('Cannot parse RoMa resume receipt') from error
        if not isinstance(report, dict):
            raise ValueError('RoMa resume receipt must be an object')
        completed = _check_resumable_predictions(destination, report, expected, staged)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        report = {**expected, 'status': 'running-unverified-roma-predictions',
                  'targets': {}, 'subject_qa_available': sorted(rasters), 'limitations':
                  'Predicted pixels are not independent keypoints or accepted camera observations.'}
        _atomic_json(report_file, report)
        completed = set()
    if len(completed) < len(targets) and predictor is None:
        predictor = RoMaPredictor(device=device, coarse_resolution=coarse_resolution,
                                  refine_resolution=refine_resolution)
    report['resumed_completed_targets'] = sorted(completed)
    report['status'] = 'running-unverified-roma-predictions'
    _atomic_json(report_file, report)
    for target in targets:
        if target in completed:
            continue
        guesses = []
        for anchor in anchors:
            # Bound inference and keep sampling deterministic by 3D track ID.
            points = sorted(observations[anchor], key=lambda row: (row['track_id'], row['feature_index']))
            if len(points) > max_tracks_per_anchor:
                stride = len(points) / max_tracks_per_anchor
                points = [points[int(i * stride)] for i in range(max_tracks_per_anchor)]
            results = predictor.predict(paths['images'] / anchor, paths['images'] / target,
                                        [point['pixel'] for point in points], dimensions[anchor])
            if len(results) != len(points):
                raise ValueError(f'RoMa returned mismatched prediction count for {anchor}')
            for point, (normalized, confidence) in zip(points, results):
                if (len(normalized) != 2 or not all(math.isfinite(v) for v in normalized)
                        or not math.isfinite(confidence) or confidence < min_confidence
                        or not all(-1 < coordinate < 1 for coordinate in normalized)):
                    continue
                px = [(normalized[0] + 1) * dimensions[target][0] / 2,
                      (normalized[1] + 1) * dimensions[target][1] / 2]
                if not (_not_background(rasters.get(anchor), point['pixel']) and
                        _not_background(rasters.get(target), px)):
                    continue
                guesses.append({'anchor': anchor, 'track_id': point['track_id'],
                                'xyz': point['xyz'], 'pixel': px,
                                'subject_candidate': True,
                                'confidence': confidence})
        consensus = agree_track_predictions(guesses)
        # The outdoor model may take minutes per pair on MPS. Detect a photo
        # edited or replaced during inference before issuing a valid-looking
        # SHA-pinned prediction receipt for stale pixels.
        for name in set(anchors + [target]):
            if _fingerprint_file(paths['images'] / name) != staged[name]['sha256']:
                raise ValueError(f'Changed staged photo during RoMa prediction: {name}')
        payload = {'target': target, 'predictions': guesses, 'status': 'predictions-not-measurements',
                   'source_photo_sha256': staged[target]['sha256']}
        output_file = destination / f'{target}.roma.json'
        _atomic_json(output_file, payload)
        report['targets'][target] = {'predictions': len(guesses), 'agreed_tracks': len(consensus),
                                      'file': output_file.name, 'sha256': _fingerprint_file(output_file)}
        _atomic_json(report_file, report)
    report['status'] = 'unverified-roma-predictions'
    _atomic_json(report_file, report)
    return report
