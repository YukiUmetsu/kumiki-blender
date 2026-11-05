"""Build an isolated COLMAP database containing candidate RoMa observations.

This does not verify matches, estimate a camera, or alter a source revision.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
from contextlib import closing
from pathlib import Path

from .sfm_cohort_recovery import _registered_names, _source_paths, _verify_staged_images
from .sfm_track_support import MAX_IMAGE_ID, _tracks
from .roma_track_consensus import agree_track_predictions


def _verified_xyz(point_file: Path, requested_ids: set[int]) -> dict[int, list[float]]:
    """Check claimed coordinates against the untouched registered 3D model."""
    if point_file.is_symlink() or not point_file.is_file():
        raise ValueError('Missing or unsafe registered 3D points')
    result = {}
    with point_file.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.split()
            if len(fields) < 8:
                raise ValueError('Malformed registered 3D point record')
            point_id = int(fields[0])
            if point_id not in requested_ids:
                continue
            values = [float(v) for v in fields[1:4]]
            if point_id in result or not all(map(math.isfinite, values)):
                raise ValueError('Duplicated or non-finite registered 3D point')
            result[point_id] = values
    if set(result) != requested_ids:
        raise ValueError('RoMa prediction references a missing 3D point')
    return result


def _filter_ambiguous_pixels(candidates: list[dict], original: list[tuple[float, float]],
                             *, radius: float = 2.0) -> tuple[list[dict], int]:
    """Drop all spatially conflicting synthetic observations, never pick a winner."""
    from collections import defaultdict
    buckets = defaultdict(list)
    def cell(pixel):
        return (math.floor(pixel[0] / radius), math.floor(pixel[1] / radius))
    for pixel in original:
        buckets[cell(pixel)].append((pixel, None))
    for index, item in enumerate(candidates):
        buckets[cell(item['pixel'])].append((item['pixel'], index))
    rejected = set()
    for index, item in enumerate(candidates):
        x, y = cell(item['pixel'])
        for cx in range(x - 1, x + 2):
            for cy in range(y - 1, y + 2):
                for other, other_index in buckets.get((cx, cy), []):
                    if other_index == index:
                        continue
                    if math.dist(item['pixel'], other) < radius:
                        rejected.add(index)
                        if other_index is not None:
                            rejected.add(other_index)
    return [item for index, item in enumerate(candidates) if index not in rejected], len(rejected)


def prepare_roma_candidate_database(revision, predictions_path, output, *, model=0):
    """Add multiview-consistent candidate matches to a *new* database copy.

    Predictions JSON is {target: staged filename, predictions: [{track_id,
    xyz, pixel, anchor, subject_candidate}, ...]}. Existing tracked anchor
    keypoints are reused; new target observations have zero descriptor rows
    and therefore the output database must not be used for feature matching.
    Run COLMAP geometric verification and camera registration separately.
    """
    paths = _source_paths(Path(revision), model)
    source_root, target = paths['revision'].resolve(), Path(output).resolve()
    if (Path(output).is_symlink() or target.exists() or target.is_relative_to(source_root)
            or source_root.is_relative_to(target)):
        raise ValueError('RoMa import requires a new directory outside the source revision')
    prediction_file = Path(predictions_path)
    if prediction_file.is_symlink() or not prediction_file.is_file():
        raise ValueError('RoMa predictions must be an existing regular JSON file')
    evidence = json.loads(prediction_file.read_text())
    if not isinstance(evidence, dict) or not isinstance(evidence.get('predictions'), list):
        raise ValueError('RoMa input must contain a predictions list and target filename')
    name = evidence.get('target')
    registered = _registered_names(paths['text'])
    with closing(sqlite3.connect(paths['database'].resolve().as_uri() + '?mode=ro', uri=True)) as source:
        names = {name: image_id for image_id, name in source.execute('SELECT image_id,name FROM images')}
        _verify_staged_images(paths, set(names))
        if name not in names or name in registered:
            raise ValueError('RoMa target must be an unregistered staged image')
        predictions = evidence['predictions']
        if len(predictions) > 200000:
            raise ValueError('RoMa prediction count exceeds the import limit')
        def valid_prediction(item):
            if (not isinstance(item, dict) or item.get('anchor') not in registered
                    or type(item.get('track_id')) is not int or item['track_id'] < 0
                    or type(item.get('subject_candidate')) is not bool):
                return False
            for field, size in (('pixel', 2), ('xyz', 3)):
                value = item.get(field)
                if (not isinstance(value, list) or len(value) != size or
                        any(type(n) not in (int, float) or not math.isfinite(n) for n in value)):
                    return False
            return True
        if not all(valid_prediction(item) for item in predictions):
            raise ValueError('Each prediction needs a registered anchor, integer track ID, finite pixel/xyz coordinates and boolean subject label')
        accepted = agree_track_predictions(predictions)
        if not accepted:
            raise ValueError('No track has agreeing predictions from multiple registered views')
        real_xyz = _verified_xyz(paths['sparse'] / 'text-models' / str(model) / 'points3D.txt',
                                 {item['track_id'] for item in accepted})
        if any(math.dist(item['xyz'], real_xyz[item['track_id']]) > 1e-5 for item in accepted):
            raise ValueError('RoMa 3D coordinates disagree with registered source tracks')
        track_lookup = _tracks(paths['sparse'] / 'text-models' / str(model) / 'points3D.txt')
        if track_lookup is None:
            raise ValueError('Source model needs exported 3D tracks')
        lookup = {}
        for (source_image, feature), point_id in track_lookup.items():
            key = (source_image, point_id)
            lookup.setdefault(key, []).append(feature)
        with closing(sqlite3.connect(':memory:')) as checked:
            # Load the exact database snapshot before writing any output.
            source.backup(checked)
            image_id = names[name]
            record = checked.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (image_id,)).fetchone()
            if record is None:
                raise ValueError('Target image has no extracted keypoint table')
            rows, cols, blob = record
            if cols not in (2, 4, 6) or blob is None or len(blob) != rows * cols * 4:
                raise ValueError('Target keypoint table has an invalid binary layout')
            if rows < 0 or rows > 2_000_000:
                raise ValueError('Target keypoint count exceeds safety bound')
            previous = [row[:2] for row in struct.iter_unpack('<' + 'f' * cols, blob)]
            accepted, skipped_ambiguous = _filter_ambiguous_pixels(accepted, previous)
            if not accepted:
                raise ValueError('All agreeing RoMa predictions overlap real or conflicting target keypoints')
            camera = checked.execute('SELECT c.width,c.height FROM images i JOIN cameras c ON c.camera_id=i.camera_id WHERE i.image_id=?', (image_id,)).fetchone()
            if camera is None or any(not (0 <= p['pixel'][0] < camera[0] and 0 <= p['pixel'][1] < camera[1]) for p in accepted):
                raise ValueError('RoMa candidate falls outside target image bounds')
            anchor_matches = {}
            for offset, item in enumerate(accepted):
                for anchor in item['anchors']:
                    anchor_id = names[anchor]
                    features = lookup.get((anchor_id, item['track_id']), [])
                    if len(features) != 1:
                        raise ValueError(f'Anchor {anchor} must have exactly one observation of track {item["track_id"]}')
                    anchor_matches.setdefault(anchor, []).append((features[0], rows + offset))
            desc = checked.execute('SELECT rows,cols,data FROM descriptors WHERE image_id=?', (image_id,)).fetchone()
            if desc is None or desc[0] != rows or desc[1] <= 0 or desc[2] is None or len(desc[2]) != rows * desc[1]:
                raise ValueError('Target descriptor table does not match its keypoints')
            additions = bytearray()
            for item in accepted:
                values = [float(item['pixel'][0]), float(item['pixel'][1])] + ([1.0, 0.0] if cols == 4 else [1.0, 0.0, 0.0, 1.0] if cols == 6 else [])
                additions.extend(struct.pack('<' + 'f' * cols, *values))
            checked.execute('UPDATE keypoints SET rows=?,data=? WHERE image_id=?', (rows + len(accepted), blob + additions, image_id))
            checked.execute('UPDATE descriptors SET rows=?,data=? WHERE image_id=?', (rows + len(accepted), desc[2] + bytes(len(accepted) * desc[1]), image_id))
            for anchor, pairs in sorted(anchor_matches.items()):
                anchor_id = names[anchor]
                low, high = sorted((anchor_id, image_id))
                pair_id = low * MAX_IMAGE_ID + high
                existing = checked.execute('SELECT rows,cols,data FROM matches WHERE pair_id=?', (pair_id,)).fetchone()
                if existing and (existing[1] != 2 or (existing[0] and (existing[2] is None or len(existing[2]) != 8 * existing[0]))):
                    raise ValueError(f'Invalid existing match data for {anchor}')
                matches = set(struct.iter_unpack('<II', existing[2])) if existing and existing[0] else set()
                matches.update((a, b) if anchor_id < image_id else (b, a) for a, b in pairs)
                payload = b''.join(struct.pack('<II', *pair) for pair in sorted(matches))
                checked.execute('INSERT OR REPLACE INTO matches(pair_id,rows,cols,data) VALUES(?,?,2,?)', (pair_id, len(matches), payload))
                checked.execute('DELETE FROM two_view_geometries WHERE pair_id=?', (pair_id,))
            checked.commit()
            target.mkdir(parents=True, exist_ok=False)
            with closing(sqlite3.connect(target / 'database.db')) as destination:
                checked.backup(destination)
    with paths['database'].open('rb') as stream:
        source_database_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'status': 'unverified-candidate-matches', 'target': name,
              'source_database_sha256': source_database_sha256,
              'candidate_tracks': len(accepted), 'anchor_counts': {a: len(v) for a, v in anchor_matches.items()},
              'ambiguous_predictions_skipped': skipped_ambiguous,
              'predictions_sha256': hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
              'warning': 'Target descriptors include zero placeholders. Never use this database for feature extraction or matching. Run geometric verification and PnP; no camera has been accepted.'}
    (target / 'candidate-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
