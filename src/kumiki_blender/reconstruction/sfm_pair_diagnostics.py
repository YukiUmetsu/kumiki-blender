"""Read-only matching-stage evidence for registered and unregistered photos."""
from __future__ import annotations

from contextlib import closing
import json
import sqlite3
import struct
from pathlib import Path

from .sfm_track_support import MAX_IMAGE_ID, _tracks


def diagnose_pairs(database, tracks_path, registered, targets):
    """Classify pair records without inferring camera pose or physical overlap.

    A missing DB row is unknown: some COLMAP backends do not persist zero-match
    attempts. A verified 2D match touching a track remains only a PnP candidate.
    """
    tracks = _tracks(Path(tracks_path))
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        names = {name: image_id for image_id, name in db.execute('SELECT image_id, name FROM images')}
        if not set(registered | targets) <= names.keys():
            raise ValueError('Target or registered image is missing from the matching database')
        result = []
        for target in sorted(targets):
            for anchor in sorted(registered):
                a, b = sorted((names[target], names[anchor]))
                pair_id = a * MAX_IMAGE_ID + b
                raw = db.execute('SELECT rows, cols, data FROM matches WHERE pair_id=?', (pair_id,)).fetchone()
                verified = db.execute('SELECT rows, cols, data, config FROM two_view_geometries WHERE pair_id=?', (pair_id,)).fetchone()
                raw_count = raw[0] if raw is not None else None
                valid_geometry = verified is not None and verified[3] not in (0, 1)
                verified_count = verified[0] if valid_geometry else 0
                if raw is None:
                    stage = 'no-match-record / attempted-unknown'
                elif raw_count == 0:
                    stage = 'no-tentative-matches'
                elif verified is None:
                    stage = 'geometric-verification-record-missing'
                elif not valid_geometry or not verified_count:
                    stage = 'geometric-verification-rejected'
                else:
                    stage = 'verified-2d-matches'
                track_matches = None
                if tracks is not None and valid_geometry and verified_count:
                    if verified[1] != 2 or verified[2] is None or len(verified[2]) != verified_count * 8:
                        raise ValueError(f'Invalid verified match data for pair {pair_id}')
                    matches = struct.iter_unpack('<II', verified[2])
                    anchor_column = 0 if names[anchor] == a else 1
                    track_matches = int(sum((names[anchor], int(k)) in tracks for match in matches for k in (match[anchor_column],)))
                elif tracks is not None:
                    track_matches = 0
                result.append({'target': target, 'anchor': anchor, 'stage': stage,
                               'tentative_matches': raw_count, 'verified_matches': verified_count,
                               'anchor_observations_on_3d_tracks': track_matches})
    return result


def write_pair_visual(database, images, tracks_path, pair, output):
    """Show tentative rejected, verified and track-linked keypoints on real photos."""
    import numpy as np
    from PIL import Image, ImageDraw

    tracks = _tracks(Path(tracks_path)) or {}
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        name_to_id = {name: image_id for image_id, name in db.execute('SELECT image_id,name FROM images')}
        a, b = sorted((name_to_id[pair['target']], name_to_id[pair['anchor']]))
        pair_id = a * MAX_IMAGE_ID + b
        ids = {v: k for k, v in name_to_id.items()}
        raw = db.execute('SELECT rows,cols,data FROM matches WHERE pair_id=?', (pair_id,)).fetchone()
        ver = db.execute('SELECT rows,cols,data,config FROM two_view_geometries WHERE pair_id=?', (pair_id,)).fetchone()
        keypoints = []
        for image_id in (a, b):
            rows, cols, blob = db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (image_id,)).fetchone()
            keypoints.append(np.frombuffer(blob, dtype='<f4').reshape(rows, cols)[:, :2])
        def decode(row):
            if row is None or row[0] == 0:
                return []
            if row[1] != 2 or row[2] is None or len(row[2]) != row[0] * 8:
                raise ValueError('Invalid pair match blob')
            return [tuple(map(int, item)) for item in np.frombuffer(row[2], dtype='<u4').reshape(-1, 2)]
        raw_matches = decode(raw)
        verified = set(decode(ver)) if ver and ver[3] not in (0, 1) else set()
    canvases = []
    for image_id in (a, b):
        with Image.open(Path(images) / ids[image_id]) as im:
            im = im.convert('RGB')
            original = im.size
            im.thumbnail((600, 440))
            canvases.append((im.copy(), original))
    width = sum(im.width for im, _ in canvases)
    height = max(im.height for im, _ in canvases)
    sheet = Image.new('RGB', (width, height + 42), (18, 18, 18))
    sheet.paste(canvases[0][0], (0, 42)); sheet.paste(canvases[1][0], (canvases[0][0].width, 42))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 2), f"{ids[a]} | {ids[b]}", fill='white')
    draw.text((4, 20), 'red: tentative only  yellow: verified  green: verified to existing 3D track', fill='white')
    anchor_id = name_to_id[pair['anchor']]
    for left, right in raw_matches:
        if left >= len(keypoints[0]) or right >= len(keypoints[1]):
            raise ValueError('Pair references a nonexistent keypoint')
        points = []
        for j, index in enumerate((left, right)):
            im, original = canvases[j]
            x, y = keypoints[j][index]
            points.append((float(x) * im.width / original[0] + (canvases[0][0].width if j else 0),
                           float(y) * im.height / original[1] + 42))
        match = (left, right)
        anchor_key = left if anchor_id == a else right
        color = ('#ff5555' if match not in verified else
                 '#60f58b' if (anchor_id, anchor_key) in tracks else '#ffdb4d')
        draw.line((*points[0], *points[1]), fill=color, width=1)
        for x, y in points:
            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
    sheet.save(output)


def audit_matching_revision(revision, output, *, model=0, max_pairs_per_target=2):
    """Write private stage counts and highest-evidence pair visualizations."""
    from .sfm_cohort_recovery import _source_paths, plan_cohort_recovery
    if type(max_pairs_per_target) is not int or not 1 <= max_pairs_per_target <= 10:
        raise ValueError('max_pairs_per_target must be between 1 and 10')
    paths = _source_paths(Path(revision), model)
    target = Path(output).resolve()
    source = paths['sparse'].resolve()
    revision_root = paths['revision'].resolve()
    if (Path(output).is_symlink() or target.exists() or target.is_relative_to(revision_root)
            or revision_root.is_relative_to(target)):
        raise ValueError('Match audit requires an unused output outside the source reconstruction')
    plan = plan_cohort_recovery(Path(revision), model=model)
    pairs = plan['pair_stage_diagnostics']
    if not pairs:
        raise ValueError('No tentative match table is available in the source database')
    target.mkdir(parents=True, exist_ok=False)
    tracks = source / 'text-models' / str(model) / 'points3D.txt'
    chosen = []
    for name in sorted(plan['missing_target_filenames']):
        candidates = [pair for pair in pairs if pair['target'] == name and pair['tentative_matches']]
        candidates.sort(key=lambda pair: (-pair['verified_matches'], -pair['tentative_matches'], pair['anchor']))
        for pair in candidates[:max_pairs_per_target]:
            filename = f"{len(chosen):04d}-matches.png"
            write_pair_visual(paths['database'], paths['images'], tracks, pair, target / filename)
            chosen.append({'visualization': filename, **pair})
    report = {'schema_version': 1, 'source_revision': str(paths['revision']),
              'database_sha256': plan['source_database_sha256'], 'pairs': pairs, 'visualized_pairs': chosen,
              'limitations': 'No match record has unknown attempt status. A missing verification record is distinct from a rejected pair. Track links are candidates, not PnP inliers; background matches require review.'}
    (target / 'pair-diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
