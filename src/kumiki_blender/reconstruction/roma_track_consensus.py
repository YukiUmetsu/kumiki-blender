"""Conservative agreement of independently predicted photo-to-track matches."""
from __future__ import annotations


def agree_track_predictions(predictions, *, max_error_pixels=4.0):
    """Group a point's predictions only when every source view agrees.

    Inputs contain measured track IDs, original 3D positions, target pixels,
    and distinct anchor names. Two or more different anchors must agree. If
    any prediction conflicts, discard the track rather than cherry-pick a pair.
    This is candidate evidence and does not estimate or certify a camera.
    """
    import math
    from collections import defaultdict

    if not math.isfinite(max_error_pixels) or max_error_pixels <= 0:
        raise ValueError('max_error_pixels must be positive and finite')
    groups = defaultdict(list)
    for item in predictions:
        if not item['subject_candidate']:
            continue
        groups[item['track_id']].append(item)
    agreed = []
    for track_id in sorted(groups):
        items = groups[track_id]
        if len(items) < 2 or len({item['anchor'] for item in items}) != len(items):
            continue
        points = [item['pixel'] for item in items]
        positions = [item['xyz'] for item in items]
        if any(len(value) != 2 or not all(math.isfinite(float(x)) for x in value) for value in points):
            continue
        if any(len(value) != 3 or not all(math.isfinite(float(x)) for x in value) for value in positions):
            continue
        if any(math.dist(positions[0], value) > 1e-8 for value in positions[1:]):
            continue
        if any(math.dist(points[i], points[j]) > max_error_pixels
               for i in range(len(points)) for j in range(i)):
            continue
        agreed.append({'track_id': track_id, 'xyz': list(positions[0]),
                       'pixel': [sum(p[j] for p in points) / len(points) for j in range(2)],
                       'anchors': sorted(item['anchor'] for item in items)})
    return agreed
