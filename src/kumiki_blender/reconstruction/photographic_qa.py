"""Camera-registered, ray-occluded diagnostic for immutable dense revisions."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .references import ReferenceError
from .surface_coverage import (_boundary_edges, _load_triangular_ply,
                               _write_ply_points,
                               plan_surface_coverage)
from .silhouette_qa import compare_silhouettes
from .discrepancy_regions import discrepancy_regions


def _area_stratified_samples(xyz, faces, max_samples, np):
    """Sample mesh area reproducibly with weights based on triangle area."""
    # Equal-area stratification gives each unit of mesh area equal weight,
    # rather than giving each triangle equal weight.
    face_area = np.empty(len(faces), dtype=np.float64)
    for start in range(0, len(faces), 100000):
        batch = xyz[faces[start:start + 100000]]
        face_area[start:start + len(batch)] = np.linalg.norm(
            np.cross(batch[:, 1] - batch[:, 0], batch[:, 2] - batch[:, 0]), axis=1) / 2
    cumulative_area = np.cumsum(face_area)
    if not np.isfinite(cumulative_area[-1]) or cumulative_area[-1] <= 0:
        raise ReferenceError('Mesh has no finite nonzero surface area to audit')
    sample_count = max_samples
    offsets = (np.arange(sample_count) + 0.5) * (cumulative_area[-1] / sample_count)
    indices = np.searchsorted(cumulative_area, offsets, side='right')
    triangles = xyz[faces[indices]]
    # Low-discrepancy barycentric coordinates avoid repeatedly testing the
    # same center when a large triangle occupies multiple area strata.
    sequence = np.arange(sample_count, dtype=np.float64) + 0.5
    u = (sequence * 0.7548776662466927) % 1
    v = (sequence * 0.5698402909980532) % 1
    root = np.sqrt(u)
    samples = ((1 - root)[:, None] * triangles[:, 0] +
               (root * (1 - v))[:, None] * triangles[:, 1] +
               (root * v)[:, None] * triangles[:, 2])
    return samples, indices, cumulative_area[-1]


def audit_photographs(dense_revision, output, *, blender, side=256, max_samples=10000,
                      max_cameras=256, min_views=2):
    import numpy as np
    from PIL import Image, ImageDraw
    from .photo_uv_projection import _camera_coords

    if type(side) is not int or not 64 <= side <= 1024:
        raise ReferenceError('side must be an integer between 64 and 1024 pixels')
    executable = shutil.which(str(blender)) if '/' not in str(blender) else str(blender)
    if not executable or not Path(executable).is_file():
        raise ReferenceError('Blender executable not found; pass --blender with its executable path')
    plan = plan_surface_coverage(dense_revision, output, max_samples=max_samples,
                                 max_cameras=max_cameras, min_views=min_views)
    if not plan['mask_paths']:
        raise ReferenceError('Photographic QA requires pinned rectified foreground masks')
    xyz, faces = _load_triangular_ply(Path(plan['mesh']), expected_vertices=plan['mesh_vertices'],
                                      expected_faces=plan['mesh_faces'], np=np)
    samples, indices, total_area = _area_stratified_samples(xyz, faces, max_samples, np)
    triangles = xyz[faces[indices]]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2
    boundary, topology = _boundary_edges(xyz, faces, max_points=50000, np=np)
    target = Path(plan['output'])
    from .dense import _json
    sparse = Path(_json(Path(plan['source_revision']) / 'dense-report.json')['source_revision']).resolve()
    if target.is_relative_to(sparse) or sparse.is_relative_to(target):
        raise ReferenceError('Photographic QA output must be outside the sparse source revision')
    target.mkdir(parents=True, exist_ok=False)
    counts = np.zeros(len(samples), dtype=np.uint16)
    records, thumbnails = [], []
    with tempfile.TemporaryDirectory(prefix='kumiki-depth-') as temp:
        scratch = Path(temp)
        np.save(scratch / 'samples.npy', samples)
        config = {'mesh': plan['mesh'], 'samples': str(scratch / 'samples.npy'),
                  'output': temp, 'cameras': plan['cameras'], 'side': side}
        (scratch / 'config.json').write_text(json.dumps(config))
        worker = Path(__file__).with_name('blender_depth_worker.py')
        result = subprocess.run([executable, '--background', '--factory-startup', '--python-exit-code', '1',
                                 '--python', str(worker), '--', str(scratch / 'config.json')],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (target / 'blender-depth.log').write_text(result.stdout)
        if result.returncode:
            raise ReferenceError(f'Blender depth audit failed; inspect {target / "blender-depth.log"}')
        for i, camera in enumerate(plan['cameras']):
            data = np.load(scratch / f'{i:04d}.npz')
            rendered, visible = data['silhouette'], data['visible'].astype(bool)
            h, w = rendered.shape
            with Image.open(plan['mask_paths'][camera['filename']]) as mask:
                intrinsic = camera['intrinsics']
                if (mask.format != 'PNG' or mask.mode not in ('L', '1')
                        or mask.size != (intrinsic['width'], intrinsic['height'])):
                    raise ReferenceError(f"Mask geometry does not match {camera['filename']}")
                full_mask = np.asarray(mask.convert('L'))
                if not np.isin(full_mask, (0, 255)).all():
                    raise ReferenceError(f"Mask is not binary: {camera['filename']}")
                reference = np.asarray(mask.convert('L').resize((w, h), Image.Resampling.NEAREST))
            metrics, diff = compare_silhouettes(reference, rendered)
            candidates = discrepancy_regions(reference, rendered)
            projected, depth = _camera_coords(samples, camera, (w, h), np)
            valid = (np.isfinite(projected).all(axis=1) & (depth > 0) & (projected[:, 0] >= 0)
                     & (projected[:, 0] < w) & (projected[:, 1] >= 0) & (projected[:, 1] < h)
                     & (areas > 1e-12))
            ix = np.flatnonzero(valid)
            valid[ix] &= reference[projected[ix, 1].astype(int), projected[ix, 0].astype(int)] > 0
            counts += visible & valid
            tiles = []
            for row in range(4):
                for col in range(4):
                    region = (slice(row*h//4, (row+1)*h//4), slice(col*w//4, (col+1)*w//4))
                    reference_tile = reference[region] > 0
                    rendered_tile = rendered[region] > 0
                    missing = int(np.count_nonzero(reference_tile & ~rendered_tile))
                    excess = int(np.count_nonzero(rendered_tile & ~reference_tile))
                    if missing or excess:
                        tiles.append({'grid_cell': [col, row], 'reference_only_pixels': missing,
                                      'render_only_pixels': excess})
            tiles.sort(key=lambda tile: -(tile['reference_only_pixels'] + tile['render_only_pixels']))
            records.append({'filename': camera['filename'], 'visible_masked_samples': int((visible & valid).sum()),
                            'discrepancy_tiles': tiles, 'discrepancy_regions': candidates, **metrics})
            with Image.open(camera['source_photo_path']) as source:
                photo = source.convert('RGB').resize((w, h), Image.Resampling.LANCZOS)
            mesh_image = Image.fromarray(rendered).convert('RGB')
            diff_image = Image.fromarray(diff)
            # Keep the photograph untouched outside discrepant pixels.
            overlay_pixels = np.asarray(photo).copy()
            different = (reference > 0) != (rendered > 0)
            overlay_pixels[different] = ((overlay_pixels[different].astype(np.uint16) +
                                         diff[different].astype(np.uint16)) // 2).astype(np.uint8)
            overlay = Image.fromarray(overlay_pixels)
            sheet = Image.new('RGB', (w * 4, h + 24), 'black')
            for j, image in enumerate((photo, mesh_image, diff_image, overlay)):
                sheet.paste(image, (w*j, 24))
            ImageDraw.Draw(sheet).text((4, 4), camera['filename'] + '  photo | mesh | discrepancy | overlay', fill='white')
            sheet.save(target / f'{i:04d}-comparison.png')
            thumb = sheet.copy(); thumb.thumbnail((800, 200))
            thumbnails.append(thumb)
    contact = Image.new('RGB', (800, sum(t.height for t in thumbnails)), 'black')
    y = 0
    for thumb in thumbnails:
        contact.paste(thumb, (0, y)); y += thumb.height
    contact.save(target / 'contact-sheet.jpg', quality=85)
    colors = np.zeros((len(samples), 3), dtype=np.uint8)
    colors[counts == 0] = (235, 47, 50); colors[counts == 1] = (255, 141, 37)
    colors[counts == 2] = (255, 215, 48); colors[counts >= 3] = (49, 175, 92)
    _write_ply_points(target / 'visible-support-points.ply', samples, colors)
    _write_ply_points(target / 'open-boundary-points.ply', boundary, np.tile((240, 50, 225), (len(boundary), 1)))
    region_min = xyz.min(axis=0)
    span = np.maximum(xyz.max(axis=0) - region_min, 1e-9)
    cells = np.clip(((samples - region_min) / span * 4).astype(int), 0, 3)
    regions = []
    for cell in np.unique(cells, axis=0):
        selected = np.all(cells == cell, axis=1)
        n = int(selected.sum())
        regions.append({'grid_cell': cell.tolist(), 'samples': n,
                        'weak_samples': int(np.count_nonzero(counts[selected] < min_views)),
                        'zero_visible_samples': int(np.count_nonzero(counts[selected] == 0)),
                        'mean_visible_views': float(counts[selected].mean())})
    regions.sort(key=lambda region: -region['weak_samples'])
    # Rank image-space candidates for inspection; separate photographs do not
    # establish that two regions correspond to the same physical surface.
    ranked = []
    for record in records:
        for candidate in record['discrepancy_regions']:
            ranked.append({'filename': record['filename'], **candidate})
    ranked.sort(key=lambda item: (-item['interior_pixels'] / (item['pixels'] / item['image_fraction']),
                                  -item['image_fraction'], item['filename']))
    report = {'status': 'diagnostic / photographic-fidelity-not-verified', 'mesh_sha256': plan['mesh_sha256'],
              'registered_cameras': plan['registered_cameras'], 'audited_cameras': len(records),
              'sampled_faces': int(len(np.unique(indices))), 'surface_samples': len(samples), 'sampling': 'deterministic-area-stratified-barycentric',
              'mesh_surface_area': float(total_area), 'min_views': min_views,
              'weak_visible_fraction': float(np.mean(counts < min_views)),
              'zero_visible_samples': int((counts == 0).sum()), 'topology': topology,
              'per_camera': records, 'regions': regions, 'ranked_image_discrepancies': ranked[:100],
              'distribution_allowed': False,
              'limitations': 'Ray visibility tests mesh occlusion but does not prove photo consistency, surface accuracy, or correct depth. Area-stratified barycentric samples are approximate and depend on the sample budget. Silhouette discrepancies can reflect masks, occluders, alignment, or resolution. Absent geometry has no 3D sample. Review comparisons manually.'}
    (target / 'photographic-qa-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
