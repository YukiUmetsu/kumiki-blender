"""Regression tests for semantic texture ownership and character-independent search."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kumiki_blender.character_build.conflict_repair import read_repair_plan, run_conflict_repair
from kumiki_blender.character_build.semantic_projection import load_semantic_projection


def _png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)


def test_semantic_rejects_wrong_part_and_missing_views(tmp_path):
    handoff = tmp_path / 'handoff'
    (handoff / 'raw').mkdir(parents=True)
    Image.new('RGB', (4, 4), 'white').save(handoff / 'raw' / 'front.png')
    _png(tmp_path / 'uv.png', np.array([[1, 1], [0, 0]], dtype=bool))
    _png(tmp_path / 'ref.png', np.array([[0, 1, 0, 0]] * 4, dtype=bool))
    rows, cols = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    policy = load_semantic_projection({'parts': [{'id': 'tail', 'uv_mask': 'uv.png',
                                                  'view_masks': {'front': 'ref.png'}}]},
                                      tmp_path, handoff, 2, rows, cols)
    ok = np.ones(4, dtype=bool)
    np.testing.assert_array_equal(policy.allowed('front', ok, np.array([0, 1, 0, 1]),
                                                  np.array([0, 0, 0, 0])), [False, True, True, True])
    np.testing.assert_array_equal(policy.allowed('back', ok, np.array([0, 1, 0, 1]),
                                                 np.array([0, 0, 0, 0])), [False, False, True, True])
    assert ok.all()


def test_semantic_refuses_overlapping_and_misregistered_masks(tmp_path):
    handoff = tmp_path / 'handoff'
    (handoff / 'raw').mkdir(parents=True)
    Image.new('RGB', (4, 4), 'white').save(handoff / 'raw' / 'front.png')
    _png(tmp_path / 'uv.png', np.ones((2, 2), dtype=bool))
    _png(tmp_path / 'src.png', np.ones((4, 4), dtype=bool))
    part = {'id': 'one', 'uv_mask': 'uv.png', 'view_masks': {'front': 'src.png'}}
    with pytest.raises(ValueError, match='overlap'):
        load_semantic_projection({'parts': [part, {**part, 'id': 'two'}]}, tmp_path, handoff,
                                 2, np.array([0, 1]), np.array([0, 1]))
    _png(tmp_path / 'bad.png', np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError, match='dimensions'):
        load_semantic_projection({'parts': [{**part, 'view_masks': {'front': 'bad.png'}}]},
                                 tmp_path, handoff, 2, np.array([0]), np.array([0]))


def _setup(tmp_path, **changes):
    handoff = tmp_path / 'handoff'
    handoff.mkdir()
    (handoff / 'spec.json').write_text('{}')
    mesh = tmp_path / 'low.obj'
    mesh.write_text('v 0 0 0\n')
    data = {'schema_version': 1, 'evaluation_size': 128, 'final_size': 128,
            'search': {'rounds': 2, 'max_evaluations': 8,
                       'steps': [{'parameter': 'power', 'values': [2, 4, 6]}]}}
    data.update(changes)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(data))
    return handoff, mesh, config


def _evaluation(mesh, settings, size, diagnostics):
    power = settings['power']
    problem = {2: 0.12, 4: 0.08, 6: 0.02}[power]
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    texture[..., 0] = int(power * 20)
    appearance = {'views': [{'view': 'front', 'mean_delta_e': problem * 100},
                            {'view': 'back', 'mean_delta_e': problem * 110}]}
    conflict = {'surface': {'misplaced_part': problem / 2, 'contested_feature': problem / 2,
                            'unseen': 0.01, 'tone_seam': problem}}
    return texture, conflict, appearance


def test_loop_selects_improvement_and_preserves_inputs(tmp_path):
    handoff, mesh, config = _setup(tmp_path)
    before = mesh.read_bytes()
    output = tmp_path / 'repair-r01'
    report = run_conflict_repair(handoff, mesh, config, output, evaluator=_evaluation)
    assert report['status'] == 'improved'
    assert report['selected_settings']['power'] == 6
    assert report['selected_attempt_index'] != 0
    assert mesh.read_bytes() == before
    assert (output / 'texture.png').is_file()
    assert (output / 'report.json').is_file()
    with pytest.raises(FileExistsError):
        run_conflict_repair(handoff, mesh, config, output, evaluator=_evaluation)


def test_loop_guards_each_view_not_aggregate(tmp_path):
    handoff, mesh, config = _setup(tmp_path)

    def dangerous(mesh, settings, size, diagnostics):
        img, conflicts, appearance = _evaluation(mesh, settings, size, diagnostics)
        if settings['power'] == 6:
            appearance['views'][1]['mean_delta_e'] = 30
        return img, conflicts, appearance

    report = run_conflict_repair(handoff, mesh, config, tmp_path / 'guarded', evaluator=dangerous)
    assert report['status'] == 'baseline-retained'
    assert any('view_regression:back' in row['rejected'] for row in report['attempts'])


def test_invalid_settings_and_unsafe_paths(tmp_path):
    _, _, config = _setup(tmp_path)
    raw = json.loads(config.read_text())
    raw['search']['steps'] = [{'parameter': 'contest_delta_e', 'values': [200]}]
    config.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='unsupported'):
        read_repair_plan(config)
    raw['search']['steps'] = []
    raw['candidate_meshes'] = ['../outside.obj']
    config.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='Unsafe'):
        read_repair_plan(config)


def test_no_improvement_keeps_baseline(tmp_path):
    handoff, mesh, config = _setup(tmp_path)

    def uniform(mesh, settings, size, diagnostics):
        return _evaluation(mesh, {**settings, 'power': 4.0}, size, diagnostics)

    report = run_conflict_repair(handoff, mesh, config, tmp_path / 'no-improve', evaluator=uniform)
    assert report['status'] == 'baseline-retained'
    assert report['selected_attempt_index'] == 0
