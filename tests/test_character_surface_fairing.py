"""Regression tests for feature-aware smoothing and source-gated revisions."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kumiki_blender.character_build.surface_fairing import (
    SurfacePolicy, fair_surface, load_surface_policy,
)
from kumiki_blender.character_build.surface_refinement_runner import run_surface_refinement


def _sphere(latitudes=12, longitudes=24, noise=0.015):
    vertices = [[0, 0, 1.0]]
    for i in range(1, latitudes):
        polar = math.pi * i / latitudes
        for j in range(longitudes):
            angle = 2 * math.pi * j / longitudes
            perturb = noise * math.cos(7 * angle) * math.sin(3 * polar)
            radius = 1 + perturb
            vertices.append([radius * math.sin(polar) * math.cos(angle),
                             radius * math.sin(polar) * math.sin(angle),
                             radius * math.cos(polar)])
    vertices.append([0, 0, -1])
    first, last = 1, len(vertices) - 1
    faces = []
    for j in range(longitudes):
        faces.append([0, first + j, first + (j + 1) % longitudes])
    for ring in range(latitudes - 2):
        for j in range(longitudes):
            a = first + ring * longitudes + j
            b = first + ring * longitudes + (j + 1) % longitudes
            c = first + (ring + 1) * longitudes + j
            d = first + (ring + 1) * longitudes + (j + 1) % longitudes
            faces.extend([[a, c, b], [b, c, d]])
    start = first + (latitudes - 2) * longitudes
    for j in range(longitudes):
        faces.append([last, start + (j + 1) % longitudes, start + j])
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
    signed = np.einsum("ij,ij->", vertices[faces[:, 0]],
                       np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])) / 6
    if signed < 0:
        faces = faces[:, [0, 2, 1]]
    return vertices, faces


def _policy(**params):
    policy = {
        "mesh": "Body", "iterations": 8, "lambda_step": .45, "mu_step": -.32,
        "feature_angle_deg": 75, "curvature_stop": 6,
        "max_displacement_m": .06, "max_edge_strain": .5,
        "max_volume_change": .1, "min_roughness_improvement": 0.0001,
        "protected_vertex_groups": (), "editable_vertex_group": None, "reference_gate": None,
    }
    policy.update(params)
    return SurfacePolicy(**policy)


def test_sphere_fairing_improves_roughness_without_moving_protected_vertices():
    vertices, faces = _sphere()
    before = vertices.copy()
    protected = np.zeros(len(vertices), dtype=bool)
    protected[[0, len(vertices) - 1]] = True
    revised, result = fair_surface(vertices, faces, _policy(), protected=protected)
    assert result["status"] == "surface-candidate-art-review-required"
    assert result["final_roughness"] < result["baseline_roughness"]
    assert result["changed_vertices"] > 0
    assert result["volume_change_fraction"] <= .1
    assert result["maximum_displacement_m"] <= .06 + 1e-9
    np.testing.assert_array_equal(revised[protected], before[protected])
    np.testing.assert_array_equal(vertices, before)
    np.testing.assert_array_equal(faces, faces.copy())


def test_sharp_cube_preserves_all_vertices():
    vertices = np.asarray([(x, y, z) for x in (-1, 1) for y in (-1, 1)
                           for z in (-1, 1)], dtype=float)
    faces = np.asarray([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6],
                        [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
                        [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], dtype=int)
    # Repair orientation for a deliberately simple coordinate indexing order.
    orientation = np.einsum("ij,ij->", vertices[faces[:, 0]],
                            np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])) / 6
    if orientation < 0:
        faces = faces[:, [0, 2, 1]]
    result, report = fair_surface(vertices, faces, _policy(feature_angle_deg=40))
    np.testing.assert_array_equal(result, vertices)
    assert report["status"] == "no-safe-improvement"


def test_empty_and_nonmanifold_surfaces_rejected():
    vertices, faces = _sphere(latitudes=8, longitudes=12)
    with pytest.raises(ValueError, match="closed/manifold"):
        fair_surface(vertices, faces[:-1], _policy())
    with pytest.raises(ValueError, match="editable weights"):
        fair_surface(vertices, faces, _policy(), editable=np.ones(len(vertices) - 1))
    with pytest.raises(ValueError, match="protected"):
        fair_surface(vertices, faces, _policy(), protected=np.zeros(len(vertices) - 1))


def _config(tmp_path, reference=False, **params):
    plan = {
        "schema_version": 1, "mesh": "Body",
        "max_displacement_m": .01,
        "protected_vertex_groups": ["Eyes", "Mouth"],
        **params,
    }
    if reference:
        char = tmp_path / "character"
        char.mkdir(exist_ok=True)
        (char / "character-plan.json").write_text("{}")
        plan["reference_gate"] = {
            "character_plan": "character", "region": "head",
            "max_iou_regression": .005,
        }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(plan))
    return path


@pytest.mark.parametrize("extra", [
    {"mesh": "../Body"}, {"iterations": 0}, {"max_volume_change": .9},
    {"protected_vertex_groups": ["Eyes", "Eyes"]},
    {"lambda_step": float("nan")}, {"reference_gate": {"region": "head"}},
    {"unrecognized": 1},
])
def test_invalid_policies_are_rejected(tmp_path, extra):
    path = _config(tmp_path, **extra)
    with pytest.raises(ValueError):
        load_surface_policy(path)


def test_policy_resolves_registered_source_folder(tmp_path):
    policy = load_surface_policy(_config(tmp_path, reference=True))
    assert policy.reference_gate["region"] == "head"
    assert Path(policy.reference_gate["character_plan"]) == (tmp_path / "character").resolve()
    assert policy.protected_vertex_groups == ("Eyes", "Mouth")


def _fake_worker(worker_name, params, **kwargs):
    assert worker_name == "surface-refine"
    output = Path(params["output"])
    output.mkdir()
    for filename in ["baseline-source.obj", "surface-candidate-source.obj",
                     "surface-candidate.blend"]:
        (output / filename).write_text("synthetic")
    report = {"schema_version": 1, "status": "surface-candidate-art-review-required"}
    (output / "report.json").write_text(json.dumps(report))
    return report


@pytest.mark.parametrize("gate_pass", [True, False])
def test_staged_reference_gate_tracks_status_without_modifying_source(tmp_path, gate_pass):
    source = tmp_path / "source.blend"
    source.write_bytes(b"approved original blender bytes")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    config = _config(tmp_path, reference=True)
    output = tmp_path / "revision"

    def fake_gate(character, region, candidate, dest, **kwargs):
        assert region == "head"
        assert kwargs["before"].name == "baseline-source.obj"
        assert kwargs["max_regression"] == .005
        dest.mkdir()
        result = {"numeric_gate_pass": gate_pass,
                  "verified_views": ["front", "left"],
                  "failed_views": [] if gate_pass else ["left"],
                  "worst_iou_delta": 0 if gate_pass else -.06}
        (dest / "report.json").write_text(json.dumps(result))
        return result

    result = run_surface_refinement(source, config, output,
                                    worker=_fake_worker, source_gate=fake_gate)
    assert output.is_dir()
    assert result["artistic_approval"] is False
    assert result["numeric_reference_gate_pass"] == gate_pass
    assert ("reference-regression-rejected" if not gate_pass
            else "registered-candidate-art-review-required") == result["status"]
    assert (output / "model" / "surface-candidate.blend").is_file()
    assert (output / "source-review" / "report.json").is_file()
    assert (output / "SOURCE_REGRESSION_REJECTED.txt").is_file() == (not gate_pass)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        run_surface_refinement(source, config, output,
                               worker=_fake_worker, source_gate=fake_gate)


def test_staging_does_not_publish_on_worker_exception(tmp_path):
    source = tmp_path / "source.blend"
    source.write_bytes(b"approved source")
    config = _config(tmp_path)
    def failure(*args, **kwargs):
        raise RuntimeError("broken Blender geometry")
    with pytest.raises(RuntimeError, match="broken Blender"):
        run_surface_refinement(source, config, tmp_path / "revision", worker=failure)
    assert not (tmp_path / "revision").exists()
