"""Fast, Blender-free tests of the bounded collision solver contract and geometry."""
from __future__ import annotations

import json

import numpy as np
import pytest

from kumiki_blender.character_build.penetration_repair import (
    CollisionPolicy, load_collision_plan, repair_penetration,
)


def _tetra():
    # One tetrahedron tip crosses the plane x=0. Three root points must stay.
    vertices = np.array([[-0.03, 0, 0], [0.20, 0, 0],
                         [0.20, 0.12, 0], [0.20, 0, 0.12]], dtype=float)
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, triangles


def _plane(points):
    # A simulated closed surface locally approximated by its tangent plane.
    normals = np.zeros_like(points)
    normals[:, 0] = 1
    return points[:, 0].copy(), normals


def _policy(**overrides):
    cfg = {
        "moving": "Tail", "fixed": "Body", "max_displacement_m": 0.05,
        "clearance_m": 0.001, "iterations": 8, "smoothing_iterations": 2,
        "max_relative_edge_change": 0.5, "protected_vertices": (1, 2, 3),
        "min_improvement_fraction": 0.02,
    }
    cfg.update(overrides)
    return CollisionPolicy(**cfg)


def test_penetrating_tip_is_repaired_without_moving_attachment():
    vertices, faces = _tetra()
    repaired, report = repair_penetration(vertices, faces, _policy(), _plane)
    assert report["status"] == "geometric-candidate-review-required"
    assert repaired[0, 0] >= -0.0001
    np.testing.assert_array_equal(repaired[1:], vertices[1:])
    assert report["candidate_penetrating_vertices"] <= report["baseline_penetrating_vertices"]
    assert report["improvement_fraction"] > 0.9
    assert report["changed_vertices"] == 1
    assert report["max_displacement_m"] <= 0.05 + 1e-10
    assert report["flipped_or_degenerate_faces"] == 0
    np.testing.assert_array_equal(vertices[0], np.array([-0.03, 0, 0]))


def test_impossible_improvement_falls_back_to_exact_input():
    vertices, faces = _tetra()
    repaired, report = repair_penetration(
        vertices, faces, _policy(max_displacement_m=0.001, min_improvement_fraction=0.99),
        _plane)
    np.testing.assert_array_equal(repaired, vertices)
    assert report["status"] == "no-safe-improvement"
    assert report["changed_vertices"] == 0


def test_protected_penetrating_vertex_is_not_edited():
    vertices, faces = _tetra()
    repaired, report = repair_penetration(
        vertices, faces, _policy(protected_vertices=(0, 1, 2, 3)), _plane)
    np.testing.assert_array_equal(repaired, vertices)
    assert report["status"] == "no-safe-improvement"


def test_unsafe_or_degenerate_geometry_rejected():
    vertices, faces = _tetra()
    with pytest.raises(ValueError, match="Protected vertex"):
        repair_penetration(vertices, faces, _policy(protected_vertices=(99,)), _plane)
    with pytest.raises(ValueError, match="Degenerate"):
        repair_penetration(vertices, np.array([[0, 1, 1]]), _policy(), _plane)


@pytest.mark.parametrize("bad", [
    {"pairs": []},
    {"pairs": [{"moving": "Tail", "fixed": "Tail"}]},
    {"pairs": [{"moving": "Tail", "fixed": "Body", "max_displacement_m": 5}]},
    {"pairs": [{"moving": "Tail", "fixed": "Body", "protected_vertices": [-1]}]},
    {"pairs": [{"moving": "Tail", "fixed": "Body", "protected_vertices": [1, 1]}]},
    {"pairs": [{"moving": "../Tail", "fixed": "Body"}]},
    {"pairs": [{"moving": "Tail", "fixed": "Body", "unexpected": 1}]},
    {"pairs": [{"moving": "Tail", "fixed": "Body"}] * 2},
])
def test_invalid_pair_plan_is_rejected(tmp_path, bad):
    plan = tmp_path / "bad.json"
    plan.write_text(json.dumps({"schema_version": 1, **bad}))
    with pytest.raises(ValueError):
        load_collision_plan(plan)


def test_plan_is_character_independent(tmp_path):
    path = tmp_path / "quill.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "pairs": [{"moving": "Wing.L", "fixed": "Torso",
                   "protected_vertices": [0, 2, 7], "max_displacement_m": 0.002}]
    }))
    parsed = load_collision_plan(path)
    assert parsed["pairs"][0].moving == "Wing.L"
    assert parsed["pairs"][0].protected_vertices == (0, 2, 7)



def test_vertex_group_and_contact_guard_policy(tmp_path):
    config = tmp_path / "contact-guards.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "guard_pairs": [["Tail", "Satchel"], ["Wing.L", "Torso"]],
        "pairs": [
            {"moving": "Tail", "fixed": "Body",
             "protected_vertex_groups": ["TailRoot", "TailAttachment.L"],
             "protected_vertices": [10, 12]}
        ],
    }))
    parsed = load_collision_plan(config)
    policy = parsed["pairs"][0]
    assert policy.protected_vertex_groups == ("TailRoot", "TailAttachment.L")
    assert policy.protected_vertices == (10, 12)
    assert parsed["guard_pairs"] == [["Tail", "Satchel"], ["Wing.L", "Torso"]]


@pytest.mark.parametrize("bad", [
    [["Tail", "Tail"]],
    [["Tail", "Satchel"], ["Satchel", "Tail"]],
    [["Tail", "Satchel", "Body"]],
    [["Tail", "../Satchel"]],
])
def test_invalid_guard_pairs_rejected(tmp_path, bad):
    path = tmp_path / "bad-guard.json"
    path.write_text(json.dumps({"schema_version": 1, "pairs": [
        {"moving": "Tail", "fixed": "Body"}], "guard_pairs": bad}))
    with pytest.raises(ValueError, match="guard"):
        load_collision_plan(path)


@pytest.mark.parametrize("groups", [["TailRoot", "TailRoot"], ["../Tail"],
                                     [""], [1], "TailRoot"])
def test_invalid_vertex_groups_rejected(tmp_path, groups):
    path = tmp_path / "bad-groups.json"
    path.write_text(json.dumps({"schema_version": 1, "pairs": [{
        "moving": "Tail", "fixed": "Body", "protected_vertex_groups": groups}]}))
    with pytest.raises(ValueError, match="protected_vertex_groups"):
        load_collision_plan(path)


def test_legacy_plan_works_without_optional_guards(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"schema_version": 1, "pairs": [
        {"moving": "Tail", "fixed": "Body", "protected_vertices": [0]}]}))
    parsed = load_collision_plan(path)
    assert parsed["guard_pairs"] == []
    assert parsed["pairs"][0].protected_vertex_groups == ()
