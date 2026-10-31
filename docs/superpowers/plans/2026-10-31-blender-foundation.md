# Blender Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested reusable Blender modeling foundation with a headless generic lattice-tower example.

**Architecture:** A pure Python recipe and structural-planning layer emits explicit geometry parts. An optional `bpy` adapter builds meshes and exports artifacts, and a small CLI separates validation from Blender rendering.

**Tech Stack:** Python 3.11+, standard-library `unittest`, Blender Python API 4.2+ (only for asset builds), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-31-blender-foundation-design.md`

## Global Constraints

- No third-party dependencies for validation, planning or tests.
- Meters and Blender Z-up; export GLB for Kumiki-UI.
- Never overwrite an existing output directory; preserve all user data.
- Demo is generic and illustrative, never an authenticated landmark replica.
- Do not mark Blender rendering visually validated unless the executable was run and screenshots inspected.

## Review Focus

1. Malformed or `NaN` JSON inputs must fail before modeling (Task 1 test).
2. Excessive tower levels or invalid proportions must be rejected (Task 1 test).
3. Repeated builds must return identical parts and names (Task 2 test).
4. Build CLI must reject an existing output directory without changing it (Task 3 test).
5. Running `build` without `bpy` must give an actionable error and not create output (Task 3 test).

---

### Task 1: Recipe contract and validation

**Files:** `src/kumiki_blender/recipe.py`, `tests/test_recipe.py`, `examples/lattice-observation-tower.json`

**Interfaces:** `load_recipe(path: str | Path) -> TowerRecipe`; `TowerRecipe` immutable dataclass with `asset_id`, dimensions, levels, material and render parameters.

- [ ] Write tests for valid fixture, unknown properties, non-finite values, invalid `levels`, top/base span order, unsafe asset IDs and resolution limits.
- [ ] Run `python -m unittest tests.test_recipe -v` and see failures due to missing interface.
- [ ] Implement strict validation and the example recipe.
- [ ] Run focused and full unit suite, verify green.
- [ ] Commit coherent increment.

### Task 2: Deterministic architecture planner

**Files:** `src/kumiki_blender/lattice.py`, `tests/test_lattice.py`

**Interfaces:** `make_lattice_tower(recipe: TowerRecipe) -> TowerPlan`, with `Beam` and `Deck` dataclasses and stable semantic names.

- [ ] Write tests for deterministic part ordering, tapered symmetry, nonzero beams, expected brace count, decks/railings and bounds.
- [ ] Run focused tests and observe expected failures.
- [ ] Implement pure structural planner without `bpy`.
- [ ] Run focused and full suite, verify green.
- [ ] Commit increment.

### Task 3: Headless CLI and safe file outputs

**Files:** `src/kumiki_blender/cli.py`, `scripts/blender_entry.py`, `tests/test_cli.py`

**Interfaces:** `main(argv: list[str] | None = None) -> int`; subcommands `validate --recipe PATH`, `build --recipe PATH --output DIR`.

- [ ] Write tests for validate JSON status, missing `bpy`, existing directory and no output mutation.
- [ ] Run focused tests and observe expected failures.
- [ ] Implement CLI; lazy-import Blender adapter only after validation and target checks.
- [ ] Run focused and full suite, verify green.
- [ ] Commit increment.

### Task 4: Blender builder and export

**Files:** `src/kumiki_blender/blender_scene.py`, `tests/test_blender_scene_contract.py`

**Interfaces:** `build_scene(recipe: TowerRecipe, plan: TowerPlan, destination: Path) -> dict`.

- [ ] Write a static contract test for no global `bpy` import and the output naming convention; on a machine with Blender add an integration smoke test for actual files.
- [ ] Run focused tests and observe expected failure.
- [ ] Implement named beams/decks, material, area lights, camera, Cycles, save, GLB selection export, look render and manifest.
- [ ] Run Python checks and actual Blender smoke test when executable is available; explicitly report any unrun checks.
- [ ] Commit increment.

### Task 5: Developer documentation and CI

**Files:** `README.md`, `pyproject.toml`, `.gitignore`, `.github/workflows/python.yml`, `docs/ARCHITECTURE.md`

- [ ] Document one-command Python validation and Mac Blender headless invocation.
- [ ] Provide GitHub Actions standard-library tests, version requirements and honest quality status.
- [ ] Run `python -m unittest discover -s tests -v` and `python -m compileall -q src scripts`.
- [ ] Review all changed files for correctness, file-output safety and fidelity claims; fix confirmed issues.
- [ ] Commit to a feature branch and open a PR.
