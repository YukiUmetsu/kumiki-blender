# Anatomical Head-Base Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import existing human-head geometry, match it to measured statue anchors, and create a sculpt-ready, auditable `.blend` and GLB revision.

**Architecture:** A dependency-free `reconstruction/head_base.py` validates the head-fit plan and computes a stable 3D similarity transform and optional local residual field. The existing Blender importer receives an optional fitted-head context, applies the transform, adds a non-destructive facial shape key, and exports through the shared immutable revision machinery. CLI preflight occurs before launching Blender.

**Tech Stack:** Python 3.11+, standard library, Blender 4.2+ on the build path, existing GLB audit/provenance pipeline, unittest.

**Spec:** `docs/superpowers/specs/2026-11-02-anatomical-head-base-design.md`

## Global Constraints

- No model or photo downloads; no new required dependency or new procedural human mesh.
- Provenance and actual static GLB validation must happen before build/export.
- Source/target coordinates use Blender Z-up meters; source landmarks match imported GLB world coordinates.
- Never mark historical likeness approved; outputs stay artist-review-required.
- Never overwrite output directories and never silently select a different head mesh.

## Review Focus

- Degenerate, mirrored, or near-collinear landmark geometry is rejected.
- Path traversal, symlink escapes, and mismatched SHA are rejected before Blender.
- Shape key deformation remains finite and localized; suspicious large changes fail preflight.
- Import hierarchy keeps eyes, teeth and other pieces aligned while sculpting only the named face mesh.
- Missing Blender, renamed target mesh and existing output surfaces explicit errors.

---

### Task 1: Validated head-fit plan and rigid alignment

**Files:** `src/kumiki_blender/reconstruction/head_base.py`, `tests/test_head_base.py`

**Interfaces:** `load_head_fit_plan(path: str | Path) -> HeadFitPlan`, `fit_head_landmarks(plan: HeadFitPlan) -> HeadFitResult`, `transform_point(point, fit) -> tuple[float,float,float]`.

- [ ] Write failing tests for known translation/rotation/scale, invalid source paths, repeated/collinear landmarks, and inaccurate registrations.
- [ ] Run focused tests to confirm missing behavior fails.
- [ ] Implement strict JSON schema, reused GLB/provenance intake, a robust proper 3D similarity fit and per-landmark diagnostics.
- [ ] Run focused tests, verify expected outputs and commit.

### Task 2: Optional non-destructive anatomical correction field

**Files:** `src/kumiki_blender/reconstruction/head_base.py`, `tests/test_head_base.py`

**Interfaces:** `head_deformation_field(plan, fit) -> HeadDeformation`, `HeadDeformation.displace(point) -> tuple[float,float,float]`.

- [ ] Write failing tests for exact/near-anchor corrections, zero offset outside compact influence, extreme input rejection, and unchanged rigid mode.
- [ ] Verify focused tests fail for missing implementation.
- [ ] Implement bounded Wendland radial basis residual corrections, to be applied only to the named editable face mesh.
- [ ] Verify tests, commit.

### Task 3: Integrate with existing Blender revision exporter

**Files:** `src/kumiki_blender/blender_import.py`, `src/kumiki_blender/cli.py`, `tests/test_head_base_cli.py`, `tests/test_blender_scene_contract.py`

**Interfaces:** `plan-head-base --manifest PATH`; `build-head-base --manifest PATH --output DIR [--blender EXE]`; `build_imported_scene(..., head_fit=plan)`.

- [ ] Add failing CLI tests for inspection output, missing Blender/mesh errors and existing revision refusal.
- [ ] Verify red.
- [ ] Integrate transform of imported root hierarchy, optional non-destructive shape key, studio landmark markers, QA metadata, and Blender subprocess dispatch.
- [ ] Verify CLI tests, optional Blender smoke render if installed, and commit.

### Task 4: Handoff documentation and full verification

**Files:** `docs/ANATOMICAL_HEAD_BASES.md`, `examples/anatomical-head-fit.example.json`, `README.md`, `.github/workflows/python.yml` (only if required by test discovery).

- [ ] Document obtaining/inspecting a CC0 MakeHuman-exported head, rights, measured landmarks, editor workflow and limitations.
- [ ] Run focused, full unittest suite, `compileall`, catalog validation, and `git diff --check`.
- [ ] Confirm no third-party mesh shipped, commit, push and create PR against `main`.
