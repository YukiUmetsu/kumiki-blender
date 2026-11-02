# Mac-first Gaussian Splat Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one repeatable, Mac-first OpenSplat training pipeline from multiple photos or existing COLMAP cameras, with CUDA support through the same external trainer.

**Architecture:** Keep GPU training behind a subprocess adapter. Reuse existing photo reference and COLMAP workers, stage a canonical private dataset, then verify and report the splat artifact. Neither Blender's `bpy` interpreter nor the core install needs new dependencies.

**Tech Stack:** Python 3.11 stdlib, optional Pillow and PyCOLMAP for new photo inputs, external OpenSplat (Metal/CUDA/CPU), unittest.

**Spec:** `docs/superpowers/specs/2026-11-02-gaussian-splat-pipeline-design.md`

## Global Constraints

- All output directories are immutable-by-convention; existing directories are rejected.
- No source images or derived splats are committed or publicly distributed by the worker.
- Gaussian PLY is not a mesh GLB; use a separate validator.
- All external tools are invoked without a shell and with bounded timeouts.
- No network/install actions during regular unit tests.

## Review Focus

- Duplicate or visually similar image inputs must not imply verified camera coverage (Task 1).
- A symlinked image or malformed rectified image index must fail before staging (Tasks 1 and 2).
- Original-coordinate masks must never be passed as rectified masks (Task 2).
- Failure, timeout or empty output must never appear as a successful training run (Task 3).
- Changing the source reconstruction after preflight must be detected by hashing before use (Task 2).

---

### Task 1: Photo-folder intake

**Files:** Create `src/kumiki_blender/reconstruction/splat_input.py`; test `tests/test_reconstruction_splats.py`.

**Interfaces:** `load_photo_folder(path: str | Path, asset_id: str) -> ReferenceSet`, `plan_splat_input(...) -> dict`.

- [ ] Write failing tests for eight unique valid photos, invalid/duplicate images, unsafe symlinks, and count bounds.
- [ ] Run `PYTHONPATH=src python -m unittest tests.test_reconstruction_splats -v` and verify expected missing-symbol failures.
- [ ] Implement validated private photo intake and preflight using existing ReferenceSet audits.
- [ ] Run focused tests until green; commit this task.

### Task 2: Canonical COLMAP dataset staging

**Files:** Modify `src/kumiki_blender/reconstruction/splat_input.py`; test `tests/test_reconstruction_splats.py`.

**Interfaces:** `stage_splat_dataset(reconstruction: str | Path, output: str | Path, *, rectified_masks: str | Path | None = None) -> dict`.

- [ ] Write failing tests for canonical images/sparse/0 layout, mask-name conversion, mismatched masks, tampering and insufficient registrations.
- [ ] Run focused tests and observe expected failure.
- [ ] Reuse immutable COLMAP revision/index validation and copy pinned photos/camera model files into a private OpenSplat dataset.
- [ ] Run focused tests until green; commit this task.

### Task 3: External OpenSplat training and artifact audit

**Files:** Create `src/kumiki_blender/reconstruction/splats.py`; test `tests/test_reconstruction_splats.py`.

**Interfaces:** `plan_splat_revision(...) -> dict`, `build_splat_revision(...) -> dict`, `audit_splat_artifact(...) -> dict`.

- [ ] Write failing tests using an executable fake OpenSplat process for valid PLY, bad headers, failed runs, timeout, and overwrite refusal.
- [ ] Run focused tests to verify red.
- [ ] Implement subprocess execution, logs, file audits, hash-pinned manifest and truthful status.
- [ ] Run focused tests until green; commit this task.

### Task 4: One-command CLI, docs and CI

**Files:** Modify `src/kumiki_blender/cli.py`, `README.md`, `.github/workflows/python.yml`; create `docs/GAUSSIAN_SPLATS.md`; test `tests/test_reconstruction_splats_cli.py`.

**Interfaces:** `kumiki-blender plan-splats` and `build-splats`, mutually exclusive `--photos`, `--references`, `--reconstruction`.

- [ ] Write failing CLI tests for all three modes, argument validation and one-command registered-photo workflow.
- [ ] Run focused tests to verify red.
- [ ] Wire CLI to validated input → optional COLMAP → stage → OpenSplat; document Metal and CUDA executable setup and private-use limits.
- [ ] Run focused and full unit tests, compileall and diff checks; commit.
- [ ] Push the feature branch and create a PR with actual check results and GPU verification limitations.
