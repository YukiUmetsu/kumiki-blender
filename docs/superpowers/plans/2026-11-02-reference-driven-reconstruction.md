# Reference-driven Reconstruction Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make actual multi-view statue photographs usable as validated reconstruction evidence and as input to an optional sparse photogrammetry worker.

**Architecture:** A standard Python `reconstruction` package owns the source manifest, image inspection, and quality report. A separate COLMAP adapter owns external execution and immutable output. CLI and docs expose the same reusable path for every monument.

**Tech Stack:** Python 3.11+, existing Pillow `qa` extra, optional local COLMAP executable.

**Spec:** `docs/superpowers/specs/2026-11-02-reference-driven-reconstruction-design.md`

## Global Constraints

- No network downloads, no `bpy` dependency, no copyright/redistribution inference.
- Never overwrite an existing revision or promote a sparse reconstruction as a verified statue mesh.
- Keep other model generators and CLI commands backward compatible.

## Review Focus

- Symlinks and relative-path traversal must not expose local arbitrary files.
- Duplicate or repeated similar photos must not inflate reference coverage.
- Masks with wrong dimensions must be rejected before COLMAP starts.
- Missing external executable or failed reconstruction must surface explicit review-required status.
- Output and manifest must never claim archaeological or photographic fidelity.

---

### Task 1: Reference manifest and coverage audit

**Files:** `src/kumiki_blender/reconstruction/references.py`, `src/kumiki_blender/reconstruction/__init__.py`, `tests/test_reconstruction_references.py`

**Interfaces:** `load_reference_set(path) -> ReferenceSet`; `audit_reference_set(reference_set) -> dict`; `ReferenceSet` carries stable IDs and absolute validated local photo/mask paths.

- [ ] Write tests for verified PNGs, malicious/unexpected JSON, traversal, duplicate hashes, and missing viewpoints/regions.
- [ ] Run the tests, see expected failure due to missing module.
- [ ] Implement minimal manifest parsing and diagnostics.
- [ ] Run focused tests and verify green.
- [ ] Commit.

### Task 2: Safe optional COLMAP sparse reconstruction

**Files:** `src/kumiki_blender/reconstruction/colmap.py`, `tests/test_reconstruction_colmap.py`

**Interfaces:** `build_sparse_reconstruction(reference_set, output, *, colmap="colmap") -> dict`; output is immutable and records commands and reconstruction status.

- [ ] Write tests for input staging, mask conventions, missing executable, immutable output, and successful/failed controlled command execution.
- [ ] Run tests, see expected failure due to missing implementation.
- [ ] Implement stage/build/audit around executable `feature_extractor`, `exhaustive_matcher`, and `mapper`.
- [ ] Verify focused tests green.
- [ ] Commit.

### Task 3: CLI, docs and CI contract

**Files:** `src/kumiki_blender/cli.py`, `tests/test_reconstruction_cli.py`, `docs/REFERENCE_RECONSTRUCTION.md`, `README.md`, `.github/workflows/python.yml`, `examples/monument-reference-set.example.json`

**Interfaces:** CLI `reference-audit --manifest PATH` and `reconstruct-photos --manifest PATH --output DIR [--colmap EXE]`.

- [ ] Write failing CLI tests for structured reports and error codes.
- [ ] Implement CLI integration with existing error handling.
- [ ] Document editable image manifest, limitations, workflow and safe source handling.
- [ ] Run focused tests, full Python suite, compileall, and catalog validation.
- [ ] Commit, push, create PR.
