# Evidence-Guided 3D Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-destructive, mode-aware completion pipeline that imports and aligns existing 3D part candidates while preserving evidence provenance.

**Architecture:** Validate a source-relative JSON dossier with two inclusion policies and source SHA-256 checks. Reuse the established proper similarity fit and Blender revision exporter to place candidates beside, never over, the observed base geometry.

**Tech Stack:** Python 3.11+, `unittest`, dependency-free GLB audit, Blender 4.2+ for actual export.

**Spec:** `docs/superpowers/specs/2026-11-03-evidence-guided-completion-design.md`

## Global Constraints

- `historical` only exports observed scans/surveys; `cinematic` may include inferred geometry.
- Never overwrite sources or existing revision directories.
- Source-relative paths and immutable SHA-256 evidence; no third-party model execution or downloads.
- Landmark coordinates are Blender Z-up metres; at least four unique, non-coplanar anchors.
- Every output is labeled artistic-review-required; no historical likeness certification.

## Review Focus

- Symlink escape or missing candidate GLB must be rejected before output creation.
- A changed source hash must invalidate planning and building.
- Mirror/degenerate/high-error anchors must never export a candidate.
- Historical mode must not silently export inferred geometry.
- Blender unavailable or an existing output must not create a partial revision.

---

### Task 1: Manifest and policy contract

**Files:**
- Create: `src/kumiki_blender/reconstruction/completion.py`
- Test: `tests/test_completion.py`

**Interfaces:**
- Produces: `load_completion_plan(path: str | Path) -> CompletionPlan`, `evaluate_completion(plan: CompletionPlan) -> dict[str, Any]`, `CompletionError`.
- `CompletionPlan` carries `asset_id`, `mode`, `base_glb`, `base_sha256`, and `regions` with candidate GLBs, observed/inferred labels, and fitted transforms.

- [ ] Write failing tests for two modes, source hash/path validation, two unknown/invalid fields, no manifest side effects.
- [ ] Run `PYTHONPATH=src python -m unittest tests.test_completion -v`; confirm expected missing API failure.
- [ ] Implement strict source-relative manifest validation, GLB audits and mode selection.
- [ ] Run the focused test command; confirm all tests pass.
- [ ] Commit `feat: add evidence-guided completion manifest and policy`.

### Task 2: Geometric registration and report

**Files:**
- Modify: `src/kumiki_blender/reconstruction/completion.py`
- Test: `tests/test_completion.py`

**Interfaces:**
- `evaluate_completion` returns region `included`, `geometry_origin`, `transform`, `rms_error_m`, `max_error_m`, and model/evidence provenance, and rejects nonrigid/mirrored/unbounded fits.

- [ ] Write failing tests for translated/rotated/scaled anchors, mirrored/degenerate anchors, and excessive residual.
- [ ] Run focused tests; confirm expected failures.
- [ ] Integrate a proper similarity fit with measured-error thresholds, reusing existing fitting primitives.
- [ ] Run focused tests; confirm all pass.
- [ ] Commit `feat: anchor completion candidates to measured geometry`.

### Task 3: CLI and Blender assembly

**Files:**
- Modify: `src/kumiki_blender/cli.py`
- Create: `src/kumiki_blender/blender_completion.py`
- Test: `tests/test_completion_cli.py`

**Interfaces:**
- `plan-completion --manifest PATH` is side-effect-free and prints the complete JSON evaluation.
- `build-completion --manifest PATH --output DIR [--blender PATH]` invokes `build_completion_scene(plan, report, output, bpy_module=bpy)` through the existing factory-startup entrypoint pattern.

- [ ] Write failing CLI tests for plan policy, existing destination and missing Blender; check no partial output.
- [ ] Run focused CLI tests and observe missing command failures.
- [ ] Add CLI integration and safe Blender scene builder with separate measured/inferred objects, preview views and revision export.
- [ ] Run focused CLI tests; confirm all pass.
- [ ] Commit `feat: export non-destructive completion revisions`.

### Task 4: Documentation and verification

**Files:**
- Create: `docs/EVIDENCE_GUIDED_COMPLETION.md`
- Modify: `README.md`
- Modify: `tests/test_completion.py`, `tests/test_completion_cli.py` if needed for uncovered realistic failures.

- [ ] Document a complete private manifest example, strict vs cinematic mode, supplied-GLB adapter, outputs, review limitations, and future AI-runner integration.
- [ ] Run `PYTHONPATH=src python -m unittest tests.test_completion tests.test_completion_cli -v` and `PYTHONPATH=src python -m unittest discover -s tests -q`.
- [ ] Record environment-only baseline failures separately and check `git diff --check`.
- [ ] Commit `docs: document evidence-guided completion and limits`.
