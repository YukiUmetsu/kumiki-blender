# Official CC0 Anatomical Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reusable, provenance-checked GLB from MakeHuman's actual CC0 human anatomy mesh, ready for the existing landmark-fit tool.

**Architecture:** Standard-library upstream-source verification and OBJ/body parsing; isolated Blender bmesh head/neck extraction; existing revision exporter and catalog CLI wiring.

**Tech Stack:** Python 3.11+, Blender 4.2+, existing asset catalog and revision exporter, unittest.

**Spec:** `docs/superpowers/specs/2026-11-02-makehuman-head-base-design.md`

## Global Constraints

- Use official CC0 graphical source only; pin full upstream commit and SHA-256.
- No new required dependency, third-party binary committed, or novel procedural face.
- Outputs are new immutable directories and never assert photo fidelity.

## Review Focus

- Wrong SHA/download errors fail before Blender and do not create a revision.
- OBJ group parsing excludes helper/rig faces, rejects invalid coordinates.
- Head slice avoids shoulders and remains manifold after actual Blender bmesh cut.
- GLB has no armature/animations, has facial anatomy and sane metric bounds.
- Catalog builds from local or cached source without modifying old commands.

---

### Task 1: Verified upstream MakeHuman core source

**Files:** `src/kumiki_blender/head_sources/makehuman.py`, `tests/test_makehuman_source.py`

**Interfaces:** `verify_makehuman_obj(path) -> MeshSource`; `load_body_mesh(source) -> (vertices, faces)`; optional `fetch_makehuman_obj(cache_path)`.

- [ ] Write failing tests for official pin, malformed OBJ, body-only extraction and head bounds.
- [ ] Verify failing tests, implement strict small parser and SHA checks, verify green.
- [ ] Commit.

### Task 2: Blender head extraction and immutable revision

**Files:** `src/kumiki_blender/blender_head_sources.py`, `scripts/smoke_makehuman_head.py`

**Interfaces:** `build_makehuman_head(source: MeshSource, output: Path, *, bpy_module) -> dict`.

- [ ] Build actual body topology in Blender, slice at the anatomical neck, cap,
      orient/scale head and export with standard QA shots.
- [ ] In a real Blender smoke, audit manifold geometry, no full-body or helper
      pieces, facial bounds and generated GLB/source.blend.
- [ ] Commit.

### Task 3: CLI, catalog and docs

**Files:** `src/kumiki_blender/cli.py`, `src/kumiki_blender/asset_catalog.py`, `src/kumiki_blender/assets/characters/people/makehuman-neutral-head/asset.json`, `docs/ANATOMICAL_HEAD_BASES.md`, `.github/workflows/python.yml`, `tests/test_makehuman_cli.py`.

**Interfaces:** `prepare-makehuman-head --source-obj PATH --output DIR` or `--download`; `catalog build makehuman-neutral-head` uses approved cached source or download explicitly.

- [ ] Write failing CLI/catalog tests, implement CLI and source-neutral catalog wiring.
- [ ] Document source provenance, usage and how to feed output into `head-fit`.
- [ ] Run focused tests, full suite, compileall, real Blender CI and update PR #14.
