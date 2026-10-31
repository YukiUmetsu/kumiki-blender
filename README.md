# Kumiki Blender

A standalone, code-first Blender toolkit for reusable, cinematic 3D authoring. Its first milestone is a **generic lattice observation tower**: a deterministic architectural scaffold, **not** a photorealistic or dimensionally verified replica of Tokyo Tower or another real landmark.

The library separates testable Python geometry planning from Blender's `bpy` modeling and rendering. It includes two example generators: a tapered lattice observation tower and a **modular architectural portico** built from closed arch stones, concave extrusions and lathed columns. Both use a shared revision exporter for editable `.blend`, portable `.glb`, inspection views and a JSON manifest for Kumiki-UI and ContentGen consumers.

## Requirements

- Python 3.11+ for recipe validation and unit tests (no runtime pip dependencies)
- Blender 4.2+ for geometry creation, `.blend` / GLB export and Cycles preview rendering
- Tested development targets: macOS with Blender installed and a standard Python environment; Blender integration has not yet been executed in the project author's current tool environment

## Validate a recipe without Blender

```bash
PYTHONPATH=src python3 -m kumiki_blender validate --recipe examples/lattice-observation-tower.json
```

This produces deterministic machine-readable JSON: asset ID, number of structural parts, world-space bounds and recipe SHA-256.

## Build a new revision with Blender

On macOS with Blender at the standard Applications location:

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build --recipe examples/lattice-observation-tower.json \
  --output .review/lattice-observation-tower/r01
```

The `--output` directory **must not already exist**; choose `r02` for the next revision. Existing files are never intentionally removed or overwritten. A failed build may leave a partial revision for inspection; do not mistake its existence for a successful build.

On Linux or a configured Mac shell, replace the full executable path with `blender`.

| Artifact | Purpose |
|---|---|
| `source.blend` | Editable native source with a separate model collection and studio scene |
| `model.glb` | Selected named model parts, no studio floor or lights |
| `preview.png` | Cycles hero preview |
| `previews/front.png`, `previews/side.png`, `previews/detail.png` | Additional silhouette and close-up QA views |
| `manifest.json` | Dimensions, provenance, asset audit, file sizes and verification status |

The current prototype uses tapered legs, structural X bracing, **64 physical gusset plates and 256 bolt-head meshes**, horizontal rings, two observation decks, railings and an antenna. It is deliberately illustrative. **Real landmark reproduction requires measured references, additional detail, better materials and actual multiview human acceptance.**

## Reusable architecture geometry and portico study

`architecture_geometry.py` is a Blender-independent mesh kernel for **concave profile extrusion** (with triangulated caps), **lathed columns and balusters**, **semicircular arches**, and **individual arch voussoir stones with real mortar gaps**. It validates self-intersecting profiles, computes watertightness and signed volume, and supports shared mesh instances with right-handed placement frames and linear repetition.

The `portico.py` demonstration composes these primitives into an original, three-bay limestone arcade: **4 fluted columns, 33 individually modeled arch stones, 64 flutes, projecting keystones, pedestals and an extruded molding cornice**. This is an architectural craft study—not a real landmark.

Preview its structural plan without Blender:

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-portico --bays 3
```

On a Mac with Blender installed, render and export a new revision:

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-portico --bays 3 --output .review/portico/r01
```

This writes editable geometry, GLB, a hero preview and front/detail views through the **same export and audit pipeline** as the lattice tower. No Blender-based visual validation has yet been performed in this environment.

## Reusable PBR surfaces

The pure Python material catalog defines physically distinct presets for painted structural steel, galvanized metal, concrete, stone, timber and optical glass. Recipes may override sRGB color and roughness without mutating shared presets. The Blender adapter applies these to Principled BSDF materials; procedural microtexture and reliable GLB texture baking are future work, not claimed capabilities.

## Inspect a GLB without Blender

```bash
PYTHONPATH=src python3 -m kumiki_blender audit --model .review/lattice-observation-tower/r01/model.glb
```

The lightweight auditor checks the GLB 2.0 envelope, declared file length, mesh list, number of named semantic nodes and absence of external texture/buffer dependencies. It does **not** replace a full glTF validator, mesh manifold analysis or visual QA. Actual builds also check `.blend` and PNG signatures before writing their manifest.

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts
```

The Blender end-to-end test automatically runs when a `blender` executable is on `PATH`; otherwise it reports a skip. It may take several minutes to render. Tests that run only in Python cannot certify a visual result.

## Architecture

See [Architecture](docs/ARCHITECTURE.md), the [design spec](docs/superpowers/specs/2026-10-31-blender-foundation-design.md), and the [implementation plan](docs/superpowers/plans/2026-10-31-blender-foundation.md).

For photo-derived 2.5D parallax, see the [Depth Anything V2 integration decision](docs/IMAGE_TO_2_5D.md). It is an optional future pipeline, distinct from creating reference-accurate 3D landmark meshes.

## Project direction

`kumiki-blender` owns high-fidelity creation, modeling recipes, baking, source files and exports. [Kumiki-UI](https://github.com/YukiUmetsu/kumiki-ui) owns interactive Three.js scene templates, motion and cameras. [ContentGen](https://github.com/YukiUmetsu/ai-content-generation) orchestrates media production. The repositories communicate through versioned assets and explicit manifests; no runtime Python dependency is imposed on Three.js.

### Next milestones

1. Run the Blender integration test on macOS, inspect native `.blend` and imported GLB from several angles, and correct defects.
2. Create a material and texture-baking pipeline with physically meaningful surface variation and no missing texture paths after GLB export.
3. Extend the new profile, extrusion, arch and instancing kernel with roofs, window surrounds, lattice joints and repeated ornament systems.
4. Model a reference-verified landmark (proposed: Tokyo Tower), separately from the generic demonstration, and require human close-up review.
5. Add Geometry Nodes asset-library tools, LOD generation, scene-template previews, and optional asset catalog integration.

## License and assets

Code is MIT-licensed. The demonstration geometry is originally generated and has no external model dependencies. Future imported assets must preserve their own license, source, and permission to redistribute raw model files; commercial use of a render does not imply raw-model redistribution rights.
