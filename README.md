# Kumiki Blender

A standalone, code-first Blender toolkit for reusable, cinematic 3D authoring. Its first milestone is a **generic lattice observation tower**: a deterministic architectural scaffold, **not** a photorealistic or dimensionally verified replica of Tokyo Tower or another real landmark.

The library separates testable Python geometry planning from Blender's `bpy` modeling and rendering. It includes three example generators: a tapered lattice observation tower, a **modular architectural portico** built from closed arch stones and lathed columns, and a **reusable stone building facade** with detailed windows, a panelled door, staggered masonry, classical cornices and a pitched tile roof. All use a shared revision exporter for editable `.blend`, portable `.glb`, inspection views and a JSON manifest for Kumiki-UI and ContentGen consumers.

## Requirements

- Python 3.11+ for recipe validation and unit tests (no runtime pip dependencies)
- Blender 4.2+ for geometry creation, `.blend` / GLB export and Cycles preview rendering
- Blender 4.5 LTS is exercised headlessly on Linux in GitHub Actions; macOS commands below are examples, not yet independently verified

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

The `portico.py` demonstration composes these primitives into an original, three-bay limestone arcade: **4 physically fluted shafts with entasis, 16 independent spiral Ionic-style volutes, carved base and capital profiles, 33 individual arch stones, projecting keystones, pedestals and a molding cornice**. Flutes are now recessed in the actual shaft mesh rather than floating thin rods. This is an architectural craft study—not a measured real landmark.

For other architectural styles, `classical_column()` generates **Tuscan-inspired smooth**, **Doric-inspired 20-flute**, and **Ionic-inspired 24-flute and scroll-capital** assemblies with named, editable parts. It can be reused in a new building recipe without authoring Blender mesh code. Preview exact geometry and component names without Blender:

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-classical-column \
  --order ionic --height 4.55 --radius 0.36
PYTHONPATH=src python3 -m kumiki_blender plan-portico \
  --bays 2 --order doric
```

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

This writes editable geometry, GLB, a hero preview and front/detail views through the **same export and audit pipeline** as the lattice tower. Headless Blender 4.5 has rendered both models successfully; smoke-quality previews still require artistic review before a claim of cinematic realism.

## Reusable architectural-detail system

The Blender-independent `architecture_details.py` provides composable, **manifold 3D solids** rather than texture-only facades:

| API | Reusable building detail |
|---|---|
| `coursed_wall()` | Staggered stone courses, physical mortar gaps, and real rectangular voids for doors/windows; adjustable block and course sizes |
| `window_assembly()` | Framed, recessed glazing with independent panes, stiles, mullions, transoms, sill, lintel and stone jambs |
| `door_assembly()` | Raised timber panels, panel moldings, jambs, stone lintel, threshold and lathed metal knob |
| `cornice_extrusion()` / `dentil_row()` | Projecting cross-section moldings and repeated stone teeth, in real dimensions |
| `gable_roof()` | Closed front/back pitched roof slabs, solid gable ends, a ridge cap and individually sloped/staggered tiles |
| `gutter_channel()` / `rainwater_system()` | Real open half-round gutters, connected downspouts, and fastening collars along the eaves |
| `rosette_relief()` | Parametric carved floral medallions with separate leaves, raised veins and central boss |
| `box_mesh()` | Cached, closed mesh primitive for any rectangular architectural detail |

Every piece returns a named `MeshInstance` with editable material role, transform and physical geometry. You can mix these functions in **new landmark recipes** without rewriting Blender object placement. `facade.py` demonstrates composition into a fully enclosed, 3/5/7-bay stone townhouse study with consistently named `masonry`, `window`, `door`, `roof`, `dentil` and `cornice` categories.

Inspect the object plan without Blender:

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-facade --bays 3
```

Build a multi-view editable revision (use a new output directory per build):

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-facade --bays 3 --output .review/facade/r01
```

The detail kit also includes a **tileable weathered-stone PBR preset** with adjustable `weathering_strength` from 0–1, subtle directional runoff/mineral deposits and deterministic per-block shade variation. Its Blender studio uses **glass-only light-linked rectangular reflection softboxes** to reveal glazing response without brightening the stonework; those lights are excluded from the GLB. A Three.js scene needs its own reflection environment for similar results.

The demonstration is **an original architectural detail study**, not a measured replica of an existing building. Future landmark work should substitute location-specific measured proportions, ornaments, roofs, glazing, weathering and material scans, with visual QA at production resolution. For component authoring conventions and examples, see [Architectural detail kit guide](docs/ARCHITECTURE_DETAILS.md).

## Reusable PBR surfaces

The pure Python material catalog defines physically distinct presets for painted structural steel, galvanized metal, concrete, stone, timber and optical glass. Recipes may override sRGB color and roughness without mutating shared presets. A plain, untextured `studio-matte` surface is available for presentation floors, keeping floor noise separate from the modeled object.

The toolkit now **generates deterministic, seamless PBR maps locally**, with no asset-site downloads or runtime Python dependencies. For stone, weathered stone, concrete, timber and metal it creates color grain, roughness variation and a tangent-space normal map, projects metric tile UVs onto generated mesh faces, and packs the PNGs into both the editable `.blend` and the GLB. Weathered stone adds vertically oriented runoff stains and irregular mineral deposits; the map remains tileable. The normal/roughness/color images use glTF-supported Principled BSDF links—not Blender-only shader nodes that would disappear on export.

CI runs `audit --require-pbr-textures` and the official Khronos validator on each example, including a re-imported model, so missing UVs or dropped material maps fail the build. These are **generic procedural surfaces**, not yet the measured, weathered, historically specific textures required for a photorealistic landmark replica.

For more believable landmark stone, metal and wood, **supply your own reviewed PBR scan maps** from any licensed provider. The optional `--texture-packs /path/to/material-packs.json` works with both `build-facade` (exact material roles) and `import-asset` (exact source material names). It loads local base-color/roughness/OpenGL-normal PNG/JPEG images, records their SHA-256 hashes and licensing declarations, preserves UVs or creates metric UVs, packs images into `.blend`, and embeds them in GLB. No particular marketplace is required. [Material-pack recipe and usage](docs/MATERIAL_PACKS.md).

## Import a reviewed asset from any source

The first source-independent intake path accepts a **local, static, self-contained `.glb`**. It can come from a licensed Sketchfab/Fab/CGTrader download, a scan, or your own work. It **does not search marketplaces, download assets, accept rigs or animations, or assume a model's source license permits redistribution**.

First, examine the GLB **and its exact material names** without Blender:

```bash
PYTHONPATH=src python3 -m kumiki_blender inspect-asset --input /path/to/model.glb
```

Create a small provenance record like [first-party-import.json](examples/first-party-import.json) with the creator, original HTTPS permalink, license, attribution and a truthful `redistribution_allowed: true`. Optionally include `expected_sha256` to pin the exact input model. This consent field is an **explicit declaration by the operator**, not automatic legal verification. **Do not use this source-export command for licensed models that prohibit distributing their raw files**; such assets require a separate render-only workflow.

Then import, preserve existing PBR materials/textures, render inspection shots and re-export a new versioned asset package:

```bash
KUMIKI_RENDER_PROFILE=smoke /Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  import-asset --input /path/to/model.glb \
  --provenance /path/to/source.json \
  --output .review/my-import/r01
```

To improve or recolor a specific imported material without editing Python, make a JSON mapping like [restyle-portico.json](examples/restyle-portico.json). Pass its path with `--material-overrides /path/to/overrides.json` to the same `import-asset` command. Each entry names an **exact material** reported by `inspect-asset`, a Kumiki preset (`natural-stone`, `weathered-stone`, `cast-concrete`, `timber`, `painted-steel`, etc.), and optional hex `color`/`roughness` (plus `weathering_strength` for weathered stone). Unrecognized, duplicate or ambiguous material names **fail before Blender creates a revision**; no heuristic keyword-based replacement is performed.

The replacement material gets the same packed glTF-compatible albedo, roughness and normal maps used by the generated architecture. Existing UVs are respected; missing UVs receive metric planar projection. The revision manifest lists all substitutions. Leave the overrides flag out to preserve all original materials. This is **visual restyling**, not an authenticity claim or permission to redistribute an asset.

The importer uses Blender's built-in glTF reader with image packing and semantic-node extras enabled. It checks imported mesh bounds/materials, rejects missing semantic part metadata, saves editable source, and uses the same Cycles multi-view/GLB/revision audits as the procedural models. The pull-request CI round-trips our own MIT-licensed portico both unchanged and with one explicit material override; no third-party artwork is copied into the repository.

## Inspect a GLB without Blender

```bash
PYTHONPATH=src python3 -m kumiki_blender audit --model .review/lattice-observation-tower/r01/model.glb
```

The lightweight auditor checks the GLB 2.0 envelope, declared file length, mesh list, number of named semantic nodes and absence of external texture/buffer dependencies. It does **not** replace a full glTF validator, mesh manifold analysis or visual QA. Actual builds also check `.blend` and PNG signatures before writing their manifest.

Pass `--require-pbr-textures` to additionally require UV0 and embedded albedo, roughness and normal texture hookups. This is intentionally optional for older untextured GLBs and third-party source models.

After building, cross-check **every file against the manifest** before importing it into another repository:

```bash
PYTHONPATH=src python3 -m kumiki_blender audit-revision --root .review/lattice-observation-tower/r01
```

This detects missing or path-escaping files, signature errors, mismatched file sizes, stale GLB audits and missing semantic node tags. ContentGen and Kumiki-UI should treat a passing revision audit as an asset-integrity prerequisite, not a substitute for visual approval.

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts
```

The Blender end-to-end test automatically runs when a `blender` executable is on `PATH`; otherwise it reports a skip. It may take several minutes to render. Tests that run only in Python cannot certify a visual result.

### Automated Blender render QA

Every pull request now has a Blender 4.5 LTS headless job. It downloads the version-pinned Blender archive, verifies its official SHA-256 checksum, builds both sample scenes, checks the generated GLBs, and uploads the editable `.blend`, GLB, manifests and PNG previews as a **Blender QA artifact** for visual review.

The CI job uses `KUMIKI_RENDER_PROFILE=smoke`, capping previews at 640 × 360 and 8 Cycles samples. It keeps the full geometry and material settings for meaningful export validation. The normal CLI uses `production` mode, preserving the sample and resolution settings in each asset recipe. A smoke preview is **not** a photorealistic acceptance render.

CI also runs the official Khronos glTF specification validator through the
version-pinned [glTF-Transform CLI](https://github.com/donmccurdy/glTF-Transform),
alongside Kumiki's own semantic-node and revision-integrity checks. This is a
CI-only development dependency, not a runtime dependency for model generation.

To run a faster local validation before a full production render:

```bash
KUMIKI_RENDER_PROFILE=smoke /Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-portico --bays 3 --output .review/portico/smoke-r01
```

Use a new output revision for each attempt; revisions cannot overwrite old files.

## Architecture

See [Architecture](docs/ARCHITECTURE.md), the [design spec](docs/superpowers/specs/2026-10-31-blender-foundation-design.md), and the [implementation plan](docs/superpowers/plans/2026-10-31-blender-foundation.md).

For photo-derived 2.5D parallax, see the [Depth Anything V2 integration decision](docs/IMAGE_TO_2_5D.md). It is an optional future pipeline, distinct from creating reference-accurate 3D landmark meshes.

For third-party materials, Geometry Nodes, rigging, BIM and export-validation candidates, see the [GitHub Blender ecosystem research](docs/BLENDER_ECOSYSTEM.md), including commercial-use licenses and integration priorities.

## Project direction

`kumiki-blender` owns high-fidelity creation, modeling recipes, baking, source files and exports. [Kumiki-UI](https://github.com/YukiUmetsu/kumiki-ui) owns interactive Three.js scene templates, motion and cameras. [ContentGen](https://github.com/YukiUmetsu/ai-content-generation) orchestrates media production. The repositories communicate through versioned assets and explicit manifests; no runtime Python dependency is imposed on Three.js.

### Next milestones

1. Conduct human review of higher-sample Blender renders and imported GLB from several angles; correct photographic and architectural defects.
2. Apply the new source-independent PBR scan importer to measured, rights-cleared landmark-specific materials; add a height/displacement workflow and physically scaled weathering for production shots.
3. Extend the roof/window/masonry/cornice kit with diverse roof types, historic facade styles, carved ornament, measured architectural references, and a separate render-only mode for licensed reference assets that cannot be redistributed as GLB.
4. Model a reference-verified landmark (proposed: Tokyo Tower), separately from the generic demonstration, and require human close-up review.
5. Add Geometry Nodes asset-library tools, LOD generation, scene-template previews, and optional asset catalog integration.

## License and assets

Code is MIT-licensed. The demonstration geometry is originally generated and has no external model dependencies. Future imported assets must preserve their own license, source, and permission to redistribute raw model files; commercial use of a render does not imply raw-model redistribution rights.
