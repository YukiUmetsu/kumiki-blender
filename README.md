# Kumiki Blender

## New: Agent-authored AssetSpec v2 composer

The optional [AssetSpec v2 composer](docs/ASSET_SPEC_V2.md) adds one generic
validated plan/build path for assemblies of reusable geometry. Start with
`pip install -e '.[composer,qa]'`, then run
`kumiki-blender plan-asset --spec examples/desk-lamp-assembly-v2.json` or
`kumiki-blender asset-schema` to inspect the machine-readable recipe contract.
`build-asset` produces the usual editable Blender scene, GLB, inspection views
and immutable manifest. Existing generator commands and the dependency-free
kernel remain unchanged.

`visual-qa --revision .review/desk-lamp/r01 --output .review/desk-lamp/r01-inspection`
generates a labeled multi-view contact sheet and exposure metrics. Adding
`--baseline` compares revisions and highlights pixel-level regressions; artistic
quality still requires a human review. See [the visual QA guide](docs/ASSET_SPEC_V2.md#multi-view-visual-qa-pillow--optional-openimageio).

A standalone, code-first Blender toolkit for reusable, cinematic 3D authoring. Its first milestone is a **generic lattice observation tower**: a deterministic architectural scaffold, **not** a photorealistic or dimensionally verified replica of Tokyo Tower or another real landmark.

The library separates testable Python geometry planning from Blender's `bpy` modeling and rendering. It includes a tapered lattice observation tower, a **modular architectural portico** built from closed arch stones and lathed columns, a **reusable stone building facade** with detailed windows/doors/roof, a **parametric Egyptian monument study** with pyramid casing blocks and a sculpted lion-bodied Sphinx figure, and a **rigged red vintage biplane** modeled from user-provided multiview artwork. All use a shared revision exporter for editable `.blend`, portable `.glb`, inspection views and a JSON manifest for Kumiki-UI and ContentGen consumers.

## Find models in the categorized asset catalog

Every model family has one manifest in `src/kumiki_blender/assets/<category>/<subcategory>/<asset-id>/asset.json`. Existing geometry modules and CLI entry points remain compatible; discover assets by name, category, tags, or alternate names without knowing which Python module implements them:

```bash
PYTHONPATH=src python3 -m kumiki_blender catalog categories
PYTHONPATH=src python3 -m kumiki_blender catalog list --category landmarks
PYTHONPATH=src python3 -m kumiki_blender catalog search old red airplane
PYTHONPATH=src python3 -m kumiki_blender catalog list --rigged
PYTHONPATH=src python3 -m kumiki_blender catalog show red-vintage-biplane
PYTHONPATH=src python3 -m kumiki_blender catalog validate
```

Create a categorized revision directly (inside Blender):

```bash
KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup \
  --python-exit-code 1 --python scripts/blender_entry.py -- \
  catalog build red-vintage-biplane --revision r01
```

The result is `.review/vehicles/aircraft/red-vintage-biplane/r01/` containing the usual `.blend`, GLB, previews and manifest. Pass `--root /your/permanent/asset-library` for a durable asset store outside the gitignored QA directory. [Catalog design, categories and conventions](docs/ASSET_CATALOG.md).

The **Egypt cinematic relief atlas** adds a fifth illustration generator: a sculpted geographic table map with deliberately heightened Sinai/Eastern Desert relief, Mediterranean and Red Seas, Nile valley/delta and waterways, Qattara Depression, Lake Nasser, **physical irrigated parcels**, and handcrafted Giza/Luxor/Cairo/Alexandria/Aswan miniature landmarks. Distinct city districts have tiny buildings, roofs and windows; twelve inspection cameras include city-level and terrain macro views. It recreates the layout of a supplied visual reference procedurally rather than placing the image on a plane. Preview it with `PYTHONPATH=src python3 -m kumiki_blender plan-egypt-map` or build with `blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- build-egypt-map --output .review/egypt-map/r01`. [See the detailed Egypt atlas guide](docs/EGYPT_RELIEF_MAP.md). The geographic representation is illustrative, not surveyed.

The Egypt generator is now organized under `src/kumiki_blender/geography/egypt/` (terrain, geometry, Blender support, landmarks and scene), with a categorized manifest at `assets/geography/maps/egypt-relief-atlas/asset.json`. Discover or build it with `catalog show egypt-relief-atlas` and `catalog build egypt-relief-atlas --revision r01`. The original Python import paths remain valid.

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

`architecture_geometry.py` is a Blender-independent mesh kernel for **concave profile extrusion** (with triangulated caps), **lathed columns and balusters**, **semicircular and pointed Gothic arches**, and **individual arch voussoir stones with real mortar gaps**. It validates self-intersecting profiles, computes watertightness and signed volume, and supports shared mesh instances with right-handed placement frames and linear repetition.

The `portico.py` demonstration composes these primitives into an original, three-bay limestone arcade: **4 physically fluted shafts with entasis, 16 spiral Ionic-style volute ridges mounted on 16 shaped stone cushions, carved base and capital profiles, 33 individual arch stones, projecting keystones, pedestals and a molding cornice**. Flutes are recessed in the actual shaft mesh rather than floating thin rods. This is an architectural craft study—not a measured real landmark.

For other architectural styles, `classical_column()` generates **Tuscan-inspired smooth**, **Doric-inspired 20-flute**, and **Ionic-inspired 24-flute and scroll-capital** assemblies with named, editable parts. `pointed_arch_voussoirs()` builds two-arc Gothic masonry with a real pointed apex; `balustrade_assembly()` repeats individually turned stone spindles between coping rails and end piers. These can be reused in new building recipes without authoring Blender mesh code. Preview exact geometry and component names without Blender:

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-classical-column \
  --order ionic --height 4.55 --radius 0.36
PYTHONPATH=src python3 -m kumiki_blender plan-portico \
  --bays 2 --order doric
PYTHONPATH=src python3 -m kumiki_blender plan-portico \
  --bays 2 --order tuscan --arch-style pointed --balustrade
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

The Gothic-and-balustrade variant uses the same exporter and can be rendered with `build-portico --bays 2 --order tuscan --arch-style pointed --balustrade --output .review/gothic/r01`. The options are independent: pointed arches need not have railings, and railings also work with round arches.

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
| `pointed_arch_voussoirs()` | Two-circle Gothic intrados/extrados, genuinely sharp apex and mortar-separated masonry blocks |
| `turned_stone_baluster()` / `balustrade_assembly()` | Curved lathed spindles, continuous base/coping rails, end posts and post caps; checked spacing |
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

## Egyptian pyramids and Sphinx geometry

`egyptian_monuments.py` adds a reusable kit for pyramid and statue modeling:

- `pyramid_frustum()` and `pyramid_casing_stone()` create **closed tapered volumes**, with a recessed limestone core, real staggered vertical/horizontal mortar joints, individually selectable casing stones and a capstone. `pyramid_assembly()` exposes dimensions, course count, stone width, casing fraction and gaps.
- `loft_elliptical_sections()` sculpts organic solids along X/Y/Z using independently adjustable radii, offsets and cubic interpolation; it is reusable for statuary, carved rock, animal anatomy and terrain features.
- `sphinx_assembly()` composes an anatomically shaped, differentially eroded lion torso, chest, four digits per forepaw, haunches, curled tail, carved face and nemes-style headdress into named editable components. An **optional** `restored_nose=True` allows an interpretive reconstruction variant.

Preview the miniature combined study without Blender:

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-monument --kind giza-study
```

Build an editable `.blend`, a self-contained `.glb`, hero/site previews, and dedicated masonry/Sphinx closeups:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-monument --kind giza-study --output .review/egyptian/r01
```

`--kind pyramid` and `--kind sphinx` build each model independently. For a **historically scaled Great Pyramid course study**, the `khufu-petrie` preset uses 201 measured course elevations rather than the miniature's evenly spaced courses. It leaves the apex missing and batches thousands of stone blocks into 805 Blender objects:

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-monument --kind pyramid --pyramid-preset khufu-petrie
```

This is **measured course-height geometry, not a fully surveyed pyramid surface**: stone layouts, damage and repairs are procedural. The Sphinx remains an **illustrative sculpture** unless replaced by a separately licensed monument scan. Imported Sphinx models now receive a non-authoritative proportion review against published monument bounds; a human still needs to verify facial and paw fidelity. See the [Egyptian monument kit guide](docs/EGYPTIAN_MONUMENTS.md) for source licensing, evaluated candidate meshes and archaeological accuracy limits.

## Rigged vintage red biplane

The `red-vintage-biplane` generator recreates a user-supplied five-view illustration as a coherent, original 3D biplane with actual shaped fabric airfoils and rib tapes, complete upper/lower wings, wooden interplane struts and crossed flying wires, tapered fuselage, open cockpit, glazed windscreen, engine grille, sculpted wooden two-blade propeller, and separately hinged control surfaces. The Blender source exposes nine animation controls for the propeller, four ailerons, elevators, rudder and wheels. GLB exports the animation clips alongside all named geometry; six views make reference comparison possible.

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-biplane
KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- build-biplane --output .review/biplane/r01
```

The dimensions are reference-derived estimates, not measurements of any known aircraft. See the [aircraft authoring and rig guide](docs/VINTAGE_BIPLANE.md) for design choices, surface specifications, controls and export verification.

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

### Focused PR checks and manual full regression

Normal pushes and pull requests run focused **Python 3.11** regressions for
both Khufu's measured masonry and Sphinx source-mesh handling. Pull requests
add a Blender 4.5 LTS job that verifies the archive's official SHA-256
checksum, builds the Khufu pyramid and imports a first-party **Sphinx test
proxy**, then checks their GLBs, source provenance, geometry-readiness
reports and dedicated masonry/anatomical previews. Editable `.blend`,
GLB, manifests and PNG previews remain available in the **Blender QA artifact**.
This keeps the review feedback loop centered on code still changing.

**No tests have been deleted.** The complete Python test suite now runs on
**one Python version (3.11)** when invoked manually. Legacy
architectural builds/import variations, broader glTF-Transform specification
validation and revision checks are part of an **opt-in full regression**:
open GitHub Actions `Python foundation` and select **Run workflow**
(`workflow_dispatch`), or run the Python suite locally using the command
above. Python 3.12 and 3.13 CI matrices have been removed.

The CI job uses `KUMIKI_RENDER_PROFILE=smoke`, capping previews at 640 × 360 and 8 Cycles samples. It keeps the full geometry and material settings for meaningful export validation. The normal CLI uses `production` mode, preserving the sample and resolution settings in each asset recipe. A smoke preview is **not** a photorealistic acceptance render.

The manually triggered full regression also runs the official Khronos glTF
specification validator through version-pinned
[glTF-Transform CLI](https://github.com/donmccurdy/glTF-Transform),
alongside semantic-node and revision-integrity checks. It remains a
development dependency, not a runtime dependency.

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
