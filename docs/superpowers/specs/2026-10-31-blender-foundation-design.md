# Kumiki Blender Foundation — Design

## Intent

Build a standalone, reusable Blender Python authoring toolkit for cinematic 3D work. The first increment must produce an editable demonstration of a **generic** lattice observation tower, not claim to reproduce Tokyo Tower or any protected/verified landmark. World-landmark replicas require separate reference research and visual acceptance.

## Goals

- Versioned, strict JSON recipes validated without Blender.
- Deterministic, reusable architectural construction independent of `bpy`.
- A headless Blender adapter for physically constructed geometry, differentiated PBR surfaces, scene lighting, editable `.blend`, reusable `.glb`, a preview still and provenance manifest.
- Safe, non-overwriting output directories and explicit actionable failures.
- A Python-only CI suite and documented real-Blender smoke checks.

## Non-goals for the foundation increment

- No claim of photorealism or landmark accuracy before human review.
- No downloading third-party models, scraped imagery or training data.
- No GUI add-on, cloud render farm, movie exporter or ContentGen integration yet.
- No change to the existing Kumiki-UI repository.

## Constraints

- Pure-Python code must support Python 3.11+ without third-party dependencies; Blender operations run under Blender's bundled Python with `bpy`.
- Blender 4.2+ is the initial target; use feature detection for exporter differences.
- Geometry is meters, Z-up in Blender; GLB is the interchange format for Three.js.
- Revisions are immutable by default. The build command rejects existing output directories, never clears them.
- The manifest records generator identity, Blender version, recipe digest and the explicitly illustrative provenance.
- Programmatic checks never substitute for close-up/multi-angle human visual acceptance.

## Approaches

1. **All Blender operators:** Fast start but hard-to-test geometry logic and fragile dependence on Blender context.
2. **Imported models only:** Great for selected detailed assets but little reusable construction machinery; licensing and provenance vary.
3. **Hybrid (chosen):** Pure geometry specification + Blender implementation + future licensed/curated detail assets. Keeps Python unit tests fast and real asset rendering capable of evolving independently.

## Interface

`kumiki_blender.recipe.load_recipe(path) -> TowerRecipe` validates a v1 `lattice-tower` document.

`kumiki_blender.lattice.make_lattice_tower(recipe) -> TowerPlan` returns named beams/decks with meters-based coordinates and deterministic ordering; no Blender import.

`kumiki_blender.blender_scene.build_scene(recipe, plan, destination) -> dict` creates Blender data in a factory-started process and writes assets to a new destination. The CLI provides `validate` (normal Python) and `build` (Blender only).

The demonstration asset uses a generic red-painted steel lattice tower with four tapered legs, repeated cross braces, horizontal structural rings, an observation platform, and railings. It is a *structural prototype*, not a finished photorealistic landmark.

## Acceptance

- Unit tests cover valid and invalid recipes, boundaries, deterministic structure, symmetry, safe output behavior and CLI status.
- `python -m unittest discover -s tests -v` passes in a plain Python environment.
- On a Blender-equipped machine, one command writes `.blend`, `.glb`, PNG, and JSON; reviewer inspects multiple angles and the imported GLB before any cinematic approval.
- If Blender cannot run in the current environment, that limitation is reported; do not claim a render was verified.
