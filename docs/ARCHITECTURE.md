# Architecture and Output Contract

## Dependencies

`recipe.py` and `lattice.py` contain no `bpy` calls or non-standard Python dependencies. `blender_scene.py` receives an explicit Blender module and touches Blender data only when `build_scene` runs. `cli.py` exposes a normal-Python validator and dynamically calls Blender only for `build`.

## Recipe version 1

The initial generator is `lattice-tower`, a four-legged tapered structural scaffold. Each crossing truss bay receives a physical outward-facing gusset and four fastener heads. Strict JSON fields and numerical bounds are enforced; unknown fields and non-finite numbers are errors. World dimensions are in meters, with +Z up. Recipe output is deterministic given the same source and code version, and conservative bounds include fastener and plate extents.

Recipes never accept executable Python or arbitrary paths. A recipe cannot directly specify a filesystem destination; that is provided separately to the CLI.

## Scene and export

- A new Blender scene is created without deleting the caller's existing scenes.
- Model meshes are grouped in `Model / portable GLB`; studio elements are separate.
- Structural beams carry stable names and a `kumiki_part` custom property for future camera-target and semantic part bindings.
- The preview uses Cycles, a neutral sky, daylight sun, a ground plane and an architecture-framing camera. Four deterministic camera poses cover hero, front, side and construction detail, with cheaper sampling on secondary views.
- The GLB contains only selected model objects; shader compatibility and visual parity must be checked by a real glTF/Three.js import before release.
- A GLB envelope audit detects malformed exports and external texture/buffer references. The build verifies native `.blend` and PNG file signatures, then includes the audit report and output file sizes in the manifest.
- A source digest and Blender version are recorded to aid repeatability and provenance.

## Materials

`surfaces.py` defines immutable PBR presets for painted steel (dielectric coating), galvanized metal, concrete, stone, timber, and glass. `blender_scene.py` converts their sRGB colors to linear Blender shader inputs and maps metallic/roughness/transmission values to Principled BSDF. No shader-only procedural textures are silently assumed to transfer to GLB: a tested bake pipeline is required before texture-based fidelity claims.

## Artifact states

`structural-prototype / visual-review-pending` is the honest status of the first example. A generated `.blend` or a passing Python test is insufficient for photorealism approval. Production quality requires verified geometry, normal orientation, material response, near/far detail, silhouette accuracy, GLB compatibility and actual render inspections.

## Boundaries

The future Kumiki-UI adapter can load `model.glb` with its current GLTF loader, then map `kumiki_part` / stable names to semantic camera targets. The Blender library remains independent of Three.js and ContentGen.
