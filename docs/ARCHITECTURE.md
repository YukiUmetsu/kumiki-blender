# Architecture and Output Contract

## Dependencies

`recipe.py`, `lattice.py`, `architecture_geometry.py`, and `portico.py` contain no `bpy` calls or non-standard Python dependencies. `blender_scene.py` and `blender_portico.py` receive an explicit Blender module and only touch Blender data when their builders run. `cli.py` exposes normal-Python planning and audit commands, importing Blender only for build commands.

## General-purpose architecture mesh kernel

`architecture_geometry.py` provides:

- `extrude_profile()`: triangulated, manifold extrusion of a simple concave polygon. Self-intersections, coincident/collinear corners and nonfinite values are rejected before meshing.
- `revolve_profile()`: capped positive-radius solids for columns, balusters, finials and stepped moldings.
- `semicircular_arch()` and `arch_voussoirs()`: genuinely open semicircular arch voids, and individually closed stones with physical joint gaps.
- `place_mesh()` / `repeat_linear()`: deterministic, named, right-handed rigid instances sharing source mesh data without duplicating every vertex.
- `audit_mesh()`: edge orientation, boundary and non-manifold counts, signed volume and degenerate polygon checks. All generated solids pass these pure-Python inspections.

`portico.py` demonstrates composition into a reusable 1–20-bay arcade with repeated column flutes, alternate stone surfaces, projecting keystones, pedestals and a concave extruded cornice. The adapter caches Blender mesh datablocks by source geometry and material, while preserving individual object names and metadata. A passing geometry audit is **not** a substitute for real image/render QA.

## Recipe version 1

The initial generator is `lattice-tower`, a four-legged tapered structural scaffold. Each crossing truss bay receives a physical outward-facing gusset and four fastener heads. Strict JSON fields and numerical bounds are enforced; unknown fields and non-finite numbers are errors. World dimensions are in meters, with +Z up. Recipe output is deterministic given the same source and code version, and conservative bounds include fastener and plate extents.

Recipes never accept executable Python or arbitrary paths. A recipe cannot directly specify a filesystem destination; that is provided separately to the CLI.

## Scene and export

- Headless builds require an empty factory-started process and reuse its **active** scene, hiding stock startup objects without deleting their datablocks. The Blender 4.5 glTF exporter has reproducibly segfaulted when exporting meshes from an inactive newly created scene via a temporary context override. In interactive Blender, a separate scene is created and activated without deleting pre-existing scenes.
- Model meshes are grouped in `Model / portable GLB`; studio elements are separate.
- Structural beams carry stable names and a `kumiki_part` custom property for future camera-target and semantic part bindings.
- The preview uses Cycles, a neutral sky, daylight sun, a ground plane and an architecture-framing camera. Four deterministic camera poses cover hero, front, side and construction detail, with cheaper sampling on secondary views.
- The GLB contains only selected model objects; shader compatibility and visual parity must be checked by a real glTF/Three.js import before release.
- `blender_output.py` owns the shared immutable revision exporter, camera passes, manifest construction and output audits for both tower and portico models. It saves the editable `.blend` **before** baking evaluated modifiers into the transient export scene, preserving source editing while ensuring the portable GLB contains real bevel and normal details. This also avoids an observed Blender 4.5 headless glTF-export crash with the exporter's global Apply Modifiers option.
- `render_profiles.py` provides a production profile and a resource-capped smoke profile. `KUMIKI_RENDER_PROFILE=smoke` affects only sample count and preview resolution, never model mesh fidelity or GLB exports. The selected profile and its dimensions are recorded in the manifest.
- A GLB envelope audit detects malformed exports and external texture/buffer references. The build verifies native `.blend` and PNG file signatures, then includes the audit report and output file sizes in the manifest.
- `revision_audit.py` independently rechecks the full delivered asset package against its manifest, rejecting missing/tampered/path-escaping files, GLB audit mismatches and absent semantic part tags. It is the asset-integrity boundary for future consumers.
- A source digest and Blender version are recorded to aid repeatability and provenance.

### Headless Blender CI

The pull-request CI job installs a pinned official Blender 4.5 LTS Linux release and checks its SHA-256 against the Blender Foundation manifest before execution. It builds both the lattice and portico studies, audits GLB portability with local semantic checks **and** the official Khronos validator via a pinned glTF-Transform CLI, and uploads previews, editable source and manifests to GitHub workflow artifacts. Low-sample smoke renders are intended to catch broken geometry/export/camera setups; final cinematic acceptance still requires production-resolution renders and visual review.

## Materials

`surfaces.py` defines immutable PBR presets for painted steel (dielectric coating), galvanized metal, concrete, stone, timber, and glass. `blender_scene.py` converts their sRGB colors to linear Blender shader inputs and maps metallic/roughness/transmission values to Principled BSDF. No shader-only procedural textures are silently assumed to transfer to GLB: a tested bake pipeline is required before texture-based fidelity claims.

## Artifact states

`structural-prototype / visual-review-pending` is the honest status of the first example. A generated `.blend` or a passing Python test is insufficient for photorealism approval. Production quality requires verified geometry, normal orientation, material response, near/far detail, silhouette accuracy, GLB compatibility and actual render inspections.

## Boundaries

The future Kumiki-UI adapter can load `model.glb` with its current GLTF loader, then map `kumiki_part` / stable names to semantic camera targets. The Blender library remains independent of Three.js and ContentGen. Photo-derived 2.5D scenes belong to a separate optional pipeline described in [Image-to-2.5D](IMAGE_TO_2_5D.md).
