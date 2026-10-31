# Reusable Blender ecosystem research (2026-10-31)

Purpose: avoid reimplementing solved architectural modeling, rigging, materials and export-validation problems while keeping `kumiki-blender` **code-first, commercially usable and suitable for unattended Blender builds**.

Licenses below describe the linked source or the downloadable media as noted. They are **not** blanket permission to copy an add-on's implementation into this MIT-licensed repository. Verify each dependency/version when actually adopted.

## Ranked shortlist

| Project | Why it matters | License and compatibility | Proposed use |
|---|---|---|---|
| [Khronos glTF-Validator](https://github.com/KhronosGroup/glTF-Validator) | Full glTF spec diagnostics beyond our minimal envelope check: accessors, buffers, extensions and images | **Apache-2.0**; Blender-independent; cross-platform | Mandatory post-export CI QA, zero errors; preserve separate Kumiki semantic-part checks |
| [glTF-Transform](https://github.com/donmccurdy/glTF-Transform) | GLB validation, inspection, geometry/texture optimization and compression | **MIT**; Node CLI | Optional optimization for Three.js with a regression test that names/extras survive |
| [ambientCG](https://ambientcg.com/) / [importer](https://github.com/ninofiliu/blender-ambientcg-addon) | PBR map sets with sensible Blender Principled BSDF wiring | **Assets CC0**; importer **GPL-3.0-or-later**, [official extension](https://extensions.blender.org/add-ons/ambientcg-material-importer/versions/) supports Blender 4.2+ | Independent small PBR material builder; use add-on as optional authoring reference, not copied source |
| [NodeToPython](https://github.com/BrendanParmer/NodeToPython) | Turn visually developed Geometry Nodes and shader graphs into Python recipes | **GPL-3.0**; explicitly covers Blender 4.2–5.2 and macOS | Optional **authoring tool** for reusable node graphs; inspect generated recipes before incorporation |
| [Sverchok](https://github.com/nortikin/sverchok) / [nodebpy](https://github.com/BradyAJohnston/nodebpy) | Advanced parametric shaping and concise procedural Geometry/Shader/Compositor Nodes definitions | Both **GPL-3.0**; Sverchok explicitly lists Blender 4.5, nodebpy compatibility requires a smoke test | Prototyping aids installed separately, no mandatory library dependency or source vendoring |
| [IfcOpenShell / Bonsai](https://github.com/IfcOpenShell/IfcOpenShell) | IFC/BIM geometry, measured walls/doors/windows and dimensional building semantics | IfcOpenShell core **LGPL-3.0-or-later**; Bonsai add-on **GPL-3.0-or-later**; test Apple M4 builds and pin a stable version | Future reference-accurate architecture import/authoring path, not part of the small core |
| [trimesh](https://github.com/mikedh/trimesh) | Mesh watertightness, volume, bounds, repair and GLB parsing | **MIT**; Python, some features need optional dependencies | Optional secondary geometry QA; keep stdlib kernel audit as the fast baseline |
| [COLMAP](https://github.com/colmap/colmap) | True multi-view reconstruction from photographs instead of a single-image depth illusion | **BSD-3-Clause**; macOS support, computationally expensive | Independent reference acquisition tool when enough landmark photos and image rights exist |
| [Rokoko Studio Live for Blender](https://github.com/Rokoko/rokoko-studio-live-blender) | Mocap and retargeting for animated human subjects | **LGPL-3.0** in LICENSE (README MIT badge is stale); proprietary Rokoko Studio dependencies for some workflows; 4.5 unverified | Optional rigging route; lower priority than architecture/materials |

Additional useful options: [Material Maker](https://github.com/RodZill4/material-maker) (**MIT**, offline PBR texture authoring), [BlenderProc](https://github.com/DLR-RM/BlenderProc) (**GPL-3.0**, scripted render/normal/depth passes), [BlenderGIS](https://github.com/domlysz/BlenderGIS) (**GPL-3.0**, georeferenced landscapes; 4.5 compatibility unverified), and [meshoptimizer/gltfpack](https://github.com/zeux/meshoptimizer/tree/master/gltf) (**MIT**, fast GLB optimization).

The [Poly Haven Public API](https://github.com/Poly-Haven/Public-API) is retained only as an **optional known-URL asset source**, not a recommended discovery workflow or pipeline requirement. Its media are CC0, but its API source is AGPL-3.0 and its live API has [separate terms](https://github.com/Poly-Haven/Public-API/blob/master/ToS.md). The collection and search have not matched the project author's practical landmark needs; avoid depending on it or assuming a requested asset exists there.

## Adoption sequence

1. **Export QA (integration underway):** run the official Khronos spec validator through glTF-Transform 4.5.1 in the GitHub Blender job after our own manifest/semantic checks. Only optimize as a separate, explicitly requested step, never silently replace the source GLB. The native Blender smoke render is still being debugged; validator adoption does not imply that CI is green.
2. **Realistic materials and lighting:** favor self-contained Blender shader patterns (micro-bump, grime, mortar, subtle roughness variation) with deterministic parameters and GLB-safe baking. Optional licensed CC0 textures or HDRIs may be useful for specific shots, but no external asset catalog—including Poly Haven—should be required, and search results are not a substitute for asset suitability checks.
3. **Authoring acceleration:** experiment with Geometry Nodes for façade details, cornices, windows, roofs and ornament. NodeToPython/nodebpy/Sverchok can be external creators while published recipes remain self-contained, testable Python.
4. **Measured landmarks:** add IfcOpenShell or COLMAP as optional import/reference steps when high-quality reference geometry exists. Do not pretend a procedurally plausible tower is an authenticated landmark replica.
5. **Character rigs:** use Blender's bundled Rigify where possible; consider Rokoko only if character motion is a requirement for the scene.

## Licensing and export boundaries

- **CC0 PBR media** can be used broadly in commercial projects; preserve provenance anyway. Asset license and API/software license are not the same thing.
- **GPL add-ons** may be run for commercial creation, but copying their code into an MIT library raises distinct redistribution and license-compliance obligations. Keep them optional unless resolved.
- **GLB optimizers** may merge/flatten objects, remove extras or change normals. Treat named semantic nodes, scale, materials and preview visual parity as regression invariants.
- **Third-party photos/landmark references** have separate rights. The fact a procedural model or renderer is open-source does not clear depicted trademarks, private property, or an imported asset's redistribution terms.

No tool listed here is a dependency of `kumiki-blender` yet. This is a researched, prioritized integration map, not an assertion that all extensions have been run with our version of Blender.
