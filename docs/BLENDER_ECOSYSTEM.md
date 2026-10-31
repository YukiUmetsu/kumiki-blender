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

glTF-Transform 4.5.1 is pinned as a **CI-only** validator. None of the other projects are runtime dependencies of `kumiki-blender` yet. This is a researched integration map, not an assertion that every extension has been run with our version of Blender.

## Model and texture marketplaces (selection, not automatic ingestion)

| Source | Best use | Reusable asset rights and pipeline decision |
|---|---|---|
| [Sketchfab downloadable models](https://sketchfab.com/3d-models) | Real landmarks, individual sculptures, photogrammetry and public-domain/CC releases | Strong first stop for an **exact named building**, but downloadable models have individual Creative Commons conditions. The [Download API guidelines](https://sketchfab.com/developers/download-api/guidelines) require visible attribution/license information. Capture each model's author and license; no unreviewed mass downloads. |
| [CGTrader](https://www.cgtrader.com/) | High-detail commercial landmark meshes with editable files | Useful as an **offline artist reference or a licensed cinematic-shot dependency**. Its [royalty-free license](https://www.cgtrader.com/pages/terms-and-conditions) generally forbids redistributing the purchased model by itself, so do not publish raw files as open GitHub artifacts. |
| [Fab](https://www.fab.com/) | Environment/architecture packs and sometimes GLB/Blender landmark models | [Fab Standard License](https://www.fab.com/eula) permits commercial use in larger projects but forbids standalone asset redistribution. This may suit rendered videos, not a freely downloadable source GLB. |
| [BlenderKit](https://www.blenderkit.com/) | Direct Blender browsing of reusable doors, columns, windows, props, materials and node groups | A practical **authoring environment**, with free and paid assets. Verify specific asset license and any source-file distribution rights before copying into the toolkit. |
| [Open Heritage 3D](https://www.openheritage3d.org/) | Measured LiDAR/photogrammetry reference geometry for historic sites | Dataset rights are **per-project**, often with CC terms including NC; datasets can be very large and may need a reconstruction/retopology process before glTF export. Preserve DOI and source record. |
| [ambientCG](https://ambientcg.com/) | CC0 surface maps where a procedural shader is insufficient | A useful **optional** texture catalog, not a model marketplace; only import selected exact assets with checksums and known mapping scale. |
| [Poliigon](https://www.poliigon.com/) / [KitBash3D](https://kitbash3d.com/) | Commercial film-quality textures or architectural environment kits | More suitable for render-only productions than an open raw-asset pipeline. [Poliigon specifically prohibits distribution of embedded models, baked textures, and accessible scene files](https://help.poliigon.com/en/articles/8749749-asset-use-licensing); KitBash3D also uses [license-tier and asset-use restrictions](https://kitbash3d.com/pages/licenses). |

**Design rule (implemented for static GLBs):** import a *specific reviewed local file* with explicit author, original source, license and checksum. Do not build an automated marketplace crawler or treat a purchase/render license as permission to publish `.blend` or `.glb` source. The `inspect-asset` / `import-asset` commands and local provenance JSON are source-agnostic; individual services are replaceable discovery surfaces rather than required infrastructure. Raw redistribution rights are a required operator assertion, not an inferred marketplace guarantee. Rigs, FBX/OBJ intake and a render-only license mode are still future work.
