# Kumiki Blender

Local-first tools for **Blender asset creation, reference-guided character modeling, photographic 3D reconstruction, and animation export**. The repository combines Python geometry/validation code with native Blender workers, reusable asset recipes, and source-aware quality checks.

This is a **toolkit, not a one-click photorealistic model generator**. Automated steps produce inspectable candidates; source likeness, anatomical accuracy, materials, and animation still need visual review.

## Get started

Run commands from the repository root. The core Python package requires **Python 3.11+**. Use **Blender 4.5 LTS** for the most consistently tested native workers; individual workflows may support other versions.

### Validate a procedural asset without Blender

```bash
python -m pip install -e .
python -m kumiki_blender validate --recipe examples/lattice-observation-tower.json
```

Validation checks the recipe and reports its structural data. It does not render or create a 3D model.

### Build an editable 3D asset

```bash
blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- build --recipe examples/lattice-observation-tower.json --output .review/lattice-observation-tower/r01
```

This procedural example generates a Blender scene, GLB, previews, and a revision manifest. Choose an **unused output directory** for each run. For fast inspection renders, prefix the command with `KUMIKI_RENDER_PROFILE=smoke`; smoke previews are not production-quality evidence.

### Start from character reference images

```bash
python -m pip install -e '.[character-build]'
kumiki-character-build check /path/to/character-references
kumiki-character-build handoff /path/to/character-references --height-m 0.35
```

The reference directory must contain the expected named views. The handoff prepares registered images, masks, and an initial blockout; it **does not** by itself produce a finished, rigged character. Follow the [character build guide](docs/CHARACTER_BUILD.md) for view naming, mesh fitting, textures, retopology, and rigging.

## Command-line interfaces

| Entry point | Use |
| --- | --- |
| `kumiki-blender` | Procedural assets, catalog, GLB audits, photography-based reconstruction, physics, and export |
| `kumiki-character-build` | Character-reference preparation, geometry, textures, anatomy, rigging, and source regression checks |
| `kumiki-character-qa` | Independent character-part, silhouette, appearance, and depth review |
| `scripts/blender_entry.py` | Run `kumiki-blender` subcommands inside Blender's Python environment |

Each interface exposes its currently supported arguments through `--help` (and `<command> --help` for a subcommand). The root CLI and the character CLI are different tools; their subcommands are not interchangeable.

## Character modeling and animation

Character work is organized around **registered source images and separately editable parts**. The following capabilities are implemented; none implies automatic approval of a finished sculpt.

| Stage | Available commands | Documentation |
| --- | --- | --- |
| Reference preparation and silhouette fitting | `check`, `handoff`, `compare`, `fit` | [Character build](docs/CHARACTER_BUILD.md) |
| CharacterSpec and isolated part revisions | `plan-character`, `plan-region`, `compare-region`, `audit-region-revision` | [Production v2](docs/CHARACTER_PRODUCTION_V2.md), [source revision gate](docs/CHARACTER_REGION_REVISION_GATE.md) |
| Registered landmarks and editable anatomical parts | `triangulate-landmarks`, `resolve-anatomy-landmarks`, `plan-anatomy`, `build-anatomy`, `fit-landmark-surface` | [Anatomy authoring](docs/CHARACTER_ANATOMY_AUTHORING.md), [geodesic landmark fitting](docs/CHARACTER_GEODESIC_LANDMARK_FIT.md) |
| Surface and joint corrections | `refine-region`, `refine-surface`, `refine-joint-topology`, `repair-penetrations` | [Surface refinement](docs/CHARACTER_SURFACE_REFINEMENT.md), [hinge topology](docs/CHARACTER_HINGE_TOPOLOGY.md), [conflict repair](docs/CHARACTER_CONFLICT_REPAIR.md) |
| UV textures and part-aware baking | `texture`, `parts`, `texture-parts`, `resolve-conflicts`, `audit-texture-revision` | [Character build](docs/CHARACTER_BUILD.md), [texture revision gate](docs/CHARACTER_TEXTURE_REVISION_GATE.md) |
| Rig and deformation validation | `plan-rig-controls`, `apply-rig-controls`, `validate-rig`, `audit-topology` | [Production v2](docs/CHARACTER_PRODUCTION_V2.md), [animation baking](docs/CHARACTER_ANIMATION_BAKE.md) |
| Facial animation and performance export | `plan-visemes`, `animate-visemes`, `plan-performance`, `export-performance`, `export-talking-performance` | [Visemes](docs/CHARACTER_VISEME_ANIMATION.md), [performance export](docs/CHARACTER_PERFORMANCE_EXPORT.md) |

Anatomical features are authored as actual Blender geometry, not merely simulated with smooth shading. Feature-preserving surface fairing operates on eligible **pre-rig, closed meshes** and can reject excess deformation or source-silhouette regressions. It is not adaptive remeshing or a replacement for artist-directed topology.

For independent review of a character sheet against an existing Blender model, install `.[character-qa]` and consult [Character Fidelity Review](docs/CHARACTER_FIDELITY_REVIEW.md). Optional local SAM 2.1 and MoGe checkpoints can assist mask and depth analysis; their model weights are **not** included in the core package.

### Incremental region planning and release evidence

When a single part changes, `plan-region-batch` records the affected region dependencies and a bounded execution schedule; `resume-region-batch` checks SHA-256 receipts before reporting which tasks are eligible to run. **These commands plan and audit work; they do not launch Blender workers or certify completed parts.** See [incremental region planning](docs/CHARACTER_REGION_BATCH.md).

The `character-benchmark` command collects existing source-registered region comparisons, texture checks, saved Blender rig/pose reports, and an animated GLB into one provenance-checked release-evidence report. Missing evidence or failed numeric checks remain visible; a numeric pass still requires artistic and animation review. See [character benchmark](docs/CHARACTER_M0_BENCHMARK.md) and the [manifest example](examples/character-benchmark.example.json).

## Photographs to 3D

There are two distinct reconstruction outputs:

| Workflow | Output and limitation | Guide |
| --- | --- | --- |
| `plan-photo-model` / `build-photo-model`, or the more guided `build-statue-glb` | Camera-registered **polygon mesh** candidates and Blender/GLB exports when sparse registration and dense meshing succeed; dependent on suitable overlapping images and local external tools | [Photo to GLB](docs/PHOTO_TO_GLB.md), [dense reconstruction](docs/DENSE_STATUE_RECONSTRUCTION.md) |
| `plan-splats` / `build-splats` | A view-dependent **Gaussian splat**, useful for visual reconstruction but **not** an editable, riggable polygon mesh | [Gaussian splats](docs/GAUSSIAN_SPLATS.md) |

For source selection, image masks, camera coverage, and non-destructive scan finishing, see [reference reconstruction](docs/REFERENCE_RECONSTRUCTION.md). Optional [evidence-guided completion](docs/EVIDENCE_GUIDED_COMPLETION.md) produces inferred candidates separately from verified reconstruction. Neither approach guarantees a faithful likeness from incomplete photos.

Photogrammetry uses optional Python dependencies (for example `.[photogrammetry]`) and may require locally installed **COLMAP, OpenMVS, OpenSplat, Blender, or AI checkpoints**, depending on the path selected. Consult the relevant workflow guide rather than assuming an optional extra installs all native executables.

## Procedural assets, catalog, and motion

| Area | Capabilities | Guide |
| --- | --- | --- |
| Reusable asset families | Search, validate, and build catalog entries with `kumiki-blender catalog` | [Asset catalog](docs/ASSET_CATALOG.md) |
| Declarative assemblies | Validate/compile AssetSpec v2, then build editable Blender assemblies | [AssetSpec v2](docs/ASSET_SPEC_V2.md) |
| Architecture and monuments | Parametric building details, facades, porticos, terrain studies, Egyptian monument studies | [Architecture details](docs/ARCHITECTURE_DETAILS.md), [Egyptian monuments](docs/EGYPTIAN_MONUMENTS.md), [Egypt relief map](docs/EGYPT_RELIEF_MAP.md) |
| Blender animation bridges | Export baked rigid-body motion and topology-stable deformations for downstream viewers | [Motion bridge](docs/KUMIKI_MOTION_BRIDGE.md), [deformation bridge](docs/KUMIKI_DEFORMATION_BRIDGE.md) |
| Imported models and materials | Inspect GLBs, record source/provenance, apply explicit material overrides, and produce reviewable revisions | [Material packs](docs/MATERIAL_PACKS.md), [asset integrity](docs/ARCHITECTURE.md) |

Browse available assets:

```bash
kumiki-blender catalog list
kumiki-blender catalog validate
```

## Outputs and quality gates

A standard procedural build commonly produces `source.blend`, `model.glb`, `preview.png`, additional views, and `manifest.json`. Other workflows use their own documented names and candidate/report layouts.

- **Revisions:** Most build and QA commands require a new output path rather than overwriting approved inputs. The default `.review/` location is intended for local, gitignored revisions; use durable storage for assets you must retain.
- **Integrity:** Run `kumiki-blender audit --model /path/to/model.glb` for structural GLB checks, and `kumiki-blender audit-revision --root /path/to/revision` for supported revision packages.
- **Reference fidelity:** Registered camera masks, source hashes, contour/texture metrics, and part-level overlays can reveal regressions. A passing numeric gate **does not establish likeness, physical correctness, or animation readiness**.
- **Licensing:** Source photographs, model weights, imported meshes, and texture packs retain their own rights and provenance. An asset's presence in a revision does not authorize redistribution.

## Development and testing

```bash
python -m compileall -q src scripts
PYTHONPATH=src python -m unittest discover -s tests -v
```

Install the optional dependencies needed by the test modules you run. The complete suite includes Blender-specific and optional-tool tests; Python-only results cannot substitute for headless Blender or visual inspection.

GitHub Actions runs **focused pull-request checks** and native Blender smoke tests. The broader historical regression is available through the `Python foundation` workflow's manual `workflow_dispatch` run. CI smoke previews are diagnostic artifacts, not artistic acceptance renders.

For internal design and module organization, see [Architecture](docs/ARCHITECTURE.md). The repo's build/export outputs can be consumed by [Kumiki-UI](https://github.com/YukiUmetsu/kumiki-ui) and [ContentGen](https://github.com/YukiUmetsu/ai-content-generation); those projects are not runtime dependencies of this Python package.

## License

The **code** is MIT-licensed (see [LICENSE](LICENSE)). Third-party models, images, pretrained weights, and other content are subject to their respective licenses. Verify source rights and attribution before exporting or distributing derived assets.
