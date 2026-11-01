# Asset catalog: organizing hundreds of future 3D models

The goal is **one place to find a model**, even when it fits more than one
subject. The registry stores lightweight metadata in source control; it does
not duplicate meshes or force an invasive rename of existing Python modules.

## The two-layer structure

1. **Source/catalog** (tracked in git): one `asset.json` per reusable model
   *family*. Its directory is `src/kumiki_blender/assets/<category>/<subcategory>/<id>/`.
   The manifest points to generator/adaptor modules, docs, search tags,
   aliases, and the existing CLI build command. No hardcoded master index is
   maintained: the catalog discovers each manifest on disk.
2. **Built models** (separate storage): default
   `.review/<category>/<subcategory>/<id>/r01/`, `r02/`, etc. Each revision
   contains `source.blend`, `model.glb`, `manifest.json` and previews. Use
   `--root /path/to/persistent-model-library` for archives you must keep;
   `.review` is for throwaway QA and is gitignored.

Example source hierarchy:

```text
src/kumiki_blender/assets/
├── architecture/
│   ├── buildings/stone-facade/asset.json
│   └── components/{classical-column,modular-portico}/asset.json
├── infrastructure/towers/lattice-observation-tower/asset.json
├── landmarks/monuments/
│   ├── giza-pyramid-study/asset.json
│   ├── giza-sphinx-study/asset.json
│   └── giza-ensemble-study/asset.json
└── vehicles/aircraft/red-vintage-biplane/asset.json
```

## Category roadmap

| Primary category | Useful subcategories | Examples of future work |
|---|---|---|
| `architecture` | `buildings`, `components`, `interiors` | Homes, temples, doors, columns, staircases |
| `infrastructure` | `towers`, `bridges`, `transport` | Bridges, roads, rail stations, power pylons |
| `landmarks` | `monuments`, `statues`, `historic-sites` | Sphinx, Statue of Liberty, castles |
| `geography` | `maps`, `terrain`, `cities` | 3D Egypt maps, cities, topographic globes |
| `vehicles` | `aircraft`, `road`, `rail`, `watercraft` | Biplanes, cars, trains, ships |
| `everyday` | `furniture`, `electronics`, `tools`, `kitchen` | Desks, chairs, lamps, books, laptops |
| `nature` | `plants`, `trees`, `rocks`, `water`, `animals` | Pines, flowers, cliffs, waterfalls |
| `characters` | `people`, `creatures`, `costumes` | Historical figures, stylized characters |
| `environments` | `interiors`, `exteriors`, `sets` | Rooms, forests, streets, studios |
| `science` | `anatomy`, `space`, `mechanisms` | Solar systems, engines, scientific diagrams |
| `visual-effects` | `particles`, `volumetrics`, `motion-kits` | Smoke, dust, cloud effects, reusable effects |

These are conventions, **not a frozen allowlist**. Add a new lowercase-slug
category or subcategory when useful; keep depth at two grouping levels before
the stable asset ID. Do not create a folder for every possible tag.

An item has **one primary home**; secondary uses belong in `tags` and
`aliases`. For example, the red biplane stays in `vehicles/aircraft` even if
also used in a history documentary or a children's video. Search supports all
those contexts without symlinks or duplicate assets.

## Discovery and building

```bash
PYTHONPATH=src python3 -m kumiki_blender catalog list
PYTHONPATH=src python3 -m kumiki_blender catalog categories
PYTHONPATH=src python3 -m kumiki_blender catalog search Giza
PYTHONPATH=src python3 -m kumiki_blender catalog search old plane
PYTHONPATH=src python3 -m kumiki_blender catalog list --tag stone
PYTHONPATH=src python3 -m kumiki_blender catalog list --rigged
PYTHONPATH=src python3 -m kumiki_blender catalog show red-vintage-biplane
PYTHONPATH=src python3 -m kumiki_blender catalog path red-vintage-biplane --revision r02
PYTHONPATH=src python3 -m kumiki_blender catalog validate
```

Run the build from the repository checkout with Blender's Python:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  catalog build red-vintage-biplane --revision r01
```

For a permanent catalog of finished, approved models:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  catalog build red-vintage-biplane --revision r02 \
  --root /path/to/permanent-model-library
```

The catalog delegates to the **existing** `build-biplane`, `build-monument`,
`build-facade`, `build-portico`, or recipe-based `build` pipeline. It will not
overwrite an existing revision, and adding a category does not change the
geometry engine. A component without a standalone build command is marked
`"build": null` and exposes its plan API instead.

## Adding a new model

1. Choose one category/subcategory and a stable lowercase hyphenated ID.
2. Create `src/kumiki_blender/assets/<category>/<subcategory>/<id>/asset.json`.
3. Declare `schema_version: 1`, `id`, `title`, `description`, `kind`, `tags`,
   `aliases`, `rigged`, `animated`, `sources`, `plan`, and `build`.
4. Point `sources` to real paths in the repository; keep implementation code
   in the existing geometry and Blender-adapter APIs. New model code may use
   category subpackages when they become substantial, with compatibility
   imports for public APIs.
5. Run `catalog validate` and the Python test suite. Build a new revision and
   inspect its silhouettes, materials, rig and GLB before approving it.

The catalog indexes **model families**, not every parametric permutation.
For example, `giza-pyramid-study` can produce illustrative and measured-size
variants; `stone-facade` can produce different bay counts. Use a descriptive
output revision and the build manifest to preserve those exact parameters.

The metadata is included in the installable Python wheel. Files used only for
Blender rendering and large binary previews are **not** included in the wheel.
Avoid committing exported `.blend`/GLB collections directly into the code
repository; use a versioned persistent asset store or object storage instead.
Geometry modules, Blender adapters, shaders and PBR material packs remain
shared technical infrastructure rather than duplicated into each asset folder.

## Limits and compatibility

- Existing `build-*`, `plan-*`, `import-asset` commands and historic `.review`
  locations are unchanged. Nothing is deleted or migrated automatically.
- Search is text/tag-based over checked-in model metadata, not mesh similarity
  or photo-based retrieval.
- The `catalog build` command uses the family's default parameters. For
  non-default parameters, use its existing `build-*` command with your chosen
  output path; catalog metadata points to the implementation and docs.
- The `lattice-observation-tower` recipe resides in `examples/`; its catalog
  build requires a source checkout with that example available.
