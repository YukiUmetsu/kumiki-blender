# Explore the model catalog

[← Home](https://github.com/YukiUmetsu/kumiki-blender/wiki)

The **catalog** is a searchable list of model families already known to this project. Think of it like a library: instead of writing new modeling code, start by seeing what already exists.

## See what's available

In the repository folder, run:

**macOS / Linux**

```bash
python -m kumiki_blender catalog list
python -m kumiki_blender catalog search plane
python -m kumiki_blender catalog show red-vintage-biplane
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m kumiki_blender catalog list
.\.venv\Scripts\python.exe -m kumiki_blender catalog search plane
.\.venv\Scripts\python.exe -m kumiki_blender catalog show red-vintage-biplane
```

There are currently **12 cataloged model families** in the inspected repository revision. Examples include:

| Family name | What it represents | Beginner note |
| --- | --- | --- |
| `lattice-observation-tower` | Steel observation tower | Best first exercise |
| `red-vintage-biplane` | Red propeller airplane | Includes movable/animated parts |
| `stone-facade` | Detailed stone building face | Doors, windows and roof |
| `giza-pyramid-study` | Egyptian pyramid | An illustrative/research model, not a full survey |
| `giza-sphinx-study` | Sphinx sculpture study | Not a photorealistic historical scan |
| `mount-fuji-iconic-diorama` | Stylized Fuji, trees and torii | Artistic proportions, not measured geography |
| `mount-fuji-survey-terrain` | Terrain-based mountain | Different goal from the iconic diorama |
| `egypt-relief-atlas` | Physical tabletop map | Terrain, river and miniature landmarks |
| `makehuman-neutral-head` | Neutral human head base | Advanced, uses separately sourced assets |

The list can grow. Use `catalog list` for the authoritative current list, and `catalog categories` to explore topics.

## Find a model's instructions

```bash
python -m kumiki_blender catalog show lattice-observation-tower
```

For Windows, prefix this command as shown above with `.\.venv\Scripts\python.exe` instead of `python`.

`show` explains which source files and build command are associated with the model. For example, the observation tower uses `examples/lattice-observation-tower.json` as its recipe.

## Build an existing model by its name

For example, the airplane can be built through the catalog. The basic shape of the command is:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  catalog build red-vintage-biplane --revision r01
```

This is the **Linux / Blender-on-PATH form**. On macOS, substitute `/Applications/Blender.app/Contents/MacOS/Blender` for `blender`. On Windows PowerShell, use the `& $blender ...` single-line style from [Make Your First 3D Model](https://github.com/YukiUmetsu/kumiki-blender/wiki/Make-Your-First-3D-Model#step-3-build-the-model), replacing everything after `--` with `catalog build red-vintage-biplane --revision r01`.

The catalog writes revisions under a category-based path such as `.review/vehicles/aircraft/red-vintage-biplane/r01`. Existing revisions are not overwritten. **Some advanced catalog entries need optional data, downloaded assets or special setup.** Read their linked technical documentation before attempting them.

## Where these catalog entries live

The names and descriptions are in `src/kumiki_blender/assets/<category>/<subcategory>/<asset-id>/asset.json`. These are **metadata files**, not the finished 3D models. The actual geometry is created when you build.

For the full organization rules, see the [technical catalog documentation](https://github.com/YukiUmetsu/kumiki-blender/blob/main/docs/ASSET_CATALOG.md).
