# Understand files and folders

[← Home](https://github.com/YukiUmetsu/kumiki-blender/wiki)

When you download the repository, you are downloading **instructions and tools**, not a giant collection of ready-made finished models.

## The folders you will see

| Path | Plain-English meaning |
| --- | --- |
| `README.md` | Project overview and technical examples |
| `examples/` | Example 3D recipes and input manifests |
| `src/kumiki_blender/` | Python code that describes and builds model geometry |
| `src/kumiki_blender/assets/` | Searchable catalog entries for model families |
| `scripts/blender_entry.py` | Entry point that runs the build inside Blender |
| `docs/` | Deeper technical guides and design decisions |
| `tests/` | Checks used by developers to catch bugs |
| `.review/` | **Generated** files from local builds; normally hidden and not tracked in Git |
| `.venv/` | Your local Python environment if you followed this wiki |

## Five file types to recognize

| Extension | What it contains | Typical app |
| --- | --- | --- |
| `.json` | Text-based instructions, settings or build information | Text editor |
| `.py` | Python code | Text editor (usually for developers) |
| `.blend` | Editable 3D scene with objects, cameras and lights | Blender |
| `.glb` | Portable 3D model, often with embedded textures and animation | Blender, Three.js, compatible 3D viewers |
| `.png` | A flat preview picture of the object | Photos, Preview, image viewer |

### Why are `.blend` and `.glb` different?

`source.blend` is your **workshop**: it can include the object, studio floor, camera, lights and editable Blender settings. `model.glb` is the **packaged object** to load in other software. A GLB usually does not look exactly like the original render unless the destination provides appropriate lighting and environment.

### What does `manifest.json` do?

It records the files and measurements of a particular build, plus integrity and provenance information. Use `audit-revision` to check those outputs are consistent. The manifest is **not** a second 3D model.

## Revisions protect your work

By default, you build to a folder ending in `r01`, `r02`, `r03`, etc. This lets you compare earlier and later versions. Do not point the builder at an existing folder; it refuses to overwrite it. **A failed build may still have created a partial folder.**

`.review` is intended for inspection and can be lost when you clean your development workspace. Keep final approved assets somewhere permanent. For catalog builds, `--root /path/to/permanent-model-library` selects an external asset library. See the [catalog storage guide](https://github.com/YukiUmetsu/kumiki-blender/blob/main/docs/ASSET_CATALOG.md).

## Where does this fit in the larger video workflow?

- **Kumiki Blender** makes and exports 3D models.
- [Kumiki-UI](https://github.com/YukiUmetsu/kumiki-ui) can use models in interactive Three.js video scenes and motion graphics.
- [ContentGen](https://github.com/YukiUmetsu/ai-content-generation) organizes wider media-production workflows.

They are separate projects. You do **not** need either of the other two repositories for your first model.
