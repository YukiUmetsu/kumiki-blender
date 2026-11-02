# Next steps: after your first tower

[← Home](https://github.com/YukiUmetsu/kumiki-blender/wiki)

Once you have built, opened and customized a tower, here is a useful learning order. You do not need to learn everything at once.

## Level 1: build more existing models

1. Browse the [model catalog](https://github.com/YukiUmetsu/kumiki-blender/wiki/Explore-the-Model-Catalog).
2. Try an architectural object like the stone facade or portico. Compare the front and detail pictures.
3. Try the red biplane, which has rigged movable parts.
4. Read [files and formats](https://github.com/YukiUmetsu/kumiki-blender/wiki/Understand-Files-and-Folders) so you know what to archive.

## Level 2: learn basic Blender editing

The [official Blender manual](https://docs.blender.org/manual/en/latest/) explains the Viewport, selecting objects, transforming, materials, camera and rendering. Suggested order:

1. Select, move, rotate and scale an object.
2. Change a material color and roughness.
3. Move the camera and a light, then render one image.
4. Save a copy of the `.blend` file and export a `.glb`.

These edits are **manual Blender changes**. A later code-generated revision will not automatically include them.

## Level 3: build a new reusable object

For code-driven assemblies, read [AssetSpec v2](https://github.com/YukiUmetsu/kumiki-blender/blob/main/docs/ASSET_SPEC_V2.md). It describes structures assembled from reusable geometry, with optional Python packages:

```bash
python -m pip install -e '.[composer,qa]'
python -m kumiki_blender plan-asset --spec examples/desk-lamp-assembly-v2.json
```

The example is a desk-lamp assembly. On Windows, replace `python` with `.\.venv\Scripts\python.exe` in both commands. Building and rendering an AssetSpec requires Blender and an output revision; see that technical guide for exact options. **Not every artistic shape can be described with a few declarative parts.**

## Level 4: improve realism

Learn why a technically valid model may still look artificial:

- **Shape:** silhouette, dimensions and proportions must match references.
- **Surface:** look at fine carving, seams, wear, scale-appropriate roughness and texture.
- **Lighting:** judge material response under multiple angles, not only one flattering hero view.
- **Comparison:** use front, three-quarter and side reference views, then inspect the GLB, not just the render.

Advanced references in this repository:

| Goal | Technical guide |
| --- | --- |
| Better stone, metal and wood | [Material packs](https://github.com/YukiUmetsu/kumiki-blender/blob/main/docs/MATERIAL_PACKS.md) |
| Bring in a licensed GLB | [Import workflow in README](https://github.com/YukiUmetsu/kumiki-blender#readme) |
| More sophisticated assemblies | [AssetSpec v2](https://github.com/YukiUmetsu/kumiki-blender/blob/main/docs/ASSET_SPEC_V2.md) |
| Compare many renders | [Visual QA in AssetSpec guide](https://github.com/YukiUmetsu/kumiki-blender/blob/main/docs/ASSET_SPEC_V2.md) |
| Reconstruct from multiple photos | [Reference reconstruction](https://github.com/YukiUmetsu/kumiki-blender/blob/main/docs/REFERENCE_RECONSTRUCTION.md) |
| Start a statue face from a head model | [Anatomical head bases](https://github.com/YukiUmetsu/kumiki-blender/blob/main/docs/ANATOMICAL_HEAD_BASES.md) |

## Level 5: use assets in a video

You can import `model.glb` into a compatible 3D application or web renderer. In this project ecosystem, [Kumiki-UI](https://github.com/YukiUmetsu/kumiki-ui) handles Three.js scenes and motion; [ContentGen](https://github.com/YukiUmetsu/ai-content-generation) coordinates broader content production. Those are separate systems, each with its own setup.

## A good quality habit

Keep a small folder of attributed reference pictures, `preview.png` outputs from each revision, and notes on what visibly improved or remains wrong. A model is **finished** when it serves the desired shot and survives close visual review, not merely when a command succeeds.
