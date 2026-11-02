# Licensed statue scans: high-resolution sculpt to portable LOD

## Why this stage exists

A polygon-decimated face will lose small wrinkles, eyelids, stone pitting,
hair grooves and sharp drapery creases. Normal, albedo and roughness texture
baking transfers much of the **existing** high-poly surface appearance to a
smaller GLB. The original licensed scan stays editable in a `.blend` for later
artistic sculpting. Baking cannot invent missing geometry, repair wrong facial
proportions or turn an unrelated 3D model into a historical statue.

The worker consumes an **existing rights-cleared, static GLB** (from a scan or
artist sculpt). It does not download 3D models or automatically manufacture a
face from internet photos. Reference-only images and unauthorized scan sources
must **not** be baked into redistributed assets.

## Command line

Make a provenance record using the same schema as
[`examples/first-party-import.json`](../examples/first-party-import.json),
with the actual creator, source permalink, license, required attribution,
verified `redistribution_allowed: true`, and ideally `expected_sha256`.

```bash
# Safe host-Python preflight; no Blender or output writes.
kumiki-blender plan-scan-lods \
  --source /private/scan/high-resolution.glb \
  --provenance /private/scan/provenance.json \
  --target-triangles 120000 \
  --map-size 2048 --ray-distance-m 0.05 \
  --output .review/statue/lod-r01

# Runs Blender 4.2+ in the background if not already inside Blender.
kumiki-blender build-scan-lods \
  --source /private/scan/high-resolution.glb \
  --provenance /private/scan/provenance.json \
  --target-triangles 120000 \
  --map-size 2048 --ray-distance-m 0.05 \
  --output .review/statue/lod-r01 --render-previews
```

The builder:

1. Checks source GLB integrity, static geometry and explicit redistribution
   rights. Fails before creating a revision if the source is unsupported.
2. Imports and retains original meshes/materials in the editable `.blend`.
   Each of 1–12 source mesh parts receives an aligned, decimated low-poly copy.
   Separate the statue's **head** from the body if you need a dedicated 2K/4K
   detail atlas for eyes, mouth, ears and hair.
3. Re-unwraps low-poly UVs into non-overlapping islands. On each part, Cycles
   performs **selected-to-active** ray-based normal, diffuse-color and
   roughness bakes from its original high-resolution surface/materials.
4. Connects the baked textures to a glTF-compatible Principled BSDF and
   exports only the lower-poly objects as `model-lod.glb`. Checks the output
   has embedded normal, albedo, roughness, and UV0 texture paths.
5. Optionally renders `source-high.png` and `baked-low.png` from one exact
   shared camera for visual inspection. Records face counts and the reduction
   fraction in `lod-report.json`.

The output also contains `scan-lod.blend` (editable high+low meshes and packed
maps) and `maps/lod-XX-{normal,color,roughness}.png`. The `.blend` and
source scan remain subject to the original source license. A successful
normal bake is **not** evidence of close-up anatomical accuracy. Inspect
seams, projective bake artifacts, sharp silhouettes and facial microdetail
manually before promotion into videos.

## Practical limits and quality expectations

- Use a sufficiently clean scan. Extremely noisy or non-manifold scans need
  manual cleanup and authentic provenance review before baking.
- Choose a ray distance that encloses real decimation deviations but avoids
  projecting normals from the *other side* of thin ears, hair, drapery and
  folds. The configurable distance is in metres, 0.0001–2.0.
- `target_triangles` is an approximate budget. The Decimate modifier preserves
  parts with fewer polygons and may not hit the number exactly.
- Large scans have 1–12 parts, up to 10M source faces and up to 96M aggregate
  texture texels; map sizes are 256, 512, 1024, 2048 or 4096 px. Prepare
  separate regions for larger projects rather than exhausting RAM.
- This first pass bakes **base color, roughness and tangent-space normals**.
  It does not create displacement maps, adaptive topology, multi-resolution
  subdivision, skin microgeometry, color-managed photometric matching or
  automatic manual-sculpt corrections. Texture accuracy still depends on
  the source scan's original lighting and material quality.
- `--render-previews` makes quick comparison frames, not independent
  historical-likeness acceptance tests. Also run the calibrated cameras,
  silhouettes and facial/drapery landmarks described in
  [reference reconstruction](REFERENCE_RECONSTRUCTION.md).
