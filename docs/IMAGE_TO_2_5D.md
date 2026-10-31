# Image-to-2.5D: Depth Anything V2 integration decision

## Decision

**Use Depth Anything V2 Small as an optional depth estimator for photo-derived 2.5D scenes.** Do not make it a dependency of the Blender architectural modeling kernel. Do not claim that a single predicted depth map is a reconstructed, metrically accurate 3D landmark.

Official sources:

- Code and license distinctions: https://github.com/DepthAnything/Depth-Anything-V2
- Metric variants: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
- Video-consistent option: https://github.com/DepthAnything/Video-Depth-Anything

The project's **Small checkpoint is Apache-2.0**; the Base, Large and Giant weights are **CC BY-NC 4.0**. The maintainers confirmed that the source code itself is Apache-2.0 and the noncommercial restriction applies to the larger model weights: https://github.com/DepthAnything/Depth-Anything-V2/issues/81. Treat metric checkpoints as independent artifacts requiring their own license checks before commercial use. Never automatically download a noncommercial weight merely because the source code is permissive.

## Where this belongs

| Task | Module / consumer | Reason |
|---|---|---|
| Estimate depth | Optional local AI preparation worker (ContentGen) | PyTorch/OpenCV and model weights are heavy; allow Apple MPS inference on Mac |
| Preserve original image, rights, depth map, masks | ContentGen asset preparation | Source traceability and commercial licensing |
| Separate foreground/background, refine edges, inpaint occluded holes | Image-preparation stage | Avoid stretched silhouettes and black gaps during camera moves |
| Animate camera through 2.5D layers or textured relief mesh | Kumiki-UI/Three.js | Efficient parallax, responsive camera paths, reusable scene recipes |
| Offline physically lit scene or composite | Blender | Only when a 3D plate, shadow catcher, or Cycles render is advantageous |
| Reference-verified arches, buildings and monuments | `kumiki-blender` architectural geometry | Depth prediction is not measured or back-face architectural geometry |

## Recommended asset contract (planned, not implemented)

```json
{
  "schema_version": 1,
  "asset_type": "photo-depth-2.5d",
  "source_image": "source.png",
  "source_license": "REQUIRED: original source and rights",
  "depth_model": "depth-anything-v2-small",
  "depth_model_license": "Apache-2.0",
  "depth_map": "depth16.png",
  "depth_semantics": "relative; conversion direction explicitly recorded",
  "foreground_masks": ["mask-subject.png"],
  "inpainted_background": "background-filled.png",
  "camera": { "fov_degrees": 45, "maximum_parallax_degrees": 6 },
  "review_status": "pending"
}
```

The contract above is illustrative. The production implementation must validate filenames, dimensions, actual rights and model identifiers. Store a lossless high-bit-depth or floating-point depth image, never a pretty colormap as the only raw depth source.

## Quality gates

1. Normalize predicted depth *with an explicit near/far convention*. The default relative-depth map is not a reliable real-world distance field; don't label its output “meters.”
2. Segment independent subjects where helpful. Otherwise a single displaced photo behaves like rubber at depth boundaries.
3. Expand/inpaint newly exposed background and check foreground edges at the maximum scripted camera movement.
4. Clamp camera parallax conservatively; one photograph has no correct information about the backs of buildings or objects.
5. Inspect the exported scene at hero and edge-of-motion viewpoints; reject stretched textures, visible empty gaps and impossible geometry.
6. For video input, investigate the creators' *Video Depth Anything* temporal model instead of naïvely running single-frame estimation on every frame.
7. Preserve source-image licensing and depth model/weight licensing in the final media manifest.

## Implementation sequence

1. Optional, pinned local MPS-enabled depth inference adapter using the Apache-2.0 Small checkpoint; no heavyweight base dependency.
2. Pure-Python normalized depth-map validation and camera-calibrated grid/segmented-layer geometry contracts.
3. Image edge cleanup, masks and inpainting, plus a fixed motion-safety limit.
4. GLB or image/layer manifest export and a Kumiki-UI scene recipe consuming it.
5. ContentGen job orchestration, retry/provenance and visual regression videos comparing still, start, midpoint and end frames.

This is an **approved direction**, not a claim that depth inference or the 2.5D renderer already exists in this repository.
