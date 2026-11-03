# Blender ↔ Kumiki silhouette parity (manual QA)

This bridge checks whether a Blender physics asset projects onto **the same pixels** after import into Kumiki's Three.js renderer. It is a geometry/camera/pose check, **not** a beauty score: PBR, depth, lighting, shadows, textures, artistic fidelity, and audible impact quality still require separate review.

The camera and frame list live in one JSON plan (in `kumiki-ui/src/craft/fixtures/blender-rotational-impacts/visual-parity-plan.json`). Its SHA-256, the GLB SHA-256, dimensions, ordered frames, and renderer identity are verified in each capture receipt. Neither renderer can silently substitute a different asset/camera.

The native render also verifies the loaded `source.blend` belongs to a **kumiki-blender physics revision**: adjacent `manifest.json` must identify the same asset/action, cover every requested frame, match recorded source/model sizes, and contain a sibling GLB byte-identical to Kumiki's imported model. `source_blend_sha256` and `revision_manifest_sha256` are recorded in the native receipt. Rendering into that immutable revision directory is rejected.

## One-time example: real Bullet swinging rod

From the `kumiki-blender` repository, with Blender installed (replace `blender` with your executable path):

```bash
# The output directory must not exist. This simulates the real Bullet scene.
blender --background --python scripts/blender_physics_rotational_contacts_smoke.py -- /tmp/kumiki-rot-parity-source

# A fresh output directory for native PNGs and capture.json.
blender --background /tmp/kumiki-rot-parity-source/physics-r01/source.blend \
  --python scripts/blender_bridge_parity_render.py -- \
  --plan ../kumiki-ui/src/craft/fixtures/blender-rotational-impacts/visual-parity-plan.json \
  --model ../kumiki-ui/src/craft/fixtures/blender-rotational-impacts/model.glb \
  --output /tmp/kumiki-rot-parity-native
```

In a **second terminal**, in `kumiki-ui`:

```bash
npm run storybook
```

In another terminal, in `kumiki-ui`:

```bash
npx playwright install chromium # only once, if Chromium is not installed
node scripts/capture-blender-bridge-parity.mjs --dry-run
node scripts/capture-blender-bridge-parity.mjs --output /tmp/kumiki-rot-parity-web
```

Back in `kumiki-blender`:

```bash
kumiki-blender bridge-parity \
  --plan ../kumiki-ui/src/craft/fixtures/blender-rotational-impacts/visual-parity-plan.json \
  --blender-dir /tmp/kumiki-rot-parity-native \
  --web-dir /tmp/kumiki-rot-parity-web \
  --output /tmp/kumiki-rot-parity-review
```

Open `contact-sheet.png` and `report.json` in the review directory. Overlays: **green** is shared silhouette, **red** appears only in Blender, **blue** only in Kumiki. The report contains per-frame intersection-over-union (IoU) and normalized bounding-box center error; thresholds default to IoU ≥ 0.90 and center error ≤ 0.035. A failed threshold exits with status 3 and `review-required`. Empty/full/mismatched/altered captures fail closed. Review the source images when thresholds fail. This tool does not automatically alter source geometry.

### Reusing for another asset

1. Provide an authoritative Blender `.blend` with native animated geometry, a verified GLB and matching baked Kumiki action.
2. Write a plan with frame numbers, GLB SHA-256, Storybook story/action IDs and a **world-space, replace-shot** camera preset. Use the same JSON preset in the Storybook story. Camera space is Y-up in the plan; Blender conversion is managed by the renderer script. Only perspective projection is supported in this version.
3. In the Storybook story, expose `data-km-parity-action` and a range input `data-km-parity-seek`. Pass `visualQaMode="silhouette"`, `pixelRatio={1}`, and the plan camera when the `kmSilhouette=1` URL flag is present. Do not remove interactive normal camera/lighting controls.
4. Capture both sides and compare. Plan/GLB changes require **fresh** capture directories on both sides.

The browser capture is intentionally **not part of GitHub Actions or FFmpeg tests**. It requires interactive software GL/WebGL and should run locally on macOS or in a dedicated visual QA environment.
