# Blender deformation → Kumiki animated mesh

For cloth, soft-body motion, wind-driven fabric, and topology-preserving
deformer animation, `build-deformation` captures **real forward-evaluated
Blender mesh vertices** into standard glTF **shape keys (morph targets)**.
The resulting `model.glb` contains the animation itself: Kumiki/Three.js can
play or scrub its named clip without a browser physics simulation, proprietary
vertex format, per-shot JSX/Three.js code, or a PM2 transform sidecar.

## Agent-friendly workflow

1. Create or import an editable Blender mesh. Set `obj["kumiki_part"] =
   "cloth-panel"` on the target, and configure actual cloth/soft-body or
   modifier motion. Save `source.blend` (keep it as the editable authority).
2. Write one bounded JSON plan, e.g.

   ```json
   {"schema_version":1,"asset_id":"sailing-flag","action_id":"wind-ripple",
    "target":"role:cloth-panel","duration_frames":48,"sample_stride":1,
    "max_vertices":4000}
   ```

3. Validate without Blender and bake with Blender 4.5:

   ```bash
   kumiki-blender plan-deformation --spec plan.json
   blender --background source.blend --python-exit-code 1 \
     --python scripts/blender_entry.py -- \
     build-deformation --spec plan.json --output output/cloth-r01
   ```

   The output is immutable: `source.blend` (original modifier rig),
   `model.glb` (native morph animation), `manifest.json` (audit and provenance),
   and `deformation-plan.json` (bake parameters). Revisions cannot overwrite.

4. Load the asset as usual:

   ```ts
   const factory = await loadBlenderRevisionFromUrl("/assets/cloth-r01/manifest.json");
   const hero = factory();
   hero.pose?.({ id: "wind-ripple", frame: 24, fps: 30 });
   hero.dispose(); factory.release();
   ```

   The GLB clip is reusable in Storybook, deterministic timeline scrubbing,
   and ordinary offline video renderers. There is no extra JSON motion file;
   `files.model` already contains the animated geometry.

## Validation, quality, limits

- A single target mesh per plan, selected uniquely by `name:` or `role:`.
  Its semantic role and existing PBR material/UVs are preserved. Multi-mesh
  scenes can be baked independently, or authored with native glTF armatures.
- The source and every evaluated frame **must have identical vertices, faces,
  and polygon indices**. Topology-changing simulations (fluid meshing,
  remeshing, changing particles) are rejected, not silently distorted.
- Blender's evaluated dependency graph is sampled **in forward order at every
  frame**, even when only every Nth frame becomes a morph target. Shape-key
  weights use linear interpolation between selected frames, with no easing
  overshoot. Fast cloth motion needs stride `1` for acceptable temporal fidelity.
- At most **64 morph targets**, **12,000 source vertices**, and **450,000
  vertex-target pairs**. These are security/performance budgets, not guarantees
  of playback quality. Large, production cloth scenes may need mesh decimation,
  LODs or a different vertex-cache playback method.
- Existing shape keys or topology-altering modifiers are refused because they
  can invalidate correspondence. Preserve a source `.blend` for future re-bakes.
- **Not a cloth solver**: Kumiki replays sampled Blender mesh geometry. It does
  not compute cloth collisions, re-tension fabric, or apply live wind. Moving
  the camera and lights is safe; changing physical object constraints requires
  a new Blender simulation.
- Actual Blender 4.5 Cloth smoke generates a 12-target `cloth-sag` clip from
  81 vertex samples; the GLB is structurally audited for one native `weights`
  animation channel. Kumiki Storybook QA verifies exact rest/end and
  out-of-order playback. Pixel-level Blender ↔ WebGL render comparison remains
  a separate release gate.

For rigid bodies and collision SFX, use
[Kumiki Motion Bridge](KUMIKI_MOTION_BRIDGE.md) (PM2) instead. The two formats
coexist in the same renderer, but deforming vertices are **not** rigid-body
transform tracks.

Blender source documentation: [glTF 2.0 exporter — shape keys](https://docs.blender.org/manual/en/dev/addons/import_export/scene_gltf2.html).
