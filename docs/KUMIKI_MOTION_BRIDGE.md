# Blender physics clips → Kumiki PM2

Use Blender's native rigid-body solver and constraints for genuine collisions, then convert its exported glTF keyframes into a portable, frame-addressed motion action. No per-object JavaScript or live solver is needed in Kumiki-UI.

## One command after baking

```bash
# Convert the only clip in a Blender-exported GLB (no Blender installation needed here):
kumiki-blender export-motion --model model.glb --output baked-actions.json

# Choose an animation when there are several; optional measured impact cues:
kumiki-blender export-motion --model model.glb --clip "RigidBodyAction" \
  --contacts contacts.json --output baked-actions.json
```

The output is a **PM2 action array**, ready to supply to the pending Kumiki-UI `loadBlenderRevision(..., { bakedActions })` bridge. It includes 30-fps reference frames, per-node absolute local translation/quaternion/scale, and any explicitly supplied contact timestamps. A full source GLB glTF clip is still playable in the existing asset bridge; the sidecar adds a renderer-neutral, bounded motion contract and SFX cues.

## Prepare a Blender animation

1. Make the object a real rigid body, choose collision shapes/margins, mass, restitution and friction; configure constraints and scene scale. Test at normal playback speed in Blender.
2. Assign stable tags to **the actual animated nodes** before GLB export, e.g. `obj["kumiki_part"] = "hardware"`. If multiple nodes share a role, Kumiki assigns stable canonical IDs like `kumiki:hardware:1` from the GLB scene hierarchy.
3. Bake rigid-body motion to keyframes using Blender's **Object → Rigid Body → Bake to Keyframes** or its Python equivalent `bpy.ops.rigidbody.bake_to_keyframes(frame_start=1, frame_end=121, step=1)` with the intended rigid bodies selected. Store the editable `.blend` source.
4. Export **binary glTF (.glb)** with animations and custom properties/extras enabled. Preserve the same node hierarchy and names for both the GLB and subsequent Kumiki import. Do not author a second coordinate conversion: this command samples the GLB's already converted Y-up local transforms.
5. Run `export-motion`, then visually compare important Blender frames and the Kumiki Storybook/MP4 output: first contact, rebound, joint alignment, final rest and material lighting.

To annotate genuine measured impacts, supply an optional `contacts.json` array:

```json
[
  {"frame": 15.25, "target": "kumiki:hardware:1", "speedMps": 3.1, "material": "metal"},
  {"frame": 37, "target": "kumiki:hardware:1", "speedMps": 0.8, "material": "metal"}
]
```

These cues are **not inferred from animation curves**. They must come from simulation contact reporting or artist verification; the converter never fabricates collision velocities. Missing cues yield an empty array, and unknown node IDs or invalid numbers fail validation.

In Kumiki-UI:

```ts
const bakedActions = await (await fetch("/assets/baked-actions.json")).json();
const factory = await loadBlenderRevision({ manifest, model: glbBytes }, {
  license: "Original asset", bakedActions,
});
const hero = factory();
hero.pose?.({ id: bakedActions[0].id, frame: 24, fps: 30 });
const soundCues = factory.bakedContactEvents(bakedActions[0].id);
hero.dispose(); factory.release();
```

## Accuracy and limits

- Only glTF **node-local translation/rotation/scale** clips with FLOAT accessors and `LINEAR` or `STEP` interpolation are accepted. Rotation interpolation uses shortest-path quaternion slerp. `CUBICSPLINE`, morph/cloth vertex cache and arbitrary bone animation without semantic node tags fail explicitly. Mixed STEP/LINEAR channels on one node are rejected because PM2 has per-track interpolation.
- Names and role ordinals follow the exported GLB's default scene DFS order, not raw Blender collection order. Avoid editing GLB hierarchy after creating the sidecar. A node with a matrix instead of animatable TRS is rejected.
- Duration: up to 120 seconds at 30-fps reference, 128 animated nodes, 120,000 sampled transform keys, 4,096 contact cues. Large sequences should be split into clips for rendering and editing. The source GLB is capped at 512 MiB; animation accessors are read via bounded file offsets without loading all mesh geometry into memory.
- `export-motion` is an interchange converter, **not a rigid-body physics solver or an artistic fidelity certificate**. Genuine Blender reference-frame comparison remains mandatory. Cloth/soft-body geometry needs morph targets or a separate vertex-cache pipeline.

Blender references: [Rigid Body Python API](https://docs.blender.org/api/4.5/bpy.ops.rigidbody.html), [Rigid Body Introduction](https://docs.blender.org/manual/en/latest/physics/rigid_body/introduction.html).
