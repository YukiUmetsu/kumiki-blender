# Blender physics clips → Kumiki PM2

Use Blender's native rigid-body solver and constraints for genuine collisions, then convert its exported glTF keyframes into a portable, frame-addressed motion action. No per-object JavaScript or live solver is needed in Kumiki-UI.

## Automatic Blender rigid-body shot (no per-object Python)

A validated JSON plan configures mass, gravity, friction, restitution, collision shapes, start offsets, solver quality and timeline for any **already modeled** `.blend` scene. It reuses the existing mesh and materials rather than generating replacement geometry. Start with [the desk-lamp fastener-drop example](../examples/desk-lamp-physics-shot.json), which targets the `base-fasteners` and `weighted-base` roles exported by the AssetSpec v2 desk lamp. Its optional `ground` is a simulation-only, unexported collider that catches falling parts; adjust `height_m`, `half_extent_m`, `thickness_m`, `friction` and `restitution` to suit any scene.

```bash
# Validate on normal Python first, no Blender needed:
kumiki-blender plan-physics --spec examples/desk-lamp-physics-shot.json

# After building the AssetSpec desk-lamp scene:
blender --background .review/desk-lamp/r01/source.blend \
  --python-exit-code 1 --python scripts/blender_entry.py -- \
  build-physics --spec examples/desk-lamp-physics-shot.json \
  --output .review/desk-lamp/physics-r01
```

The output directory must **not exist**. The original `.blend` is never overwritten. A physics bundle contains:

| File | Purpose |
| --- | --- |
| `source.blend` | Editable copy with configured rigid bodies, before keyframe baking |
| `model.glb` | Existing geometry/materials and actual baked glTF animation |
| `baked-actions.json` | PM2 transforms converted from the GLB; all disjoint per-object clips merged safely |
| `physics-plan.json` | Canonical reusable input plan, including explicit material/solver settings |
| `manifest.json` | Schema-1 metadata and GLB counts for the Kumiki-UI asset bridge |

Use `baked-actions.json` in `loadBlenderRevision(..., { bakedActions })` and select the generated action in Storybook with the portable camera/lighting inspector. This motion-only bundle has **no rendered preview images** and is not a full visual-QA revision; review Blender viewport frames and rendered video before production approval. If the glTF exporter cannot represent a shot faithfully (non-TRS clips, overlapping actions, untagged nodes), the build fails explicitly.

Physics is assigned only to explicitly targeted editable meshes. `name:<exact Blender object name>` selects one mesh (and automatically gives an untagged selected object a safe `kumiki_part` role); `role:<kumiki_part>` applies a rule to all matching meshes. All selected bodies are checked for overlaps and the animation is sampled from frame zero at a 30-fps reference rate, producing exactly `duration_frames` of playback. In headless Blender 4.5, the worker samples evaluated Bullet transforms and inserts native object keyframes because the standard `bpy.ops.rigidbody.bake_to_keyframes` operator requires an unavailable UI keying context. The generic shot does not guess arbitrary collision forces or contacts from gravity equations.

### Reusable hinges, sliders and welded joints

Use [the mechanical joint example](../examples/mechanical-constraints-physics-shot.json) for a real Blender-native three-joint demonstration. Each rule names two already-configured rigid bodies; `role:` is allowed only if it uniquely selects one body. `pivot_m` is the world-space joint anchor and `axis` is a unit vector. Hinge rotation uses the constraint helper's Z axis; slider translation uses its X axis, automatically oriented to `axis`.

```json
{"id":"door-pin","kind":"hinge","body_a":"name:door-frame","body_b":"name:door-panel","pivot_m":[0,0,1],"axis":[0,0,1],"limits_deg":[-70,70]}
```

Other supported types: `slider` with `limits_m` and `fixed` without limits. All use Blender Bullet constraints, not handcrafted keyframes. Joint helpers remain in the editable `.blend` but are excluded from the portable GLB. Conflicting role selectors and invalid/out-of-order limits fail before the bake.

### Ground-impact audio cues

If `ground` exists, the builder conservatively derives audio cue candidates from **sampled Bullet transforms and actual mesh bounds**: a body must approach the authored ground plane, have measurable downward speed and subsequently remain supported or rebound. Example body option: `"impact_material":"metal-on-floor"`. The PM2 action contains fractional-frame `{frame,target,speedMps,material}` records, aligned to canonical GLB semantic part IDs.

This is **not** Bullet's native contact impulse/manifold: timings are frame-interpolated estimates, and glancing hits or body-to-body contacts may not be detected. The worker suppresses pass-throughs, ambiguous IDs and unverified non-ground contacts. Missing evidence means **no cue**, not an invented sound. Use the independent [`export-motion --contacts`](#one-command-after-baking) path for artist-verified or externally measured contacts.

## One command after baking

```bash
# Convert the only clip in a Blender-exported GLB (no Blender installation needed here):
kumiki-blender export-motion --model model.glb --output baked-actions.json

# Choose an animation when there are several; optional measured impact cues:
kumiki-blender export-motion --model model.glb --clip "RigidBodyAction" \
  --contacts contacts.json --output baked-actions.json

# If Blender emitted one disjoint clip per rigid body:
kumiki-blender export-motion --model model.glb --merge-clips --output baked-actions.json

# Blender may emit LINEAR position but STEP constant rotation on one object;
# this explicit opt-in preserves every 30-fps rendered frame, not subframes:
kumiki-blender export-motion --model model.glb --merge-clips --resample-mixed --output baked-actions.json
```

The output is a **PM2 action array**, ready to supply to the pending Kumiki-UI `loadBlenderRevision(..., { bakedActions })` bridge. It includes 30-fps reference frames, per-node absolute local translation/quaternion/scale, and any explicitly supplied contact timestamps. A full source GLB glTF clip is still playable in the existing asset bridge; the sidecar adds a renderer-neutral, bounded motion contract and SFX cues.

## Prepare a Blender animation

1. Make the object a real rigid body, choose collision shapes/margins, mass, restitution and friction; configure constraints and scene scale. Test at normal playback speed in Blender.
2. Assign stable tags to **the actual animated nodes** before GLB export, e.g. `obj["kumiki_part"] = "hardware"`. If multiple nodes share a role, Kumiki assigns stable canonical IDs like `kumiki:hardware:1` from the GLB scene hierarchy.
3. Bake rigid-body motion to keyframes using Blender's **Object → Rigid Body → Bake to Keyframes** in the interactive application, or use `build-physics` in headless Blender. The `bake_to_keyframes` operator is not supported in Blender 4.5 background mode because it invokes a UI keying set. Store the editable `.blend` source.
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

- Only glTF **node-local translation/rotation/scale** clips with FLOAT accessors and `LINEAR` or `STEP` interpolation are accepted. Rotation interpolation uses shortest-path quaternion slerp. `CUBICSPLINE`, morph/cloth vertex cache and arbitrary bone animation without semantic node tags fail explicitly. Mixed STEP/LINEAR channels on one node are rejected by default; `--resample-mixed` converts them to per-frame linear PM2 tracks for 30-fps video, with no guarantee of subframe parity.
- Names and role ordinals follow the exported GLB's default scene DFS order, not raw Blender collection order. Avoid editing GLB hierarchy after creating the sidecar. A node with a matrix instead of animatable TRS is rejected.
- Duration: up to 120 seconds at 30-fps reference, 128 animated nodes, 120,000 sampled transform keys, 4,096 contact cues. Large sequences should be split into clips for rendering and editing. The source GLB is capped at 512 MiB; animation accessors are read via bounded file offsets without loading all mesh geometry into memory.
- `export-motion` is an interchange converter, **not a rigid-body physics solver or an artistic fidelity certificate**. Genuine Blender reference-frame comparison remains mandatory. Cloth/soft-body geometry needs morph targets or a separate vertex-cache pipeline.

Real Blender QA: `scripts/blender_physics_smoke.py` creates two falling metal cubes and an unexported floor, simulates them in Blender 4.5, and checks that the GLB has exactly two semantic animated nodes, neither penetrates the floor and at least one grounded SFX cue is measured. `scripts/blender_physics_constraints_smoke.py` verifies all three native constraint kinds: pendulum hinge falls, slider moves within its bounds, welded body remains fixed. The PR CI additionally bakes the **actual 12 fasteners** in the existing desk-lamp AssetSpec, checks trajectories and nonzero grounded contact cues, and uploads `.blend`, GLB, PM2 and manifest artifacts for independent inspection.

Blender references: [Rigid Body Python API](https://docs.blender.org/api/4.5/bpy.ops.rigidbody.html), [Rigid Body Introduction](https://docs.blender.org/manual/en/latest/physics/rigid_body/introduction.html).
