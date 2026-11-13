# Bake native Blender character controls to portable GLB clips

This is a **single-action, skeletal-animation** integration slice of the [Character Production v2](CHARACTER_PRODUCTION_V2.md) workflow. It is Blender-local, cloud-free and does not add any runtime/model dependency or copy source from GPL add-ons. The original character art and Blender source are never overwritten.

## Prerequisites

1. Create a reviewed rig with `kumiki-character-build rig`.
2. Add a non-deforming animator control layer with `plan-rig-controls` / `apply-rig-controls`.
3. Open the resulting `controlled.blend` in Blender and keyframe the relevant `CTRL_...` bones. For example, rotate `CTRL_upper_wing.L` over frames 1–24 to fold a wing.
4. Save a **new** animated source `.blend`. The armature must have one active Action and **no NLA tracks** in this initial version. Do not rely on unstored pose edits.

## Commands

```bash
kumiki-character-build plan-animation-bake \
  --blend .review/quill/animated-controls.blend \
  --clip wing_fold --start 1 --end 24 --fps 24 \
  --output .review/quill/wing-fold-plan-r01

kumiki-character-build bake-character-animation \
  --plan .review/quill/wing-fold-plan-r01/bake-plan.json \
  --output .review/quill/wing-fold-export-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

Outputs:
- `baked.blend`: a **separate, editable revision** with IK/FK constraints visually baked onto the deform bones, non-deforming `CTRL_` bones removed, and the active baked Action renamed to the requested clip.
- `animated.glb`: selected skinned character meshes plus the armature and sampled deform-bone animation, without Blender-only `CTRL_` nodes.
- `report.json`: source/provenance, sampled frame range, maximum position/rotation discrepancy between controlled and baked poses, bones removed, skin/joint/animation/channel counts and remaining limitations.

The worker refuses stale source inputs, missing animation, NLA tracks, unskinned rigs, out-of-bounds frames, unsupported exporter modes, invisible/missing GLB animation channels or pose parity beyond the configured tolerance. Default limits are **2 mm** position and **2°** rotation error in the rig's pose matrix, measured at the selected sample frames. A successful numerical audit still produces **exported-visual-parity-review-required**, not an automatic cinematic-quality certificate.

## Implementation boundary

Blender control rigs (IK, FK, pole targets) are *authoring tools*. glTF clients such as Kumiki-UI should receive only evaluated deformer skeleton animation. The bake process snapshots the evaluated pose, uses native Blender visual-keying bake, removes controller bones and compares frame-by-frame deformation-bone pose matrices before exporting GLB. The GLB is then independently parsed for actual skin and animation channels.

This first iteration intentionally handles only **one active skeletal Action**. It does not combine multiple NLA tracks, morph-target/viseme actions, simulations or geometry-node feather deformation; those require separately validated timelines. It also does not retarget animation between distinct species rigs. Existing texture/material names may be preserved by Blender's glTF exporter, but a later independent Blender–Kumiki-UI render-parity audit is still required for final approval.

### Verification

The script `scripts/smoke_character_controls_topology_blender.py` exercises a synthetic animated avian rig under the official pinned Blender 4.5 LTS binary. It creates control keyframes, bakes the action, verifies that an animated skin is present in GLB and checks pose parity. This smoke test is wired into the character-specific GitHub Actions workflow and does not use private Pip or Quill artwork.

These tests establish **runtime functionality**, not source fidelity or profession-level rig quality.
