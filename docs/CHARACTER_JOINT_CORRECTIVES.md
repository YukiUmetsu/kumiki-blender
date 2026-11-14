# Artist-sculpted joint correctives (Blender, source locked)

**Problem:** Linear skinning and even dual-quaternion preserve-volume skinning do not fully model folding of Pip's tail/paws or Quill's feathered wing joints. Correcting weights alone cannot supply absent elbow, knee or wing-fold forms.

This feature adds real **rest-space sculpt geometry as relative shape keys** on existing skinned mesh objects, then drives those shapes from a specific bone's local rotation. It cannot infer correct anatomy, replace hand-authored sculpts or approve a finished character.

## Prerequisites

1. Start from a **reviewed skinned rig** Blender file with all pose bone transforms at neutral, and any existing viseme/shape-key values at zero. The source remains immutable.
2. In Blender, duplicate the body-part **mesh datablock** with Shift+D (not a linked duplicate). Preserve exactly the same vertex and polygon order and object world transform. Name it, for example, WingRaiseSculpt.L.
3. Remove **all modifiers** from that duplicate, including Armature and subdivision. Hide it from rendering (hide_render=True), but keep it in the saved Blender scene.
4. Sculpt the duplicate's vertices **in the original rest-space coordinate system**. Use character reference close-ups for the intended wing/paw/tail fold. Do **not** rotate or translate the duplicate object or change topology. The duplicate is the **full desired rest-space corrective shape**, not just a delta.
5. Save the .blend file, and author a JSON configuration. The model must already have an acceptable silhouette at rest; this feature only changes posed deformation.

Example joint-corrective-config.json:

    {
      "armature": "CharacterRig",
      "correctives": [
        {
          "mesh": "Wing.L",
          "target_object": "WingRaiseSculpt.L",
          "shape_key": "CORR_WingRaise.L",
          "bone": "upper_wing.L",
          "axis": "Z",
          "direction": "positive",
          "start_degrees": 10,
          "full_degrees": 50,
          "max_displacement_m": 0.008,
          "min_changed_vertices": 4,
          "max_changed_vertices": 2000
        }
      ]
    }

**Coordinate contract:** axis is the *local bone rotation channel* (X, Y or Z), not a world-space direction. With the example, the corrective is 0% at 10° and below, ramps in magnitude to 100% at 50°, and remains 0% on opposite-direction bends. The shape-key geometry is evaluated **before** the Armature modifier. If you use an IK or FK controller, select the **actual deform bone** whose evaluated local rotation responds to the controller and verify its sign in Blender.

## Produce a new candidate

    kumiki-character-build plan-joint-correctives \
      --blend /project/characters/quill/rigged.blend \
      --config /project/characters/quill/joint-corrective-config.json \
      --output /project/review/quill/correctives-plan-r01

    kumiki-character-build bind-joint-correctives \
      --plan /project/review/quill/correctives-plan-r01/joint-corrective-plan.json \
      --output /project/review/quill/correctives-r01

The plan records a **SHA-256 of the saved source Blender file**. A modified source requires a new plan. The worker will reject wrong vertex order, mismatched topology, object-transform drift, missing joints, duplicate shape-key names, unhidden or modified target meshes, out-of-budget displacement/vertex counts, nonfinite coordinates, flipped faces, active non-neutral poses or non-neutral shape keys.

The output is an independent corrected.blend with a report.json containing changed-vertex counts, maximum sculpt offsets and driver assignments. It never overwrites source files. All existing UV/material data and unrelated scene meshes are retained. Corrective shape keys contain the approved sculpt vertices; **only driver values vary with the pose**.

## Required validation before use

- Check **neutral pose** against the original source image: corrected and uncorrected rest geometry must match.
- Test **both bend directions**, several intermediate angles and the full authored range. Look for pinching, volume loss, clipping and visible hard transitions.
- Run validate-rig on corrected.blend using a pose battery that includes relevant joint angles. Use silhouette and source-reference comparisons rather than relying only on edge-stretch metrics.
- Retain old revised source assets as checkpoints; a geometry corrector never earns automatic artistic acceptance.

## Portable GLB animation: rig controls + joint correctives

Blender driver expressions themselves cannot be exported to glTF. To make
their **evaluated deformation** portable, the animation worker now supports
explicit framewise sampling and verifies the resulting GLB binary. First animate
the approved controlled rig, with the same skinned mesh/shape keys from the
corrected Blender revision (both the controller Action and the CORR_ shape keys
must be in one saved .blend). Use an original controlled rig with no NLA
tracks or existing facial shape-key Action:

    kumiki-character-build plan-animation-bake \
      --blend /project/characters/quill/animated-corrected.blend \
      --clip wing_fold \
      --start 1 --end 96 --step 1 --fps 24 \
      --bake-correctives \
      --output /project/review/quill/wing-fold-bake-plan-r01

    kumiki-character-build bake-character-animation \
      --plan /project/review/quill/wing-fold-bake-plan-r01/bake-plan.json \
      --output /project/review/quill/wing-fold-glb-r01

The worker:

1. Validates the source .blend SHA-256, one selected skeletal Action, and each
   reviewed CORR_ driver bound to a local deform-bone rotation channel.
2. Samples source driver values at **every integer frame** before IK/FK baking.
3. Bakes the rig visually, removes controllers, and checks both deform-bone
   motion and corrective weight parity against the source.
4. Removes Blender-only drivers in the **new revision**, inserts regular
   shape-key Action keyframes, and exports a single glTF animation with
   skeletal joint channels and morph-weight channels.
5. Reads the exported GLB's binary float32 accessors to verify each named
   shape target's actual sampled weights (maximum difference 0.025).
   It rejects wrong target ordering, missing channels, invalid timestamps,
   duplicate mesh nodes, and mismatched weights rather than claiming success.

Output: baked.blend, animated.glb and report.json. Check
report.json -> morph_parity for each object and corrective. Running the worker
without --bake-correctives on a driven shape still fails **closed**, as before.
This path is intentionally opt-in and requires --step 1 to avoid dropping
subframe joint-corrective changes.

**Remaining limitations:** shape-key facial Actions and multiple NLA tracks are
not yet combined into this exporter. There is no automatic source-faithful
sculpting or visually certified silhouette in this change. Use source-view
render comparisons, gesture range checks, and the ContentGen renderer for final
artistic approval.

**Export limitation:** Do not use raw GLB export to preserve Blender drivers.
Use the checked --bake-correctives workflow above.

## Verification

CI includes pure-Python contract tests and a native Blender 4.5 smoke that builds a wing with a sculpted duplicate, binds the corrective, and confirms the neutral geometry and opposite bend remain identical while the intended bend actually changes evaluated vertices. This proves a functional technique, **not Pip/Quill source fidelity**. Artist-reviewed geometry and animation on the character source are still required.
