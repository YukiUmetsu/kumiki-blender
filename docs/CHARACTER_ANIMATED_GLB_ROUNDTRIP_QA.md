# Animated GLB evaluated-surface roundtrip QA

The existing character animation baker already verifies **skinned animation
channels** and, when requested, checks **binary morph-weight accessors** for
art-directed facial visemes and joint corrections. This new stage asks an
additional, materially different question:

> After exporting and re-importing the actual GLB in a fresh Blender scene,
> does its evaluated character geometry match the saved baked Blender rig at
> meaningful moments in the animation?

A valid GLB may contain the expected channels but still produce a displaced
paw, folded wing, incorrect skinning matrix, or silently lost mouth deformation.
This check detects sufficiently large **evaluated 3D geometry** differences,
without assuming glTF preserves vertex indices or quad topology.

## Usage

Use the *actual* outputs of the approved bake-character-animation command:

    kumiki-character-build audit-animated-glb \
      --blend /project/review/quill/combined-glb-r01/baked.blend \
      --glb /project/review/quill/combined-glb-r01/animated.glb \
      --start 1 --end 96 --fps 24 \
      --samples 7 \
      --max-p95-error-m 0.002 \
      --max-max-error-m 0.01 \
      --output /project/review/quill/combined-glb-r01/roundtrip-qa-r01

Options:

- --armature exact source rig object name; default CharacterRig
- --samples 3–11 equally distributed distinct integer frames (default 5)
- --max-vertices memory guard, default 150,000 evaluated vertices per mesh (additional 3,000,000 total stored source vertex-frames cap)
- --max-p95-error-m allowed 95th-percentile symmetric vertex-nearest-neighbor drift (default 2 mm)
- --max-max-error-m allowed maximum vertex and bounding-box drift (default 10 mm)
- --blender optional explicit local Blender executable path

Each source mesh bound to the specified rig must exist by the **same exact
name** in the imported GLB. Extra imported meshes, missing parts, invalid
geometry, large coordinate errors, or changed source files are failures.

The worker uses **two independent Blender scenes**:

1. Load the precise saved baked source and sample evaluated, *world-space*
   vertex coordinates at each selected frame, after all modifiers, armature
   poses, and shape-key animation.
2. Reset Blender to an empty scene, import the GLB from disk, and evaluate
   the imported skinned/morph-animated meshes at the same FPS and frame numbers.
3. Compute symmetric nearest-vertex errors from source to GLB and GLB to
   source using Blender's KD-tree. Separately compare all six bounding
   coordinates; this catches gross relocation or orientation errors.
4. Produce a report with per-part, per-frame vertex counts, worst distance,
   95th-percentile distance, bounding-box error, and any failed budget.

The inputs are SHA-256 locked before and after QA. The original .blend
and .glb files are **never rewritten**. The command returns exit code 1 on
structural-fail and writes a readable report.json in either case.

## Interpreting results

Passing status: evaluated-glb-roundtrip-review-required.

This means structural geometry parity was measured within the requested
thresholds at sampled frames. The passing result **does not** certify final
character identity, color fidelity, texture quality, feather topology, smooth
inter-frame motion, or cinematographic appearance.

For Quill, review both wing folding and mouth articulation against the
original illustrations. For Pip, check paw anatomy, arm gesture, leg
weighting, and the carefully positioned tail in the real 3D rig.
Sampling more frames reduces the risk of missing a brief deformation glitch,
but at most 11 frames are sampled per command. Targeted tests with narrow
frame ranges can probe suspect movements.

This metric is **bidirectional nearest evaluated vertices**, not exact
continuous triangle-to-triangle Hausdorff distance. A GLB may split
vertices at UV/material seams; comparing only vertex index 42 to vertex
index 42 would be incorrect. High-density or heavily decimated geometry
may require an explicit quality plan and source-visual comparison.

Run the native Blender character tests for automated confidence and then
inspect renders in ContentGen/Kumiki-UI at the intended size and camera
angles. A pass cannot approve artwork on its own.
