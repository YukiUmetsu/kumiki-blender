# Joint-local hinge topology refinement (pre-rig)

This Blender-first operation adds **real local quad-edge density** around reviewed skeletal hinge positions. It builds on the existing plan-topology and audit-topology workflows, which previously only measured insufficient joint contour coverage.

**This is a conservative, source-preserving retopology candidate**, not an automatic professional edge-flow solution. It is intended for a mostly quad, closed **pre-rig** low-poly cage. The mesh must not yet contain shape keys, skin vertex groups, deforming modifiers or linked-library data. Improve geometry before skinning/visemes and do not run on production meshes with authored morph shapes.

## Workflow

1. Make a reviewed joints.json and a stable mostly-quad retopology .blend. The mesh object must match --mesh-name.
2. Generate a native topology review plan for the particular joints:

~~~bash
kumiki-character-build plan-topology \
  --joints /path/to/reviewed-joints.json \
  --mesh-name CharacterCage --joint elbow.L --joint elbow.R \
  --output .review/character/topology-r01
~~~

3. Plan bounded axial loop cuts in one or more reviewed hinge regions:

~~~bash
kumiki-character-build plan-joint-refine \
  --blend /path/to/retopo/character-pre-rig.blend \
  --topology .review/character/topology-r01/topology-plan.json \
  --joint elbow.L --joint elbow.R --cuts 2 \
  --max-added-vertices 25000 \
  --output .review/character/hinge-plan-r01
~~~

4. Apply locally with Blender 4.5+:

~~~bash
kumiki-character-build refine-joint-topology \
  --plan .review/character/hinge-plan-r01/joint-refine-plan.json \
  --output .review/character/hinge-candidate-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
~~~

The worker locates existing quad-edge strips that cross each reviewed 3D joint plane and travel along the corresponding bone axis. It creates bounded cuts with native BMesh subdivision, retains source vertices, and rejects missing quad bands, nonmanifold topology, UV/material-name changes, source rest-volume regressions >2%, and damage to unrelated meshes.

Outputs are an immutable **hinge-retopo.blend**, hinge-retopo.obj (legacy interchange), hinge-retopo-source.obj (Z-up source QA), and a report with before/after topology, per-joint cut counts, original vertex/UV/material protection and explicit quality limitations. The original .blend is never overwritten.

## Mandatory follow-up

Use audit-topology on the new cage, inspect actual concentric loop flow in Blender, compare part silhouettes to original registered reference masks, and **regenerate** skin weights and facial morphs only after approving the topology. Run validate-rig to test deformation before calling a hinge animation-ready.

Caveats:
- BMesh local subdivision adds sampling density, not automatically anatomically correct loops or skin weights.
- A complicated wing pivot with branch geometry, extraordinary poles, non-quads or unregistered joint centers must be corrected manually or via a separate guided patch workflow.
- This algorithm refuses rigged/shape-key meshes to prevent destruction of final animation or materials.
- Native Blender synthetic tests cover a quad-cylinder hinge; real Pip/Quill meshes still need individual source reference and pose QA.

No additional libraries, online models or commercial dependencies are required.
