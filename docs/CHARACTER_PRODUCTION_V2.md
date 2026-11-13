# Character Production v2: implementation guide

This is the **first runnable integration slice** of the [Blender-first production roadmap](https://github.com/YukiUmetsu/kumiki-blender/pull/60). It adds a source-verified semantic region registry, immutable region plans, registered full-frame part comparisons, bounded native Blender region refinement, and a Blender rig pose-audit worker. It **does not yet implement** topology-guided animation loops, complete animator IK/FK rigs, ARAP/Open3D, facial visemes, grooming, or a character performance/action library.

The legacy \`kumiki-character-build\` workflow remains unchanged; new commands are additive. No new dependency or external model is mandatory. This is local-only and works with Blender 4.5 LTS where relevant.

## 1. Prepare a CharacterSpec v2

Copy [character-v2.example.json](../examples/character-v2.example.json) beside its referenced masks and replace placeholders. The sample mask path **does not exist until you provide it**.

Rules:
- Semantic IDs are stable: \`head\`, \`eye.L\`, \`eye.R\`, \`lid.L\`, \`tail\`, \`coat\`, etc.
- The source frame is fixed at **1600 × 1200 px**; the character is 1000 px tall, the soles at row 1100 and the figure horizontally registered with the existing \`handoff\` convention.
- Region \`reference_masks\` are *full-frame binary PNGs*. A 128+ grayscale pixel or varying alpha marks the region. Cropped PNGs are not accepted; images are not recentred or rescaled to inflate metrics.
- Use \`verified_views\` only after a human or trusted calibration process verifies the camera, pose, scale, and mask placement. Images with uncertain registration can be listed under \`reference_masks\` but cannot contribute to numeric registered-view scores.
- \`source_confidence=estimated\` and \`unsupported\` regions are useful for planning but must not pretend to have verified source geometry. \`source_confidence=verified\` alone does not verify camera registration; the view must also be included explicitly.
- Parent and \`depends_on\` relationships form a directed acyclic graph. Source masks and mesh files are path-guarded, checksum-pinned, and must remain within the source folder without symlinks.
- \`attachment\` is descriptive in this stage: it records whether geometry should be rigid, parented to a bone/part, or skinned; it is not an automatically implemented skinning operation.

Create frozen plans:

\`\`\`bash
python -m pip install -e '.[character-build]'
kumiki-character-build plan-character \
  --spec /path/to/character/character.json \
  --output .review/characters/example/plan-r01

kumiki-character-build plan-region \
  --character .review/characters/example/plan-r01 \
  --region eye.L \
  --output .review/characters/example/eye-L-r01
\`\`\`

The region task records the exact source-spec and region-input hashes, prerequisite regions, **downstream invalidations**, and the unrelated regions that must be preserved. A region edit never automatically rebuilds the entire character.

For example, if \`lid.L\` depends on \`eye.L\`, then editing \`eye.L\` invalidates \`lid.L\` but not \`eye.R\`. The graph can support distributed workers later, but memory-hungry Blender work should remain serial on M4/24 GB.

Mutating an input file after planning causes validation to reject the old plan. Create a new revision instead of silently accepting stale art.

## 2. Compare a part against its *verified* source

Give \`compare-region\` a part-only mesh already positioned in the same world frame as the handoff reference:

\`\`\`bash
kumiki-character-build compare-region \
  --character .review/characters/example/plan-r01 \
  --region eye.L \
  --mesh /path/to/eye-left-in-character-frame.ply \
  --output .review/characters/example/eye-L-qa-r01 \
  --min-iou 0.85
\`\`\`

Only explicitly \`verified_views\` are measured. The output contains:
- \`report.json\` with IoU, missing/extra pixels and bounding boxes, per view.
- \`overlays/front.png\`, etc. Green = overlapping model/reference, red = missing model surface, blue = extra model surface.
- An \`artistic-review-required\` status when no numeric threshold fails. Passing a silhouette metric **does not certify 3D likeness**.

To reject regression against an earlier candidate:

\`\`\`bash
kumiki-character-build compare-region \
  --character .review/characters/example/plan-r01 --region eye.L \
  --mesh /path/to/revised-eye.ply \
  --baseline .review/characters/example/eye-L-qa-r01/report.json \
  --max-regression 0.01 \
  --output .review/characters/example/eye-L-qa-r02
\`\`\`

Scores compare identical registered camera/mask positions, with no free fitting. Candidate meshes are not automatically centred, scaled or rotated. If your region is presently embedded in one big mesh, extract and place it first; a full-body mesh cannot be passed off as an eye region. Meshes with invalid topology or missing facets are rejected by existing \`read_mesh\` checks.

## 3. Locally refine one Blender mesh without changing unrelated meshes

Use the example [character-region-sculpt.example.json](../examples/character-region-sculpt.example.json). Set the exact Blender object name, an expected vertex count and one or more sphere-shaped correction fields:

\`\`\`json
{
  "schema_version": 1,
  "object": "Eye.L",
  "expected_vertex_count": 800,
  "max_displacement_m": 0.008,
  "operations": [
    {
      "center_m": [0.035, -0.064, 0.26],
      "radius_m": 0.014,
      "move_m": [0, -0.004, 0],
      "falloff": "smoothstep"
    }
  ]
}
\`\`\`

\`\`\`bash
kumiki-character-build refine-region \
  --blend /path/to/assembled-character.blend \
  --plan /path/to/sculpt.json \
  --output .review/characters/example/eye-sculpt-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
\`\`\`

The Blender worker applies world-space displacement to the named mesh in memory, with zero movement outside each requested radius and a smooth falloff at its boundary. It rejects source files inside the target output, linked library data, shape-key meshes (until a shape-key-aware implementation exists), missing vertices, excessive displacement, and face normal flips. All other meshes are hashed before and after editing. It writes a **new** \`refined.blend\` and \`region.obj\`, never the original.

The worker deliberately **does not** synthesize eyes, eyelids, individual feathers or hidden anatomy. It is a safe geometric correction step, not a substitute for authoring those shapes. Run \`compare-region\` on the resulting part geometry in a consistent frame, then perform visual and posed QA.

## 4. Test rig deformation with native Blender

The [character-rig-pose-qa.example.json](../examples/character-rig-pose-qa.example.json) defines named bone rotations and optional mesh collision pairs. Update the exact armature/bone names and coordinate axes for each character.

\`\`\`bash
kumiki-character-build validate-rig \
  --blend /path/to/rigged.blend \
  --plan /path/to/rig-poses.json \
  --output .review/characters/example/pose-qa-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
\`\`\`

The worker evaluates Blender's actual modified skinned mesh in neutral and nominated poses, reporting:
- Unweighted, over-four-influence and non-normalized vertices on bound meshes.
- Deformation topology changes.
- P95/max relative edge stretch; P95 face area change.
- Volume change only where both meshes are valid closed manifolds.
- World-space BVH triangle-overlap candidates for explicitly named collision pairs.
- An optional hard upper threshold for P95 relative edge stretch.

A structural failure produces exit code 1, and a passing structural audit retains **deformation-review-required**, because neither volume nor stretch proves that a wing, tail, or face looks correct. A contact intersection is an advisory review candidate, not an assumed penetration severity. Set \`render:true\` only if the source scene already has an appropriate active camera. The source \`.blend\` is never saved.

### Optional native Preserve Volume skinning

The pre-existing Blender rig builder now accepts \`--preserve-volume\`, which enables the Armature modifier's Preserve Volume (dual-quaternion) deformation before measuring range of motion. It is **opt-in** to retain the original linear-blend skinning baseline:

\`\`\`bash
kumiki-character-build rig retopo/low.obj joints.json \
  --output .review/characters/example/rig-dual-quaternion-r01 \
  --preserve-volume --render-rom
\`\`\`

Compare the standard and volume-preserving reports and posed renders. Preserve Volume can reduce collapses but will not resolve bad bone placement, collisions, or unnatural wing/tail articulation. The actual Blender-specific improvement remains subject to a local smoke run.

## 5. Tests and runtime boundaries

Run the added CPU tests:

\`\`\`bash
PYTHONPATH=src:tests python -m unittest \
  tests.test_character_spec_v2 \
  tests.test_character_region_sculpt_plan \
  tests.test_character_pose_spec
\`\`\`

Also run the original character test suite; no existing command should change behavior. The new Python schema and mask-comparison stages require the existing \`character-build\` dependencies, but the versioned schema, DAG and rig/sculpt plan validators themselves use only Python's standard library.

Real Blender tests are **not** performed by these Python unit tests; use a local Blender executable with real mesh and rig fixtures. The feature is intentionally incomplete until real assets and Blender/Kumiki export checks have been exercised.

## 6. Remaining roadmap (not implemented in this slice)

1. Fully semantic Blender geometry authoring and optional Open3D ARAP deformation.
2. Non-destructive seam bridging and retopology guided by eyes, mouth, wing, hip, knee and tail bends.
3. Rigify/species profile adaptors, IK/FK controls, weight-region exclusion masks and corrective shape keys.
4. Facial expression and viseme shape-key generation with optional MIT Rhubarb Lip Sync.
5. Groomed bird feathers/fur, secondary deformation, animation actions, and validated skinned GLB playback.

Prioritize a synthetic owl/squirrel fixture and then independently reviewed Pip and Quill model runs before declaring these modeling or rigging workflows production-ready. Continue using immutable private character assets rather than committing copyrighted reference artwork or unlicensed third-party meshes.
