# Source-guarded, feature-preserving surface refinement

\`kumiki-character-build refine-surface\` is a **non-destructive pre-rig stage** for
surface lumps left by retopology, local sculpting, and rough source-image
reconstruction. It runs locally on Blender 4.5+ with NumPy; it does not
require a cloud AI model, SciPy inside Blender, or a new paid dependency.

The output is a *candidate*, not a reconstructed character approved for use.
Smoothness and silhouette fidelity are different; preserving both matters.

## Example

Copy [the reusable policy](../examples/character-surface-fairing.example.json) and
edit it for your character's **existing, separate pre-rig mesh object**.

~~~bash
kumiki-character-build refine-surface \
  --blend /path/to/approved-character.blend \
  --config /path/to/surface-policy.json \
  --output /path/to/surface-refine-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
~~~

The output contains \`report.json\`, \`model/report.json\`, and, if numerical
surface improvement passed the safety guards:

- \`model/surface-candidate.blend\`: new independent, editable Blender scene
- \`model/baseline-source.obj\`: original part, **world-space Z-up** for camera QA
- \`model/surface-candidate-source.obj\`: revised part, **world-space Z-up**
- \`model/surface-candidate.obj\`: traditional Blender-export Y-up interchange

The source .blend is never overwritten. The original object's vertex IDs,
triangles/quads, UV layers, materials and unrelated objects are retained.

## How it works

The NumPy solver constructs the triangle adjacency of a closed, outward
surface. It identifies high-dihedral **sharp edges** and high-curvature
vertices from the untouched original; these receive reduced or zero smoothing
weight. Blender artist-authored groups may explicitly protect the eyes, beak,
tail tip, paw creases or garment boundaries.

Each iteration performs bounded positive/negative Laplacian relaxation
(Taubin-style fairing), with movement clamped relative to the **original**
surface. It only retains iterations which reduce sampled one-ring roughness
and pass all hard guards:

- maximum per-vertex world-metre displacement
- maximum relative edge stretch
- maximum closed-mesh volume change
- no flipped or degenerate faces
- protected named vertex groups untouched

\`curvature_stop\` controls suppression of smoothing near distinctive shapes;
\`feature_angle_deg\` locks sharp edge vertices. Both are *geometry*
parameters, not artificial ways to adjust acceptance thresholds.

\`editable_vertex_group\` optionally limits smoothing to artist-painted
vertices. Blend weights between 0 and 1 supply continuous falloff. Every
name must exist and a group needs at least one weight of 0.5 or greater.
\`protected_vertex_groups\` always override editable weights.

For a 0.35 m character, begin with \`max_displacement_m\` between 0.001 and
0.004. Don't apply aggressive smoothing to the muzzle or eye regions before
protecting those high-value features.

## Optional verified reference gate

For a CharacterSpec v2 with approved, source-registered full-frame part masks,
add the following to the policy (replace the path and region):

~~~json
{
  "reference_gate": {
    "character_plan": "/path/to/frozen/character-plan-folder",
    "region": "tail",
    "max_iou_regression": 0.005,
    "min_iou": null
  }
}
~~~

The command exports the **same Blender object**, before and after, as
registered Z-up OBJs and invokes the existing
\`gate_region_revision\` machinery. It compares *independently verified*
reference views, preserves source-camera registration, and generates
per-view overlays and a baseline/candidate report.

If **any view** regresses beyond the configured tolerance, the top-level
status is \`reference-regression-rejected\` and a
\`SOURCE_REGRESSION_REJECTED.txt\` marker is written. The output remains
inspectable, but do **not** promote its \`.blend\` to production. The CLI exits
with a nonzero status on registered-source regression.

If no CharacterSpec gate is supplied, the top-level status is
\`surface-candidate-art-review-required\`: geometric roughness alone
does not establish likeness.

**Critical:** A CharacterSpec region is a *part-only reference mask*. The
specified Blender object must represent that same part in the same world
frame. Passing a whole body as a tiny eye or tail region is invalid evidence.
If you have no verified part masks, omit the gate; do not fabricate them.

## Pipeline placement

Run **after** coarse silhouette fitting, volume modeling, retopology and
local anatomy placement, but **before** facial shape keys and skinning.
Then perform \`audit-region-revision\`, \`audit-topology\`,
\`refine-joint-topology\` if necessary, and \`validate-rig\` after skinning.

The tool rejects modifiers, linked or shape-key meshes and any surface that
is open, non-manifold, degenerate or not outward oriented. It will not
silently merge cloth, fur cards, overlapping eyes or wings into a single
volume. Those are separate semantic pieces that require independent modeling.

### Limitations and next extensions

- This release **does not subdivide or retopologize**. Adaptive edge-loop
  placement is a separate source-preserving operation.
- The score is geometric one-ring regularity, not perceived attractiveness.
- Closed-surface volume preservation and no face flips cannot prove
  no self-intersections or no artistic detail loss.
- A registered silhouette pass cannot verify facial anatomy, color,
  surface normals, hidden depth, or animation. Independent visual QA remains
  essential.
- A rough source mesh can have genuinely important protrusions; protect
  those regions before smoothing, rather than overfitting a numerical score.
