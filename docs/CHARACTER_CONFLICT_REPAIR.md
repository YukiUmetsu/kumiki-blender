# Reusable character conflict-repair loop

This feature is opt-in and character-independent. It works with the existing
kumiki-character-build handoff directory and an OBJ with UV coordinates.
It uses the actual camera-view conflict diagnostics, not a generative guess,
and keeps a candidate only if its score improves without damaging any
reference view beyond configurable tolerances.

It addresses source-to-surface **texture conflicts and projection seams**,
not general mesh intersection or rig collision correction. It does not
automatically sculpt anatomy or claim that 2D agreement proves 3D fidelity.

## Run

Install the existing build extra and copy the example policy:

    python -m pip install -e '.[character-build]'
    cp examples/character-conflict-repair.example.json /path/to/repair.json
    kumiki-character-build resolve-conflicts \
      /path/to/handoff /path/to/retopo/low.obj \
      --config /path/to/repair.json --output /path/to/repair-r01

Each run writes an immutable, new output folder:
- texture.png — selected bake at final_size.
- preview.png — accepted search-resolution candidate.
- report.json — baseline, every attempted setting, per-view errors, selected
  parameters, scores, rejections and source hashes.
- diagnostics/turnaround.png, owners.png and conflicts.png — final view audit.
- mesh.obj — copied only if an externally supplied compatible mesh candidate won.

If no trial passes, the baseline remains selected. Inputs are never changed.
Use a fresh output directory for another revision.

## What is automatically optimized

The JSON search list contains explicit values per parameter. The solver starts
from the baseline, tests single-parameter alternatives, accepts the best
guarded improvement, and can iterate from that winner on subsequent rounds.
It stops at the evaluation budget or when a round brings no improvement.

These search parameters are allowed: power, diagonal_weight, min_cosine,
region_px, min_region_px, occlusion_px, level_reach_px.

Importantly, conflict-identification thresholds are *not* searchable: simply
raising a threshold to suppress red or yellow errors is not a fix.
The objective combines fixed misplaced-part, contested-feature, unseen-surface,
tone-seam, per-view appearance and per-view silhouette terms. Guardrails compare against the original
baseline, not just against the previous accepted candidate. Every view has its
own maximum allowed mean Delta E increase and silhouette IoU decrease, and the solver checks that the unseen
area does not grow too much. The final-resolution selection is revalidated
against the final-resolution baseline.

For speed on a local Mac, start with evaluation_size 256 or 512,
final_size 1024 and max_evaluations 6-12, then increase after a useful run.

## Semantic ownership: prevent tail/scarf bleed

This is optional. A single-semantic character is not required. Add a
semantic object to the JSON policy. Each part has:

- a binary UV-atlas PNG where white texels belong to that part;
- binary PNG masks in the **raw reference image coordinate frame** for the
  cameras that actually show that part.

Example fragment (paths are relative to repair.json; supply real files):

    "semantic": {
      "parts": [
        {
          "id": "tail",
          "uv_mask": "masks/tail-uv.png",
          "view_masks": {
            "front": "masks/front-tail.png",
            "right": "masks/right-tail.png",
            "back": "masks/back-tail.png"
          }
        },
        {
          "id": "scarf",
          "uv_mask": "masks/scarf-uv.png",
          "view_masks": {
            "front": "masks/front-scarf.png",
            "right": "masks/right-scarf.png"
          }
        }
      ]
    }

If a surface texel belongs to the tail, it will only take color from a
pixel inside a tail mask for that source. Missing masks do not implicitly
authorize a projection. UV part masks may be nearest-resized, but source view
masks must exactly match the raw PNG dimensions. Part UV masks must be disjoint.
If no trusted image can paint a labeled part, the bake **fails explicitly**
instead of borrowing a nearby scarf or skin color. Unlabeled UV texels retain
legacy selection behavior.

Create reliable masks from authored part geometry / material IDs or a
reviewed segmentation output. A color-coded conflict turnaround is *not* a
semantic segmentation mask. Multiple separate Blender objects can be mapped
into a UV atlas using material/part IDs before applying this policy.

## Geometric penetration repair: opt-in Blender worker

The texture loop cannot fix tail/torso or wing/clothing **mesh intersections**.
The companion command works on a native `.blend` containing separate named
mesh objects, using an explicitly directed movable/fixed pair policy:

```bash
cp examples/character-penetration-repair.example.json /path/to/collision.json
# Replace Tail / Body with the exact object names in the source .blend.
# Replace protected_vertices with the actual attachment/root indices.
kumiki-character-build repair-penetrations \
  --blend /path/to/character.blend \
  --plan /path/to/collision.json \
  --output /path/to/collision-repair-r01
```

The worker measures triangle overlap using Blender's BVH, estimates signed
vertex clearance against a **closed, outward-facing fixed mesh**, then
generates bounded local displacements for the movable object. Protected
vertices remain unchanged. It smooths displacements across mesh neighbors,
backs off steps which flip faces or exceed edge-stretch limits, and rolls back
any candidate increasing triangle intersections or previously accepted pair
conflicts. The original .blend is never saved over.

On improvement, `collision-candidate.blend` and `report.json` are produced.
If there is no safe improvement, it writes only a report and retains all
input geometry. It does **not** silently treat allowed physical contact as
a defect; you must specify only *forbidden* intersection pairs.

Safeguards and limitations:
- Moving geometry must be an editable mesh without shape keys/modifiers,
  with stable vertex indices and no mirrored transforms. Fixed geometry must
  be a closed/manifold, outward-oriented mesh without modifiers.
- `protected_vertices` should cover tail roots, wing sockets, clothing
  attachment seams, etc.; these indices are specific to each character.
- Vertex clearance approximates penetration on the fixed mesh; BVH overlap
  detects triangle crossing but does not measure penetration volume. The
  result may be **partial** and always requires artistic inspection.
- This stage deliberately does not modify UVs, skeletons, animation keys,
  unselected body parts, or trusted source references.
- A geometry candidate should be followed by `compare` / `compare-region`,
  the existing `validate-rig` pose battery, and a new texture conflict report
  before acceptance. There is no automatic claim of screen-space fidelity.
- If the fixed target is open cloth, hair cards, or non-manifold geometry,
  this repair step refuses to guess inside/outside. Use a reviewed proxy
  collision surface or a different constraint mechanism.
- Use conservative world-meter displacement budgets (for a 0.35 m character,
  start with a few millimeters). No generative model is necessary.

## Optional geometry candidates

The configuration may include:

    "candidate_meshes": ["candidates/tail-tucked-r02.obj"]

This option evaluates **pre-authored or externally generated** alternatives
against the same cameras using the same unmodified UV layout and face indices.
The selected OBJ is copied to mesh.obj if it wins. The repair loop does not
generate or certify those geometry edits. Verify real anatomical accuracy,
attachments, and penetration with Blender and pose QA before accepting
a changed shape. This mechanism is designed for later integration with
bounded region sculpting and a collision detector.

## Limits and extension points

- Registered camera and source masks remain evidence supplied by the artist.
  A better score cannot prove that camera alignment is correct.
- Appearance comparisons are numeric, not an independent artistic review.
- Semantic labels are supplied, not inferred by the loop.
- A high surface-conflict count may be caused by wrong geometry or inconsistent
  art. Stop texture optimization when reference geometry needs correction.
- Geometry candidates must pass a separate per-view silhouette guard; texture
  errors alone cannot justify shrinking the character to hide missing regions.
- The library entrypoint is run_conflict_repair(..., evaluator=optional_callback);
  an alternative character pipeline can supply its own candidate evaluator.
  Callback output must keep conflict definitions and view comparisons stable.

The original commands (conflicts, texture, fit, compare-region, validate-rig)
are unaffected unless resolve-conflicts or semantic arguments are used.
