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
tone-seam and per-view appearance terms. Guardrails compare against the original
baseline, not just against the previous accepted candidate. Every view has its
own maximum allowed mean Delta E increase, and the solver checks that the unseen
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
- The library entrypoint is run_conflict_repair(..., evaluator=optional_callback);
  an alternative character pipeline can supply its own candidate evaluator.
  Callback output must keep conflict definitions and view comparisons stable.

The original commands (conflicts, texture, fit, compare-region, validate-rig)
are unaffected unless resolve-conflicts or semantic arguments are used.
