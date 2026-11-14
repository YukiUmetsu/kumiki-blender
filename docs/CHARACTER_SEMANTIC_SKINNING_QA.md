# Semantic skinning quality gate: prevent cross-region bone leakage

This adds a conservative, explicit **skin-weight isolation check** to the existing `kumiki-character-build validate-rig` pose battery. It addresses a common source of strange character animation: well-normalized weights can still be **anatomically wrong**. For example, moving Quill's spine should not drag vertices deep inside a wing, and Pip's tail should not receive unrelated head weights.

## Why this is distinct from existing QA

The previous rig check detects zero weights, more than four deform influences per vertex, and non-normalized totals. A vertex with `upper_wing.L=0.9` plus `spine=0.1` is normalized, but may produce incorrect motion. This addition verifies an explicit mapping from each user-authored **semantic region vertex group** to an allowed set of existing deform bones. It **never** automatically repaints weights or presumes anatomy based on object names.

## Blender preparation

1. On the original skinned character, paint a **non-deforming** vertex group marking an anatomical zone, e.g. `REGION_WingLeft` (or `REGION_Tail`). This group is a marker only: it must not match a deform bone.
2. Give included vertices a membership weight of at least **0.5**. Other vertices are not audited under that zone.
3. Decide which deform bones are legitimately allowed to influence those vertices. Include adjacent articulating bones around joint boundaries. Do not choose an unnecessarily restrictive set that would reject intentional deformation.
4. Save the revised `.blend`, then run a new pose QA revision with `weight_zones` in the pose plan.

Add this optional key to the existing `schema_version: 1` pose plan:

```json
"weight_zones": [
  {
    "mesh": "Wing.L",
    "region_group": "REGION_WingLeft",
    "allowed_bones": ["upper_wing.L", "fore_wing.L"],
    "max_outside_weight": 0.02,
    "min_vertices": 20
  }
]
```

The `mesh` must be a real mesh bound to the specified armature, `region_group` must exist on it, and every `allowed_bones` entry must be an existing deform bone. `min_vertices` is a hard coverage floor: an empty or poorly marked zone **cannot pass**. `max_outside_weight` applies to the **sum** of all disallowed deform-bone weights per included vertex, not each bone individually. Values from other non-deforming marker groups are excluded from the skin-weight sum.

## Use and evidence

```bash
kumiki-character-build validate-rig \
  --blend /path/to/rigged.blend \
  --plan /path/to/region-aware-pose-qa.json \
  --output .review/quill/rig-pose-with-weight-zones-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

New output in `report.json`: `rest_weight_zones` for each zone, with selected vertex count, required minimum, violation count, maximum disallowed fraction, and up to 12 indexed offending vertices/bones. Any violation marks the report `structural-fail`, which the existing source-locked character benchmark treats as a failed pose gate. The benchmark also exposes `weight_zone_count` and `failed_weight_zones`, and independently refuses a malformed receipt that claims success while its zone data reports a violation. The source `.blend` remains unchanged.

## Acceptance and limitations

- Run native Blender smoke tests including a deliberately weight-leaking wing that retains total weight 1.0; only the semantic audit should detect this particular error.
- Inspect Pip/Quill anatomy and painted region boundaries. Region groups and permitted bones are **artist-authored facts**, not guessed from the model.
- Weight isolation alone cannot certify deformation quality, silhouette, mass, collision behavior or correct rigging. Continue the independently registered source-image checks, articulated pose tests and visual contact-sheet reviews.
- Unmarked vertices are outside this new scope; the existing general skinning checks still cover them.
- For feather fans, gliding membranes, jaw/neck and flexible tails, plan separate explicit zones with allowances for overlapping joint influences. No automatic skin repair is done.
