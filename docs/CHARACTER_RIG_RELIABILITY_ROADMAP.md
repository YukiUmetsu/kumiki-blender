# Source-faithful, animation-ready characters: implementation roadmap

**Goal:** Reproducible original-art fidelity and usable Blender .blend, skinned mesh, animator controls, facial motion, and portable GLB animation. Improve one anatomical region without inadvertently replacing other approved work. The character's source art, not a plausible generic species template, is the visual authority.

## Status and order of work

| Priority | Capability | State / acceptance evidence |
|---|---|---|
| P0 | M0 evidence baseline and M1 source-locked region planning | Implemented as standalone tools; real Pip/Quill golden camera and performance evidence still needed |
| P0 | **Rig deformation QA** | **This PR:** optional enforced nonzero evaluated mesh motion, neutral-relative contact counts, immutable source hash; existing weight/stretch checks remain. Does not itself fix skinning |
| P0 | M2 source-guided anatomical mesh refinement | Next: eye, muzzle and wing/tail region fixture revisions, measured against verified original source views and protected neighboring parts |
| P0 | M3 non-destructive assembly | Stable anchors and seams; do not voxel-merge deforming or textured pieces by default |
| P0 | M4 deformation-aware topology and UVs | Per-hinge edge loops and registered color/detail bake regression, plus edge-flow QA |
| P0 | M5 species-specific Blender rigs | Squirrel forelimbs, paws, gliding membrane and tail; owl wings, feather fans, feet, head, glasses. Native IK/FK and skin weights; deform-only export skeleton |
| P0 | M6 full validation | Expand this first gate with structural mesh checks, whitelistable collision semantics, evaluated pose/frame sequence, contact sheets and real Pip/Quill regression |
| P1 | M7 performances | Explicit blink, gaze, mouth/viseme, gesture and locomotion actions with actual facial mesh shapes |
| P1 | M8 delivery parity | Blender-to-GLB-to-Kumiki-UI motion/visual parity; versioned artifacts, export manifests, deterministic QA |
| P1 | M9 batch production | Region dependency/cache reuse, bounded Mac M4 memory, auto-resume and stop-on-regression policies |

This builds on [the existing detailed production plan](superpowers/plans/2026-11-13-character-production-pipeline-v2.md), not a competing pipeline. Current bake/export facilities and source registration already exist; extend them rather than introducing a new rig or animation engine.

## This slice: rig pose regression gate

The v1 pose QA schema remains valid. Optionally add:

```json
{
  "min_pose_displacement_m": 0.0005,
  "max_new_collision_pairs": 0
}
```

Use these keys *alongside* the existing `armature`, `poses`, `collision_pairs`, `max_p95_edge_stretch` and `render` fields. `max_new_collision_pairs` requires at least one explicit `collision_pairs` entry. The gate evaluates triangle intersections between named meshes at a neutral pose and each requested pose, then counts newly intersecting triangle **pairs**. It can distinguish a pre-existing decorative joint overlap from a new wing/body crossing, but it **cannot** infer actual penetration depth or intended contacts. The operator must choose collision pairs that should not newly intersect.

`min_pose_displacement_m` rejects dead/ineffective requested pose tests that move none of the evaluated mesh vertices (for example, a controller with no skin influence or a constraint overriding the pose). Measurements use unregistered world-space mesh coordinates; root movement also counts. No single metric certifies a well-skinned character. Apply the limit only to meaningful deforming test poses, and keep reviewing rendered poses.

## Local acceptance sequence

1. Record source SHA-256 and generate a fresh `validate-rig` report on a saved Blender rig using `examples/character-rig-pose-qa.example.json`.
2. Test a controlled joint that visibly moves and a deliberately disconnected bone. Confirm the disconnected pose **fails**, not that the pipeline creates a misleading success receipt.
3. Capture original neutral overlaps; pose wing across torso and confirm **new** intersections fail when the explicit contact limit is zero.
4. Check existing nonzero overlaps are reported for review but do not become false *new* contacts.
5. Run `character-benchmark` with the pose receipt, reviewed reference-region reports, texture gates and animated GLB. Exit success still means artistic review required.
6. For Pip and Quill, capture front, side, three-quarter and rear reference-aligned renders plus bend extremes and animation clips. Do **not** mark production-ready until real source fidelity and movement are reviewed.

Tests: `python -m unittest tests.test_character_deformation_metrics tests.test_character_pose_spec`. Blender smoke and character-specific GitHub checks are separate.

**Known limits:** This is a reliability gate, not an automatic high-quality 3D model generator. It does not implement M2–M9, prove accurate hidden anatomy, detect all self-intersections inside a single mesh, or evaluate physical collision volume.
