# Character Production Pipeline v2: Blender-first source-fidelity, sculpting, rigging, and animation

**Status:** Research and implementation proposal; no implementation claimed.
**Research date:** 2026-11-13.
**Target:** local Apple Silicon M4 / 24 GB, Blender 4.5 LTS as stable reference, Python 3.11+ outside Blender. Linux headless CI.
**Scope:** character modeling, rigging, character animation, QA and GLB delivery. Existing photo/monument reconstruction and geometric asset authoring remain intact.

## 0. Decision and rationale

The highest-value work is **not another entire-character generative model**. It is a production-oriented part-aware workflow, anatomy-aware topology, species-appropriate animator rigs, verified deformation, and reliable exports.

Maintain two classes of truth:
- **Reference evidence:** original turnaround images, landmarks, region masks, source hashes, whether a view is verified and consistent with other views.
- **Editable model:** real meshes, UVs, materials, armatures, weights, shape keys, actions, and quality results.

Never improve one whole-character IoU by silently destroying eyes, facial features, wing detail, or motion quality. Do not claim a plausible hidden surface to be source-authenticated. Repainted/AI-completed views are hypotheses, never higher-authority references than originals.

### Verified repository foundations

Read before implementation:
- [docs/CHARACTER_BUILD.md](../CHARACTER_BUILD.md): reference ingest/check/handoff, multi-view Hunyuan shape, fit, placement, project, retopo, joints, rig, part assembly and repaint, with Pip/Quill results.
- [docs/CHARACTER_FIDELITY_QA.md](../CHARACTER_FIDELITY_QA.md) and [docs/CHARACTER_FIDELITY_REVIEW.md](../CHARACTER_FIDELITY_REVIEW.md): per semantic part registered 2D comparisons and limitations.
- [docs/KUMIKI_DEFORMATION_BRIDGE.md](../KUMIKI_DEFORMATION_BRIDGE.md) and [docs/KUMIKI_MOTION_BRIDGE.md](../KUMIKI_MOTION_BRIDGE.md): portable deformation/motion handoff.
- [docs/ARCHITECTURE.md](../ARCHITECTURE.md): immutable revisions, asset provenance, GLB integrity and Blender headless restrictions.
- [docs/BLENDER_ECOSYSTEM.md](../BLENDER_ECOSYSTEM.md): prior ecosystem/licensing review.

Read code:
- `src/kumiki_blender/character_build/{views,checks,handoff,fit,texture,rig,assembly,swap}.py`
- `src/kumiki_blender/character_build/blender_{retopo,rig,assemble,fur}.py`
- `src/kumiki_blender/character_fidelity/{blender_capture,evaluate,advanced_metrics}.py`
- `src/kumiki_blender/{glb_audit,revision_audit,blender_output}.py`
- `tests/test_character_build.py` and `scripts/smoke_character_build_blender.py`.

Current specific shortcomings:
1. `blender_retopo.py`: global QuadriFlow; fallback creates grid quads with no protected deformation loops; Smart UV Project prioritizes speed, not semantic islands.
2. `rig.py`: biped-like mesh cross-section heuristic for limbs/hips, limited wing/tail semantics. `blender_rig.py`: automatic heat weights + nearest-bone fallback and limited one-axis ROM volume checks. No animator IK/FK controls or automated region collision checks.
3. `blender_swap.py` and `blender_assemble.py`: voxel unions can destroy microdetail, UVs, names, and predictable deformation boundaries. Assembly can already preserve separate objects; prefer it for moving parts.
4. `character_fidelity` handles appearance and silhouette in rest pose, not full animated fidelity or semantic motion.
5. The documented Pip tail volume change at 45° and Quill wing/body interference make deformation validation a release blocker, not a cosmetic metric.
6. Hunyuan3D-2mv, TRELLIS.2, ComfyUI repaint, SAM2 and existing texture/mask tools already exist; **do not reimplement or require extra cloud models**.

## 1. Design principles

1. **Blender-native first**: bpy, BMesh, Geometry Nodes, modifiers, constraints, shape keys, rig/animation APIs, and native glTF exporter.
2. **Source-of-truth `.blend`**: retain untouched high-resolution sculpt(s), canonical editable assembly, armature and original materials. A GLB is a delivery representation, not the master.
3. **Non-destructive, region-local changes**: changing an eye or mouth must not remesh a tail or reset a finished rig. Use reference-pinned independent part revisions and tracked parent/attachment constraints.
4. **Semantic geometry**: distinguish eyes vs decals, feathers vs wings, teeth vs muzzle, clothing vs fur, and rig deform bones vs animator controls.
5. **Distinct quality axes**: registered shape/placement, appearance, volumetric anatomy, topology, skinning, posed deformation, visual export parity. Never hide a failure inside a single averaged score.
6. **Mac-local by default**: no CUDA requirement, monthly subscriptions, large mandatory checkpoints, network-dependent runtime steps, or bundled generated assets. Heavy Python libs run in a separate venv/process instead of modifying Blender's Python.
7. **Unattended deterministic runs**: validated machine-readable contracts, immutable outputs, reproducible seeds, bounded memory, resumable stages, explicit degraded/fail statuses, useful error reports.
8. **Strict source licensing**: only MIT or Apache-2.0 software may be added as independent repository dependencies without separate license review. Blender/Rigify are installed native GPL tools; use their public interfaces and do not copy GPL source into MIT modules. Preserve all third-party notices.

## 2. System architecture

```text
Original artwork and parts -- source hashes, view consistency, semantic masks
               |
      CharacterSpec v2 (schema + cameras + semantic hierarchy)
               |
       Region tasks (head / eyes / muzzle / torso / wings / tail / costume)
          |         |              |
       Blender      local Open3D   optional existing 3D generator
       tools         ARAP edit      proposal only
          \         |              /
         per-part mesh + anchors + authored-vs-inferred provenance
               |
      compare-region (original artwork + geometry + colors + placement)
               |
      assemble-character (non-destructive attachment policies)
               |
      retopo-character (joint/face edge-flow guides + UV + bakes)
               |
      rig-character (species profile + weights + controls + correctives)
               |
      validate-character (pose suite, self-collision, warping, animated QA)
               |
      facial / feather / fur + action library (optional but controlled)
               |
      export-actor (editable blend + portable GLB + actions + manifest)
               |
       Kumiki-UI / ContentGen playback and renderer parity
```

Preserve the existing `kumiki-character-build` commands and test fixtures; implement v2 as additive modules/commands. Increment schema version with explicit migration. Do not silently reinterpret old specs.

### 2.1 Proposed semantic contract

Add `src/kumiki_blender/character_build/character_spec_v2.py` with bounded, validated dataclasses or Pydantic in an *optional* extra, retaining no hard runtime dependency. Example:

```json
{
  "schema_version": 2,
  "character_id": "character-example",
  "height_m": 0.35,
  "up_axis": "Z",
  "forward_axis": "-Y",
  "source_ref_manifest": "references/manifest.json",
  "profile": "stylized-biped-with-tail",
  "regions": [
    {
      "id": "eye.L",
      "parent": "head",
      "type": "eyeball",
      "reference_masks": {"front": "masks/front-eye-L.png"},
      "anchor_ids": ["eye.L.center", "eye.L.inner", "eye.L.outer"],
      "attachment": {"mode": "bone-parent", "bone": "head"},
      "geometry_policy": "separate",
      "symmetry": {"partner": "eye.R", "verified": false},
      "deformation_policy": "rigid",
      "detail_critical": true,
      "source_confidence": "verified"
    }
  ]
}
```

Rules:
- Regions have stable IDs, disjoint semantic ownership and explicit non-overlap/occlusion relationships.
- Capture true intrinsic source masks and full-frame registered masks separately; no per-region recentering to improve apparent accuracy.
- Use `source_confidence` values such as `verified`, `estimated`, `unsupported`; never fabricate hidden-view claims.
- Mirror only verified symmetric anatomy, never accessories/paint variations by assumption.
- Per-region geometry policy: `separate`, `joined-nondestructive`, `weld-required`; avoid any destructive voxel-union default.
- A region may have an explicit cage, rest mesh, sculpt hi-res, animation mesh, material assignment, UV set and attach rig module.
- Unique paths, SHA-256 hashes and protected relative paths; no arbitrary expression evaluation or Python snippets.

### 2.2 Revision/artifact layout

```text
.review/characters/<id>/<revision>/
  source-spec.json
  sources-manifest.json
  parts/<region>/source.blend
  parts/<region>/sculpt-high.glb
  parts/<region>/region-report.json
  parts/<region>/overlays/*.png
  assemblies/assembled.blend
  topology/topology.blend
  topology/low.obj
  topology/bakes/*.png
  rig/rigged.blend
  rig/joints.json
  rig/weights-report.json
  rig/qa/pose-*.png
  performance/actions.json
  export/source.blend
  export/character.glb
  export/manifest.json
  export/validator-report.json
  export/parity/*.png
```

Use existing immutable revision exporter where practical. Keep external generators' raw outputs and any large images in gitignored private storage, not in Git. Include source hashes, generator parameters, Blender version, rig preset, external executable versions and optional library licenses in manifests.

## 3. Milestones and engineering work

### M0. Capture a reproducible baseline and test harness — P0, prerequisite

- Record `main` SHA, Blender version and current Pip/Quill candidate results. Run current `check`, `compare`, `appearance`, `retopo`, `rig --render-rom`, existing smoke script.
- Establish a small synthetic permissively licensed owl-like and squirrel-like rig/mesh fixture for CI; keep real character references local/private if their redistribution rights are not explicit.
- Add a `character-benchmark` test runner producing per-stage CPU/RSS/time, vertex counts, region masks, silhouette scores, UV quality, GLB asset/clip inventory and pose screenshots.
- Establish golden cameras (front, both sides, 3/4, rear, overhead), stable render profile, image hash parity. Render at low resolution in CI, high resolution locally.
- **Exit:** the baseline report can be rerun without editing source art; known failures are captured before optimization.

### M1. Semantic region registry + part task orchestration — P0

- Add `character_spec_v2.py`, `regions.py`, `region_plans.py`, tests, and additive CLI:
  - `plan-character --spec ... --output ...`
  - `plan-region --character ... --region eye.L --output ...`
  - `compare-region --character ... --region eye.L --mesh ... --output ...`
- Reuse `character_fidelity` mask reviews, anchor registration and source color/depth diagnostics. Add region geometry roles, dependencies, provenance, occlusion and candidate version tracking.
- Scheduler: independent regions may run in parallel on capable machines, **but default to serial heavy Blender workers on 24 GB M4**; RAM and process caps, resume receipts, deterministic ordering.
- On replacement: invalidate only dependent body attachments, UV/skin derivatives and final export; leave unrelated region products unchanged.
- **Exit:** rebuilding `eye.L` leaves `eye.R`, body, tail and costume geometry SHA hashes unchanged unless explicitly dependent; all regions have traceable source links.

### M2. Native shape refinement and high-detail region authoring — P0

- New `blender_region_sculpt.py` supporting bounded landmark-driven changes using Blender proportional editing/BMesh, Shrinkwrap, Lattice, Multires where suitable, and face/eye/appendage-specific construction.
- Add optional `open3d` extra in an external Python worker for **ARAP** editing with constrained anchor sets; preserve untouched region boundaries, check triangle inversion and displacement falloff. Run CPU and exchange PLY/OBJ plus contract JSON; do not import large Open3D runtime into embedded Blender Python.
- Define repeatable anatomy templates (not guaranteed anatomical inference): spherical eye volume + separate eyelids, muzzle/cheek surface controls, ear thickness, wing primary/secondary layering, finger/paw digits, tail cross sections. Parameterize dimensions from approved landmarks, not fixed chibi defaults.
- Distinguish two modes: `reference-faithful` (no unsupported invention) vs `cinematic-completion` (hypothesized hidden surfaces with provenance).
- Support explicit local artist override and resumed refinement; no hidden destructive remeshing or false perfection status.
- **Exit:** at least three distinct part edits improve the target region's registered alignment without changing unrelated source-derived features, material colors or attachment transforms.

### M3. Non-destructive assembly and seam engineering — P0

- Extend `assembly.py` and `blender_assemble.py`: stable transforms from >=3 non-collinear anchors, seam compatibility, attachment to deform or rigid bones, explicit mesh-material provenance.
- Prefer separate meshes for eyeballs, glasses, buttons and feather layers. For welded skin, add guided seam topology / boundary ring matching, matched UVs and normal reconciliation.
- Optional `manifold3d` (Apache-2.0) solid booleans only on approved static/watertight parts; not the default for joint-adjacent surfaces or texture-bearing facial regions.
- Do not use voxel union for deforming parts without measuring and explicitly accepting lost topology/UV detail.
- **Exit:** rerunning one part replacement preserves other part hashes, named meshes, materials, UV maps, attachment IDs and local rig relationships.

### M4. Deformation-aware retopology and texture protection — P0

- Maintain QuadriFlow as a coarse option but **do not call its output animation-ready solely because it consists of quads**.
- Add `topology_guides.py` and `blender_character_topology.py`. Region/landmark-guided loop targets: eyes, mouth/jaw, neck, shoulders, wing hinges, hips, knees, ankles and tail root. Produce guide cages/patches that are bridged and shrinkwrapped to high-res meshes.
- Add edge-flow QA: edge/quad counts, extraordinary poles near hinge paths, stretched/skinny faces, seam boundaries, UV distortion, nonmanifold/intersections, displacement-to-high-res distances.
- Native Blender Unwrap with explicit seams should be first choice; keep Smart UV for non-critical props; evaluate optional MIT `xatlas` on a fixture if native unwrap repeatedly fails.
- Add local normal/albedo bake comparison that rejects lost eyes, clothing edges, facial painting and feather borders.
- Freeze topology before facial shape keys; version topology and invalidate morph/weight outputs when it changes.
- **Exit:** loop guides exist for all prescribed deforming joints; source camera silhouette/albedo do not regress beyond calibrated tolerance; retopologized model passes deterministic deform tests.

### M5. Species-specific rigs and weight systems — P0

- Add `rig_profiles.py` and `blender_character_rig_v2.py`. Built-in profiles:
  - stylized biped + tail (Pip-like);
  - bird/biped + wings, layered feathers (Quill-like);
  - humanoid Rigify backend when compatible;
  - quadruped Rigify backend when compatible.
- Blender-native mechanics: deform bones separate from control bones, IK/FK and pole targets for limbs, eye aim, feet contact, Bendy Bones / Spline IK for flexible tails, limited stretch and clean root motion.
- Use **Rigify** as a native Blender-installed adapter for supported body plans, not as a dependency whose GPL implementation is copied to the MIT repository. Always produce an export-only deform skeleton and bake constraint-driven motions before GLB export.
- Skinning: auto heat weights first; semantic bone-region exclusion/allowed masks; distance/geodesic-based fallback; normalization and <=4 influences/vertex at the GLB boundary. Never hide broad heat-binding failures with nearest-bone weights.
- Enable Armature Preserve Volume where appropriate, and region-gated Corrective Smooth and driven corrective shape keys where necessary. Those are techniques to test, not a promise of perfect deformation.
- **Exit:** no unweighted deforming vertices, no empty required deform bones, consistent foot/wing/tail movement, and no unintended body movement when moving a local control.

### M6. Deformation and animated-fidelity validation — P0, release gate

- Add `blender_character_pose_qa.py` and `character_build/deformation_metrics.py`, additive `validate-character --blend ... --plan ... --output ...`.
- Pose battery: head turns, both wing/arm bends, elbows/knees, tail flex and twist, walking extremities, facial blinks, and simultaneous compound poses. Samples should be bounded and deterministic rather than a combinatorial explosion.
- Checks:
  1. Mesh connectivity, numerical stability, zero-length bones, missing/invalid skin weights and violated influence limits.
  2. Triangle flips, local area/stretch ratio, edge-length distortion, excessive shear at deformation bands.
  3. BVH interpenetration for unrelated regions (wing/torso, glasses/face, feet/ground), with explicitly whitelisted intended contacts.
  4. Volume-change check **only for watertight, meaningful regions**; report self-intersection instead of treating it as valid volume conservation.
  5. Reference-camera per-part silhouette/placement preservation in compatible poses. Never compare a different animation pose as a registered neutral source.
  6. Normal/highlight, albedo and UV/texture pinching at the same camera, pose and lighting.
- Output per-bone/region heatmaps, worst-pose contact sheets, machine-readable status `pass | review | fail`, and a region-specific prioritized repair report.
- Proposed preliminary gates (calibrate on baseline; avoid false accuracy claims): no NaN, 0 missing required deform-bone influences, <=4 weights after export normalization, no new inverted faces in critical regions, no new non-whitelisted collisions, no high-fidelity region regressions >1 percentage point in verified silhouette IoU. Surface strain and volume limits are report-only until tuned to real rigs.
- **Exit:** existing Pip-tail and Quill-wing failure cases are identified automatically and repaired outcomes can be scored by the same unmodified fixtures.

### M7. Facial expressions, lip sync, gaze and secondary anatomy — P1

- Native shape-key/morph specification for mouth open/closed, smile, frown, brow, eyeL/R blink, eye gaze, cheek and muzzle deformations. Use actual geometry where required, not texture-only animation.
- Add `facial_profile.py`, `blender_character_face.py` and `character_build/visemes.py`. Where jaw and eyeballs are distinct, let bones/objects move; for soft areas use shape keys and drivers.
- Use MIT [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) as an **optional offline viseme-cue adapter**. It was designed for 2D mouth poses: map cues to actual Blender shape-key/bone animations, smooth timing, and maintain manual override. English acoustic recognizer vs language-independent phonetic recognizer have different precision; test on Japanese rather than assuming supported phonemes.
- Keep audio and cue processing separate from Blender; import validated timestamp JSON, bake to NLA Actions and sample after export.
- **Exit:** a character can blink, change gaze, express two emotions and visibly speak a test audio line without breaking facial geometry/texture; exported GLB reproduces cues in a reference player.

### M8. Feathers, fur and clothing secondary motion — P1

- Extend existing `blender_fur.py` and scatter tooling, do not replace it. Native Geometry Nodes hair guides, root masks and curves-on-surface for editable Blender grooming.
- For birds: explicit primary/secondary feather instances with tapered 3D meshes or cards, socket anchors, layered orientation, compatible folding rigs; never fake a wing as one featureless plate.
- For squirrels/furry species: groom region masks, root/tip color gradients, fur density/direction presets. Keep low-complexity baked cards/meshes for portable GLB and source-quality strands in Blender.
- Secondary animation via keyframes/rigging or separately validated bake; physics modifiers remain optional. No browser hair simulator requirement.
- **Exit:** feathers/hair move with the rig, no obvious unsupported clipping in pose battery, and portable export retains approved appearance under a documented vertex/material budget.

### M9. Animation library and production exports — P1

- Define canonical `actions.json`: `idle`, `blink`, `look-at`, `wave`, `walk`, `turn`, `talk`, `wing-fold`, `tail-swish`, with capability-based opt-in per species.
- Generate/actions/bake in native Blender NLA. Track root motion, fps, duration, action names, target rig profile, required bones/shape keys and loopability.
- Export authoring `source.blend`; portable skinned+morphphed `character.glb`; all audited metadata and preview renders. Bake control-rig constraints to supported glTF pose-bone, object-transform and shape-key animation channels.
- Reuse existing Khronos validator in CI and local `glb_audit`; add animated import/probe to check skins, joint hierarchy, morph names, action names and sampled pose transforms. Keep a Three.js/Kumiki frame-parity test.
- Optional glTF-Transform / gltfpack optimization is a **separate derived artifact**, not a destructive rewrite; explicitly check semantic names, NLA clips, morph targets, materials, skins and screenshot parity before accepting it.
- **Exit:** Blender and Kumiki-UI produce visually consistent agreed camera+pose frames for representative animated clips, and asset identity survives export/import.

### M10. Experimental assistive models and automation — P2; only after core gates

- **Local-first:** benchmark MIT/Apache tools that run on Mac CPU/MPS, provided their exact weights have suitable commercial rights and memory usage fits 24 GB.
- **CUDA-only, optional external workers:** MIT [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens), [UniRig](https://github.com/VAST-AI-Research/UniRig), [PartCrafter](https://github.com/bzi33/partcrafter). Their public published setups require NVIDIA/CUDA; do not pretend they run locally on this Mac without porting work.
- **Human motion reference, experimental:** Apache-2.0 [VideoIntoBlenderModel](https://github.com/s0ck3t/VideoIntoBlenderModel) or MIT [vid2anim](https://github.com/mehulcode12/vid2anim). Test rights, maturity, macOS support and results on a permissive humanoid fixture; not a generic Pip/Quill rig solver.
- Optional Apache-2.0 MediaPipe may suggest landmarks or mocap on humans, but must not infer owl/squirrel anatomy as if they were people.
- Models generate **candidates** only. Feed candidates through existing placement/fidelity metrics plus v2 animated QA. Never automatically replace approved high-detail regions based on a whole-body score.

## 4. Dependency and licensing decision record

| Tool | License | Default | Intended function | Key constraint |
|---|---|---|---|---|
| Blender 4.5 LTS / bundled Rigify | Native Blender GPL distribution | yes (Blender), conditional Rigify | geometry, rigging, controls, animation, export | Do not vendor/copy add-on source into MIT modules |
| trimesh | MIT | optional QA extra, adopt early | closedness, components, geometry repair, audit | Pin version; cross-check Blender import/export semantics |
| manifold3d | Apache-2.0 | optional | robust operations on approved solids | No claim of UV/skin/shape-key preservation |
| Open3D | MIT | optional external worker | constrained ARAP mesh edits | Verify Mac ARM wheel, memory, triangle mapping and anchoring |
| xatlas | MIT | optional | UV atlas after topology approval | Prefer Blender seam-aware unwrap; UV atlas may duplicate verts |
| Rhubarb Lip Sync | MIT | optional | local viseme timestamps | 2D cue tool, not a 3D face rig; English vs other languages differ |
| glTF-Transform | MIT | existing tooling | validate/inspect derived GLB | Preserve names, metadata, skins, actions, morphs |
| glTF-Validator | Apache-2.0 | existing tooling | spec QA | Does not certify artistic quality |
| meshoptimizer/gltfpack | MIT | optional separate export | minimize GLB cost | May prune/merge nodes; preserve `extras` and morph names |
| GroomLab | Apache-2.0 | defer | advanced grooming | Released Blender integration not proven Mac-local |
| Rigger / CurioMesh | GPL-3.0 | no repository dependency | conceptual comparisons only | Separate license review needed; not MIT/Apache |
| SkinTokens / UniRig | MIT | defer | AI skeleton + weights | Published NVIDIA/CUDA inference |
| PartCrafter | MIT | defer | proposed decomposed meshes | Published CUDA-first workflow |

License audit on every adopted dependency: exact repo license text and version, NOTICE if required, transitive binaries/weights/assets, independent runtime vs copied source. Do not assume a GitHub license badge also licenses trained checkpoints or bundled examples.

Links for verification:
- [Blender Rigify basic usage](https://docs.blender.org/manual/en/4.5/addons/rigging/rigify/basics.html)
- [Armature Preserve Volume](https://docs.blender.org/manual/en/4.5/modeling/modifiers/deform/armature.html)
- [Corrective Smooth](https://docs.blender.org/manual/en/4.5/modeling/modifiers/deform/corrective_smooth.html)
- [Shape Keys](https://docs.blender.org/manual/en/4.5/animation/shape_keys/introduction.html)
- [Spline IK](https://docs.blender.org/manual/en/4.5/animation/armatures/posing/bone_constraints/inverse_kinematics/spline_ik.html)
- [Open3D ARAP](https://www.open3d.org/docs/release/tutorial/geometry/mesh_deformation.html)
- [Open3D macOS ARM support](https://www.open3d.org/docs/release/arm.html)
- [Rhubarb license](https://github.com/DanielSWolf/rhubarb-lip-sync/blob/master/LICENSE.md)
- [glTF exporter animation constraints](https://github.com/KhronosGroup/glTF-Blender-IO/blob/main/docs/blender_docs/scene_gltf2.rst)

## 5. Testing, quality gates and performance

### Tests by level

**Unit (fast, default CI):** v2 schema invalid fields/unsafe paths/cycles; region dependency invalidation; camera/anchor math; line/contour and edge-flow metrics; deform weight checks; scene manifest and GLB animation envelope assertions.

**Blender Linux headless (smoke, limited CI):** small procedural biped and bird fixtures; scripted eye edit; 2+ bones and skin; one wing/tail fold pose; shape keys and drivers; rig control bake; exported GLB validator; do **not** run full expensive reconstruction CI.

**Mac M4 local (extended acceptance):** real Pip and Quill private source manifests and candidate meshes; body-part changes, topology and UV bakes, animation battery with renders; 24 GB memory probe and peak-RSS report; repeat after Blender upgrades.

**Browser parity (separate optional nightly/manual check):** glTF imported by Kumiki-UI with identical pose/frame/camera; compare silhouettes and sampled transforms; reference 3D rendering may differ slightly due to lighting/tonemapping.

### Blocking rules

- **Fail** if source contract broken, unverified registered camera silently applied, missing critical region, invalid topology, empty or unweighted required deform bones, broken GLB, wrong action/skin/morph inventory, nonfinite metrics, destructive source overwrite.
- **Review** if source views contradict one another, supported geometry is incomplete, an estimated hidden region is unverified, or pose/appearance measures outside calibrated bands.
- **Pass** only when structural and selected quantitative gates succeed, exports verify, and explicit character/region evidence is not contradicted. High-fidelity artistic approval remains separate from machine validation.

### Resource budget and safe execution

- Default one heavy Blender process at a time on a 24 GB Mac, with local subprocess timeouts and checkpointed stage outputs.
- First run low-resolution QA; sculpt/texture 4K only when source supports it and quality probes justify it. Preserve real geometric detail regardless of smoke render profile.
- Keep Blender's embedded Python small; Open3D, ML inference, external texture workers and converters in isolated local environments.
- Never download weights automatically or enable network access unless specifically configured.
- On stage failure: write diagnostic logs and untouched input refs; no fallback that fabricates a passing full-body mesh.
- Tune budgets using measurements from M0, not assumed fixed elapsed time or fabricated performance claims.

## 6. Proposed file-level implementation map

| New module | Responsibility | Existing integration |
|---|---|---|
| `character_build/character_spec_v2.py` | versioned semantic source/mesh/rig contract | `views.py`, `checks.py`, `handoff.py` |
| `character_build/regions.py` | region ownership, attachments, dependency graph | `assembly.py`, `character_fidelity` |
| `character_build/region_plans.py` | deterministic region revisions / candidate orchestration | `__main__.py`, `swap.py` |
| `character_build/blender_region_sculpt.py` | native local detail modeling | existing `blender_common.py` |
| `character_build/arap_worker.py` | optional Open3D external ARAP | `region_plans.py` |
| `character_build/topology_guides.py` | joint, face, seam geometry constraints | `blender_retopo.py` |
| `character_build/blender_character_topology.py` | quad loops, UV, bake + audits | `blender_retopo.py`, `blender_common.py` |
| `character_build/rig_profiles.py` | species presets / native Rigify selection | `rig.py` |
| `character_build/blender_character_rig_v2.py` | controls, binding, correctives, deform skeleton | `blender_rig.py` |
| `character_build/deformation_metrics.py` | deformation and weight diagnostics | `character_fidelity` |
| `character_build/blender_character_pose_qa.py` | bounded pose battery with renders | `blender_rig.py`, `blender_output.py` |
| `character_build/facial_profile.py` | facial controls, morph contract | `rig_profiles.py` |
| `character_build/visemes.py` | offline cue → morph/bone action | existing animation bridge |
| `character_build/character_export.py` | manifest and GLB animated verification | `glb_audit.py`, `revision_audit.py` |

Avoid a second giant orchestration framework; the existing CLI/workers, manifests and revision practices should be extended.

## 7. Delivery order, integration slices and agent assignment

Dependencies: **M0 → M1 → M2+M3 → M4 → M5 → M6**. M7/M8 can begin after M5, M9 after M6. M10 last.

1. **PR-A Baseline & semantic specs:** M0, M1. Most suitable for a strong architecture/modeling agent; support agents can write schemas, unit tests and fixture documentation.
2. **PR-B Local anatomy / ARAP / independent part edits:** M2, M3. High-intelligence 3D agent for source-accurate eye/face/wing geometry; ordinary agents for adapter plumbing and provenance tests.
3. **PR-C Topology:** M4. Specialist geometry/rigging agent; cannot delegate edge-flow artistic criteria to a generic code-only agent.
4. **PR-D Rig and pose QA:** M5, M6. Specialist rigging agent, separate QA agent that tests real pose data and rejects self-evaluated claims.
5. **PR-E Facial/feather/animation exports:** M7–M9. Split facial controls, procedural grooming, audio cue adapters and animated glTF audits into independent PRs.
6. **PR-F Optional learned proposals:** M10 only when native rig/QA pipeline works, budget permits, and model license+runtime requirements are verified.

Always preserve tests in the repository. Keep expensive Blender/Cycles and browser render suites outside routine PR CI unless a change directly touches that subsystem; unit tests plus small Blender smoke and glTF validator remain required. Do not reduce quality criteria merely to save CI credits.

## 8. First implementation task (highest ROI)

Create PR-A only:
- Introduce schema v2 and stable semantic region IDs.
- Add `plan-character`, `plan-region`, `compare-region` as additive commands.
- Reuse existing per-part fidelity metrics and immutable artifact layout.
- Demonstrate a single eye-only change with registered source masks and protected untouched regions on a synthetic test fixture.
- Add versioned tests and a baseline pose/asset manifest (without committing private Pip/Quill images).
- Report concrete measurements. No claims about production-quality 3D artistry until independent Blender renders and deformation tests pass.

**Success criterion:** an automated agent can isolate, rebuild and measure one character part without restarting the entire body generation or worsening any unrelated part. This is the foundation that makes subsequent detailed modeling, rigging and batch character production economical.
