# M0: Reproducible character evidence benchmark

This implements the missing **M0 benchmark aggregator** from the original
[Character Production Pipeline v2 roadmap](superpowers/plans/2026-11-13-character-production-pipeline-v2.md).

It collects evidence from the existing character tools into a source-locked,
repeatable report. It does **not** generate a character, re-run Blender,
download AI weights, or claim synthetic geometry proves Pip/Quill likeness.

## Prerequisites

Create a CharacterSpec v2 plan using **plan-character**.
Produce immutable region reports using **audit-region-revision**, color reports
using **audit-texture-revision**, pose reports using **validate-rig**, and a
real portable animated GLB using the performance export workflow.

**Important:** validate-rig now records source_blend and source_blend_sha256.
Re-run that worker on the *saved* Blender rig revision before benchmarking.
Older pose reports without its original .blend fingerprint are refused.

## Manifest

Create a JSON manifest (schema_version 1). All paths are relative to the
manifest file unless explicitly absolute. A complete example lives at
[examples/character-benchmark.example.json](../examples/character-benchmark.example.json).

~~~json
{
  "schema_version": 1,
  "character_plan": "qa/character-plan-r01",
  "required_checks": ["regions", "textures", "pose", "glb"],
  "required_regions": ["eye.L", "eye.R", "wing.L", "wing.R"],
  "required_poses": ["head_turn", "wing_fold_L", "wing_fold_R"],
  "required_clips": ["idle", "talk", "wing_fold"],
  "region_reports": [
    {"region": "eye.L", "report": "qa/eye-L-r02/report.json"},
    {"region": "eye.R", "report": "qa/eye-R-r02/report.json"}
  ],
  "texture_reports": [
    {
      "id": "basecolor",
      "report": "qa/texture-r02/report.json",
      "inputs": {
        "handoff_spec": "handoff/spec.json",
        "mesh_obj": "models/character.source.obj",
        "baseline_texture": "textures/basecolor-r01.png",
        "candidate_texture": "textures/basecolor-r02.png",
        "aligned_front": "handoff/aligned/front.png"
      }
    }
  ],
  "pose_report": {
    "report": "qa/rig-poses-r02/report.json",
    "blend": "models/character-rig-r02.blend"
  },
  "glb": "export/character-r02.glb",
  "runtime_observations": []
}
~~~

This is **an illustration, not a measured Pip/Quill result**.
For each texture report, supply **every** key in the original input_sha256
mapping, including further aligned source-camera PNGs. The collector hashes
these files and rejects stale art, geometry, UVs and textures. Matching
camera names alone do not independently prove texture camera alignment.

For an incomplete first run, omit missing reports, and the resulting report
will list required missing evidence explicitly. Unless limited in the
manifest, all four QA categories are required.

## Command

~~~bash
kumiki-character-build character-benchmark \
  --manifest local/character-benchmark.json \
  --output local/benchmarks/pip-r001
~~~

A new output contains **report.json** with source SHA-256, verified region
coverage, silhouette/contour regression summaries, texture color checks,
pose/skinning details, GLB joint/skin/morph/action inventory, failures and
missing evidence. **SUMMARY.md** is a concise readiness table.

Exit 0 means **numeric-gates-passed-artistic-review-required**. Exit 1
means a required gate failed or its evidence is missing. Invalid source
data, hashes, or provenance return error 2 without a final benchmark.

## Verification boundaries

- Region reports must match CharacterSpec source and per-region hashes,
  the complete verified camera set, and unchanged baseline/candidate mesh.
- Texture reports must match every supplied source file fingerprint.
- Pose reports must match the exact Blender source .blend SHA, list skin
  weights and tested poses, and preserve actual screenshot filenames.
  Collision candidates remain advisory contact diagnostics.
- The GLB is inspected by reading only the JSON chunk, not multi-GB
  model binaries. It inventories skinned joints, meshes, morph targets,
  clips and actual deform-joint/morph-weight animation channels.
- Optional wall/CPU/RSS observations are **externally supplied** and
  always labeled as such, not instrumented by the collector.

This completes one concrete M0 engineering deliverable, but the full M0
acceptance criteria also require real Pip and Quill original source masks,
golden front/side/three-quarter/rear images, reproducible real worker
resource measurements and Blender pose/preview renders. M2–M9 source
likeness and animation playback remain **independent release gates**.
