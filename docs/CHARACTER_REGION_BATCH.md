# M1: Incremental region batch planning and safe resume

This is the **dependency orchestration and provenance** portion of milestone M1
from the original [Character Production Pipeline v2](superpowers/plans/2026-11-13-character-production-pipeline-v2.md).
It builds on the existing frozen CharacterSpec v2 and single-region plans.

**Goal:** Reworking a single eye must not rebuild every squirrel or owl
body part. Its attached eyelid must be invalidated, while the opposite
eye, tail, and unrelated costume geometry remain on their reviewed revisions.

## Create a deterministic batch

~~~bash
kumiki-character-build plan-region-batch \
  --character .review/pip/character-plan-r01 \
  --changed eye.L \
  --output .review/pip/eye-left-batch-r01
~~~

Multiple --changed flags are allowed. The planner reads the validated
CharacterSpec dependency DAG and produces **region-batch-plan.json** with:

- Original CharacterSpec source hash and each region input hash.
- Changed roots, downstream invalidations, topological build order.
- Explicit scheduled dependencies versus preserved upstream parents.
- The exact list of unaffected regions to keep unchanged.
- Proposed deterministic execution waves and a fixed batch SHA-256.

By default at most **one heavy Blender worker** is planned for a 24 GiB host.
The conservative model reserves 4 GiB for the host and **14 GiB per**
simultaneous heavy Blender worker. If you explicitly request two workers:

~~~bash
kumiki-character-build plan-region-batch \
  --character .review/pip/character-plan-r01 \
  --changed eye.L --changed eye.R \
  --max-workers 2 --memory-gib 32 \
  --output .review/pip/parallel-eyes-batch-r01
~~~

The resource budget is a safety **planning cap**, not measured peak RSS,
a process launcher, or a promise of available memory. On M4/24 GiB, the
default remains serial. No parts are built or approved by this command.

## Resume without accepting stale geometry

Before a changed eye is built, a preserved head attachment must have an
explicit stable artifact hash. Supply a JSON **receipts.json** file:

~~~json
{
  "schema_version": 1,
  "batch_sha256": "replace-with-the-plan-batch-sha256",
  "preserved": {
    "head": {
      "artifact": "approved-head.blend",
      "sha256": "replace-with-exact-original-sha256"
    }
  },
  "completed": []
}
~~~

Receipts are resolved relative to the receipts file (absolute paths work).
An initial audit without receipts is allowed; it reports exactly which
preserved parent artifacts are missing and **does not invent readiness**.

~~~bash
kumiki-character-build resume-region-batch \
  --plan .review/pip/eye-left-batch-r01/region-batch-plan.json \
  --receipts .review/pip/receipts-r01.json \
  --output .review/pip/queue-status-r01
~~~

Once eye.L is built and independently reviewed, add a completion record
pointing to its derived Blender/OBJ artifact and its SHA-256, including
the exact upstream head revision:

~~~json
{
  "region": "eye.L",
  "artifact": "reviewed-eye-left.blend",
  "sha256": "replace-with-new-eye-artifact-sha256",
  "dependency_sha256": {
    "head": "replace-with-approved-head-sha256"
  }
}
~~~

Then create a **new receipts file revision** and run resume-region-batch
into a **new status output directory**. The next eligible task, lid.L,
will become ready only after the reviewed eye artifact matches its
declared SHA. Its completion receipt must bind the eye.L SHA as an
upstream dependency.

This prevents a later eye or head change from silently reusing an old
eyelid. Task completion receipts may be stored in any order; the plan
verifies them in source dependency order.

## Outputs and exit codes

**region-batch-plan.json**: immutable source-locked task list and maximum
safe concurrency. **batch-status.json**: completed parts, verified original
parent receipts, pending tasks, next eligible Blender tasks and missing
upstream evidence.

Plan exit 0 means a valid plan was written. Resume exit 0 means all tasks
have artifact receipts, **not** that artistic or animation QA passed.
Resume exit 1 means tasks remain pending or upstream evidence is missing.
Changed original art, tampered plan/receipt, invalid artifact SHA or
unsafe memory configuration returns an error without a completed status.

This is intentionally a **planner/resume auditor only**, not a hidden
unattended agent. The user or a separate supervised worker runs the
existing build-anatomy, refine-region, fit-landmark-surface and Blender
commands, then supplies the resulting approved files and hashes.
After finishing tasks, run the M0 character-benchmark, per-view source
gates and actual posed deformation checks before delivery.

## Remaining M1 work

The future execution engine must enforce live memory/process caps, durable
crash-safe job resumes, independent region workers with bounded wall time,
and dependency-aware cache invalidation of UVs, weights, shape keys and
exported GLBs. This slice **does not** claim those execution controls or
real Pip/Quill likeness have been validated.
