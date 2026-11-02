# Official CC0 anatomical head source

## Intent

Give Kumiki monument authors a genuine human-anatomy starting mesh immediately,
without manually hunting for a suitable GLB for every statue. This follows the
head-fitting foundation in PR #14 and leaves historical likeness review intact.

## Existing constraints

The asset catalog indexes buildable **families**, with no large binary model
assets in git; Blender builds immutable revisions. The head-fit consumer already
accepts a reviewed static GLB and mandatory provenance. Source faces require
front/profile QA, not primitive-generated replacements.

## Selected approach

Add `makehuman-neutral-head` to `characters/people` as an official-asset-backed
family. A small standard-library parser fetches/validates one **pinned** upstream
MakeHuman `base.obj`, whose header and official license state CC0, and isolates
the actual body/face topology. A Blender adapter extracts the head/neck region,
closes its neck cut and emits a reusable editable `source.blend`, `model.glb`,
preview set and source provenance through the common exporter. A local OBJ path
allows reproducible offline builds, with no new pip dependency.

## Alternatives

- Require users to install MPFB and export manually: broad options but prevents
  one-command CI/reproducibility and forces an extra editor workflow.
- Commit the upstream OBJ/GLB to git: easy for consumers, but grows every clone
  and diverges from the catalog's intentional external asset-store architecture.
- Procedurally invent a new head: explicitly contrary to the quality objective.

## Non-goals

- No claim that MakeHuman is a photorealistic statue or a likeness to Liberty,
  Redeemer, or the Sphinx. No new face primitive generation or automatic 2D fit.
- No downloading third-party community clothing/hair/eyes without separate
  license verification. No modification of the monument model branches yet.
- No redistribution of MakeHuman executable/add-on code (AGPL/GPL); only the
  documented CC0 graphical base mesh is consumed.

## Acceptance criteria

- Remote content pinned by upstream commit and SHA-256; wrong/replaced bytes
  rejected **before** Blender. No output directory overwritten.
- Local OBJ and cached path work without network; download only when requested.
- Extracted head contains real nose, eyelids, lips, ears, jaw and neck from the
  source topology, is correctly oriented in Blender Z-up meters, has a closed
  neck cut, and no unrelated full-body/helper geometry.
- Blender exports editable `.blend` and portable GLB with front/profile/hero
  previews, CC0/source hash provenance, and honest artistic-review status.
- Catalog discovery/build, focused Python tests, and actual Blender CI smoke
  prove the production workflow, not merely manifest output.
