# Evidence-Guided 3D Completion (initial integration)

Use this workflow when an existing photo-to-GLB mesh is incomplete and you
have a candidate part from an attributed scan, artist, or generative model.
It **does not itself run image/3D generation**: `supplied-glb` is the first
interchangeable output adapter. Obtain the candidate GLB separately, inspect
its source/model license, and place it alongside the private manifest.

The builder retains the original mesh and aligns additional parts **as separate
editable Blender objects**. No invisible geometry is claimed to be observed;
no automatic seam welding or photorealistic quality certification occurs.

## Two modes

| Mode | Included geometry | Excluded geometry |
| --- | --- | --- |
| `historical` | Pinned, attributed `observed-scan` and `observed-survey` inputs | `generated`, `artist-hypothesis` |
| `cinematic` | Both observed and inferred candidates | None solely by origin, but all must pass geometric checks |

Even `observed-scan` and `observed-survey` are **source declarations**; the
tool checks a syntactically valid HTTPS URL and stated license, but cannot
authenticate an archaeological scan. Human review is still mandatory.
Historical mode never automatically restores a destroyed historical feature.

## 1. Prepare a private manifest

This illustrative structure requires real source GLBs and their exact hashes.
Place `base.glb`, `left-ear.glb`, and `completion.json` in the same private
directory. `base.glb` is the *existing reconstruction*; `left-ear.glb` is a
separate completed part, not a replacement statue model.

```json
{
  "schema_version": 1,
  "asset_id": "great-sphinx",
  "mode": "cinematic",
  "base_origin": "observed",
  "base_glb": "base.glb",
  "base_sha256": "REPLACE_WITH_64_CHARACTER_LOWERCASE_SHA256",
  "regions": [
    {
      "id": "left-ear",
      "candidate_glb": "left-ear.glb",
      "candidate_sha256": "REPLACE_WITH_64_CHARACTER_LOWERCASE_SHA256",
      "origin": "generated",
      "generator": {
        "adapter": "supplied-glb",
        "model_id": "microsoft/TRELLIS.2-4B"
      },
      "max_error_m": 0.02,
      "anchors": [
        {"id": "a", "source_xyz_m": [0, 0, 0], "target_xyz_m": [0, 0, 0]},
        {"id": "b", "source_xyz_m": [1, 0, 0], "target_xyz_m": [1, 0, 0]},
        {"id": "c", "source_xyz_m": [0, 1, 0], "target_xyz_m": [0, 1, 0]},
        {"id": "d", "source_xyz_m": [0, 0, 1], "target_xyz_m": [0, 0, 1]}
      ]
    }
  ]
}
```

**Replace every illustrative coordinate and hash.** Real anchors must name
identical surface features in both parts, with at least four non-coplanar
correspondences. Use Blender **Z-up metres**, including the world coordinates
of any GLB imported with Blender's glTF axis conversion. Coordinates are not
GLB binary glTF Y-up positions. The importer fits one proper rotation,
uniform scale, and translation, rejecting mirror or high residuals. Configure
`max_error_m` for the actual scale/detail being matched.
`base_origin` is always required: use `observed` for a real photo/scan-derived
base and `inferred` for an AI/artist-derived base. Historical mode refuses an
inferred base. A reimported cinematic asset retains its per-mesh inferred tags;
declaring it observed cannot erase those tags or make it historical evidence.

For an attributed scan/survey, change the region to:

```json
{
  "origin": "observed-scan",
  "source_url": "https://example.org/actual-scan-source",
  "license": "CC0-1.0"
}
```

These three lines replace the `origin` and `generator` fields in the example;
the `source_url` must identify a real source when used, not the sample URL.
Other valid origin values: `observed-survey`, `artist-hypothesis`, `generated`.
Generated candidates must record `generator.adapter` and `generator.model_id`.
The adapter/model string is **provenance metadata**, not code that gets
downloaded or executed. The tool does not infer commercial reuse rights from
a generator name, license label, or source URL.

Hashes on macOS:

```bash
shasum -a 256 base.glb left-ear.glb
```

The manifest and files stay private; do not commit third-party photos,
unlicensed candidate geometry or private generated models to Git.

## 2. Plan and build

```bash
kumiki-blender plan-completion --manifest /private/statue/completion.json

# Requires installed Blender 4.2+ and a NEW revision directory:
kumiki-blender build-completion \
  --manifest /private/statue/completion.json \
  --output .review/great-sphinx/completion-r01
```

The planning command is read-only. Its JSON report includes inclusion/exclusion
for each region, source SHA-256, transform, per-landmark error, worst residual,
and an explicit review-required verdict. Planning rejects invalid paths,
symlinks that escape the manifest directory, non-GLB/animated sources, hash
mismatches, nonfinite data, ambiguous landmark geometry, and residuals beyond
your declared limit. Blender builds refuse existing destinations; reruns must
use `r02`, `r03`, etc.

A successful Blender build writes:

- `source.blend`: original base mesh plus independent candidate objects.
- `model.glb`: portable combined scene geometry, not automatically welded.
- `preview.png` and `previews/*.png`: hero, front, profile, rear and top views.
- `manifest.json`: per-region inclusion, measured anchor fit, candidate
  provenance and `kumiki_geometry_origin` tags (`observed` or `inferred`).

## 3. Review before approving

1. Verify source attribution and rights; a URL is not proof of authenticity.
2. In Blender, inspect the seams/overlap and surface continuity between the
   base and imported parts; the build has not fused them.
3. Compare the original photographs with actual-camera renders using
   [reference reconstruction](REFERENCE_RECONSTRUCTION.md),
   [silhouette QA and landmark QA](REFERENCE_RECONSTRUCTION.md#measure-region-specific-facial-hair-and-drapery-likeness).
   Synthetic novel views may inspire candidates but **must not** count as
   independent real-camera measurements.
4. Only after manual review should an artist perform non-destructive blending,
   retopology, texture finishing and publication-ready asset approval.

## Next integrations

- Add isolated generator runner adapters (initial candidates: SPAR3D on CPU/
  tested MPS, multiview generation, CUDA-hosted TRELLIS.2) only after licensing,
  memory and platform smoke tests. Each runner would emit a pinned GLB and
  record its exact model checkpoint/seed/source photos.
- Automate low-confidence region suggestions from visibility and surface
  coverage reports, then provide landmark editing/preview UI.
- Add true geometry boundary fitting and reviewed seam blending, followed by
  real-photo reprojection gates before artifact approval.

These future phases are not implemented by `build-completion` today.
