# Evidence-Guided 3D Completion

Use this workflow when an existing photo-to-GLB mesh is incomplete and you
have a candidate part from an attributed scan, artist, or generative model.
The repo can now **suggest low-coverage regions**, **generate illustrative
missing-region images** through a local ComfyUI workflow, and **run an ungated
image-to-3D model** to create a candidate GLB. You can still supply your own
scan/mesh through the `supplied-glb` adapter. Neither generator fits an ear,
face, or other part onto the original model automatically: measured anchors
are required by the evidence-gated assembly step below.

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

## 0. Identify gaps and generate candidates (optional)

These generation commands produce **inferred** content, suitable only for
`cinematic` mode. Historical mode still accepts only genuine observed scans/
surveys. Nothing generated is silently added to your real-photo SfM database.

### Suggest under-photographed regions

```bash
kumiki-blender reference-audit --manifest /private/sphinx/references.json > /private/sphinx/reference-audit.json
kumiki-blender suggest-completion --audit /private/sphinx/reference-audit.json --min-views 2
```

This identifies absent/single-view **reference annotations** and missing camera
views. It cannot determine whether those parts are missing from the GLB or
prove the photographs have geometric overlap. Verify candidates in Blender.

### Option A: Inpaint a missing view using local ComfyUI

1. Run the locally installed ComfyUI server with Metal support and an
   **already-installed, independently licensed** inpainting checkpoint.
2. Export a ComfyUI workflow in **API format**. It must contain a `LoadImage`
   node and a `SaveImage` node. The `LoadImage` node reads your private PNG;
   its alpha channel may supply an inpainting mask, depending on your graph.
3. Execute (substitute the node IDs used in your API workflow):

```bash
kumiki-blender generate-completion-image \
  --image /private/sphinx/ear-mask.png \
  --workflow /private/sphinx/inpaint-api.json \
  --image-node 2 --output-node 9 \
  --server http://127.0.0.1:8188 --require-mps \
  --output /private/sphinx/image-candidate-r01
```

The adapter communicates **only with localhost** and refuses HTTP redirects.
It accepts a conservative allowlist of built-in local ComfyUI image/inpainting
nodes, blocking custom and cloud-partner nodes. It submits the supplied graph,
waits for one PNG, and records exact input,
workflow, and output SHA-256 digests. If `--require-mps` is supplied, it fails
before image upload unless ComfyUI reports an MPS device. No model is downloaded
by this adapter and there is **no interactive model-approval flow**. Install an
ungated model yourself if the local workflow does not already have one. The
adapter cannot certify the workflow checkpoint's license. ComfyUI may retain
the uploaded file in its local `input/`
directory; delete it there under your normal local data-retention policy.
Do not reuse generated pixels as independent photo evidence.

### Option B: Generate a 3D candidate from an isolated image using TripoSR

Use a **transparent-background segmented PNG/WebP** of a single part, for
example a cropped ear. Your existing SAM2 masks can prepare this input.

```bash
kumiki-blender generate-completion-part \
  --image /private/sphinx/segmented-ear.png \
  --device mps --resolution 128 --chunk-size 2048 \
  --output /private/sphinx/ear-part-r01
```

This uses only the **public, ungated, MIT-licensed** `stabilityai/TripoSR`
checkpoint. On first use, the `huggingface_hub` dependency downloads the public
model without an authentication token, click-through access request, or manual
approval. Use `--offline` to require an already-cached model; use `--revision`
with a 40-character checkpoint repo commit for repeatability. The provenance
report records the actual resolved commit and SHA-256 of `model.ckpt` and
`config.yaml`, alongside the GLB digest and actual compute device.

**Runtime setup:** Install PyTorch, Pillow, NumPy, OmegaConf, `huggingface_hub`,
scikit-image, trimesh and the upstream [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
Python package (`tsr`) in the Python environment that runs `kumiki-blender`.
TripoSR's other upstream dependencies are also required. This optional runtime
is not installed by the base Kumiki package. For Apple Silicon, use a PyTorch
build with `torch.backends.mps.is_available() == True`. Kumiki runs neural
inference on MPS and performs only marching cubes on CPU via scikit-image;
the upstream `run.py` silently forces CPU when CUDA is absent, so **do not use
that script for MPS**. Use `--device auto` to prefer MPS, then CUDA; it refuses
unaccelerated execution unless you explicitly select `--device cpu`.

Run `kumiki-blender completion-doctor` first. It checks the optional imports
and visible MPS/CUDA devices without downloading weights or submitting a job.

CI includes an Apple Silicon MPS tensor and CPU marching-cubes smoke test,
but **full TripoSR inference has not been exercised on the user's M4**; upstream
model operators may still require compatibility fixes. Start at
resolution 128 on a 24 GB Mac and use `--resolution 64` if memory is tight.
The input image stays private; generated outputs are `candidate.glb` and
`provenance.json`, not a completed monument. TripoSR was trained for single-
image general 3D objects, so a historical stone ear/face still needs human
shape, texture, and source-photo comparison.

### Assembly handoff

Copy the `candidate.glb` into your private completion manifest folder. Set
`origin: generated` and `generator.adapter: triposr-local`, with
`generator.model_id: stabilityai/TripoSR` and the candidate SHA-256 from its
provenance report. Then supply four or more measured, noncoplanar source/target
landmarks (in Blender Z-up metres). Use `plan-completion` and
`build-completion` below. An image-only candidate is not valid mesh geometry;
first generate a GLB from a segmented image.

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

## Remaining quality work

- Real **mesh-surface/registered-camera visibility** gap detection beyond photo
  metadata; interactive landmark editing and auto-correspondence proposals.
- Non-destructive, quality-gated geometry seam blending and retopology.
- Registered real-camera reprojection and source-photo likeness checks for each
  generated candidate before publication. Synthetic views are not ground truth.
- A real Apple Silicon M4 end-to-end inference benchmark, plus optional model
  adapters only if their platform support and commercial-use rights are proven.

These are not yet automated by `build-completion`; retain artist review.
