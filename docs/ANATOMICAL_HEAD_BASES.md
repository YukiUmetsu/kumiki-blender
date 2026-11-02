# Anatomical head base → editable monument sculpture

Use this when a primitive-generated statue face looks generic, compressed or
anatomically inaccurate. Start with **real 3D human-head geometry** from a
reviewed local GLB, then fit that geometry to measured monument landmarks.

This is a **geometry authoring foundation**, not a one-click photorealistic
reconstruction or proof of historical likeness.

## 1. Prepare the anatomical base

- Create/export a *neutral, static* human-head mesh from an established tool
  such as [MakeHuman / MPFB](https://www.makehumancommunity.org/) (its **core
  exported graphical assets** are CC0). A licensed artist-authored mesh or an
  actual monument scan can also be used. For a non-CC0 add-on/third-party mesh,
  confirm export and redistribution rights separately.
- Export a **static GLB without armatures or animation**. For local corrections,
  export the neutral base without preexisting morph targets. Keep the original
  editable asset too.
- Give the facial mesh an explicit stable name such as `Head`. The imported GLB
  can include other parts, e.g. eyes or hair; only the named mesh receives the
  optional local sculpt correction. All imported parts receive the same rigid
  transform.
- Write the existing [source provenance manifest](../examples/first-party-import.json)
  for the actual source, with verified `redistribution_allowed: true`. A pinned
  `expected_sha256` is recommended. Restricted meshes may be inspected privately
  elsewhere but **cannot** go through this publication-ready importer.

## 2. Measure the monument before fitting

Create a landmark plan following
[`examples/anatomical-head-fit.example.json`](../examples/anatomical-head-fit.example.json).
The example coordinates are illustrative only; **do not use them as Christ the
Redeemer survey coordinates.** Place the JSON next to your reviewed `head.glb`
and `head-provenance.json`.

Each landmark names the identical anatomical point on both the base and the
target, in **Blender Z-up meters**. Source xyz must match the Blender-imported
GLB's original world coordinates, *not* the glTF Y-up binary axis positions.
Set `source_front_axis` to the direction the original head faces (`-Y` by
default); this orients the front, profile and three-quarter QA cameras.
Use at least four noncollinear correspondences, preferably 8–20 distributed
across both eye corners, nose bridge/tip, ears, jawline, forehead and chin.
Include depth from genuine front/profile/three-quarter measurements; a single
2D image cannot provide verified 3D target points.

The [reference reconstruction pipeline](REFERENCE_RECONSTRUCTION.md) supplies
registered cameras and existing silhouette/landmark QA. Independent survey
data, scans and artist-confirmed multi-view correspondences remain important.

- `fit_mode: rigid` applies only a **proper rotation, positive uniform scale and
  translation**; it preserves the base anatomy exactly.
- `fit_mode: local-sculpt` additionally creates a smoothly deformed Blender
  **shape key** from remaining per-landmark offsets. `radius_m` controls where
  each landmark's correction tapers to zero. Large/unstable changes are refused;
  choose a better anatomical source instead of forcing a generic head into a
  very different shape.

## 3. Review the fit, then build a revision

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-head-base \
  --manifest /your/private/reference/head-fit.json

PYTHONPATH=src python3 -m kumiki_blender build-head-base \
  --manifest /your/private/reference/head-fit.json \
  --output .review/redeemer-head/r01
```

The first command requires no Blender. It prints actual scale/orientation,
per-landmark post-fit errors, source digest and an explicit artistic-review
status. The second starts an installed Blender 4.2+ (or use `--blender PATH`)
and produces the shared revision bundle:

| File | Purpose |
| --- | --- |
| `source.blend` | **Editable** anatomical base, correction shape key and visible target landmark empties |
| `model.glb` | Portable, aligned head; the active correction is baked into export geometry |
| `preview.png` and `previews/*.png` | Studio inspection angles; no claim of likeness |
| `manifest.json` | Rights/source hash, source/target residuals, mode and review limitations |

Never reuse an existing output directory; create `r02` for the next revision.
Keep the working `.blend` as the editable source of truth. Render its front,
three-quarter and profile views against independently sourced monument images,
and use `render-reference-scene`, `silhouette-qa`, and `landmark-qa` before
calling any result photo-faithful.

## Limitations and next phase

- Landmark fitting does not reconstruct missing eyelid/hair/beard detail,
  weathering, garment folds or damage. These require artist-directed sculpting
  and high-resolution evidence; a fitted human head alone is not a statue.
- The Great Sphinx's eroded historical geometry is a poor match for a generic
  modern head. Prefer a verified archaeological scan or multi-view survey first.
- Future work: integrate a reviewed CC0 base-head asset catalog and a tightly
  constrained 2D/3D landmark authoring UI, plus in-repo multi-angle acceptance
  fixtures for each monument. No external binary is bundled in this PR.
