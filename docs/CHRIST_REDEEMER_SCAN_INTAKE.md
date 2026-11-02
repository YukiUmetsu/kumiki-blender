# Christ the Redeemer — source-verified scan intake and anatomy gate

This ledger distinguishes **the statue on Corcovado** from miniatures,
replicas, generic Christ sculptures, and unproven third-party reconstructions.
The project aims at a photo-faithful 3D model; at the current quality gate, the
export is still an original *approximation*, not a facial likeness match.

## Authentic scan candidates

| ID | Source and producer | Authenticity and surface evidence | Access/rights | Modeling decision |
| --- | --- | --- | --- | --- |
| S1 | [Pix4D / PUC Rio / Aeryon, Projeto Redentor (2015)](https://www.pix4d.com/blog/mapping-christ/) · [statue-only 3D preview](https://sketchfab.com/3d-models/christ-the-redeemer-rio-fa8c4b24898c40ec86a40449cec47474) | **Verified real monument.** 2,090 drone images, 134.4M points, 2.5M-triangle full textured mesh; authors report individual soapstone tiles visible on the face. Three measured dimensions cross-checked with 1924 drawings. | Public viewer; **no verified permission** to redistribute the mesh. Do not extract or embed viewer data. | **Primary geometric reference** for registered front/profile/three-quarter visual comparisons, not a source mesh. |
| S2 | [Baukunst Patrimonio Virtual, Cristo Redentor survey (2016)](https://baukunst.es/projects/17) · [2018 3D viewer](https://sketchfab.com/3d-models/cristo-redentor-christ-the-redemeer-107e94e081d7464f93267b79fa9372c7) | **Verified monument/site** by producer description. Combined terrestrial/aerial photogrammetry for restoration; viewer reports ~3M triangles / 1.5M vertices. Facial surface quality needs detailed preview inspection before claiming a usable reconstruction. | Public viewer, download/reuse rights unverified. | Independent scan corroboration; **reference-only**. |
| S3 | [Archéomatique, Christ the Redeemer Rio](https://skfb.ly/RW8x) (credited in [mapart3d's Rio model](https://sketchfab.com/3d-models/rio-de-janeiro-lagoa-26e7c6606d6a43308a917735024af408)) | Rio monument is named, but survey method, capture date and close-up facial accuracy are unverified. | Cited **CC BY-NC**; unsuitable for commercial redistribution without additional permission. | Secondary inspiration only; not a commercial source mesh. |

## Excluded as authoritative geometry

| ID | Source | Why it fails the authentic-scan gate |
| --- | --- | --- |
| X1 | [TerraPrinter free model](https://terraprinter.com/landmarks/christ-the-redeemer/) | The maker explicitly derives it from a CC0 **MDesigns7 hand-modeled miniature**, then remeshes it into ~113k triangles for printing. This is a real-monument *interpretation*, **not a survey of the real statue**. The license does not establish likeness. |
| X2 | [Brian Trepanier Sketchfab model](https://sketchfab.com/models/90d53b3cb4b042ce845a24baa679109e/embed) | Download indicated, but the model's scan provenance, facial detail and item-specific license are not verified. Other collections mix map-style extracts and artist works. |
| X3 | [Commercial scan-based asset](https://www.turbosquid.com/3d-models/3d-christ-redeemer-cristo-redentor-1363192) | Seller claims photogrammetry and retopology, but primary capture provenance and permissions to redistribute derived source geometry are not established. Paid product, not a verified survey. |

No suitable **redistribution-cleared, authentic high-detail face scan** was
identified as of November 2026. A high triangle count, free download, or CC0
label alone does not establish anatomical or historical accuracy. Do not
silently substitute replicas for the surveyed monument.

## Reusable *anatomical donor*, not target evidence

| ID | Producer and original source | Reuse decision | What it can and cannot prove |
| --- | --- | --- | --- |
| A1 | [Blender Studio / community, Human Base Meshes v1.4.1](https://www.blender.org/download/demo-files/) · [official archive](https://download.blender.org/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.4.1.zip) | **CC0**, as marked by Blender's official asset page. The repository contains a gzip-compressed, 3-part OBJ extraction of its realistic sculpting head and two sclera objects in `assets/landmarks/christ_redeemer/head_anatomical_base_cc0.obj.gz`. | Quad-topology eyelids, nose, cheeks, mouth, ears and cervical forms provide a sculptable *human anatomical scaffold*. **It is not Christ the Redeemer, does not prove the monument's geometry, and must be resculpted against real-subject evidence.** |

The extraction was made from Blender Studio's `GEO-head_sculpting_realistic`
and its paired `.sclera.L` / `.sclera.R` meshes. It does not include iris
geometry, textures, or third-party viewer data. The original Blender asset's
CC0 license applies to this extracted geometry. This donor remains optional
and separate from the published procedural GLB.

## Measured geometry and current uncertainty

| Evidence | Value | Class | Intended use |
| --- | --- | --- | --- |
| Pix4D survey / 1924 original drawing | Statue height 30.00 / 30.03 m | Measured | Lock body scale and head height |
| Pix4D survey / 1924 original drawing | Pedestal 8.09 / 8.00 m | Measured | Lock pedestal-to-body transition |
| Pix4D survey / 1924 original drawing | Span 29.42 / 29.40 m | Measured | Lock fingertip span |
| [Paul Landowski archive](https://www.paul-landowski.com/en/le-christ-de-rio/) | Head 3.75 m; hand 3.20 m | Artist documentation | Facial cage/hand anchor sizes |
| S1/S2 viewer and real close-up photos | Brow, eyelid and nostril undercuts | Observed, camera-dependent | Sculpture; **do not infer unseen cavities from one angle** |
| Low-angle head reference photo supplied with project | Downturned mouth, tapered chin, hair framing | Observed | Clay macro and jaw geometry |

## Current anatomy failure and required method switch

The previous 38,560-vertex parametric head is **not** a credible likeness:
Gaussian eyes look like thin slits, the nose is indistinct from the front,
the forehead/chin depth relationship is wrong in profile, and the separate
hair curtain fails to merge into the neck/shoulders like the scan silhouette.

Use an **anatomical control cage with explicit facial planes and undercuts**,
sculpt in neutral clay, and align **front + profile + three-quarter** views to
S1/S2 screen previews and real photographs. Do not add high-frequency stone
detail, material noise or extra lip/eye tubes to compensate for missing forms.

Reference-driven priorities:

1. Bow the whole head to the actual profile, tracking forehead, nose, chin and
   back-hair outline together; avoid matching one reference camera only.
2. Reconstruct primary head mass and cheek/jaw planes, then actual eye sockets,
   nose projection/alar underside, and lip-chin transition.
3. Treat hair as a continuous sculpted back-of-head and shoulder volume rather
   than an isolated thin shell.
4. Fit robe folds from shoulder, belt/stole tension and gravity, not periodic
   sinusoidal bands; validate a low angle and rear/side view.
5. Recheck the full figure scale and artifacts before an incremental GLB update.

Use `scripts/reference_compare.py` for labeled, unaligned comparison sheets.
Only use `--aligned` when the cameras were genuinely registered. Local reference
screenshots are inspection-only and must not enter GLB, PR media or source code.

## Updated craftsmanship hold — November 2026

The repeated landmark/heightfield prototypes `landmark_r56` through `r59`
could not recover coherent eye-socket, nasal-alar and lip planes across front
and oblique views. Per the method-switch gate, **that representation is frozen**:
it remains an optional technical comparison, not the accepted face.

A distinct `--face-method anatomical` clay workflow now uses A1's **real
3D quad topology**, subdivided to a sculptable surface, with author-directed
mandible/chin expansion, upper-eyelid closure and mouth-corner shaping. This
is a geometry-foundation change, not extra Gaussians on the old ellipsoid.

The next anatomy study fitted the **sagittal nose surface** itself, rather
than attaching a separate nose primitive. Its first pass (`anatomy_r07`)
created an over-projected beak in profile; the second (`anatomy_r08`)
reduces the tip and gives the dorsum a flatter, longer plateau. Neither
iteration is treated as a verified likeness: the cameras remain
unregistered, and scan screenshots cannot supply reliable millimetric
depths. The two attempts to create a freestanding cross-section hair
shell (`anatomy_r05`/`r06`) produced a detached hood in both frontal and
side clay views. **That hair-shell method is rejected and is not included
in the generator.** Reconstruct hair from the head's actual surface or a
rights-cleared monument scan, not by further tuning the rejected shell.

It can be regenerated from the CC0 OBJ bundled in the repository:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/landmarks/christ_redeemer.py -- \
  --head-study --face-method anatomical --verify \
  --views .review/christ_redeemer/anatomy-study \
  --shot head_front --shot head_profile --shot head_three_quarter \
  --shot head_low --size 850 --samples 24
```

### Hair reconstruction study — still on HOLD

Two direct photographs of the **actual monument** clarify the silhouette:
[Landowski archive low-angle close-up](https://www.paul-landowski.com/portfolio/le-christ-redempteur/)
and [frontal face/shoulder photograph](https://generationvoyage.fr/christ-redempteur-rio-histoire/).
The hair is a close-fitting center-parted crown with a few broad carved
sweeps; its long side margins are relatively narrow and join the shoulder,
not a free-standing tall hood. These third-party photographs were used as
**local, reference-only** views; their files are not included in the repository
or model export. Camera intrinsics and registered depth remain unknown.

| Experiment | Geometry strategy | Actual front/profile verdict |
| --- | --- | --- |
| `scalp_r09` | Expand the anatomy cage's posterior scalp and neck vertices | **REJECTED:** still bald from the front and a swollen, indented rear silhouette in profile. |
| `scalp_r10` | Solid, scalp-intersecting rear and temple-to-shoulder volumes | **INVALID RENDER:** Blender's new objects had stale transform matrices; a geometry-position regression is now tested. |
| `scalp_r11` | Correctly transformed broad solid drapes | **REJECTED:** oversized wig-like side masses and a separate hood shape in profile. |
| `scalp_r12` | Narrowed reference-observed silhouette, removed crown top blob, C1-smooth longitudinal profiles | **HOLD / METHOD FROZEN:** more compact than r11, but still reads as a smooth rigid hood, not integrated stone hair. Face and crown mismatch remain clear. |

These are **artist-inferred draft volumes**, not target-verified geometry.
The optional `--integrated-hair` flag reproduces the editable r12 scalp study,
but does not replace the full-figure procedural export:

```bash
blender -b --factory-startup --python scripts/landmarks/christ_redeemer.py -- \
  --head-study --face-method anatomical --integrated-hair --verify \
  --views .review/christ_redeemer/hair-study \
  --shot head_front --shot head_profile --shot head_three_quarter --shot head_low
```

**Required next method switch:** reconstruct the *hairline and nape as sculpted
surface topology* registered to the real subject, or use licensed authentic
scan geometry. Another parameter change to the same ellipsoidal sweeps is not
an acceptable third attempt. The face and hair remain a visual HOLD.

### Anatomy-surface hair study — topology checkpoint (also HOLD)

The next candidate `--scalp-surface-hair` changes **the source of its geometry**:
it extracts one connected high-density scalp quad patch from the CC0 anatomy
mesh, preserves its exact anatomical surface correspondence, relaxes the
boundary loops, and extends the side and rear skin over the neck. This is not
another ellipsoidal loft, and it leaves the donor's sculpted face intact.

```bash
blender -b --factory-startup --python scripts/landmarks/christ_redeemer.py -- \
  --head-study --face-method anatomical --scalp-surface-hair --verify \
  --views .review/christ_redeemer/scalp-surface \
  --shot head_front --shot head_profile --shot head_three_quarter --shot head_low
```

The first `scalp_surface_r13` clay views exposed a stepped forehead boundary,
copied ear bumps and an undersized side curtain. `r14` smooths the boundary
and extends the selected neck/temple surface; it **does improve contour
continuity** but the forehead still looks bald, the ear forms protrude, and
the hair terminates as side bulges rather than long carved locks. Real-subject
front and low-oblique photographs were inspected against both renders in
**unregistered, qualitative** comparison sheets. **Both surface passes fail
the likeness gate**; this method is now frozen as an editable sculpt foundation,
not promoted to the full figure or GLB.

No currently authorized authentic monument mesh or registered photo-camera
set constrains the actual scalp-to-shoulder surface. Subsequent genuine
sculpting/retopology requires that target evidence or manual artist-directed
3D fitting; more procedural hair masks cannot establish photo fidelity.

| Reconstruction gate | Decision | Evidence / unresolved work |
| --- | --- | --- |
| Authentic subject foundation | **HOLD** | S1/S2 clearly depict the monument, but no authorized high-detail scan mesh is available; A1 is only a generic donor. |
| Front/profile/oblique camera registration | **HOLD** | Scan-viewer screenshots and real photographs are unregistered. No focal-length/camera solve or valid numerical landmark residual exists. |
| Primary face geometry | **HOLD — improving** | Anatomical quad mesh and 3D jaw/chin sculpt resolve the earlier featureless mask, but donor eye openings, nose, ear/hairline and lips remain noticeably unlike Landowski's head. |
| Multi-view clay acceptance | **HOLD** | Actual front, profile, oblique and low-angle renders are required after each distinct sculpt change; the last render does not yet justify release. |
| Photo-faithful GLB publication | **BLOCKED** | No new anatomical head has passed the visual or camera gate. The existing exported GLB remains labeled an approximation. |

Next required work is to calibrate **actual monument** front/profile/oblique
reference cameras or secure authorized high-detail survey data, then fit the
source-specific facial planes and close the eyelid anatomy. Neither the donor
license nor a 52k-vertex sculpt is evidence of target likeness.

### Hand-traced crown locks and temple sheets — November 2026 HOLD

The distinct `--guided-hair` experiment abandons the threshold-selected scalp
patch: six artist-directed, surface-projected stone locks follow the crown,
and two separately retopologized ruled sheets trace narrow temple-to-neck
control edges. The work uses the CC0 donor only. The control edges are
*hand-interpreted* from genuine photographs, **not registered measurements**.

```bash
blender -b --factory-startup --python scripts/landmarks/christ_redeemer.py -- \
  --head-study --face-method anatomical --guided-hair --verify \
  --views .review/christ_redeemer/guided-hair \
  --shot head_front --shot head_profile --shot head_three_quarter --shot head_low
```

- **r15 / rejected:** crown locks start to show the reference's center-parted
  sweep, but the two large side panels stand upright above the head and form
  triangular capes in profile. Facial anatomy remains generic.
- **r16 / rejected:** manually redrawn panel edges eliminate the standing
  horns and reduce sagittal thickness, but the long locks still read as
  separate, stiff ribbons running off the skull. Their roots and tips show
  discontinuities in three-quarter/profile views, and the frontal eye, nose,
  jaw and lip features remain unlike the monument.

**Two actual four-angle clay render passes failed. The guided-sheet hair
representation is frozen; do not attempt a third spline-width correction.**
The default full-figure GLB is unchanged. The optional Blender study is
editable for an artist, but is not a photographically accepted likeness.
Next reconstruct the face and hair silhouette *together* in a registered
camera space using a sculpted shared control cage and verified source images,
or obtain explicit reuse rights for a high-resolution survey of the monument.

### First camera-fitting diagnostic — November 2026 (HOLD)

The reusable `camera_registration.py` perspective solver fits 3D head-surface
points to annotated pixels from a photo of the **actual Rio monument**. Its
sculpt-pixel probing script uses ray casts into the donor's *real quad mesh*,
not guessed 3D depths. The self-contained [front annotation JSON](CHRIST_REDEEMER_FRONT_ANCHORS_R16.json)
records 15 paired photo/model pixels and source provenance without embedding
the copyrighted reference photograph. The [fit report](CHRIST_REDEEMER_FRONT_CAMERA_FIT_R16.json)
records projected points, camera parameters and residuals.

```bash
# Open the separately saved r16 HOLD `.blend` from the review handoff.
blender -b /path/to/christ_redeemer_guided_hair_hold.blend \
  --python scripts/landmarks/probe_face_anchor_pixels.py -- \
  --input docs/CHRIST_REDEEMER_FRONT_ANCHORS_R16.json \
  --output .review/christ_redeemer/front-anchors-3d.json
PYTHONPATH=src python scripts/fit_reference_camera.py \
  .review/christ_redeemer/front-anchors-3d.json \
  --output .review/christ_redeemer/front-camera-fit.json
```

The initial fit reaches **28.738 px RMS / 67.384 px maximum reprojection
error** at 1536 × 1024 source resolution; the optimized camera also hits its
allowed distance/target bounds. The chin anchor misses by **67.38 px**, nose
bridge by **42.3 px**, and outer eyelids by roughly **31–40 px**. This result
is **not** a valid camera registration and must not be used for an aligned
photo overlay or claimed as survey-level evidence. Manual point annotation,
unknown lens distortion and fundamentally inaccurate donor face geometry
all contribute; individual residuals cannot yet be attributed to a specific
mesh deformation.

Before accepting any facial sculpt, capture a second independent authentic
profile/oblique view with enough overlapping facial points, refine the
landmark correspondences, and fit per-photo perspective cameras. Then
sculpt the **shared 3D facial and hair silhouette** against all views.

## Additional openly licensed comparison models (2026-11-02)

These are **not verified authentic scans of the Rio monument**. Their
public Sketchfab pages state Creative Commons Attribution (CC BY),
which is suitable for *licensed comparison* subject to preserving
attribution and checking the exact downloadable asset and terms.

| Candidate | Published geometry | License shown | Source |
|---|---:|---|---|
| Parlin Sitorus, *Christ the Redeemer (Cristo Redentor)* | 55.6k triangles | CC BY | https://sketchfab.com/3d-models/christ-the-redeemer-cristo-redentor-157dbdb9d8654651a5e5c9af7b412398 |
| JuanG3D, *Christ the Redeemer* (3D print) | 58.8k triangles | CC BY | https://sketchfab.com/3d-models/christ-the-redeemer-d143e5357efd42529777438d0f72c79c |
| Walter Araujo, *Cristo redentor* | 14.8k triangles | CC BY | https://sketchfab.com/3d-models/cristo-redentor-christ-the-redeemer-27c2c6a36001409b9f6879051e6217e1 |

**Download status:** no model geometry was fetched or imported; the
public Sketchfab download action may require the asset owner's
account/OAuth. Do not scrape the 3D viewer or assume a screenshot
licenses the underlying mesh. The Pix4D and Baukunst authentic
photogrammetric surveys remain better fidelity references, but
redistribution/download rights are not established. The paid
photogrammetry models on TurboSquid/Free3D have editorial-use
restrictions and are not automatically suitable for commercial
video production.

**Next best-quality path:** obtain a properly licensed real-monument
scan or a CC BY reference model and compare its **frontal, profile
and low-angle** silhouette against the editable clay. Rebuild the
topology where necessary; do not treat generic donor anatomy as
the actual monument.
