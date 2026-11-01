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
