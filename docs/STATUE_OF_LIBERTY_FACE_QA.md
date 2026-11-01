# Statue of Liberty — reference-calibrated facial acceptance gate

## r56 — hard measured multi-view release gate

The updated reconstruction workflow requires authenticated **real-target 3D
geometry** or a genuinely sculpted, reference-calibrated foundation before
detail promotion. The existing procedural r53 face and rejected r54/r55
control-cage trials do not satisfy that condition. The NPS documents a
2007 exterior scan and improved exterior capture during its 2019 survey,
but no licensed high-resolution **usable head mesh** has been obtained from
those projects; neither a decorative miniature nor a museum-face replica is
equivalent to the installed copper monument.

| Release gate | Decision | Evidence / missing work |
| --- | --- | --- |
| Foundation authenticity | **BLOCKED** | No rights-cleared real-head scan, calibrated 3D survey mesh or approved anatomical sculpt |
| Camera correspondence | **BLOCKED** | HAER photos have unknown intrinsics/pose; face-display frontal is not the installed head |
| Continuous facial anatomy | **HOLD** | Watertight surface exists but realistic orbits, alae, lip masses and cheek/jaw planes do not |
| Front/profile/three-quarter QA | **HOLD** | Reference comparisons expose major mismatches; no calibrated pixel-grid segmentation/landmarks |
| Photo-faithful GLB release | **HOLD** | Do not publish a new model as more realistic based on tessellation or render success alone |

The versioned [HOLD manifest](STATUE_OF_LIBERTY_FACE_HOLD.json) records these
explicitly. The new QA scripts fail closed: `geometry_compare.py` measures
binary silhouette IoU plus diagonal-normalized landmark RMS only when masks,
photos and named landmarks occupy the same registered pixel grid; optional
baseline regression blocks numerical backsliding. `reconstruction_gate.py`
requires all three actual image pairs, documented camera calibration,
threshold rationales, authentic foundation evidence, and a passing human
anatomy review before returning `PASS`. No artificial masks or landmark
annotations were invented for the unregistered HAER photos.

```bash
# Expected exit code 2 and a truthful HOLD, not a passing release:
PYTHONPATH=src python3 scripts/reconstruction_gate.py \
  docs/STATUE_OF_LIBERTY_FACE_HOLD.json

# Once genuine matching and registered views exist, evaluate them individually:
python3 scripts/geometry_compare.py \
  --reference-mask front-ref-mask.png --render-mask front-render-mask.png \
  --reference-landmarks front-ref-points.json \
  --render-landmarks front-render-points.json \
  --registration calibrated --camera-evidence "Matched survey control points" \
  --min-iou 0.90 --max-landmark-error 0.02 \
  --output front-metrics.json
```

The sample numeric limits in the second command are **illustrative**; actual
thresholds require justified camera/survey uncertainty. Numerical PASS cannot
override a visible facial defect, nor recover hidden 3D structure.

For a sculptor, the clay review can display an **editable scaled-measurement
overlay** on the front camera. It uses the NPS feature ratios and shows where
the existing illustrative mesh departs from those guides; it never changes
the portrait geometry or pretends to constrain unknown side depth:

```bash
blender --background --factory-startup --threads 6 --python-exit-code 1 \
  --python scripts/blender_liberty_face_clay.py -- \
  --measurement-guides \
  --output .review/landmarks/statues/statue-of-liberty/measurement-guides-next
```

The generated `clay.blend` retains named measurement curves, dimensions in
scene metadata, and clean three-quarter/profile renders with guides hidden.

## Source ledger and confidence

| ID | Primary source | Observation / intended use | Evidence and rights | Confidence |
| --- | --- | --- | --- | --- |
| NPS-M | [NPS statue statistics](https://www.nps.gov/stli/learn/historyculture/statue-statistics.htm) | Head, eye, nose and mouth measurements | **Measured**, factual source; reference only | High for listed dimensions |
| HAER-F | [Jet Lowe, straight face elevation, HAER NY-138-227](https://commons.wikimedia.org/wiki/File:STRAIGHT_ELEVATION_OF_FACE,_BUILT_BY_FRENCH_COPPERSMITH_IN_FULFILLING_THEIR_MASTERY_OF_SMITHING._-_Statue_of_Liberty,_Liberty_Island,_Manhattan,_New_York,_New_York_County,_NY_HAER_NY,31-NEYO,89-227.tif) | Eyeball/lid openings, nasal alae, lip form and cheek planes of a coppersmith-fabricated face **display**, not necessarily the installed head | **Observed**, public-domain NPS/HAER photo; reference only | Medium for installed-statue anatomy |
| HAER-P | [Jack Boucher, installed face profile, HAER NY-138-241](https://commons.wikimedia.org/wiki/File:DETAIL,_PROFILE_VIEW_OF_LEFT_SIDE_OF_FACE_-_Statue_of_Liberty,_Liberty_Island,_Manhattan,_New_York,_New_York_County,_NY_HAER_NY,31-NEYO,89-241_%28CT%29.tif) | Nose dorsum, lip projection, cheek/mandible depth, brow and hair relationship | **Observed**, public-domain NPS/HAER photo; reference only | High for contour, medium for precise depth due perspective |
| NPS-S | [NPS HDP 2019 laser/photogrammetry documentation](https://www.nps.gov/articles/laser-scan-lady-liberty.htm) | Establishes that a modern real-statue survey exists | **Observed** survey report, but no reusable external mesh obtained | High for survey existence, unknown for access/rights |
| SK-F | [avredu, scanned museum face](https://sketchfab.com/3d-models/statue-of-liberty-face-model-in-the-museum-1a09198686344d279a8220cb7bdb7573) | Rotatable model preview for general spatial understanding, **not** a scan of the installed head | **Inspiration only**, CC BY-NC-SA; do not import or redistribute | Low/medium |

The original photograph copyrights, 3D model rights and provenance are
separate. **No reference photograph or third-party scan is incorporated into
the GLB or generator.** Use the public-domain HAER images only to inspect and
compare the original independently authored mesh.

## Measured calibration target

NPS measurements are **head height 5.26 m**, **head width 3.05 m**,
**eye width 0.76 m**, **nose length 1.37 m**, and **mouth width 0.91 m**.
The existing procedural model's head is 3.52 m high. Scaling the NPS feature
ratios *to that illustrative head*, without claiming real monument scale,
gives these horizontal/vertical construction targets:

| Feature | NPS-derived ratio target (m) | r53 procedural estimate (m) | Observation |
| --- | ---: | ---: | --- |
| Head width | 2.04 | 2.29 actual geometry span | Cranium is broad; hair/crown comparison still needed |
| Individual eye width | 0.509 | ~0.63 field support | Slit-like surface, no correct sclera/iris/lid volume |
| Nose length | 0.917 | ~0.9 relief extent | Tip, bridge and alar transition still anatomically wrong |
| Mouth width | 0.609 | ~0.86 field support | Mouth is wide and too flat in clay |

These are **design diagnostics**, not claims of millimeter-accurate image
registration. Feature relief-field support is not the same thing as a
geometrically measured eyelid or lip opening.

## r54–r55 strategy-change experiments — **rejected**

The earlier r49–r53 Gaussian/ellipsoid face had repeatedly failed likeness
inspection. Following the method-switch rule, we prototyped a separate
anatomical cage with independent frontal/rear depth and width control points
constrained by the NPS ratios, then varied the iris, lid and lip relief.
Fixed front, three-quarter and true profile neutral-clay renders were compared
with the two HAER photographs at matched **rough crop** scale, not treated as
registered overlays.

Both trials **failed visual QA**:

1. **Front / anatomy:** flat frontal planes; eye openings without genuine
   eyeball and lid forms; nose/nostril shaped like a small cross; cheeks,
   philtrum and lips drastically unlike the photographed face display.
2. **Profile / silhouette:** abrupt nose-to-lip recess, broad square jaw,
   short lip mass, too-simple bowl ears, incomplete hair/temple integration.
3. **Multi-angle:** a narrower head alone improved measured width but did not
   produce anatomical likeness. A second feature pass still did not pass.

**Decision:** reject and do not promote the attempted cage into
`build-liberty --detailed-face`. The previously shipped r53 model and GLB
remain the latest deliverable. More Gaussian detail would not solve these
primary-shape failures. Next reconstruction should use a truly suitable
reuse-authorized scan of the actual face (if one becomes available), or a
high-resolution manually sculpted and retopologized facial foundation
calibrated from multiple photographic views.

## Repeatable multi-angle comparison

The checked-in neutral-clay renderer deliberately **excludes** crown/hair,
torch, tablet and materials so anatomy remains visible, and preserves stable
front / three-quarter / profile cameras. The sheet tool labels both images
and requires an explicit crop, without claiming pixel registration.

```bash
blender --background --factory-startup --threads 6 --python-exit-code 1 \
  --python scripts/blender_liberty_face_clay.py -- \
  --output .review/landmarks/statues/statue-of-liberty/face-qa-next

# Download/reference the HAER images yourself from the public-domain pages
# above, then use a calibrated crop appropriate to the specific source view:
python3 scripts/reference_compare.py /path/to/haer-front.jpg \
  .review/landmarks/statues/statue-of-liberty/face-qa-next/front.png \
  --ref-crop 155 348 820 1080 \
  --output .review/landmarks/statues/statue-of-liberty/face-qa-next/front-compare.png
```

The shown crop fits the 960 px HAER-F thumbnail referenced in the ledger;
larger originals and other photos require different crop coordinates.
These checks are **required before** accepting new face geometry, regardless
of whether mesh topology tests or GLB export audits pass.
