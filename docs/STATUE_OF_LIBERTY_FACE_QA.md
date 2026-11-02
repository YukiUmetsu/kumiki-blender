# Statue of Liberty — reference-calibrated facial acceptance gate

## r61 — actual-head archival anatomy atlas + parallax feasibility (**HOLD**)

This iteration changes the **source evidence and sculpt workbench**, not the
still-rejected face mesh. From the Library of Congress/NPS Historic American
Engineering Record, the reference studio now packs **eleven distinct primary
photographs of the actual installed copper statue** in three editable scenes:
the inherited continuous donor sculpt, **six 1984 whole-head/side/crown
views** (HAER 236–241), and **five 1985 restoration-era anatomical close-ups**
(HAER 177–181). Unlike the museum coppersmith face and miniature replicas,
these really depict the installed monument. All source JPEGs are individually
SHA-256 verified and packed unchanged into the new `.blend`.

Especially useful new observations:

- [HAER 240](https://www.loc.gov/pictures/item/ny1251.color.570009c/)
  shows the **left three-quarter cheek/jaw** and hair/crown junction;
  [238](https://www.loc.gov/pictures/item/ny1251.color.570007c/) and
  [239](https://www.loc.gov/pictures/item/ny1251.color.570008c/)
  show the sculpted hair direction, large rolled side-lock, and ear recession.
- [HAER 178](https://www.loc.gov/pictures/item/ny1251.photos.120436p/)
  reveals the **orbital and nostril undercuts from below**;
  [179](https://www.loc.gov/pictures/item/ny1251.photos.120437p/)
  isolates the eyelid/eye form, and
  [181](https://www.loc.gov/pictures/item/ny1251.photos.120439p/)
  shows the distinctive external alar wing, inner nostril and lip volume.
- [HAER 180](https://www.loc.gov/pictures/item/ny1251.photos.120438p/)
  contains a scale stick in the worker's hand. **Its depth and angle relative
  to the face are not registered**, so it cannot be used as a direct 3D
  measurement. Restoration damage and patching in these photos must not be
  sculpted as an intended original facial feature.

An optional OpenCV/SIFT study tested whether the six color views could support
an actual camera reconstruction. Using broad **head-only ROI rectangles**,
14 of 15 image pairs had fewer than 20 tentative descriptor matches or fewer
than 15 epipolar inliers. The remaining pair, **238/239**, produced 88
tentative matches, 79 homography inliers and 76 epipolar inliers: these are
close/redundant side-hair views, not an independent depth baseline. The
best other profile candidate **240/241** had just 23 tentative matches and
10 epipolar inliers. These are exploratory *correspondence counts*, not a
calibration or likeness score. Scaffold clutter, different illumination,
viewpoint, crop, and dark smooth copper still preclude responsible 3D
triangulation from this photo subset. The pose and shape remain **unrecovered**.

```bash
python3 scripts/fetch_liberty_haer.py --output .review/liberty-haer

# Optional only: pip install opencv-python-headless
python3 scripts/liberty_haer_photo_overlap.py \
  --reference-dir .review/liberty-haer \
  --output .review/liberty-r61-photo-overlap.json

/path/to/blender-4.5/blender --background --factory-startup \
  --python-exit-code 1 --python scripts/blender_liberty_reference_studio.py -- \
  --clay-blend .review/liberty-r60-spatial-study/anatomy-stroke-study.blend \
  --reference-dir .review/liberty-haer \
  --output .review/liberty-r61-reference-studio
```

**Scale caution:** NPS lists *chin to cranium* as **5.26 m**, while the old
r59 donor study deliberately used an **illustrative 3.52 m** head-height
proxy. That earlier donor scale must not be propagated as an actual monument
measurement. See [NPS dimensions](https://www.nps.gov/stli/learn/historyculture/statue-statistics.htm).
The atlas does not rescale or promote the generic head; it is for an artist
to perform a genuinely source-constrained sculpt from anatomical primary
forms. No new photo-faithful GLB, claimed scan, camera alignment or released
facial geometry is justified. **Craftsmanship status: informed reference
foundation; model acceptance remains HOLD.**

## r60 — reversible 3D anatomy trial, donor route frozen (**HOLD**)

One additional *distinct* volumetric sculpt experiment was conducted on the
r59 full, continuous Blender Studio anatomical mesh, rather than the rejected
procedural Liberty face or the earlier failed miniature face-patch graft.
Eight deliberately restrained spatial strokes were placed on actual 3D face
surface hits (front-facing mesh raycasts), modifying **2,102 base vertices**
with a maximum base-vertex displacement of **0.129 m**. The changes reside in
an editable Blender **shape key**; the original donor head remains recoverable
in the same file. This is an **artist-inferred** exercise, not calibrated
photogrammetry or historical survey depth.

```bash
blender --background --factory-startup --threads 6 --python-exit-code 1 \
  --python scripts/blender_liberty_3d_sculpt_study.py -- \
  --input .review/liberty-r59-whole-head/whole-head-sculpt.blend \
  --output .review/liberty-r60-spatial-study --resolution 640

# Then use the existing HAER studio generator to pack the three originals:
blender --background --factory-startup --threads 6 --python-exit-code 1 \
  --python scripts/blender_liberty_reference_studio.py -- \
  --clay-blend .review/liberty-r60-spatial-study/anatomy-stroke-study.blend \
  --reference-dir .review/liberty-haer \
  --output .review/liberty-r60-studio
```

**Visual finding:** r60 is less caricatured than the rejected r59 broad-band
pass and has modestly fuller lower lip/chin volumes in both side views, but
these changes **do not close the underlying subject-likeness gap**. Both the
real [HAER 237 right](https://www.loc.gov/pictures/item/ny1251.color.570006c/)
and [HAER 241 left](https://www.loc.gov/pictures/item/ny1251.color.570010c/)
head profiles show a much heavier continuous orbital overhang, a broad
nearly vertical cheek-to-jaw plane, distinct alar-nostril geometry, larger
projected sculpted lips, deep lip/chin undercut, a thick rounded lower chin,
and characteristic hair/crown overlap. The r60 mesh is still an anatomically
generic, bald CC0 human bust with no actual target-specific survey evidence.
All side-by-side sheets are **qualitative**: HAER camera intrinsics and pose
remain unknown. No measured comparison score is asserted.

**Method decision: FREEZE donor-only procedural deformation.** The earlier
r59 broad-band pass made a cartoonish brow, and this r60 more localized
shape-key trial improved small volumes without establishing Bartholdi's
unique installed-head anatomy. Following the Craft Director two-failed-trial
trigger, do not apply a third parameter pass to this CC0 donor. Need a
**rights-cleared actual installed-head scan/survey** or a genuinely skilled,
reference-constrained manual head sculpt and camera fitting. The U.S. NPS
[2019/2007 documentation](https://www.nps.gov/articles/laser-scan-lady-liberty.htm)
confirms high-resolution surveys were performed, but no accessible licensed
high-detail head mesh has been verified. The former [NPS virtual tour](https://home.nps.gov/articles/000/statue-of-liberty-tour.htm)
has been unsupported since May 2025, and its panoramas would not themselves
substitute for recoverable 3D surface. Production Statue code and r53 GLB
remain unchanged; photo-faithful GLB release remains **HOLD**.

## r59 — isolated continuous anatomy sculpt foundation (**HOLD**)

The earlier r41 experiment grafted a generic CC0 donor face into a separately
scanned **miniature** and left an unacceptable border seam, detached features
and mismatched anatomy. This trial changes the **underlying source topology**:
the entire, uncut Blender Studio CC0 human head (Dan Ulrich, Human Base Meshes
v1.2.0) is retained as **one continuous editable anatomical surface**, with
its original eye geometry and two-level Multires sculpt stack. Its source is
SHA-256 verified, and parent/world matrices are preserved before eye objects
are unparented so the original sockets and globes stay aligned. There is no
face-patch graft, replica scan substitution or Gaussian facial relief.

The cranium is scaled to the NPS *height-to-width ratio* at the earlier
illustrative 3.52 m head height, using a visually inspected donor **chin
landmark and cheek/cranium-width section** rather than the source file's
shoulder/bust bounds. These source section landmarks are approximate and
not evidence of actual-monument anatomy. Fore-aft depth is **inferred from
the donor**, not measured from the actual statue. Four neutral-clay cameras
(front, three-quarter, both profiles) are saved in the new editable `.blend`.
The existing HAER reference-board generator can add the three authentic
installed-head photographs as packed images in a separate scene.

```bash
python3 scripts/fetch_liberty_anatomy.py \
  --output .review/liberty-donor/source.blend
blender --background --factory-startup --threads 6 --python-exit-code 1 \
  --python scripts/blender_liberty_whole_head_sculpt.py -- \
  --source .review/liberty-donor/source.blend \
  --output .review/liberty-r59-whole-head --resolution 640
blender --background --factory-startup --threads 6 --python-exit-code 1 \
  --python scripts/blender_liberty_reference_studio.py -- \
  --clay-blend .review/liberty-r59-whole-head/whole-head-sculpt.blend \
  --reference-dir .review/liberty-haer \
  --output .review/liberty-r59-studio
```

**Visual verdict:** usable as a *human sculpting starting surface*, not an
accepted Statue of Liberty head. The real [HAER 241 left profile](https://www.loc.gov/pictures/item/ny1251.color.570010c/)
and [HAER 237 right profile](https://www.loc.gov/pictures/item/ny1251.color.570006c/)
show a dramatically deeper supraorbital mass, broader near-vertical cheek
plane, prominent undercut nasal alae, projected lips and a large rounded chin.
The untouched donor renders show shallow brow, generic narrow jaw, weak
under-lip/chin structure and a smooth tall bald cranium. Its ears are
incorrectly exposed without Bartholdi's characteristic coiffure. A focused
automatic broad-plane deformation of this new donor was also rendered but
**rejected**: it made a cartoonish jutting eyebrow and squared jaw. That
deformation is deliberately **not** in the reproducible script or promoted
artifacts; hand-sculpted volume and photographic camera constraints remain
necessary. The old r53 GLB and production Statue mesh remain unchanged.

**Release gate:** authenticity of the CC0 donor is verified, authenticity as
*the installed monument* is not; camera correspondence is absent and all
three facial likeness views remain **HOLD**. This is an experimental, separately
editable alternate foundation, not a photo-faithful reconstruction. Its
presence does not lift the existing `STATUE_OF_LIBERTY_FACE_HOLD.json` gate.

## r58 — dual-profile clay inspection with persistent cameras (**HOLD**)

The sculpt renderer now saves **four independently selectable orthographic
cameras** in its `.blend` (front, three-quarter, profile facing left, and
opposite-profile facing right), plus a PNG for each. This enables qualitative
comparison against both actual installed-head archival profiles, rather than
mistaking the two sides for interchangeable views. It does not assign survey
camera poses, change the facial mesh, or pass the release gate.

```bash
blender --background --factory-startup --threads 6 --python-exit-code 1 \
  --python scripts/blender_liberty_face_clay.py -- \
  --measurement-guides --resolution 640 \
  --output .review/liberty-face-r58-four-views

# Same-facing side-by-side study, *not* pixel-registered evidence:
python3 scripts/reference_compare.py \
  .review/liberty-haer/haer-237-right-profile-large.jpg \
  .review/liberty-face-r58-four-views/opposite-profile.png \
  --ref-crop 230 250 700 800 \
  --output .review/liberty-face-r58-four-views/right-side-study.png
python3 scripts/reference_compare.py \
  .review/liberty-haer/haer-241-left-profile-large.jpg \
  .review/liberty-face-r58-four-views/profile.png \
  --ref-crop 170 240 660 820 \
  --output .review/liberty-face-r58-four-views/left-side-study.png
```

The model's current profile is still too planar around the brow/cheek and
cannot reproduce the real nose-tip / lip / chin projection. It remains the
previously rejected illustrative geometry, not a new sculpt foundation. The
new cameras make the next genuine anatomy rebuild easier to inspect and
prevent single-side tunnel vision.

## r57 — authentic installed-head reference studio, still **HOLD**

Three archival color photographs attributed in the HAER index to Jet Lowe show the **actual installed
copper head**: [HAER 236](https://www.loc.gov/pictures/item/ny1251.color.570005c/)
(right oblique), [HAER 237](https://www.loc.gov/pictures/item/ny1251.color.570006c/)
(right profile), and [HAER 241](https://www.loc.gov/pictures/item/ny1251.color.570010c/)
(left profile). These are more directly relevant than the HAER 227 coppersmith
display face. The NPS/HAER images are public-domain U.S. federal works. Their
photographic poses are **not** known. The side images include scaffold poles,
deep shadows and nonuniform exposure; none is a calibrated orthographic scan.

With these real-target views beside the existing r53 clay render, the failure
is especially visible in the nose-to-upper-lip profile, narrow eyelid slit,
missing brow/eyeball volume, broad flat cheek and chin planes, and detached ear
form. Do not mistake a cropped hairless clay surface for a finished monument
head. No new facial surface, photo-faithfulness score, or GLB is promoted here.

The new studio generator packs the actual archival photos **into a separate
Blender reference-board scene** and keeps the editable model in the primary
sculpt scene. Its Blender Text datablock contains the ordered anatomical
sculpt brief. Original pixels remain unwarped; the reference board is not
projected onto a 3D surface and cannot qualify as a registered likeness test.
The extremely dark right-profile preview has display-only gamma correction,
explicitly labeled in the board; its original pixel data stays packed unchanged.

```bash
mkdir -p .review/liberty-haer
curl -fL https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/570005cv.jpg \
  -o .review/liberty-haer/haer-236-oblique-large.jpg
curl -fL https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/570006cv.jpg \
  -o .review/liberty-haer/haer-237-right-profile-large.jpg
curl -fL https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/570010cv.jpg \
  -o .review/liberty-haer/haer-241-left-profile-large.jpg

blender --background --factory-startup --threads 6 --python-exit-code 1 \
  --python scripts/blender_liberty_face_clay.py -- \
  --measurement-guides --output .review/liberty-face-r57-baseline
blender --background --factory-startup --threads 6 --python-exit-code 1 \
  --python scripts/blender_liberty_reference_studio.py -- \
  --clay-blend .review/liberty-face-r57-baseline/clay.blend \
  --reference-dir .review/liberty-haer \
  --output .review/liberty-face-r57-studio
```

Open `liberty-face-sculpt-studio.blend`: scene **01** contains the editable
face and optional NPS scale guides; switch to scene **02** for the packed,
attributed source photographs. The generated `haer-reference-board.png` is a
visual aid, **not** a camera-calibrated comparison. Neither the 2D images nor
the archive board is exported to the statue model. The studio renderer requires
all three source files and refuses to overwrite an existing revision.

**Next sculpt gate:** a legitimate manual continuous anatomical rebuild, then
front/left/right/three-quarter clay inspection with independent human review;
calibrated source cameras or real-target scan are additionally needed for a
photo-faithful release. The r53 GLB remains the previously available preview.

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
| HAER-P | [Jet Lowe, installed face profile, HAER NY-138-241](https://commons.wikimedia.org/wiki/File:DETAIL,_PROFILE_VIEW_OF_LEFT_SIDE_OF_FACE_-_Statue_of_Liberty,_Liberty_Island,_Manhattan,_New_York,_New_York_County,_NY_HAER_NY,31-NEYO,89-241_%28CT%29.tif) | Nose dorsum, lip projection, cheek/mandible depth, brow and hair relationship | **Observed**, public-domain NPS/HAER photo; reference only | High for contour, medium for precise depth due perspective |
| HAER-O | [Jet Lowe, installed head oblique, HAER NY-138-236](https://www.loc.gov/pictures/item/ny1251.color.570005c/) | Actual brow overhang, deep orbits, nasal/cheek transition and hairline | **Observed**, public-domain NPS/HAER photograph; camera unregistered | High for qualitative form |
| HAER-R | [Jet Lowe, installed right profile, HAER NY-138-237](https://www.loc.gov/pictures/item/ny1251.color.570006c/) | Nose-tip / philtrum / lips / chin silhouette, opposite-side asymmetry check | **Observed**, public-domain NPS/HAER photograph; poor shadow exposure | High for visible contour, low for depth measurement |
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
