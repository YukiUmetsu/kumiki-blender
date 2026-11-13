# Character build: reference images to a fitted, rigged model

`kumiki-character-build` turns a checked set of orthographic character
reference images into 3D data. The data includes a blockout, a silhouette fit
of any mesh, vertex colours, feather or fur cards, quad retopology, a rig, and
part assembly. Each stage measures its result against the approved images.

The reference images come from the `character-modeling-references` skill.
The skill names the views and gives the prompts.
[Character fidelity QA](CHARACTER_FIDELITY_REVIEW.md) stays the detailed
part-by-part review. This tool writes a manifest for it.

## Install

```bash
python -m pip install -e '.[character-build]'
kumiki-character-build --help
```

The `blender-setup`, `retopo`, `rig` and `assemble` commands need Blender 4.2
or newer. The other commands need only NumPy, SciPy, scikit-image and Pillow.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Name the images | `ingest SRC map.json REFS` | Copies with view names: `front.png`, `left.png`, `front_left.png`, ... |
| 2. Check them | `check REFS` | `REFS/check/report.md`, mirror overlays, hull overlays |
| 3. Make 3D data | `handoff REFS --height-m 0.35` | `REFS/handoff/`: aligned views, masks, `blockout.obj`, `spec.json`, generator inputs |
| 4. Fit a mesh | `fit HANDOFF mesh.glb --output fitted.ply` | The mesh, deformed to match every view |
| 5. Colour it | `project HANDOFF fitted.ply --output coloured.ply` | Vertex colours from the views |
| 6. Add cards | `scatter coloured.ply --colour 94,70,58 --output feathers.obj` | Feather, fur or scale cards on one colour region |
| 7. Retopology | `retopo fitted.ply --output retopo/` | `low.obj` (quads, UVs), `low_normal.png`, `low.glb` |
| 8. Joints | `joints HANDOFF fitted.ply --limbs wings --output joints.json` | Editable joint positions |
| 9. Rig | `rig retopo/low.obj joints.json --output rig/ --render-rom` | `rigged.glb`, range-of-motion report and renders |
| Any time | `compare HANDOFF model.glb` | IoU, missing, extra and height error for each view, with overlays |
| Parts | `assemble plan.json --output asm/` | Parts placed with anchor pairs, seams measured |
| Blender | `blender-setup HANDOFF --output refs/ --save` | Reference cameras, image empties and blockout in a `.blend` |

Each command writes a new file or folder. It does not overwrite an output.

## Views and frame

- View names are camera positions: `front`, `front_left`, `left`, `back_left`,
  `back`, `back_right`, `right`, `front_right`. `three_quarter` is an alias of
  `front_right`, and `three_quarter_back` is an alias of `back_left`.
- Axes are Blender axes. Z is up, the character faces -Y, and its right side is at -X.
- The frame is 1600 x 1200 px. Each view is scaled so the figure is 1000 px
  tall, stands on row 1100 and is centred on column 800. The `REF_CAM_<view>` cameras
  from `blender-setup` render exactly this frame. Diagonal views are shifted to
  the camera frame by the offset that the hull carve measures.
- Blender's numpad 3 "Right" view shows the character's left side.

## What each check means

- **Mirror checks.** In orthographic projection the back outline is the mirror
  image of the front outline. A failure means perspective, a changed pose or a
  redrawn part.
- **3D agreement.** The tool carves a voxel hull from all full-body views. For
  each view it reports the fraction of the silhouette that the hull cannot
  explain. A high value means that no single 3D shape can make all the views.
  The front and the primary side carve exactly. The other views get a
  tolerance of 1% of the height.
- **Direction.** A diagonal view warns only when another diagonal direction
  fits clearly better. Front and back 3/4 silhouettes are often near mirror
  images, so a small difference decides nothing.

## Stages in detail

- **Blockout.** The visual hull with rounded cross-sections (a superellipse,
  exponent 2.5, inscribed in each front x side rectangle). The rounding keeps
  the front and side silhouettes. Use `--rounding 0` for the box hull when wide
  parts (wings held away from the body) need the box corners.
- **Fit.** Vertices outside a reference silhouette move in. Outline vertices
  inside it move out. The front and primary side have double weight. Only the
  displacement is smoothed, so regions that need no move keep their detail.
  For a generator mesh, the tool chooses the facing (four quarter turns) and
  scales the mesh to the reference height first.
- **Colours.** A vertex takes colour only from views where it is visible. The
  test uses a depth buffer with a slope-aware tolerance. Each view is weighted
  by how directly it sees the vertex. `--assume-symmetric` fills a missing side view with the
  mirrored opposite side. Do not use it for a one-sided accessory.
- **Relief.** `--relief-mm` moves vertices by the high-pass of the projected
  `*_clay` views. This is stylized relief from shading, not measured geometry.
  Flat clay gives no relief.
- **Retopology.** QuadriFlow, with a voxel remesh fallback. The worker scales
  the mesh to 1.5 m for QuadriFlow: QuadriFlow refuses small meshes (a 0.35 m
  figure) with a misleading "not manifold" message.
- **Joints and rig.** The spine follows the body midline, which can differ
  from x = 0 when an accessory hangs on one side. Legs are found above the
  feet, and arms or wings are the lateral mass. Landmark heights come from
  `landmarks.json` (`crown`, `face_bottom`, `shoulder`, `hip`, `knee`, in % of
  height from the top). The rig uses heat weights. Vertices without a weight
  get their nearest bone, and the report gives that fraction. Influences are
  limited to 4 for glTF.

## Results on the two reference sets (2026-11-12)

| | Quill | Pip |
|---|---|---|
| Views | front, back, left, front_left, head and foot close-ups | front, back, left, right, front_left, back_left, head close-ups |
| Mirror checks | back 0.963 PASS | back 0.81 FAIL, left 0.82 FAIL |
| 3D agreement | every view <= 0.03 unexplained | front 0.13, left 0.10, right 0.09 unexplained: FAIL |
| Cause | consistent views | the tail fans out behind the head in the front view, but the back and side views draw a narrow tail |
| Blockout IoU | 0.95 mean (3/4 view 0.89) | not built: regenerate the views first |
| Fit | 3/4 view 0.89 -> 0.93, front and side -0.01 | |
| Retopology | 10k quads, normal bake | |
| Rig | 15 bones, all vertices heat-weighted; worst range-of-motion volume change 14% (wing turned into the body) | |

## Limits

- A silhouette hull cannot place outline detail in 3D. Jagged outlines (feather
  tips) make horizontal ridges, and a part seen only from the front (a wing)
  becomes a plate. For organic detail, generate a mesh from `handoff/gen/`
  with an image-to-3D tool and `fit` it.
- With few views the hull is boxy in diagonal directions. Each extra
  orthographic view (`front_right`, `back_right`, ...) removes a corner.
- The fit cannot satisfy views that disagree. It trades between them. Anchor
  weighting keeps the front and side close.
- Joint estimates are heuristics for biped creatures. Read `joints.json` and
  correct it before `rig`.
- The limits in `checks.py` are start values from one synthetic set and two
  AI-generated sets.

## Tests

```bash
PYTHONPATH=src:tests python -m unittest tests.test_character_build
PYTHONPATH=src:tests python scripts/smoke_character_build_blender.py   # needs Blender
```
