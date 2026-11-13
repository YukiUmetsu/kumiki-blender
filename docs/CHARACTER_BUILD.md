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
| Generators | `place HANDOFF a.glb b.glb --output ranked/` | Repaired, solid, oriented candidates ranked by IoU |
| Generators | `solidify mesh.glb --output solid.ply` | A hollow or badly wound mesh rebuilt as one solid |
| Generators | `swap BASE DONOR --region "y>0.05" --output sw/` | One region taken from another generation |
| Blender | `blender-setup HANDOFF --output refs/ --save` | Reference cameras, image empties and blockout in a `.blend` |

Each command writes a new file or folder. It does not overwrite an output.

## Image-to-3D route (recommended for detail)

A shape from silhouettes knows only the outlines. Ears, a snout and a tail
volume need an image-to-3D generator. The pipeline then places, scores,
corrects, colours and rigs the generated mesh.

1. **Generate.** [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) runs on
   Apple Silicon through [trellis-silicon](https://github.com/sanchez-kim/trellis-silicon)
   (MPS and Metal). Give it the aligned views with a transparent background
   (`handoff/aligned/<view>.png`), one view per run. A 3/4 view works best:
   on Pip, front-left scored 0.85, right side 0.80 and front 0.69.
2. **Rank.** `place HANDOFF a.glb b.glb ... --output ranked/` repairs each
   candidate (UV seams, face winding, hollow shell) and finds its facing in
   3 degree steps. The side views tell the front from the back. Then it scores
   each candidate against every view. Run several seeds and keep the best.
3. **Correct.** `fit HANDOFF ranked/best.ply --no-normalise --smooth-radius-px 40`.
   A large smoothing radius changes proportions only and keeps the generated
   detail. The default radius (4) crumples a detailed mesh.
4. **Finish.** `project`, `joints`, `retopo` and `rig` as for a blockout.
5. **Optional part swap.** `swap BASE DONOR --region "y>0.05" --region "z>0.04"`
   aligns the donor to the base with a two-way similarity ICP outside the
   region, then cuts and fuses closed solids. It works when a box separates the
   part. It does not work when the part wraps around the body.

TRELLIS.2 exports a hollow, inconsistently wound shell, so `place` and
`solidify` rebuild it as one solid. On a 24 GB Mac the 512 pipeline peaked at
26 GB (memory compression) and took about 5 minutes per mesh.

Setup notes for trellis-silicon:

- Install Eigen (`brew install eigen`) and build `o_voxel` with
  `CPLUS_INCLUDE_PATH=/opt/homebrew/include/eigen3`, or it falls back to a
  slower texture baker.
- TRELLIS.2 loads the RMBG-2.0 background remover at start, which is gated and
  licensed for non-commercial use. Inputs with a transparent background never
  use it. In `TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py`, wrap the
  `rembg_model` load in `try/except OSError` and set it to `None`.

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

- **Blockout.** The tool blurs each view's 2D signed distance field by 12 px.
  The blur removes the wobble of fur and feather fringes. It extrudes each field
  along its camera and intersects them with a smooth maximum. The smooth maximum
  (6 px) rounds the edges where views meet. Floating pieces with less than 5% of the body's area are dropped. The
  blur shortens thin tips (ears, tufts) by about 2% of the height. The fit
  restores most of it. Options: `--outline-blur-px`, `--round-px`.
- **Fit.** Vertices outside a reference silhouette move in. Outline vertices
  inside it move out. The targets use the same blurred outline as the
  blockout, so the fit does not groove the surface row by row. The front and
  primary side have double weight. Only the displacement is smoothed. A flip
  guard cancels each move that turns a triangle more than 90 degrees from its
  start. A final local smoothing removes each crease sharper than 90 degrees. The
  report gives the remaining folds (0 on both test characters). For a
  generator mesh, the tool chooses the facing (four quarter turns) and scales
  the mesh to the reference height first.
- **Colours.** A vertex takes colour only from views where it is visible. The
  test uses a depth buffer with a slope-aware tolerance. Each view is weighted
  by (n . d)^4, and 3/4 views count 0.1, so they fill only what the front, side
  and back views see poorly. Otherwise an AI 3/4 view paints its own face
  beside the front view's face (ghost eyes). `--assume-symmetric` fills a missing side view with the
  mirrored opposite side. Do not use it for a one-sided accessory.
- **Relief.** `--relief-mm` moves vertices by the high-pass of the projected
  `*_clay` views. This is stylized relief from shading, not measured geometry.
  Flat clay gives no relief.
- **Retopology.** QuadriFlow refuses a mesh with folded triangles, sliver
  triangles, floating debris or more than about 250k faces. It gives one
  misleading "not manifold" message for all of them, and its result also
  depends on the mesh's absolute size. Large meshes are decimated to 150k faces
  first. The quads are then relaxed and shrink-wrapped onto the high mesh. The worker removes
  debris, merges vertices closer than a tenth of the median edge, tries
  QuadriFlow at several sizes, then at a voxel-remeshed copy. If every try
  fails, a voxel remesh sized to the target face count makes grid-aligned
  quads. The report then warns that they have no edge loops.
- **Joints and rig.** The spine follows the body midline, which can differ
  from x = 0 when an accessory hangs on one side. Legs are found above the
  feet. Arms or wings are the lateral mass. Tail joints sit at centroids of
  distance bands from the tail root, so they follow a fanned or curled tail. Landmark heights come from
  `landmarks.json` (`crown`, `face_bottom`, `shoulder`, `hip`, `knee`, in % of
  height from the top). The rig uses heat weights. Vertices without a weight
  get their nearest bone, and the report gives that fraction. Influences are
  limited to 4 for glTF.

## Results on the two reference sets (2026-11-12)

| | Quill | Pip, first set | Pip, remade with the updated skill |
|---|---|---|---|
| Views | front, back, left, front_left | front, back, left, right, front_left, back_left | the same six views |
| Back mirror | 0.963 | 0.81 FAIL | 0.935 |
| Left / right mirror | | 0.82 FAIL | 0.905 |
| 3D agreement (worst view unexplained) | 3% | 13% FAIL | 5.7% WARN (left drawn slightly larger) |
| Blockout IoU | 0.944 | not built | 0.940 |
| Fitted IoU, folds | 0.955, 0 | | 0.959, 0 |
| Retopology | 11k quads, QuadriFlow | | 11k quads, QuadriFlow |
| Low-poly IoU | 0.952 | | 0.957 |
| Rig | 15 bones, all heat-weighted | | 18 bones with a 3-bone tail, all heat-weighted |
| Worst range-of-motion volume change | 14% (wing into the body) | | 43% (tail bent 45 degrees into the back) |

Remade Pip through TRELLIS.2 used the front-left view, seed 42, the best of
four seeds. Results:

- Placed IoU 0.848, fitted IoU 0.880, height error -0.4%.
- 87% of the surface coloured from the views.
- QuadriFlow retopology with 13k quads, low-poly IoU 0.877.
- An 18-bone rig with every vertex heat-weighted.

All four seeds read the fanned tail as two lobes. The fit narrows the gap.

The first Pip set drew a fanned tail in the front view and a narrow tail in
the back and side views. The remade set used the skill's volume lock and
turnaround strip, and the views agree.

## Limits

- A shape from silhouettes cannot place outline detail in 3D, and a part seen
  only from the front (a wing) becomes a plate. Heads come out boxy with few
  views. For organic detail, generate a mesh from `handoff/gen/` with an
  image-to-3D tool and `fit` it.
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
