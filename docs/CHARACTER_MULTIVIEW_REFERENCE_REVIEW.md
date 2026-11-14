# Multi-view character source-art review (Pip and Dr. Quill)

This step addresses a real gap left by native rig, animation-export and GLB
roundtrip testing: mathematically correct mesh deformation does not imply
the 3D character resembles the original artwork.

It creates **frame-locked, orthographic clay renders** at known camera
directions and makes paired contact sheets alongside the actual source art.
The result is a visual review package, not an automatic artistic pass.

## Inputs

A usable skinned and optionally animated saved Blender source revision is
required. This does **not** convert an illustrated PNG or Character Studio
layer ZIP into a faithful volumetric mesh. Build and approve the source model
first.

With the canonical Blender modeling frame (+Z up, -Y front), cameras are
available for front, right, back, left, front_right, front_left, back_right,
back_left, top. Choose at most six per command.

You may also pass an optional source-art reference manifest. **Each view must
already be one image, not an uncut multi-panel sheet.** Front is the
design-lock anchor; any other full-body view is compared visually against it.
Perspective illustrations and unregistered art cannot be used to compute
pixelwise silhouette scores.

Example reference manifest JSON:

    {
      "schema_version": 1,
      "views": {
        "front": "/project/pip/references/front.png",
        "right": "/project/pip/references/right.png",
        "back": "/project/pip/references/back.png"
      },
      "regions": [
        {
          "id": "eyes",
          "object": "Head",
          "vertex_group": "REGION_Eyes",
          "views": {
            "front": "/project/pip/references/head_front_clay.png"
          }
        },
        {
          "id": "tail",
          "object": "Tail",
          "vertex_group": null,
          "views": {
            "right": "/project/pip/references/part_tail_right.png"
          }
        }
      ]
    }

All referenced paths must be local image files. Each PNG/JPEG/WEBP and the
saved Blender source are locked by SHA-256 before Blender launches and
rechecked after capture. The art files are never committed or changed.

For regions, use either the exact name of a character mesh object (for an
entire separate tail, wing or eye assembly), or an existing semantic
vertex group on that mesh. The vertex-group option makes close-ups around
the modeled anatomy inside a larger mesh, while keeping the **whole visible
character** in context. This avoids hiding a collar, bag strap or joint
connection just to make one body region look better. A region needs 3+
positively weighted vertices and no topology-changing evaluated modifier.

## Command

    kumiki-character-build render-character-review \
      --blend /project/characters/pip/approved-rig.blend \
      --armature CharacterRig \
      --frames 1,12,24 \
      --views front,right,back,left,front_right \
      --references /project/pip/references/review-manifest.json \
      --resolution 720 \
      --output /project/review/pip/multiview-r01

This renders the same explicit Blender frames across chosen orthographic views.
There is a **48-render maximum** across full character, anatomy regions,
camera angles and animation frames. Split work into separate immutable
revision outputs for larger audits.

A square 720-pixel Workbench neutral-clay render is produced for each
region/view/frame. The camera target and ortho-scale are computed from the
*union of the evaluated geometry across requested frames*, so they **do not
recenter or zoom** as a paw, wing or tail moves. The source rig is not saved
or changed, and unrelated scene scenery is hidden from the temporary
review captures. Character mesh accessories parented to the rig are retained.

Results include:

- images/full-front-f00001.png and corresponding requested views/frames
- images/eyes-front-f00012.png or part-specific images when configured
- sheet-full-f00001.png and sheet-eyes-f00012.png, with the source art beside
  the exact clay render and explicit camera labels
- report.json: art/source SHA locks, view and pose identity, evaluated
  full/region camera bounds, visible parts, and links to all frames/sheets

This command creates an **inspection-only** result:
source-visual-review-required. It never reports “source-faithful” merely
because rendering and SHA checking succeeded.

## Quality review expectations

For Pip: inspect the approved face/eye scale and arrangement, animal (not
human) hand anatomy, relaxed shoulder height, bag/strap drape, and tail
attachment/side placement. Use neutral and actual hand-raising/walking frames;
the camera must not move between pose samples. Check the tail from both sides
and at rear three-quarter angles.

For Dr. Quill: inspect the beak, large eyelids, brow/eye proportions, feather
clusters, neck transition, wing/tail root joints, and costume layering against
the actual master illustration and verified orthographic views. Review
corrective wing deformation at neutral and raised-wing limits.

Compare each close-up independently; do not let a good full-body image hide
weak eyes or paws. Check that every piece attached to the rig appears in
the output. A front-facing thumbnail alone is not enough for volumetric QA.

### What this does NOT do

- It does **not** align a sketch to a different orthographic camera to
  artificially inflate an intersection-over-union score.
- It does **not** automatically measure likeness from unregistered 2D source
  art, or approve a character based on a generic pixel difference.
- Workbench neutral clay intentionally ignores final PBR textures, fur,
  fabric materials, and final lighting.
- Images with perspective distortion or different poses are paired only for
  manual appearance review.
- Topology-changing modifiers may prevent reliable vertex-group indexing;
  use a same-topology review LOD or a separate mesh focus instead.
- The command reviews a saved Blender rig, not direct Image-to-3D output.
  It does not synthesize missing Pip or Quill geometry.

For numeric comparisons from **registered** artwork and camera frames,
use the existing CharacterSpec v2 region-compare workflow and silhouette
masks instead. This turntable tool provides the missing visual evidence
package so that a human or separate vision reviewer can identify which
body part deserves focused sculpting.

## CI evidence

The native Blender smoke builds a rigged synthetic character with a moving
wing and a parented accessory, renders full-body and wing close-ups in
multiple orthographic views, checks camera stability across animation
frames and source hashes, and confirms that the wing's rendered pose changes.
Synthetic CI does not establish artistic resemblance to Pip/Quill. Real
character QA needs their completed local .blend models and approved images.
