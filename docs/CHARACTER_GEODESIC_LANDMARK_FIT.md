# Geodesic landmark fitting for source-faithful mesh parts

The registered orthographic landmark solver already recovers a candidate 3D position of eyes, beak tips, muzzle features or other source points from original images. Native anatomy can build **new** meshes around those points.

This workflow adds the missing second operation: move specific vertices in an **existing detailed pre-rig mesh** toward those positions, deforming only its own connected surface neighborhood. It is designed for iterative facial/wing/tail-part correction on approved meshes, not an AI reconstruction model.

## Why use geodesic distances?

A simple spatial brush near the face can accidentally deform a disconnected ear, eye or tongue sitting just millimeters away. This solver finds shortest paths **along actual mesh edges** from each manually associated source vertex. It applies a smooth bounded falloff only to vertices reachable within that surface distance.

The tool does not guess which source landmark belongs to which vertex. You supply that stable vertex index, and any critical boundary vertex can be marked protected; protected vertices do not move **or transmit influence** across a branch.

## Step 1. Triangulate the source landmarks

Use [multi-view reference landmarks](CHARACTER_LANDMARK_TRIANGULATION.md) and original 1600x1200 aligned artwork:

~~~bash
kumiki-character-build triangulate-landmarks \
  --spec /path/to/registered-landmarks.json \
  --output .review/characters/pip/source-points-r01
~~~

The solver verifies photo hashes, view-pair observability and 2D reprojection errors. Landmark IDs are permanent, e.g. eye.L.corner, beak.tip, cheek.L.outer.

## Step 2. Write a small fit specification

Supply the exact Blender object name, vertex count, constraints and source vertex associations:

~~~json
{
  "schema_version": 1,
  "mesh": "FacePatch",
  "expected_vertex_count": 169,
  "max_displacement_m": 0.005,
  "max_relative_edge_change": 0.25,
  "protected_vertices": [0, 12, 156, 168],
  "pins": [
    {
      "id": "eye.L.corner",
      "source_vertex": 84,
      "radius_m": 0.035
    }
  ]
}
~~~

Values are examples only, **not Pip/Quill measurements**. Select reviewed vertex indices in the source Blender cage, not indices from a different retopology revision. The target 3D point is sourced directly from the corresponding triangulated landmark report: no arbitrary XYZ edit goes into the fit spec.

A fit spec supports 1–24 independent landmarks, a bounded displacement, maximum edge-length strain, protected boundary vertices, and an existing unrigged mesh of 300,000 vertices or fewer.

## Step 3. Plan and run the native Blender worker

~~~bash
kumiki-character-build plan-landmark-fit \
  --blend /path/to/approved-unrigged-character.blend \
  --landmarks .review/characters/pip/source-points-r01/landmarks.json \
  --spec /path/to/fit.json \
  --output .review/characters/pip/fit-plan-r01

kumiki-character-build fit-landmark-surface \
  --plan .review/characters/pip/fit-plan-r01/landmark-fit-plan.json \
  --output .review/characters/pip/landmark-fit-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
~~~

The output contains a separate **landmark-fitted.blend**, part-only **landmark-fitted.obj** (legacy interchange), **landmark-fitted-source.obj** (canonical source camera), and **report.json**. The source .blend is never overwritten, and source Blender / landmark report / input specification hashes are checked.

The new part mesh retains original vertex indices, UV corner coordinates, material slots and mesh topology. All unrelated meshes are verified unchanged. Multiple overlapping landmark influences are blended, but the explicitly pinned vertex is moved exactly to the triangulated target. Incompatible pins, boundary moves, oversized displacements, surface flips, collapsed faces and excessive edge strain fail without saving an output.

## Explicit interchange versus source-camera OBJ formats

Every character-build exporter retains the original OBJ filename in the historical
**Y-up / -Z-forward interchange convention**, matching existing Blender importers.
Do not silently alter those files or pass a canonical source OBJ to the default importer.

For registered source-art comparisons, each exporter also emits a separate
**Z-up, -Y-forward source-frame OBJ** (`*-source.obj`, or `parts/<id>.source.obj`).
Blender's OBJ operator preserves native coordinates with `forward_axis=Y, up_axis=Z`;
this does not change the character's own -Y-facing anatomy.
Use that one with `compare-region`, never the legacy file.

Native Blender tests verify both that source-frame vertices stay in original
world coordinates and that legacy OBJ files still import in the prior frame.

## Required source and deformation QA

Run **compare-region** with verified source-image masks and inspect the original front, side and three-quarter drawings. Then run joint/topology audits and **validate-rig** after re-skinning.

This tool does *not* infer hidden anatomy, source textures, vertex correspondences, or semantic control points from images. It cannot operate on a mesh with shape keys, vertex groups, armature modifiers or parented deform bones: fit the pre-rig sculpt or cage first. Nonmanifold meshes and intersections between separate objects still need review.

A passing numerical fit is a **geometry candidate**, not a claim that Pip or Quill now matches the character sheet or can be reliably animated.
