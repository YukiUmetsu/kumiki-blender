# Source-locked geodesic sculpting for character parts

This incremental M2 implementation extends the existing **refine-region** command; it does not replace source-matched anatomical modeling, eye/feather authoring, skinning, or source QA.

## Why

A Euclidean sculpt sphere can pass through thin geometry. Moving the front of an eyelid may also distort a nearby back surface or separate muzzle shell, even when its vertices are disconnected in the mesh. **Geodesic** mode uses shortest paths along mesh edges from an explicit, reviewed seed vertex and never crosses disconnected topology or pinned vertex barriers.

Unlike a Blender sculpt brush's visual effect, this workflow is serialized, bounded and repeatable. It preserves other mesh objects, mesh topology, UV coordinates, material slots and the original `.blend` file. Changes remain candidates pending registered-source and deformation review.

## 1. Inspect the saved source revision

```bash
kumiki-character-build inspect-region-sculpt-source \
  --blend /path/to/character-approved.blend \
  --object Eye.L \
  --anchor 0.035 -0.053 0.275 \
  --output .review/pip/eye-L-probe-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

This writes `report.json` without changing the input `.blend`. It includes:

- `expected_vertex_count`, `expected_mesh_sha256`: a fingerprint of vertex positions and polygon connectivity for this one named mesh.
- `anchor_candidates[].nearest_vertex`, `distance_m`: the closest vertex to each requested **world-space** anchor.

The suggested seed is **not automatically verified** to belong to the intended anatomical surface. Inspect the mesh/point in Blender before using it. A seed too far from the requested 3D anchor is rejected. The mesh digest does not cover materials or UVs, so original reference-image and texture QA remain mandatory.

## 2. Create a source-locked sculpt plan

Use the *actual* hash, count, and reviewed seed returned by the probe. Values below are illustrative and must be replaced:

```json
{
  "schema_version": 1,
  "object": "Eye.L",
  "expected_vertex_count": 800,
  "expected_mesh_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "protected_vertex_indices": [40, 41, 42],
  "max_displacement_m": 0.008,
  "operations": [
    {
      "center_m": [0.035, -0.053, 0.275],
      "radius_m": 0.014,
      "move_m": [0, -0.004, 0],
      "falloff": "smoothstep",
      "distance_mode": "geodesic",
      "seed_vertex": 125,
      "max_seed_distance_m": 0.003
    }
  ]
}
```

- `seed_vertex` and `protected_vertex_indices` refer to the **source mesh's vertex order**, not landmark labels, vertex-group names or indices from a separate OBJ.
- Protected indices receive **zero movement**. In geodesic mode they also block surface traversal; pin a full boundary loop to isolate eyelids/wing roots from neighboring tissue.
- Radius measures **edge-path distance from the seed** in world metres, rather than straight-line distance from the given `center_m`. The anchor-to-seed tolerance is independently enforced.
- The source mesh SHA-256 and vertex count are **mandatory** for geodesic seed operations, so unchanged counts with changed vertex positions do not silently move the wrong vertices.
- The existing `euclidean` mode and older sculpt plans remain supported. Geodesic mode does not fill missing geometry or construct anatomical parts.

## 3. Run and independently validate

```bash
kumiki-character-build refine-region \
  --blend /path/to/character-approved.blend \
  --plan /path/to/eye-L-geodesic-sculpt.json \
  --output .review/pip/eye-L-geodesic-r01 \
  --character .review/pip/character-plan-r01 \
  --region eye.L \
  --max-regression 0.01 \
  --max-contour-regression 0.03 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

`refine-region` returns an independent `refined.blend`, the legacy handoff `region.obj`, source-frame `region-source.obj`, and a report with the source mesh fingerprint, movement budget, protected-vertex count and per-operation method/affected vertices. It rejects a stale fingerprint, changed count, excessive displacement, flipped face normals, modifications to other mesh objects or a mismatched loaded source.

With `--character` and `--region`, the command **automatically** executes the existing original-art `audit-region-revision` numeric gate. The character plan must contain verified camera views and either an original `source_mesh` or an explicit `--qa-before` source-frame OBJ/PLY/GLB. A failed per-view silhouette or contour check exits **1** and leaves the rejected sculpt with a `source-fidelity/report.json` and `source-fidelity-receipt.json` for diagnosis. Do not promote that candidate. Without these flags, `refine-region` retains legacy manual-review behavior.

Only `region-source.obj` uses the source-fidelity gate's registered Z-up coordinates. Continue with `validate-rig` on a rigged revision after geometry acceptance, and inspect each verified source view. Numerical acceptance is **never** a claim of feature-film quality or faithful hidden anatomy.

## Test coverage and remaining work

- `tests/test_character_surface_distance.py`: thin disconnected sheet, protected boundary, shortest-path cutoff, bad geometry.
- `tests/test_character_region_sculpt_plan.py`: additive schema, source fingerprint, seed/tolerance bounds.
- `tests/test_character_region_sculpt_source_qa.py`: source regression blocks promotion, even when Blender sculpting succeeds.
- `scripts/smoke_character_controls_topology_blender.py`: native Blender saved-mesh inspection and geodesic sculpt exercise (subject to official Blender availability).

Still required: real Pip/Quill source-registered multi-view benchmarks, sculpting detailed eye/feather/hand parts, deformation-aware retopology, species-specific weight painting, correctives, animated rigs, and Blender-to-Kumiki-UI parity.
