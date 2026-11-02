# Advanced Photographic Material Finishing

## Intent

Extend the existing private, hash-pinned OpenMVS OBJ/MTL/UV-atlas finishing pipeline to produce more relightable surfaces from multi-angle photo reconstructions. Keep original imagery and geometry immutable, and require visual review before any claim of physically measured reflectance or photo fidelity.

## Approaches

1. **Unsupervised image-only intrinsic decomposition:** automatically treat low-frequency image intensity as illumination. Convenient but risks erasing authentic material color and patina and inventing reflectance, especially across cast-shadow edges.
2. **Guided, opt-in finishing (selected):** user-authored UV-space material masks, optional illumination and defect masks, roughness priors, and genuine height maps. These constrain bounded non-destructive corrections and make outcomes reproducible. A conservative low-frequency fallback is permitted, clearly labeled heuristic.
3. **Learned multi-view inverse rendering:** potentially stronger but requires calibrated illumination, a learned model and substantial new GPU infrastructure. Defer rather than quietly claim photo-perfect reconstruction.

## Boundaries

- Use a strict local JSON manifest and hashed masks/maps. No hidden downloads, no arbitrary execution, no modification of source photos.
- Base color de-lighting operates on linear-light RGB under per-material masks, with an explicit guidance map when possible, limited correction strength and EV, and before/after QA images.
- Defect removal applies only to user-supplied masks, preserving unmarked scratches/patina; small masked regions are conservatively filled from neighboring valid pixels.
- Roughness is a per-material prior or supplied measurement, **not** an inference from diffuse color; use a per-atlas roughness map for portable GLB.
- Displacement uses externally supplied height maps; Cycles gets a Blender displacement node, while GLB is explicitly documented not to encode true displacement.
- Provide an optional UDIM package with OBJ UV remapping, 1001+ texture tiles and a provenance manifest, independent of the portable GLB pipeline.
- Fail closed on invalid paths, missing masks, mismatched dimensions, overlapping material masks and tampered auxiliary maps.

## Non-goals

- No automatic recovery of occluded geometry, fine relief or absent photographic detail.
- No claim that shadow-free albedo, optical roughness or height can be uniquely recovered from ordinary photos.
- No change to existing finishing output unless the advanced manifest is requested.

## Validation

Synthetic gradients and segmented materials demonstrate that guided de-lighting reduces known lighting differences; source bytes are unchanged. Tests check defects, alpha/padding protection, roughness-mask values, 16-bit height preservation, UDIM remapping, hash validation and CLI wiring. Rendering in Blender remains a separate integration check where the Blender executable exists.
