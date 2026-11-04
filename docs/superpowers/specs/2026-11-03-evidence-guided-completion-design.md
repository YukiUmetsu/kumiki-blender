# Evidence-Guided 3D Completion: Initial Integration Design

## Intent

Kumiki Blender can reconstruct only surfaces supported by the source photographs.
It needs a non-destructive way to add *candidate* geometry from scans, artists,
or generative models while retaining the original reconstruction and provenance.
The intended consumers are historical documentary and cinematic video assets.

## Goals

- Two explicit modes, `historical` and `cinematic`, sharing one candidate workflow.
- Historical builds may incorporate only geometry with identified observational
  evidence (scan or survey); generative/artist hypotheses remain review-only.
- Cinematic builds may incorporate generated geometry, always labeled inferred.
- Accept existing portable GLB candidate outputs from interchangeable generation
  tools, with a recorded adapter/model identity and input hashes. The first
  adapter is `supplied-glb`: users can bring outputs of SPAR3D, TRELLIS, WaLa,
  Hunyuan or a manually sculpted/verified scan without installing those models.
- Validate source files, ownership boundaries, per-region 3D anchor constraints,
  and mode policy before building; generate a read-only JSON planning report.
- Build new, editable `.blend` and `.glb` revisions through Blender without
  touching source files; mark each imported component as observed or inferred.
- Report actual geometric anchor error and source SHA-256, retaining limitations.

## Non-goals for this initial PR

- Installing, downloading, or executing third-party AI weights, model code, or
  photo services automatically. Concrete model runner adapters follow after
  platform/license smoke tests, using the same candidate contract.
- Pretending newly generated views constitute independent photographic evidence.
- Automatic historical authentication, mesh welding, watertight guarantees,
  texture inpainting, automatic region discovery, or photographic likeness
  certification. Those require separate validation and iteration.
- Altering or replacing the original reconstructed mesh or source project.

## Inputs and modes

The JSON manifest is bounded and source-relative. It explicitly declares the
base geometry origin (`observed` or `inferred`); historical mode refuses an
inferred base, including any previously tagged inferred meshes inside it.
Each region names an existing
GLB, its evidence class (`observed-scan`, `observed-survey`, `artist-hypothesis`,
`generated`), and generator provenance (`adapter`, optional `model_id`). A
reference to the original base GLB and each candidate is pinned by SHA-256;
paths cannot escape the manifest's directory, including through symlinks.

Each region supplies >=4 unique, non-coplanar source/target landmark pairs in
Blender Z-up metres. The coordinates describe the same physical features in
the candidate and base mesh. Proper similarity registration returns scale,
rotation, translation, RMS and worst residual, with a mandatory per-region
limit. Invalid, mirrored, degenerate, or high-error registrations are rejected.

`historical`: a generated or artist hypothesis is reported as excluded and
is never exported into the model. `cinematic`: those candidates are included
as inferred geometry, not measured geometry. Neither mode automatically
asserts publication approval; even observational sources require review.

## Architecture

- `reconstruction/completion.py`: manifest parsing, provenance, candidate
  policy and geometric fit report, independent of `bpy` and external AI libs.
- `blender_completion.py`: isolated Blender scene, imports of base and permitted
  candidates, proper transforms of candidate hierarchies, semantic tags,
  standard revision exporter and preview views.
- `cli.py`: `plan-completion` and `build-completion`; errors are actionable;
  builds refuse existing destinations and require installed Blender.
- `docs/EVIDENCE_GUIDED_COMPLETION.md`: manifest example, explicit limitations,
  future model adapter contract, and review steps.

The base mesh stays intact. Initial assembly deliberately keeps each candidate
as a separate imported mesh (no destructive Boolean union or implicit seam
repair). Alignment error measures supplied landmarks only; photographic
validation remains a separate required review stage using existing tools.

## Alternatives considered

1. Generated images fed into COLMAP: rejected as ground truth because synthetic
   views can strengthen a wrong geometry estimate.
2. Replace the reconstruction with a complete image-to-3D asset: fast but loses
   measured surfaces and repeatable provenance.
3. **Selected:** constrain separate candidate parts to known 3D anchors,
   retain measured source, and keep the evidence/inference boundary explicit.

## Acceptance

- Read-only planning prints per-region inclusion decision and fit errors.
- Historical mode excludes generated hypotheses; cinematic mode includes them.
- Invalid paths, GLBs, source digests, mode, anchors, and residuals fail closed.
- Builds produce immutable Blender/GLB revisions with candidate provenance.
- Tests cover both policies, alignment, validation, CLI preflight, and output
  refusal without requiring downloads, Blender, or external AI weights.
