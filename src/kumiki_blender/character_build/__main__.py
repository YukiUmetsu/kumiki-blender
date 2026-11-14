"""kumiki-character-build: reference images -> checked views -> blockout -> fitted, detailed, rigged model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _new_file(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"{path} exists. Each output is a new revision.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _new_dir(path: Path) -> Path:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} exists and is not empty. Each output is a new revision.")
    return path


def _bake_arguments(p: argparse.ArgumentParser) -> None:
    """Options that `texture` and `conflicts` share: which sources, and every threshold of the bake."""
    p.add_argument("handoff", type=Path)
    p.add_argument("mesh", type=Path, help="OBJ with UVs, e.g. retopo/low.obj")
    p.add_argument("--views", help="comma-separated drawings to use (default: all)")
    p.add_argument("--assume-symmetric", action="store_true",
                   help="paint a side without a drawing from the mirrored opposite side")
    p.add_argument("--extra", type=Path, help="manifest.json from `repaint`: add the repainted views")
    p.add_argument("--power", type=float, default=4.0,
                   help="view quality = cos^power: higher keeps each view to the surface it faces")
    p.add_argument("--diagonal-weight", type=float, default=1.0, help="quality factor of 3/4 views")
    p.add_argument("--min-cosine", type=float, default=0.35, help="views at a grazing angle below this do not paint")
    p.add_argument("--region-px", type=float, default=24.0,
                   help="radius of the neighbourhood that decides which view paints (aligned px)")
    p.add_argument("--min-region-px", type=float, default=40.0, help="smaller regions join a neighbour")
    p.add_argument("--occlusion-px", type=float, default=30.0,
                   help="trust a view less this near a place where one part hides another (0: off)")
    p.add_argument("--slack-px", type=float, default=60.0,
                   help="how far apart two drawings may place the same feature")
    p.add_argument("--contest-delta-e", type=float, default=20.0,
                   help="two views show different things at a point above this difference (0: no contested features)")
    p.add_argument("--agree-delta-e", type=float, default=15.0,
                   help="scale of the score loss for disagreement with the other drawings (0: no objections either)")
    p.add_argument("--no-level", action="store_true", help="do not level the tone across seams")
    p.add_argument("--level-reach-px", type=float, default=150.0, help="distance over which a tone offset fades")


def _bake_settings(args: argparse.Namespace) -> dict:
    return dict(views=args.views.split(",") if args.views else None, mirror_missing=args.assume_symmetric,
                power=args.power, diagonal_weight=args.diagonal_weight, min_cosine=args.min_cosine,
                region_px=args.region_px, min_region_px=args.min_region_px, occlusion_px=args.occlusion_px,
                slack_px=args.slack_px, contest_delta_e=args.contest_delta_e, agree_delta_e=args.agree_delta_e,
                level=not args.no_level, level_reach_px=args.level_reach_px)


def _spec(handoff: Path) -> dict:
    return json.loads((handoff / "spec.json").read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kumiki-character-build", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan-region-batch", help="plan deterministic source-locked rebuild tasks with safe Blender worker concurrency")
    p.add_argument("--character", type=Path, required=True, help="frozen CharacterSpec v2 plan directory")
    p.add_argument("--changed", action="append", required=True, help="changed region ID; repeat for multiple roots")
    p.add_argument("--max-workers", type=int, default=1, help="simultaneous heavyweight Blender task ceiling")
    p.add_argument("--memory-gib", type=int, default=24, help="local host memory budget, default M4 24 GiB")
    p.add_argument("--output", type=Path, required=True, help="new immutable batch plan revision")

    p = sub.add_parser("resume-region-batch", help="audit source-bound receipts and list unblocked next Blender part tasks")
    p.add_argument("--plan", type=Path, required=True, help="previous region-batch-plan.json")
    p.add_argument("--receipts", type=Path, help="optional preserved and completed part SHA receipts")
    p.add_argument("--output", type=Path, required=True, help="new immutable batch progress snapshot")

    p = sub.add_parser("character-benchmark", help="collect source-locked region, color, rig and GLB evidence into one review")
    p.add_argument("--manifest", type=Path, required=True, help="frozen CharacterSpec and existing stage report paths")
    p.add_argument("--output", type=Path, required=True, help="new immutable benchmark report + SUMMARY.md")

    p = sub.add_parser("plan-landmark-fit", help="source-lock geodesic mesh fit to explicitly triangulated 3D landmarks")
    p.add_argument("--blend", type=Path, required=True, help="approved unrigged source .blend")
    p.add_argument("--landmarks", type=Path, required=True, help="registered landmark report")
    p.add_argument("--spec", type=Path, required=True, help="explicit source-vertex to landmark mapping")
    p.add_argument("--output", type=Path, required=True, help="immutable source-hashed fit plan")

    p = sub.add_parser("fit-landmark-surface", help="refine one pre-rig mesh locally toward source landmarks along its surface")
    p.add_argument("--plan", type=Path, required=True, help="validated source-landmark geodesic fit plan")
    p.add_argument("--output", type=Path, required=True, help="new landmark-fitted.blend + OBJ revision")
    p.add_argument("--blender")
    p.add_argument("--character", type=Path, help="CharacterSpec v2 plan for automatic registered-view source QA")
    p.add_argument("--region", help="semantic region to audit, e.g. eye.L; requires --character")
    p.add_argument("--qa-before", type=Path, help="optional baseline canonical mesh when region has no source_mesh")
    p.add_argument("--max-regression", type=float, default=.01, help="max permitted source IoU decline in any view")
    p.add_argument("--max-contour-regression", type=float, default=.03, help="maximum permitted source boundary F1 decline in any view")
    p.add_argument("--min-iou", type=float, help="optional per-camera absolute source silhouette floor")

    p = sub.add_parser("triangulate-landmarks", help="recover registered 3D anatomical points from verified 2D views")
    p.add_argument("--spec", type=Path, required=True, help="source-registered 2D landmark observations JSON")
    p.add_argument("--output", type=Path, required=True, help="immutable 3D landmark report directory")

    p = sub.add_parser("resolve-anatomy-landmarks", help="substitute triangulated 3D landmarks into anatomical control points")
    p.add_argument("--spec", type=Path, required=True, help="anatomy template with explicit {landmark: ID} markers")
    p.add_argument("--landmarks", type=Path, required=True, help="source-verified landmarks.json")
    p.add_argument("--output", type=Path, required=True, help="new resolved-anatomy.json with provenance")

    p = sub.add_parser("plan-anatomy", help="source-lock explicit 3D eye, lid and contour-loft anatomy recipes")
    p.add_argument("--blend", type=Path, required=True, help="approved source Blender character .blend")
    p.add_argument("--spec", type=Path, required=True, help="source-authored anatomical parts JSON")
    p.add_argument("--output", type=Path, required=True, help="new immutable anatomy plan revision")

    p = sub.add_parser("build-anatomy", help="build source-anchored editable Blender anatomical parts")
    p.add_argument("--plan", type=Path, required=True, help="anatomy-plan.json")
    p.add_argument("--output", type=Path, required=True, help="new anatomy.blend + part-only OBJ files")
    p.add_argument("--blender")

    p = sub.add_parser("plan-character", help="validate CharacterSpec v2 and freeze a source-hashed character plan")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new immutable character plan directory")

    p = sub.add_parser("plan-region", help="isolate one anatomical region and its dependencies")
    p.add_argument("--character", type=Path, required=True, help="directory with character-plan.json")
    p.add_argument("--region", required=True, help="stable semantic region id, e.g. eye.L")
    p.add_argument("--output", type=Path, required=True, help="new immutable region task directory")

    p = sub.add_parser("audit-region-revision", help="reject a part edit that worsens any registered source silhouette")
    p.add_argument("--character", type=Path, required=True, help="source-hashed CharacterSpec v2 plan folder")
    p.add_argument("--region", required=True, help="verified semantic region ID, e.g. eye.L")
    p.add_argument("--before", type=Path, help="canonical baseline mesh; defaults to source_mesh in CharacterSpec")
    p.add_argument("--after", type=Path, required=True, help="new canonical source-frame OBJ, PLY or GLB")
    p.add_argument("--max-regression", type=float, default=.01, help="maximum allowed per-camera IoU decline")
    p.add_argument("--max-contour-regression", type=float, default=.03, help="maximum allowed per-camera 2-pixel contour F1 decline")
    p.add_argument("--min-iou", type=float, help="optional absolute per-camera source silhouette floor")
    p.add_argument("--output", type=Path, required=True, help="new immutable before/after report and overlays")

    p = sub.add_parser("compare-region", help="compare a region mesh with verified registered source masks")
    p.add_argument("--character", type=Path, required=True, help="directory with character-plan.json")
    p.add_argument("--region", required=True)
    p.add_argument("--mesh", type=Path, required=True, help="aligned OBJ, PLY or GLB candidate")
    p.add_argument("--output", type=Path, required=True, help="new comparison directory")
    p.add_argument("--min-iou", type=float, help="optional per-view geometry threshold; no automatic art approval")
    p.add_argument("--baseline", type=Path, help="prior report.json to reject registered-view regressions")
    p.add_argument("--max-regression", type=float, default=0.01, help="maximum per-view drop in IoU")

    p = sub.add_parser("plan-talking-performance", help="plan a baked body Action plus offline mouth cues")
    p.add_argument("--blend", type=Path, required=True, help="baked deform Action .blend with unkeyed authored mouth shapes")
    p.add_argument("--cues", type=Path, required=True, help="offline Rhubarb-style mouthCues JSON")
    p.add_argument("--clip", required=True, help="single output animation clip name")
    p.add_argument("--start", type=int, required=True, help="inclusive skeletal Action start frame")
    p.add_argument("--end", type=int, required=True, help="inclusive skeletal Action end frame")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--armature", default="CharacterRig")
    p.add_argument("--mouth-object", default="Mouth")
    p.add_argument("--ramp-frames", type=float, default=1.5)
    p.add_argument("--time-alignment-tolerance-s", type=float, default=.05)
    p.add_argument("--output", type=Path, required=True, help="new immutable automated performance plan")

    p = sub.add_parser("export-talking-performance", help="auto-key existing mouth shapes and export synchronized body+face GLB")
    p.add_argument("--plan", type=Path, required=True, help="talking-plan.json")
    p.add_argument("--output", type=Path, required=True, help="new combined.blend and performance.glb revision")
    p.add_argument("--blender")

    p = sub.add_parser("plan-performance", help="lock one simultaneous skeletal and facial Action timeline")
    p.add_argument("--blend", type=Path, required=True, help="source .blend with both active Actions")
    p.add_argument("--clip", required=True, help="single output glTF animation name")
    p.add_argument("--start", type=int, required=True, help="inclusive start frame")
    p.add_argument("--end", type=int, required=True, help="inclusive end frame")
    p.add_argument("--armature", default="CharacterRig")
    p.add_argument("--mouth-object", default="Mouth")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--time-alignment-tolerance-s", type=float, default=.05)
    p.add_argument("--output", type=Path, required=True, help="new immutable performance plan")

    p = sub.add_parser("export-performance", help="combine active deform-bone and shape-key Actions into one GLB clip")
    p.add_argument("--plan", type=Path, required=True, help="performance-plan.json")
    p.add_argument("--output", type=Path, required=True, help="new combined.blend and performance.glb")
    p.add_argument("--blender")

    p = sub.add_parser("plan-visemes", help="plan authored facial morph motion from Rhubarb mouth-cue JSON")
    p.add_argument("--blend", type=Path, required=True, help="existing authored face Blender .blend")
    p.add_argument("--cues", type=Path, required=True, help="local Rhubarb JSON containing mouthCues")
    p.add_argument("--mouth-object", required=True, help="exact Blender mesh object containing shape keys")
    p.add_argument("--clip", required=True)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--first-frame", type=int, default=1)
    p.add_argument("--ramp-frames", type=float, default=1.5)
    p.add_argument("--output", type=Path, required=True, help="new immutable face animation plan")

    p = sub.add_parser("animate-visemes", help="write authored shape-key animation and a morph-target GLB")
    p.add_argument("--plan", type=Path, required=True, help="viseme-plan.json")
    p.add_argument("--output", type=Path, required=True, help="new visemes.blend and face-animated.glb")
    p.add_argument("--blender")

    p = sub.add_parser("plan-animation-bake", help="lock an animated Blender control rig and one clip range")
    p.add_argument("--blend", type=Path, required=True, help="animated source control-rig .blend")
    p.add_argument("--clip", required=True, help="filename-safe clip name")
    p.add_argument("--start", type=int, required=True, help="inclusive start frame")
    p.add_argument("--end", type=int, required=True, help="inclusive end frame")
    p.add_argument("--step", type=int, default=1, help="bake every Nth frame")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--armature", default="CharacterRig")
    p.add_argument("--max-position-error-m", type=float, default=.002)
    p.add_argument("--max-rotation-error-deg", type=float, default=2.)
    p.add_argument("--bake-correctives", action="store_true",
                   help="sample Blender joint-driven shape keys and verify glTF morph animation")
    p.add_argument("--output", type=Path, required=True, help="new immutable bake plan directory")

    p = sub.add_parser("bake-character-animation", help="visually bake Blender controls into deform-only GLB animation")
    p.add_argument("--plan", type=Path, required=True, help="bake-plan.json")
    p.add_argument("--output", type=Path, required=True, help="new baked.blend / animated.glb revision")
    p.add_argument("--blender")

    p = sub.add_parser("plan-rig-controls", help="generate native FK/IK animator control rig from reviewed joints")
    p.add_argument("--joints", type=Path, required=True, help="validated joints.json from existing character rig")
    p.add_argument("--profile", required=True, choices=("generic-biped", "stylized-biped-tail", "avian-biped-wings"))
    p.add_argument("--armature", default="CharacterRig")
    p.add_argument("--pole-angle", type=float, default=0.0, help="explicit knee pole angle degrees")
    p.add_argument("--output", type=Path, required=True, help="new controls plan folder")

    p = sub.add_parser("apply-rig-controls", help="add optional IK/FK control bones to an existing rig .blend")
    p.add_argument("--blend", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True, help="controls.json from plan-rig-controls")
    p.add_argument("--output", type=Path, required=True, help="new editable controlled.blend")
    p.add_argument("--blender")

    p = sub.add_parser("plan-joint-refine", help="source-lock local quad-loop density changes around reviewed hinge joints")
    p.add_argument("--blend", type=Path, required=True, help="pre-rig quad cage .blend")
    p.add_argument("--topology", type=Path, required=True, help="existing topology-plan.json")
    p.add_argument("--joint", action="append", required=True, help="reviewed joint name, repeat for multiple hinges")
    p.add_argument("--cuts", type=int, default=2, help="additional axial cuts per crossing quad edge, 1–4")
    p.add_argument("--max-added-vertices", type=int, default=25000)
    p.add_argument("--axial-alignment-min", type=float, default=.6)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("refine-surface", help="feature-preserving pre-rig surface fairing with optional source QA")
    p.add_argument("--blend", type=Path, required=True, help="approved editable source .blend")
    p.add_argument("--config", type=Path, required=True, help="validated smoothing and protection policy JSON")
    p.add_argument("--output", type=Path, required=True, help="new immutable candidate/QA revision")
    p.add_argument("--blender", help="Blender executable")

    p = sub.add_parser("refine-joint-topology", help="add bounded Blender BMesh hinge loops on an unskinned quad cage")
    p.add_argument("--plan", type=Path, required=True, help="joint-refine-plan.json")
    p.add_argument("--output", type=Path, required=True, help="new hinge-retopo.blend and OBJ revision")
    p.add_argument("--blender")

    p = sub.add_parser("plan-topology", help="generate joint-specific topology cross-section review zones")
    p.add_argument("--joints", type=Path, required=True)
    p.add_argument("--mesh-name", default="Low", help="exact Blender mesh object name")
    p.add_argument("--joint", action="append", help="specific named joint; repeat (default key hinges)")
    p.add_argument("--min-coverage", type=float, default=0.7)
    p.add_argument("--min-edges", type=int, default=8)
    p.add_argument("--strict", action="store_true", help="fail numeric thresholds instead of report-only")
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("audit-topology", help="evaluate joint topology and optional Blender review rings")
    p.add_argument("--blend", type=Path, required=True, help="existing retopologized .blend")
    p.add_argument("--plan", type=Path, required=True, help="topology-plan.json")
    p.add_argument("--annotate", action="store_true", help="save new editable .blend with joint guide rings")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blender")

    p = sub.add_parser("validate-rig", help="native Blender pose tests for a skinned .blend")
    p.add_argument("--blend", type=Path, required=True, help="existing rigged source .blend")
    p.add_argument("--plan", type=Path, required=True, help="bounded pose battery JSON")
    p.add_argument("--output", type=Path, required=True, help="new immutable report directory")
    p.add_argument("--blender", help="local Blender executable path")
    p = sub.add_parser("repair-rig-weights",
                       help="bounded semantic skin-weight correction without changing source mesh or .blend")
    p.add_argument("--blend", type=Path, required=True, help="original saved rig .blend")
    p.add_argument("--pose-plan", type=Path, required=True,
                   help="validated pose QA JSON with explicit weight_zones")
    p.add_argument("--output", type=Path, required=True, help="new immutable repaired .blend or preview report")
    p.add_argument("--max-removed-weight", type=float, default=.15,
                   help="maximum total foreign influence removed from one vertex; default .15, max .5")
    p.add_argument("--max-changed-vertices", type=int, default=2000,
                   help="maximum repaired vertices across all marked regions; default 2000")
    p.add_argument("--dry-run", action="store_true", help="preview affected vertices without editing a .blend")
    p.add_argument("--blender", help="local Blender executable path")


    p = sub.add_parser("plan-joint-correctives",
                       help="SHA-lock artist-sculpted joint corrective targets to a saved rig")
    p.add_argument("--blend", type=Path, required=True, help="rigged source .blend with hidden sculpt duplicates")
    p.add_argument("--config", type=Path, required=True, help="reviewed joint-corrective-config.json")
    p.add_argument("--output", type=Path, required=True, help="immutable joint-corrective-plan.json revision")

    p = sub.add_parser("bind-joint-correctives",
                       help="attach authored rest-space shape keys and pose-angle drivers in a new .blend")
    p.add_argument("--plan", type=Path, required=True, help="source-locked joint-corrective-plan.json")
    p.add_argument("--output", type=Path, required=True, help="new corrected.blend revision")
    p.add_argument("--blender", help="Blender executable path")

    p = sub.add_parser("repair-penetrations", help="bounded BVH collision repair for named Blender object pairs")
    p.add_argument("--blend", type=Path, required=True, help="original editable .blend file")
    p.add_argument("--plan", type=Path, required=True, help="directed collision-pair JSON with protected anchors")
    p.add_argument("--output", type=Path, required=True, help="new revision directory")
    p.add_argument("--blender", help="local Blender binary")

    p = sub.add_parser("inspect-region-sculpt-source",
                       help="read-only Blender mesh SHA and candidate surface seed vertices")
    p.add_argument("--blend", type=Path, required=True)
    p.add_argument("--object", required=True, help="exact Blender mesh object name")
    p.add_argument("--anchor", nargs=3, type=float, action="append",
                   metavar=("X", "Y", "Z"), help="candidate world-space seed anchor, repeatable")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blender")

    p = sub.add_parser("refine-region", help="bounded local sculpt edit; preserve all other .blend objects")
    p.add_argument("--blend", type=Path, required=True, help="editable source Blender file")
    p.add_argument("--plan", type=Path, required=True, help="region sculpt correction JSON")
    p.add_argument("--output", type=Path, required=True, help="new revision containing refined.blend and region.obj")
    p.add_argument("--blender")
    p.add_argument("--character", type=Path, help="frozen CharacterSpec v2 plan for automatic original-art QA")
    p.add_argument("--region", help="region ID to source-audit; requires --character")
    p.add_argument("--qa-before", type=Path, help="previous Z-up source-frame part for source QA")
    p.add_argument("--max-regression", type=float, default=.01)
    p.add_argument("--max-contour-regression", type=float, default=.03)
    p.add_argument("--min-iou", type=float)



    p = sub.add_parser("ingest", help="copy generator output into canonical view names")
    p.add_argument("source", type=Path)
    p.add_argument("mapping", type=Path, help='JSON {"file.png": "front", ...}')
    p.add_argument("destination", type=Path)

    p = sub.add_parser("check", help="check reference images alone, against each other and in 3D")
    p.add_argument("folder", type=Path)

    p = sub.add_parser("trim", help="cut from a view the part that its opposite view has no room for")
    p.add_argument("folder", type=Path)
    p.add_argument("view", help="the view to trim, e.g. front")
    p.add_argument("--against", help="the view to trust (default: the opposite view, e.g. back)")
    p.add_argument("--output", type=Path, required=True, help="new .png (RGBA)")
    p.add_argument("--slack-px", type=int, default=15, help="outline difference that is not a part (aligned px)")
    p.add_argument("--min-thickness-px", type=int, default=40, help="only parts this thick are cut")

    p = sub.add_parser("handoff", help="aligned views, masks, blockout, spec, generator inputs")
    p.add_argument("folder", type=Path)
    p.add_argument("--height-m", type=float, required=True, help="soles to highest point, metres")
    p.add_argument("--voxel-px", type=int, default=5, help="blockout voxel in aligned pixels (1000 px = height)")
    p.add_argument("--tolerance-px", type=float, default=10.0)
    p.add_argument("--round-px", type=float, default=6.0, help="edge rounding where views meet (aligned px)")
    p.add_argument("--outline-blur-px", type=float, default=12.0,
                   help="low-pass the outlines so fur or feather fringes do not groove the surface (0 keeps them)")

    p = sub.add_parser("compare", help="model silhouettes against the references")
    p.add_argument("handoff", type=Path)
    p.add_argument("source", type=Path, help="mesh (.obj .ply .glb) or folder of blender-setup renders")
    p.add_argument("--output", type=Path, help="new report folder (default: next to the source)")

    p = sub.add_parser("fit", help="deform a mesh until its silhouettes match the references")
    p.add_argument("handoff", type=Path)
    p.add_argument("mesh", type=Path)
    p.add_argument("--output", type=Path, required=True, help=".ply or .obj")
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--views", nargs="*", help="limit to these views")
    p.add_argument("--no-normalise", action="store_true", help="the mesh already stands in the handoff frame")
    p.add_argument("--smooth-radius-px", type=float, default=4.0,
                   help="displacement smoothing radius in fit-grid pixels (about 1.3 mm each for a 0.32 m figure); "
                        "use about 40 for a detailed generator mesh, so only proportions change")
    p.add_argument("--lattice", help="nx,ny,nz cells: deform only through a coarse lattice (best for generator "
                                     "meshes), or `auto`: 4,4,8 then 8,8,16, each kept only if it does no harm")
    p.add_argument("--max-distortion", type=float, default=0.03,
                   help="with --lattice auto: largest share of edges stretched or squeezed by more than 1.5")
    p.add_argument("--max-new-folds", type=int, default=0, help="with --lattice auto: folds that a stage may add")

    p = sub.add_parser("project", help="vertex colours from the views; optional clay relief")
    p.add_argument("handoff", type=Path)
    p.add_argument("mesh", type=Path)
    p.add_argument("--output", type=Path, required=True, help=".ply keeps the colours")
    p.add_argument("--subdivide", type=int, default=0, help="split each triangle in four this many times")
    p.add_argument("--relief-mm", type=float, default=0.0, help="stylized relief from *_clay views")
    p.add_argument("--assume-symmetric", action="store_true",
                   help="fill a missing side view with the mirrored opposite side (not for one-sided accessories)")

    p = sub.add_parser("scatter", help="feather, fur or scale cards over a region")
    p.add_argument("mesh", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--colour", help="r,g,b of the region (needs vertex colours)")
    p.add_argument("--max-delta-e", type=float, default=18.0)
    p.add_argument("--z-min-m", type=float, default=float("-inf"))
    p.add_argument("--z-max-m", type=float, default=float("inf"))
    p.add_argument("--y-min-m", type=float, default=float("-inf"), help="the back of the figure is +Y")
    p.add_argument("--y-max-m", type=float, default=float("inf"))
    p.add_argument("--density", type=float, default=20000.0, help="cards per square metre")
    p.add_argument("--length-mm", type=float, default=12.0)
    p.add_argument("--width-mm", type=float, default=6.0)
    p.add_argument("--lift-deg", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("joints", help="estimate an editable joints.json for the rig")
    p.add_argument("handoff", type=Path)
    p.add_argument("mesh", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limbs", choices=("arms", "wings"), default="arms")
    p.add_argument("--tail", action="store_true")

    p = sub.add_parser("texture", help="bake a UV texture from the full-resolution reference views")
    _bake_arguments(p)
    p.add_argument("--output", type=Path, required=True, help=".png")
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--diagnostics", type=Path, help="new folder for owners.png, conflicts.png and turnaround.png")
    p.add_argument("--screen-extra", type=float, metavar="DELTA_E",
                   help="drop a repaint that alone raises any drawn view's error by more than this (e.g. 0.5)")

    p = sub.add_parser("conflicts", help="find where the drawings overlap and disagree on a mesh, with pictures")
    _bake_arguments(p)
    p.add_argument("--output", type=Path, required=True, help="new folder: report.json, turnaround.png, ...")
    p.add_argument("--size", type=int, default=1024)

    p = sub.add_parser("resolve-conflicts", help="bounded, rollback-safe semantic texture conflict repair")
    p.add_argument("handoff", type=Path, help="existing character handoff folder")
    p.add_argument("mesh", type=Path, help="source OBJ with UVs")
    p.add_argument("--config", type=Path, required=True, help="character-independent repair policy JSON")
    p.add_argument("--output", type=Path, required=True, help="new directory for best texture and QA report")

    p = sub.add_parser("repaint", help="repaint views of the model with a local FLUX.2 [klein] in ComfyUI")
    p.add_argument("handoff", type=Path)
    p.add_argument("mesh", type=Path, help="OBJ with UVs")
    p.add_argument("texture", type=Path, help="current texture (e.g. baked from the drawings)")
    p.add_argument("--output", type=Path, required=True, help="new folder: renders, repaints, manifest.json")
    p.add_argument("--views", help="comma-separated cameras (default: four 3/4 views and four from above)")
    p.add_argument("--subject", default="the character", help='e.g. "the cartoon squirrel"')
    p.add_argument("--server", default="http://127.0.0.1:8188", help="local ComfyUI (loopback only)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--denoise", type=float, default=1.0)
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--min-iou", type=float, default=0.75,
                   help="reject a repaint whose figure overlaps the render less than this")
    p.add_argument("--min-usable", type=float, default=0.5,
                   help="reject a repaint if less than this share of the figure keeps its outline")
    p.add_argument("--weight", type=float, default=0.8, help="quality of a repaint relative to a drawing")
    p.add_argument("--owners", type=Path,
                   help="owners.png from `texture --diagnostics` of the same texture: take the tone only from "
                        "pixels that a drawing painted")
    p.add_argument("--resume", action="store_true",
                   help="reuse finished repaints in --output whose render and settings match; repaint the rest")

    p = sub.add_parser("split-eyes", help="make the eyes separate eyeballs and let one drawing paint around them")
    p.add_argument("handoff", type=Path, help="the handoff of the head or body (e.g. split/rest)")
    p.add_argument("mesh", type=Path, help="the placed head or body mesh, to set the eyeballs on")
    p.add_argument("--output", type=Path, required=True, help="new folder: eyes/, rest/, eyes.json, anatomy.json")
    p.add_argument("--view", default="front", help="the drawing that looks at the face")
    p.add_argument("--zone", type=float, default=1.9,
                   help="the surroundings of an eye, in iris radii: only --view paints there")
    p.add_argument("--protrude", type=float, default=0.5, help="share of the eyeball's depth that stands out of the head")

    p = sub.add_parser("redraw-view", help="change one drawing so that it agrees with the design of another")
    p.add_argument("drawing", type=Path, help="the drawing to change (RGBA cut-out, or on a flat background)")
    p.add_argument("example", type=Path, help="a drawing that shows the wanted design")
    p.add_argument("--change", required=True, help='what to change, e.g. "the arms wear cream knit sleeves"')
    p.add_argument("--output", type=Path, required=True, help="new .png (RGBA)")
    p.add_argument("--subject", default="the character")
    p.add_argument("--server", default="http://127.0.0.1:8188", help="local ComfyUI (loopback only)")
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)

    p = sub.add_parser("split-views", help="split the drawings into one part and the rest, to model each alone")
    p.add_argument("handoff", type=Path)
    p.add_argument("--part", required=True, help='the part to take out, in plain words for the editor, e.g. "tail"')
    p.add_argument("--name", help="folder name of the part (default: --part with dashes)")
    p.add_argument("--output", type=Path, required=True, help="folder for <part>/, rest/ and split.json")
    p.add_argument("--views", help="comma-separated drawings that show the part (default: all)")
    p.add_argument("--subject", default="the character", help='e.g. "the cartoon squirrel"')
    p.add_argument("--behind", default="the body and the clothes", help="what the part hides, for the editor")
    p.add_argument("--keep", default="", help="what the editor must not take away with the part")
    p.add_argument("--server", default="http://127.0.0.1:8188", help="local ComfyUI (loopback only)")
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--change-delta-e", type=float, default=7.0,
                   help="inside the figure, a colour change above this belongs to the removed part")

    p = sub.add_parser("place-part", help="put a part that was modelled alone into the frame of the whole character")
    p.add_argument("handoff", type=Path, help="the part's handoff from `split-views` (masks in the character frame)")
    p.add_argument("mesh", type=Path, help="the part, with any size, position and turn about the vertical axis")
    p.add_argument("--output", type=Path, required=True, help=".ply")
    p.add_argument("--yaw-deg", type=float, help="the turn about Z, if it is known (default: search)")
    p.add_argument("--stretch", type=float, default=0.15,
                   help="how far the scale of one direction may differ from the common scale (0: one scale)")
    p.add_argument("--ignore-thinner-px", type=float, default=0.0,
                   help="drop pieces of the masks thinner than this (aligned px), e.g. a strap that the mesh lacks")
    p.add_argument("--voxel-mm", type=float, default=0.0,
                   help="rebuild the placed part as one solid with this voxel (for a hollow or badly wound mesh)")

    p = sub.add_parser("seat-part", help="move a finished part out of the body it hangs on, as one rigid piece")
    p.add_argument("body", type=Path, help="the placed mesh that the part hangs on")
    p.add_argument("part", type=Path, help="the placed part; an OBJ keeps its UVs, so its texture stays valid")
    p.add_argument("--output", type=Path, required=True, help="the moved part (.obj with UVs, or .ply)")
    p.add_argument("--metres-per-px", type=float, required=True, help="metres_per_aligned_px of the handoff spec")
    p.add_argument("--view", help="the side from which the part sits on the body, e.g. left (default: found)")
    p.add_argument("--embed", type=float, default=0.15, help="share of the part's surface that may stay inside")
    p.add_argument("--move-m", type=float,
                   help="move by this distance toward --view and measure nothing (to move a second mesh of the part alike)")

    p = sub.add_parser("join-parts", help="one mesh and one texture from placed, textured parts")
    p.add_argument("parts", nargs="+", help="each part as low.obj:basecolor.png or low.obj:basecolor.png:normal.png")
    p.add_argument("--output", type=Path, required=True,
                   help="new folder: joined.obj, basecolor.png, layout.json (and normal.png if every part has one)")
    p.add_argument("--size", type=int, default=4096)

    p = sub.add_parser("parts", help="recognise the character's parts on the mesh and in each drawing")
    _bake_arguments(p)
    p.add_argument("--output", type=Path, required=True, help="new folder: parts.json, masks, turnaround.png")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--classes", type=int, default=10, help="colour classes to start from")
    p.add_argument("--contrast", type=float, default=10.0,
                   help="two touching parts join if the colour step across their border is below this (delta E)")
    p.add_argument("--max-parts", type=int, default=24, help="join the weakest borders until at most this many are left")
    p.add_argument("--crease-radius-px", type=float, default=60.0,
                   help="shapes (tail, body, ...) are separated by creases that bend inward more sharply than this")
    p.add_argument("--bleed-share", type=float, default=0.05,
                   help="a material on less than this share of a shape, with most of it on other shapes, is taken "
                        "for a part that the bake put on the wrong shape (0: off)")
    p.add_argument("--material-delta-e", type=float, default=8.0,
                   help="parts whose mean colours differ by less than this are one material and are baked together")
    p.add_argument("--min-share", type=float, default=0.005,
                   help="a part smaller than this share of the surface joins a neighbour")

    p = sub.add_parser("texture-parts", help="bake each recognised part only from the pixels that show it")
    _bake_arguments(p)
    p.add_argument("parts", type=Path, help="folder from `parts`")
    p.add_argument("--output", type=Path, required=True, help=".png")
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--diagnostics", type=Path, help="new folder for owners.png, parts.png and turnaround.png")
    p.add_argument("--no-polish", action="store_true",
                   help="do not close the seams inside the parts or fill the unseen texels smoothly")
    p.add_argument("--tone-delta-e", type=float, default=18.0,
                   help="inside one material, level differences between two views up to about twice this")
    p.add_argument("--polish-reach-px", type=float, default=12.0,
                   help="distance from a seam over which the polish works (aligned px)")
    p.add_argument("--qa-baseline", type=Path, help="prior texture; opt into source-art QA of the newly baked candidate")
    p.add_argument("--qa-output", type=Path, help="new evidence folder; default <texture-stem>-source-qa")
    p.add_argument("--qa-max-mean-increase", type=float, default=1.5, help="allowed per-camera increase in mean Lab delta E")
    p.add_argument("--qa-max-p90-increase", type=float, default=3.0, help="allowed per-camera increase in P90 Lab delta E")
    p.add_argument("--qa-max-within-10-drop", type=float, default=.02, help="allowed fraction of source pixels leaving delta E <= 10")
    p.add_argument("--qa-max-projection-samples", type=int, default=2_000_000, help="per-camera safe UV projection sample cap")


    p = sub.add_parser("audit-texture-revision", help="reject a texture change that worsens original-art colors in any aligned camera")
    p.add_argument("handoff", type=Path, help="reference handoff with aligned images and fixed camera scale")
    p.add_argument("mesh", type=Path, help="unchanged OBJ mesh with UVs (source-frame or legacy interchange)")
    p.add_argument("before", type=Path, help="previous texture image")
    p.add_argument("after", type=Path, help="candidate texture image")
    p.add_argument("--views", help="comma-separated source cameras; default all aligned handoff views")
    p.add_argument("--max-mean-increase", type=float, default=1.5, help="maximum increase in mean Lab delta E per camera")
    p.add_argument("--max-p90-increase", type=float, default=3.0, help="maximum increase in P90 Lab delta E per camera")
    p.add_argument("--max-within-10-drop", type=float, default=.02, help="maximum decrease in fraction of pixels within delta E 10")
    p.add_argument("--min-pixels-per-view", type=int, default=150, help="minimum common reference/model pixels")
    p.add_argument("--max-projection-samples", type=int, default=2_000_000, help="preflight cap on per-camera mesh UV samples for local RAM safety")
    p.add_argument("--output", type=Path, required=True, help="new immutable per-camera QA panels and report")

    p = sub.add_parser("appearance", help="colour agreement of a textured mesh with each reference view")
    p.add_argument("handoff", type=Path)
    p.add_argument("mesh", type=Path, help="OBJ with UVs")
    p.add_argument("texture", type=Path)
    p.add_argument("--output", type=Path, required=True, help="new folder: report.json and side-by-side images")

    p = sub.add_parser("fur-mask", help="UV-space fur mask from a base-colour texture (clothing excluded)")
    p.add_argument("basecolor", type=Path)
    p.add_argument("--exclude", action="append", default=[], help="r,g,b of a no-fur colour; repeat")
    p.add_argument("--max-delta-e", type=float, default=18.0)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("place", help="repair, orient and score image-to-3D candidates against the references")
    p.add_argument("handoff", type=Path)
    p.add_argument("meshes", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True, help="new folder: placed meshes, reports, ranking.json")
    p.add_argument("--voxel-mm", type=float, default=1.0, help="solidify voxel")

    p = sub.add_parser("solidify", help="rebuild a hollow or badly wound generator mesh as one solid")
    p.add_argument("mesh", type=Path)
    p.add_argument("--output", type=Path, required=True, help=".ply or .obj")
    p.add_argument("--voxel-mm", type=float, default=1.0)

    p = sub.add_parser("swap", help="replace a region of one mesh with the same region of another")
    p.add_argument("base", type=Path)
    p.add_argument("donor", type=Path)
    p.add_argument("--region", action="append", help="condition such as y>0.02; repeat to combine")
    p.add_argument("--ellipsoid", help="cx,cy,cz,rx,ry,rz in metres: a round region (a head) instead of --region")
    p.add_argument("--voxel-m", type=float, default=0.0015, help="voxel size that fuses the pieces")
    p.add_argument("--no-align", action="store_true", help="the donor already matches the base exactly")
    p.add_argument("--align-region", action="append",
                   help="align only on these shared parts (default: everything outside --region); repeat")
    p.add_argument("--output", type=Path, required=True, help="new folder")
    p.add_argument("--blender")

    p = sub.add_parser("fur", help="strand fur with Blender's hair assets on a textured low-poly mesh")
    p.add_argument("mesh", type=Path, help="low.obj from retopo (with UVs)")
    p.add_argument("--basecolor", type=Path, required=True)
    p.add_argument("--systems", type=Path, required=True, help="JSON list of fur systems (see blender_fur.py)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, help="PLY from fur-weights (red = fur); systems use it with use_fur_attribute")
    p.add_argument("--render", action="store_true")
    p.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE"), default="CYCLES")
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--blender")

    p = sub.add_parser("fur-weights", help="per-vertex fur weights from colours projected from the drawings")
    p.add_argument("coloured", type=Path, help="mesh from `project` (vertex colours)")
    p.add_argument("--clothing-top-m", type=float, required=True, help="height above which no clothing exists")
    p.add_argument("--eye-band-m", required=True, help="low,high height of the eyes")
    p.add_argument("--output", type=Path, required=True, help=".ply (red channel = weight)")

    for name, help_text in (("blender-setup", "reference cameras and images in Blender; optional renders"),
                            ("assemble", "place parts with anchor pairs and join them"),
                            ("retopo", "quad retopology, UVs and a normal bake"),
                            ("rig", "armature, automatic weights, range-of-motion test")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("input", type=Path)
        if name == "rig":
            p.add_argument("joints", type=Path)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--blender")
        if name == "blender-setup":
            p.add_argument("--blend", type=Path, help="add the references to this .blend (it is not changed)")
            p.add_argument("--render", action="store_true")
            p.add_argument("--render-blockout", action="store_true", help="render the blockout itself")
            p.add_argument("--save", action="store_true")
        if name == "retopo":
            p.add_argument("--target-faces", type=int, default=8000)
            p.add_argument("--bake-size", type=int, default=2048)
            p.add_argument("--cage-m", type=float, default=0.005)
            p.add_argument("--colour-from", type=Path,
                           help="placement JSON from `place`: bake that source file's own texture as base colour")
        if name == "rig":
            p.add_argument("--rom-deg", type=float, default=45.0)
            p.add_argument("--render-rom", action="store_true")
            p.add_argument("--preserve-volume", action="store_true", help="enable Blender armature Preserve Volume skinning")
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run(args) -> int:
    from .views import ingest

    if args.command == "plan-region-batch":
        from .region_batch import save_region_batch
        report = save_region_batch(
            args.character, args.changed, args.output,
            max_workers=args.max_workers, memory_gib=args.memory_gib)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "resume-region-batch":
        from .region_batch import resume_region_batch
        report = resume_region_batch(args.plan, args.receipts, args.output)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "complete-awaiting-source-and-animation-qa" else 1
    if args.command == "character-benchmark":
        from .benchmark import character_benchmark
        report = character_benchmark(args.manifest, args.output)
        print(json.dumps(report, indent=2))
        return 0 if report["numeric_gates_pass"] else 1
    if args.command == "plan-landmark-fit":
        from .landmark_fit_plan import save_landmark_fit_plan
        report = save_landmark_fit_plan(args.blend, args.landmarks, args.spec, args.output)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "fit-landmark-surface":
        from .landmark_fit_plan import load_landmark_fit_plan
        from .blender import run_worker
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        if (args.character is None) != (args.region is None):
            raise ValueError("Automatic source QA requires both --character and --region")
        if args.qa_before is not None and args.character is None:
            raise ValueError("--qa-before requires a --character and --region")
        plan = load_landmark_fit_plan(args.plan)
        source_gate_before = None
        if args.character is not None:
            import math
            from .character_spec_v2 import open_plan
            from .landmark_triangulation import load_triangulated_landmarks
            from .region_revision_gate import _canonical_mesh
            if (not math.isfinite(args.max_regression)
                    or not 0 <= args.max_regression <= 1):
                raise ValueError("--max-regression must be a finite fraction from 0 to 1")
            if (not math.isfinite(args.max_contour_regression)
                    or not 0 <= args.max_contour_regression <= 1):
                raise ValueError("--max-contour-regression must be a finite fraction from 0 to 1")
            if args.min_iou is not None and (
                    not math.isfinite(args.min_iou) or not 0 <= args.min_iou <= 1):
                raise ValueError("--min-iou must be a finite fraction from 0 to 1")
            source_plan, character_spec = open_plan(args.character)
            if character_spec["character_id"] != plan["character_id"]:
                raise ValueError("The fit and CharacterSpec must refer to the same character ID")
            landmarks = load_triangulated_landmarks(plan["landmark_report"])
            if abs(character_spec["height_m"] - landmarks["height_m"]) > 1e-6:
                raise ValueError("Triangulated landmark height and CharacterSpec reference scale differ")
            region = next((r for r in character_spec["regions"] if r["id"] == args.region), None)
            if region is None or not region.get("verified_views"):
                raise ValueError("Automatic source QA requires a region with verified reference masks")
            if args.qa_before is None and "source_mesh" not in region:
                raise ValueError("Region has no source_mesh; supply --qa-before")
            source_gate_before = _canonical_mesh(
                args.qa_before if args.qa_before is not None
                else Path(source_plan["source_spec"]).parent / region["source_mesh"],
                "baseline")
        report = run_worker("geodesic-fit", {
            "plan": plan, "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=Path(plan["source_blend"]))
        if args.character is None:
            print(json.dumps(report, indent=2))
            return 0
        from .region_revision_gate import gate_region_revision
        source_qa = gate_region_revision(
            args.character, args.region, args.output / "landmark-fitted-source.obj",
            args.output / "source-fidelity", before=source_gate_before,
            max_regression=args.max_regression, min_iou=args.min_iou,
            max_contour_regression=args.max_contour_regression)
        receipt = {
            "schema_version": 1,
            "status": ("source-regression-rejected" if not source_qa["numeric_gate_pass"]
                       else "source-numeric-gate-artistic-review-required"),
            "modeling_status": report.get("status"),
            "character_id": plan["character_id"],
            "region_id": args.region,
            "modeling_report": "report.json",
            "source_qa_report": "source-fidelity/report.json",
            "modeling_blend": "landmark-fitted.blend",
            "modeling_source_frame_obj": "landmark-fitted-source.obj",
            "source_sha256": source_qa["source_spec_sha256"],
            "failed_views": source_qa["failed_views"],
            "numeric_gate_pass": source_qa["numeric_gate_pass"],
            "max_boundary_f1_2px_regression_per_view": source_qa["max_boundary_f1_2px_regression_per_view"],
            "artistic_review_required": True,
        }
        (args.output / "source-fidelity-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0 if source_qa["numeric_gate_pass"] else 1
    if args.command == "triangulate-landmarks":
        from .landmark_triangulation import triangulate_landmarks
        report = triangulate_landmarks(args.spec, args.output)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "resolve-anatomy-landmarks":
        from .anatomy_landmark_resolver import resolve_anatomy_landmarks
        report = resolve_anatomy_landmarks(args.spec, args.landmarks, args.output)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plan-anatomy":
        from .anatomy_plan import plan_anatomy
        report = plan_anatomy(args.blend, args.spec, args.output)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "build-anatomy":
        from .anatomy_plan import load_anatomy_plan
        from .blender import run_worker
        if args.output.exists():
            raise FileExistsError(f"Anatomy output revision exists: {args.output}")
        plan = load_anatomy_plan(args.plan)
        report = run_worker("anatomy", {
            "plan": plan, "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=Path(plan["source_blend"]))
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plan-character":
        from .character_spec_v2 import plan_character
        report = plan_character(args.spec, args.output)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plan-region":
        from .character_spec_v2 import plan_region
        report = plan_region(args.character, args.region, args.output)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "audit-region-revision":
        from .region_revision_gate import gate_region_revision
        report = gate_region_revision(
            args.character, args.region, args.after, args.output, before=args.before,
            max_regression=args.max_regression, min_iou=args.min_iou,
            max_contour_regression=args.max_contour_regression)
        print(json.dumps(report, indent=2))
        return 0 if report["numeric_gate_pass"] else 1
    if args.command == "compare-region":
        from .region_compare import compare_region
        report = compare_region(args.character, args.region, args.mesh, args.output,
                                min_iou=args.min_iou, baseline=args.baseline,
                                max_regression=args.max_regression)
        print(json.dumps(report, indent=2))
        return 1 if report["failed_views"] else 0
    if args.command == "repair-penetrations":
        from dataclasses import asdict
        from .penetration_repair import load_collision_plan
        from .blender import run_worker
        if not args.blend.is_file() or args.blend.suffix.lower() != ".blend":
            raise ValueError("--blend must be an existing .blend source")
        if args.output.exists():
            raise FileExistsError(f"Revision exists: {args.output}")
        validated = load_collision_plan(args.plan)
        report = run_worker("penetration-repair", {
            "plan": {"schema_version": 1, "pairs": [asdict(p) for p in validated["pairs"]],
                     "guard_pairs": validated["guard_pairs"]},
            "source_blend": str(args.blend.resolve()),
            "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=args.blend.resolve())
        print(json.dumps(report, indent=2))
        # A no-op collision check is a valid, non-destructive result.
        return 0
    if args.command == "inspect-region-sculpt-source":
        from .blender import run_worker
        if not args.blend.is_file() or args.blend.suffix.lower() != ".blend":
            raise ValueError("--blend must be a saved Blender .blend")
        if args.output.exists():
            raise FileExistsError(f"Output revision exists: {args.output}")
        report = run_worker("region-sculpt", {
            "inspect_only": True, "object": args.object, "anchors": args.anchor or [],
            "source_blend": str(args.blend.resolve()),
            "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=args.blend.resolve())
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "refine-region":
        from .region_sculpt_plan import load_region_sculpt_plan
        from .blender import run_worker
        if not args.blend.is_file() or args.blend.suffix.lower() != ".blend":
            raise ValueError("--blend must be an existing Blender .blend file")
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        if (args.character is None) != (args.region is None):
            raise ValueError("Automatic source QA requires both --character and --region")
        if args.qa_before is not None and args.character is None:
            raise ValueError("--qa-before requires --character and --region")
        plan = load_region_sculpt_plan(args.plan)
        source_gate_before = None
        if args.character is not None:
            from .character_spec_v2 import open_plan
            from .region_revision_gate import _canonical_mesh, _number
            _number(args.max_regression, "max_regression", 0, 1)
            _number(args.max_contour_regression, "max_contour_regression", 0, 1)
            if args.min_iou is not None:
                _number(args.min_iou, "min_iou", 0, 1)
            source_plan, character_spec = open_plan(args.character)
            region = next((r for r in character_spec["regions"]
                           if r["id"] == args.region), None)
            if region is None or not region.get("verified_views"):
                raise ValueError("Original-art QA requires an existing region with verified views")
            if args.qa_before is None and "source_mesh" not in region:
                raise ValueError("Region has no original mesh; supply --qa-before")
            source_gate_before = _canonical_mesh(
                args.qa_before if args.qa_before is not None
                else Path(source_plan["source_spec"]).parent / region["source_mesh"],
                "baseline")
        report = run_worker("region-sculpt", {
            "plan": plan, "output": str(args.output.resolve()),
            "source_blend": str(args.blend.resolve())},
            blender=args.blender, blend_file=args.blend.resolve())
        if args.character is None:
            print(json.dumps(report, indent=2))
            return 0
        from .region_revision_gate import gate_region_revision
        qa = gate_region_revision(
            args.character, args.region, args.output / "region-source.obj",
            args.output / "source-fidelity", before=source_gate_before,
            max_regression=args.max_regression,
            max_contour_regression=args.max_contour_regression,
            min_iou=args.min_iou)
        receipt = {
            "schema_version": 1,
            "status": ("source-regression-rejected" if not qa["numeric_gate_pass"]
                       else "source-numeric-gate-artistic-review-required"),
            "modeling_report": "report.json",
            "modeling_blend": "refined.blend",
            "modeling_source_frame_obj": "region-source.obj",
            "source_qa_report": "source-fidelity/report.json",
            "source_sha256": qa["source_spec_sha256"],
            "character_id": character_spec["character_id"],
            "region_id": args.region,
            "failed_views": qa["failed_views"],
            "numeric_gate_pass": qa["numeric_gate_pass"],
            "artistic_review_required": True,
        }
        (args.output / "source-fidelity-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0 if qa["numeric_gate_pass"] else 1
    if args.command == "plan-talking-performance":
        from .talking_plan import save_talking_plan
        report = save_talking_plan(args.blend, args.cues, args.clip, args.start,
                   args.end, args.output, fps=args.fps, armature=args.armature,
                   mouth_object=args.mouth_object, ramp_frames=args.ramp_frames,
                   time_alignment_tolerance_s=args.time_alignment_tolerance_s)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "export-talking-performance":
        from .talking_plan import load_talking_plan
        from .blender import run_worker
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        plan = load_talking_plan(args.plan)
        report = run_worker("performance", {
            "plan": plan["performance"], "viseme_plan": plan["visemes"],
            "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=Path(plan["performance"]["source_blend"]))
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plan-performance":
        from .performance_plan import save_performance_plan
        report = save_performance_plan(args.blend, args.clip, args.start, args.end,
                 args.output, armature=args.armature, mouth_object=args.mouth_object,
                 fps=args.fps,
                 time_alignment_tolerance_s=args.time_alignment_tolerance_s)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "export-performance":
        from .performance_plan import load_performance_plan
        from .blender import run_worker
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        plan = load_performance_plan(args.plan)
        report = run_worker("performance", {
            "plan": plan, "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=Path(plan["source_blend"]))
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plan-visemes":
        from .viseme_plan import save_viseme_plan
        plan = save_viseme_plan(args.blend, args.cues, args.mouth_object,
                               args.clip, args.output, fps=args.fps,
                               first_frame=args.first_frame, ramp_frames=args.ramp_frames)
        print(json.dumps(plan, indent=2))
        return 0
    if args.command == "animate-visemes":
        from .viseme_plan import load_viseme_plan
        from .blender import run_worker
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        plan = load_viseme_plan(args.plan)
        report = run_worker("visemes", {
            "plan": plan, "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=Path(plan["source_blend"]))
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plan-animation-bake":
        from .animation_bake_plan import save_bake_plan
        report = save_bake_plan(args.blend, args.clip, args.start, args.end, args.output,
                                armature=args.armature, step=args.step, fps=args.fps,
                                max_position_error_m=args.max_position_error_m,
                                max_rotation_error_deg=args.max_rotation_error_deg,
                                bake_correctives=args.bake_correctives)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "bake-character-animation":
        from .animation_bake_plan import load_bake_plan
        from .blender import run_worker
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        plan = load_bake_plan(args.plan)
        report = run_worker("animation-bake", {
            "plan": plan, "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=Path(plan["source_blend"]))
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plan-rig-controls":
        from .rig_control_plan import save_control_plan
        report = save_control_plan(args.joints, args.profile, args.output,
                                   armature=args.armature, pole_angle_deg=args.pole_angle)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "apply-rig-controls":
        from .rig_control_plan import load_control_plan
        from .blender import run_worker
        if not args.blend.is_file() or args.blend.suffix.lower() != ".blend":
            raise ValueError("--blend must be an existing Blender .blend")
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        plan = load_control_plan(args.plan)
        report = run_worker("rig-controls", {
            "plan": plan, "source_blend": str(args.blend.resolve()),
            "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=args.blend.resolve())
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plan-joint-refine":
        from .joint_refine_plan import save_joint_refine_plan
        report = save_joint_refine_plan(args.blend, args.topology, args.joint,
                                        args.output, cuts=args.cuts,
                                        max_added_vertices=args.max_added_vertices,
                                        axial_alignment_min=args.axial_alignment_min)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "refine-surface":
        from .surface_refinement_runner import run_surface_refinement
        report = run_surface_refinement(args.blend, args.config, args.output, blender=args.blender)
        print(json.dumps(report, indent=2))
        return 1 if report["status"] == "reference-regression-rejected" else 0
    if args.command == "refine-joint-topology":
        from .joint_refine_plan import load_joint_refine_plan
        from .topology_zones import load_topology_plan
        from .blender import run_worker
        if args.output.exists():
            raise FileExistsError(f"Output revision exists: {args.output}")
        plan = load_joint_refine_plan(args.plan)
        topology = load_topology_plan(plan["topology_plan"])
        report = run_worker("joint-refine", {
            "plan": plan, "topology": topology, "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=Path(plan["source_blend"]))
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plan-topology":
        from .topology_zones import save_topology_plan
        report = save_topology_plan(args.joints, args.output, mesh=args.mesh_name,
                                    joints=args.joint, min_coverage=args.min_coverage,
                                    min_edges=args.min_edges, strict=args.strict)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "audit-topology":
        from .topology_zones import load_topology_plan
        from .blender import run_worker
        if not args.blend.is_file() or args.blend.suffix.lower() != ".blend":
            raise ValueError("--blend must be an existing retopologized Blender .blend")
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        plan = load_topology_plan(args.plan)
        report = run_worker("topology-qa", {
            "plan": plan, "annotate": args.annotate,
            "source_blend": str(args.blend.resolve()), "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=args.blend.resolve())
        print(json.dumps(report, indent=2))
        return 1 if report["status"] == "structural-review-failed" else 0
    if args.command == "validate-rig":
        from .pose_spec import load_pose_plan
        from .blender import run_worker
        if not args.blend.is_file() or args.blend.suffix.lower() != ".blend":
            raise ValueError("--blend must be an existing Blender .blend file")
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        plan = load_pose_plan(args.plan)
        report = run_worker("pose-qa", {"plan": plan, "output": str(args.output.resolve())},
                            blender=args.blender, blend_file=args.blend.resolve())
        print(json.dumps(report, indent=2))
        return 1 if report["status"] == "structural-fail" else 0
    if args.command == "repair-rig-weights":
        import hashlib
        import math
        from .pose_spec import load_pose_plan
        from .blender import run_worker
        if not args.blend.is_file() or args.blend.suffix.lower() != ".blend":
            raise ValueError("--blend must be an existing saved Blender .blend")
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        if (not math.isfinite(args.max_removed_weight) or
                not 0 < args.max_removed_weight <= .5):
            raise ValueError("--max-removed-weight must be finite in (0, 0.5]")
        if not 1 <= args.max_changed_vertices <= 10000:
            raise ValueError("--max-changed-vertices must be within 1 to 10000")
        plan = load_pose_plan(args.pose_plan)
        zones = plan.get("weight_zones", [])
        if not zones:
            raise ValueError("Pose plan needs nonempty weight_zones with explicit allowed bones")
        if any(zone["max_outside_weight"] >= args.max_removed_weight for zone in zones):
            raise ValueError("Each weight-zone outside tolerance must be below the maximum repair weight")
        digest = hashlib.sha256()
        with args.blend.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        report = run_worker("weight-repair", {
            "source_blend": str(args.blend.resolve()),
            "source_blend_sha256": digest.hexdigest(),
            "armature": plan["armature"],
            "weight_zones": zones,
            "max_removed_weight": args.max_removed_weight,
            "max_changed_vertices": args.max_changed_vertices,
            "dry_run": args.dry_run,
            "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=args.blend.resolve())
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "plan-joint-correctives":
        from .joint_corrective_plan import save_corrective_plan
        plan = save_corrective_plan(args.blend, args.config, args.output)
        print(json.dumps(plan, indent=2))
        return 0
    if args.command == "bind-joint-correctives":
        from .joint_corrective_plan import load_corrective_plan
        from .blender import run_worker
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        plan = load_corrective_plan(args.plan)
        report = run_worker("joint-correctives", {
            "plan": plan, "output": str(args.output.resolve()),
        }, blender=args.blender, blend_file=Path(plan["source_blend"]))
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "ingest":
        used = ingest(args.source, args.mapping, args.destination)
        print(json.dumps(used, indent=2))
        return 0
    if args.command == "check":
        from .checks import check_folder

        report = check_folder(args.folder)
        print((args.folder / "check" / "report.md").read_text(encoding="utf-8"))
        return 0 if report["pass"] else 1
    if args.command == "trim":
        from .reconcile import parts_without_room, trim_view
        from .views import ALIASES, load_folder, mirror_of

        name = ALIASES.get(args.view, args.view)
        against = ALIASES.get(args.against, args.against) if args.against else mirror_of(name)
        loaded, errors = load_folder(args.folder, {name, against})
        for needed in (name, against):
            if needed not in loaded:
                raise SystemExit(f"error: no usable view {needed!r} in {args.folder}: {errors.get(needed, 'missing')}")
        image, share = trim_view(loaded[name], parts_without_room(
            loaded[name].mask, loaded[against].mask, slack_px=args.slack_px, min_thickness_px=args.min_thickness_px))
        image.save(_new_file(args.output))
        print(json.dumps({"view": name, "against": against, "removed_share_of_figure": round(share, 4),
                          "output": str(args.output)}, indent=2))
        return 0
    if args.command == "handoff":
        from .handoff import build_handoff

        spec = build_handoff(args.folder, args.height_m, factor=args.voxel_px, tolerance_px=args.tolerance_px,
                             round_px=args.round_px, outline_blur_px=args.outline_blur_px)
        print(json.dumps({"views": spec["views"], "blockout": spec["blockout"]}, indent=2))
        return 0
    if args.command == "compare":
        from .handoff import compare

        report = compare(args.handoff, args.source, output=args.output)
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1
    if args.command in ("fit", "project", "joints"):
        from .mesh import read_mesh, write_mesh

        from .mesh import orient_faces, weld

        # Generator files split UV seams and can have inconsistent winding.
        spec, (mesh, _) = _spec(args.handoff), orient_faces(weld(read_mesh(args.mesh)))
        _new_file(args.output)
        if args.command == "fit":
            from .fit import fit_silhouettes
            from .handoff import reference_masks

            scale = 0.25
            masks = reference_masks(args.handoff, scale)
            if args.views:
                masks = {n: m for n, m in masks.items() if n in args.views}
            common = dict(scale=scale, iterations=args.iterations, normalise_first=not args.no_normalise,
                          smooth_radius_px=args.smooth_radius_px)
            stages = None
            if args.lattice == "auto":
                from .fit import fit_ladder

                result, stages = fit_ladder(mesh, masks, spec["height_m"], max_distortion=args.max_distortion,
                                            max_new_folds=args.max_new_folds, **common)
            else:
                cells = tuple(int(v) for v in args.lattice.split(",")) if args.lattice else None
                result = fit_silhouettes(mesh, masks, spec["height_m"], lattice_cells=cells, **common)
            write_mesh(result.mesh, args.output)
            report = {"input": str(args.mesh), "normalise": result.transform, "best_iteration": result.best_iteration,
                      "folds": result.folds,
                      "start": result.history[0], "best": result.history[result.best_iteration]}
            if stages is not None:
                report["stages"] = stages
                report["start"] = {"iteration": 0, **{k: stages[0][k] for k in ("weighted_iou", "views")}}
            _write_json(args.output.with_suffix(".json"), {**report, "history": result.history})
        elif args.command == "project":
            import numpy as np

            from .project import load_camera_images, project_colours, project_images, relief, subdivide

            for _ in range(args.subdivide):
                mesh = subdivide(mesh)
            px = spec["metres_per_aligned_px"]
            mesh, report = project_colours(mesh, args.handoff, spec["views"], px, assume_symmetric=args.assume_symmetric)
            if args.relief_mm:
                clay = load_camera_images(args.handoff, spec["views"], "_clay")
                if not clay:
                    raise ValueError("Relief needs *_clay views in handoff/aligned (clay images of full-body views)")
                grey = {n: img.mean(axis=2, keepdims=True) / 255.0 for n, img in clay.items()}
                projection = project_images(mesh, grey, px)
                mesh, relief_report = relief(mesh, np.nan_to_num(projection.values[:, 0]), projection.weight > 0,
                                             args.relief_mm / 1000)
                report["relief"] = relief_report
            write_mesh(mesh, args.output)
            report["vertices"] = len(mesh.vertices)
            _write_json(args.output.with_suffix(".json"), report)
        else:
            from .rig import estimate_joints

            report = estimate_joints(mesh, spec["height_m"], spec.get("landmarks_z_m"), limbs=args.limbs,
                                     tail=args.tail)
            _write_json(args.output, report)
        print(json.dumps(report if args.command != "fit" else {k: report[k] for k in ("normalise", "start", "best", "folds")},
                         indent=2))
        return 0
    if args.command == "texture":
        from PIL import Image

        from .texture import bake_texture, read_obj_uv

        _new_file(args.output)
        diagnostics = _new_dir(args.diagnostics) if args.diagnostics else None
        mesh, corner_uv = read_obj_uv(args.mesh)
        extra = json.loads(args.extra.read_text(encoding="utf-8"))["extra"] if args.extra else None
        settings = _bake_settings(args)
        screened = None
        if extra and args.screen_extra is not None:
            from .repaint import screen_extras

            extra, screened = screen_extras(args.handoff, mesh, corner_uv, extra,
                                            max_added_delta_e=args.screen_extra, **settings)
        image, report = bake_texture(args.handoff, mesh, corner_uv, size=args.size, extra=extra,
                                     diagnostics=diagnostics, **settings)
        if screened is not None:
            report["screen_extra"] = screened
        Image.fromarray(image).save(args.output)
        _write_json(args.output.with_suffix(".json"), report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "conflicts":
        from PIL import Image

        from .texture import bake_texture, conflict_advice, read_obj_uv

        out = _new_dir(args.output)
        mesh, corner_uv = read_obj_uv(args.mesh)
        extra = json.loads(args.extra.read_text(encoding="utf-8"))["extra"] if args.extra else None
        image, report = bake_texture(args.handoff, mesh, corner_uv, size=args.size, extra=extra, diagnostics=out,
                                     **_bake_settings(args))
        Image.fromarray(image).save(out / "texture.png")
        summary = {**report["conflicts"], "advice": conflict_advice(report),
                   **{k: report[k] for k in ("owned", "owner_colours", "regions") if k in report},
                   **{k: v for k, v in report.items() if k.startswith("seam_")}}
        _write_json(out / "report.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "resolve-conflicts":
        from .conflict_repair import run_conflict_repair

        report = run_conflict_repair(args.handoff, args.mesh, args.config, args.output)
        print(json.dumps({"status": report["status"], "selected_settings": report["selected_settings"],
                          "selected_score": report["selected_score"], "attempts": len(report["attempts"]),
                          "output": str(args.output)}, indent=2))
        return 0
    if args.command == "repaint":
        import numpy as np
        from PIL import Image

        from ..reconstruction.completion_image_generation import _LocalComfyClient
        from .repaint import default_views, repaint_views
        from .texture import read_obj_uv

        out = args.output if args.resume else _new_dir(args.output)
        mesh, corner_uv = read_obj_uv(args.mesh)
        texture = np.asarray(Image.open(args.texture).convert("RGB"))
        views = args.views.split(",") if args.views else default_views()
        owners = np.asarray(Image.open(args.owners).convert("RGB")) if args.owners else None
        manifest = repaint_views(args.handoff, mesh, corner_uv, texture, views, out, _LocalComfyClient(args.server),
                                 subject=args.subject, size=args.size, steps=args.steps, denoise=args.denoise,
                                 min_iou=args.min_iou, min_usable=args.min_usable, weight=args.weight, owners=owners)
        print(json.dumps(manifest["views"], indent=2))
        return 0
    if args.command == "split-eyes":
        from .eyes import split_eyes
        from .mesh import read_mesh
        from .texture import read_obj_uv

        out = _new_dir(args.output)
        head = read_obj_uv(args.mesh)[0] if args.mesh.suffix.lower() == ".obj" else read_mesh(args.mesh)
        report = split_eyes(args.handoff, head, out, view=args.view, zone=args.zone, protrude=args.protrude)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "redraw-view":
        from ..reconstruction.completion_image_generation import _LocalComfyClient
        from .redraw import redraw_view

        if args.output.exists():
            raise FileExistsError(f"{args.output} exists. Each output is a new revision.")
        report = redraw_view(args.drawing, args.example, args.output, _LocalComfyClient(args.server),
                             change=args.change, subject=args.subject, size=args.size, steps=args.steps, seed=args.seed)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "split-views":
        from ..reconstruction.completion_image_generation import _LocalComfyClient
        from .split_views import split_views

        report = split_views(args.handoff, args.output, _LocalComfyClient(args.server), part=args.part,
                             name=args.name, subject=args.subject, behind=args.behind, keep=args.keep,
                             views=args.views.split(",") if args.views else None, size=args.size, steps=args.steps,
                             change_delta_e=args.change_delta_e)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "place-part":
        from .handoff import reference_masks
        from .mesh import orient_faces, read_mesh, weld, write_mesh
        from .part_place import place_part

        _new_file(args.output)
        spec = _spec(args.handoff)
        mesh, flipped = orient_faces(weld(read_mesh(args.mesh)))
        placed, info = place_part(mesh, reference_masks(args.handoff, 0.25), spec["metres_per_aligned_px"],
                                  yaw_deg=args.yaw_deg, ignore_thinner_px=args.ignore_thinner_px,
                                  stretch=args.stretch)
        if args.voxel_mm > 0:
            from .hull import solidify

            placed = solidify(placed, args.voxel_mm / 1000)
        write_mesh(placed, args.output)
        report = {"source": str(args.mesh.resolve()), "faces_flipped": flipped, **info}
        _write_json(args.output.with_suffix(".placement.json"), report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "seat-part":
        from .mesh import Mesh, read_mesh, write_mesh
        from .part_place import seat_part, write_obj_uv
        from .texture import read_obj_uv
        from .views import camera

        _new_file(args.output)
        corner_uv = None
        if args.part.suffix.lower() == ".obj" and args.output.suffix.lower() == ".obj":
            try:
                part, corner_uv = read_obj_uv(args.part)
            except ValueError:
                part = read_mesh(args.part)
        else:
            part = read_mesh(args.part)
        if args.move_m is not None:
            if not args.view:
                raise SystemExit("error: --move-m needs --view")
            moved = Mesh(part.vertices + args.move_m * camera(args.view).toward, part.faces, part.colours)
            info = {"view": args.view, "moved_m": args.move_m}
        else:
            body = read_obj_uv(args.body)[0] if args.body.suffix.lower() == ".obj" else read_mesh(args.body)
            moved, info = seat_part(part, body, args.metres_per_px, view=args.view, embed=args.embed)
        if corner_uv is not None:
            write_obj_uv(moved, corner_uv, args.output, comment=f"seated: moved {info['moved_m']} m toward {info['view']}")
        else:
            write_mesh(moved, args.output)
        print(json.dumps(info, indent=2))
        return 0
    if args.command == "join-parts":
        import numpy as np
        from PIL import Image

        from .part_place import join_parts, write_obj_uv
        from .texture import read_obj_uv

        out = _new_dir(args.output)
        loaded, normals, names = [], [], []
        for item in args.parts:
            fields = item.split(":")
            if len(fields) not in (2, 3) or not all(fields):
                raise SystemExit(f"error: give each part as mesh.obj:texture.png[:normal.png], not {item!r}")
            mesh, corner_uv = read_obj_uv(Path(fields[0]))
            with Image.open(fields[1]) as picture:
                loaded.append((mesh, corner_uv, np.asarray(picture.convert("RGB"))))
            if len(fields) == 3:
                with Image.open(fields[2]) as picture:
                    normals.append((mesh, corner_uv, np.asarray(picture.convert("RGB"))))
            names.append(str(Path(fields[0]).resolve()))
        if normals and len(normals) != len(loaded):
            raise SystemExit("error: give a normal map for every part or for none")
        mesh, corner_uv, texture, layout = join_parts(loaded, size=args.size)
        out.mkdir(parents=True, exist_ok=True)
        write_obj_uv(mesh, corner_uv, out / "joined.obj", comment="joined parts; see layout.json")
        Image.fromarray(texture).save(out / "basecolor.png")
        if normals:      # the same layout: it depends only on the meshes
            Image.fromarray(join_parts(normals, size=args.size)[2]).save(out / "normal.png")
        report = {"parts": [{**row, "source": name} for row, name in zip(layout, names)], "size": args.size,
                  "vertices": len(mesh.vertices), "faces": len(mesh.faces)}
        _write_json(out / "layout.json", report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "parts":
        from .parts import find_parts
        from .texture import read_obj_uv

        out = _new_dir(args.output)
        mesh, corner_uv = read_obj_uv(args.mesh)
        extra = json.loads(args.extra.read_text(encoding="utf-8"))["extra"] if args.extra else None
        report = find_parts(args.handoff, mesh, corner_uv, out, size=args.size, classes=args.classes,
                            contrast=args.contrast, min_share=args.min_share, max_parts=args.max_parts,
                            material_delta_e=args.material_delta_e, crease_radius_px=args.crease_radius_px,
                            bleed_share=args.bleed_share, extra=extra, **_bake_settings(args))
        print(json.dumps({k: report[k] for k in ("source_confidence", "review", "shapes", "bleed", "parts",
                                                 "agreement", "advice")},
                         indent=2))
        return 0
    if args.command == "texture-parts":
        from PIL import Image

        from .parts import bake_parts
        from .texture import read_obj_uv

        if args.qa_output is not None and args.qa_baseline is None:
            raise ValueError("--qa-output requires --qa-baseline")
        new_report = args.output.with_suffix(".json")
        qa_receipt = args.output.with_suffix(".source-fidelity.json")
        qa_folder = (args.qa_output if args.qa_output is not None else
                     args.output.parent / f"{args.output.stem}-source-qa")
        for reserved in [args.output, new_report] + (
                [qa_folder, qa_receipt] if args.qa_baseline else []):
            if reserved.exists():
                raise FileExistsError(f"Output revision already exists: {reserved}")
        diagnostics = _new_dir(args.diagnostics) if args.diagnostics else None
        mesh, corner_uv = read_obj_uv(args.mesh)
        extra = json.loads(args.extra.read_text(encoding="utf-8"))["extra"] if args.extra else None
        qa_views = args.views.split(",") if args.views else None
        if args.qa_baseline is not None:
            from .texture import estimate_uv_projection_samples
            from .texture_revision_gate import _fraction, _image

            # Fail early, before the expensive multi-material bake. Candidate
            # output and original reviewed masks are always separate revisions.
            _image(args.qa_baseline)
            for key, value, hi in (
                ("qa_max_mean_increase", args.qa_max_mean_increase, 50),
                ("qa_max_p90_increase", args.qa_max_p90_increase, 80),
                ("qa_max_within_10_drop", args.qa_max_within_10_drop, 1),
            ):
                _fraction(value, key, 0, hi)
            if (type(args.qa_max_projection_samples) is not int
                    or not 10_000 <= args.qa_max_projection_samples <= 10_000_000):
                raise ValueError("--qa-max-projection-samples must be in [10000, 10000000]")
            spec = json.loads((args.handoff / "spec.json").read_text(encoding="utf-8"))
            declared = spec.get("views", [])
            selected = qa_views if qa_views is not None else declared
            if (not isinstance(selected, list) or not selected or
                    len(set(selected)) != len(selected) or
                    not set(selected) <= set(declared)):
                raise ValueError("Source QA requires distinct cameras in the original handoff")
            if not all((args.handoff / "aligned" / f"{name}.png").is_file()
                       for name in selected):
                raise ValueError("All selected source cameras need aligned reference images")
            predicted = estimate_uv_projection_samples(
                mesh, metres_per_px=spec["metres_per_aligned_px"], scale=.25)
            if predicted > args.qa_max_projection_samples:
                raise ValueError(
                    f"Projection sample budget exceeded ({predicted:,} > "
                    f"{args.qa_max_projection_samples:,}); use an approved coarser QA cage")
        _new_file(args.output)
        image, report = bake_parts(args.handoff, mesh, corner_uv, args.parts, size=args.size, extra=extra,
                                   polish=not args.no_polish, polish_reach_px=args.polish_reach_px,
                                   tone_delta_e=args.tone_delta_e,
                                   diagnostics=diagnostics, **_bake_settings(args))
        Image.fromarray(image).save(args.output)
        _write_json(new_report, report)
        if args.qa_baseline is None:
            print(json.dumps(report, indent=2))
            return 0
        from .texture_revision_gate import audit_texture_revision

        evidence = audit_texture_revision(
            args.handoff, args.mesh, args.qa_baseline, args.output,
            qa_folder, views=qa_views,
            max_mean_increase=args.qa_max_mean_increase,
            max_p90_increase=args.qa_max_p90_increase,
            max_within_10_drop=args.qa_max_within_10_drop,
            max_projection_samples=args.qa_max_projection_samples)
        receipt = {
            "schema_version": 1,
            "status": ("texture-source-regression-rejected"
                       if not evidence["numeric_gate_pass"]
                       else "texture-numeric-gate-artistic-review-required"),
            "candidate": str(args.output),
            "candidate_bake_report": str(new_report),
            "source_qa_report": str(qa_folder / "report.json"),
            "source_qa_numeric_pass": evidence["numeric_gate_pass"],
            "failed_views": evidence["failed_views"],
            "input_sha256": evidence["input_sha256"],
            "artistic_review_required": True,
        }
        _write_json(qa_receipt, receipt)
        print(json.dumps(receipt, indent=2))
        return 0 if evidence["numeric_gate_pass"] else 1
    if args.command == "audit-texture-revision":
        from .texture_revision_gate import audit_texture_revision
        report = audit_texture_revision(
            args.handoff, args.mesh, args.before, args.after, args.output,
            views=args.views.split(",") if args.views else None,
            max_mean_increase=args.max_mean_increase,
            max_p90_increase=args.max_p90_increase,
            max_within_10_drop=args.max_within_10_drop,
            min_pixels_per_view=args.min_pixels_per_view,
            max_projection_samples=args.max_projection_samples)
        print(json.dumps(report, indent=2))
        return 0 if report["numeric_gate_pass"] else 1
    if args.command == "appearance":
        import numpy as np
        from PIL import Image

        from .texture import appearance_report, read_obj_uv

        out = _new_dir(args.output)
        mesh, corner_uv = read_obj_uv(args.mesh)
        texture = np.asarray(Image.open(args.texture).convert("RGB"))
        report = appearance_report(args.handoff, mesh, corner_uv, texture, out=out)
        _write_json(out / "report.json", report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "fur-weights":
        import numpy as np

        from .mesh import Mesh, read_mesh, write_ply
        from .scatter import fur_weights

        _new_file(args.output)
        mesh = read_mesh(args.coloured)
        band = tuple(float(v) for v in args.eye_band_m.split(","))
        weights = fur_weights(mesh, clothing_top_m=args.clothing_top_m, eye_band_m=band)
        grey = np.repeat((weights * 255).astype(np.uint8)[:, None], 3, axis=1)
        write_ply(Mesh(mesh.vertices, mesh.faces, grey), args.output)
        print(json.dumps({"fur_fraction": round(float(weights.mean()), 4)}))
        return 0
    if args.command == "fur-mask":
        import numpy as np
        from PIL import Image

        from .scatter import fur_mask

        _new_file(args.output)
        texture = np.asarray(Image.open(args.basecolor).convert("RGB"))
        mask = fur_mask(texture, [tuple(int(v) for v in c.split(",")) for c in args.exclude], args.max_delta_e)
        Image.fromarray(mask).save(args.output)
        print(json.dumps({"fur_fraction": round(float((mask > 0).mean()), 4)}))
        return 0
    if args.command == "place":
        from .fit import normalise
        from .handoff import compare, reference_masks
        from .hull import solidify
        from .mesh import orient_faces, read_mesh, weld, write_ply

        stems = [path.stem for path in args.meshes]
        if len(set(stems)) != len(stems):  # outputs are named by stem
            raise ValueError(f"Candidate file names must differ: {sorted(stems)}")
        spec, out = _spec(args.handoff), _new_dir(args.output)
        out.mkdir(parents=True, exist_ok=True)
        masks = reference_masks(args.handoff, 0.25)
        ranking = []
        for path in args.meshes:
            mesh, flipped = orient_faces(weld(read_mesh(path)))
            # Scale to the reference height first, so the solidify voxel is in metres.
            placed, info = normalise(mesh, masks, spec["height_m"], spec["metres_per_aligned_px"], 0.25)
            solid = solidify(placed, args.voxel_mm / 1000)
            target = out / f"{path.stem}.ply"
            write_ply(solid, target)
            _write_json(out / f"{path.stem}.placement.json", {"source": str(path.resolve()), **info})
            report = compare(args.handoff, target, output=out / f"{path.stem}-compare")
            ranking.append({"mesh": str(path), "placed": str(target), "mean_iou": report["mean_iou"],
                            "views": {v["view"]: v["iou"] for v in report["views"]},
                            "orientation": info, "faces_flipped": flipped})
            print(f"{path.name}: mean IoU {report['mean_iou']}", file=sys.stderr)
        ranking.sort(key=lambda r: -r["mean_iou"])
        _write_json(out / "ranking.json", {"ranking": ranking})
        print(json.dumps([{k: r[k] for k in ("mesh", "mean_iou", "views")} for r in ranking], indent=2))
        return 0
    if args.command == "solidify":
        from .hull import solidify
        from .mesh import read_mesh, weld, write_mesh

        _new_file(args.output)
        source = weld(read_mesh(args.mesh))
        solid = solidify(source, args.voxel_mm / 1000)
        write_mesh(solid, args.output)
        print(json.dumps({"input_faces": len(source.faces), "faces": len(solid.faces),
                          "voxel_mm": args.voxel_mm}, indent=2))
        return 0
    if args.command == "scatter":
        from .mesh import read_mesh, write_mesh
        from .scatter import ScatterSpec, scatter_cards

        _new_file(args.output)
        colour = tuple(int(v) for v in args.colour.split(",")) if args.colour else None
        spec = ScatterSpec(colour=colour, max_delta_e=args.max_delta_e, z_min_m=args.z_min_m, z_max_m=args.z_max_m,
                           y_min_m=args.y_min_m, y_max_m=args.y_max_m,
                           density_per_m2=args.density, length_m=args.length_mm / 1000, width_m=args.width_mm / 1000,
                           lift_deg=args.lift_deg, seed=args.seed)
        cards, report = scatter_cards(read_mesh(args.mesh), spec)
        if report["cards"] == 0:
            raise ValueError("No cards: the region is empty. Check --colour, --max-delta-e and the height band.")
        write_mesh(cards, args.output)
        _write_json(args.output.with_suffix(".json"), report)
        print(json.dumps(report, indent=2))
        return 0

    from .blender import run_worker

    if args.command == "fur":
        for path in (args.mesh, args.basecolor, args.systems):
            if not path.is_file():
                raise FileNotFoundError(f"{path} does not exist")
        systems = json.loads(args.systems.read_text(encoding="utf-8"))
        for system in systems:
            if system.get("mask"):
                system["mask"] = str((args.systems.parent / system["mask"]).resolve())
        out = _new_dir(args.output)
        report = run_worker("fur", {"mesh": str(args.mesh.resolve()), "basecolor": str(args.basecolor.resolve()),
                                    "weights": str(args.weights.resolve()) if args.weights else None,
                                    "systems": systems, "render": args.render, "engine": args.engine,
                                    "samples": args.samples, "output": str(out.resolve())},
                            blender=args.blender)
        print(json.dumps({k: report[k] for k in ("strands", "renders")}, indent=2))
        return 0
    if args.command == "swap":
        from .mesh import read_mesh, weld, write_ply
        from .swap import align, parse_region, region_box

        for path in (args.base, args.donor):
            if not path.is_file():
                raise FileNotFoundError(f"{path} does not exist")
        ellipsoid = [float(v) for v in args.ellipsoid.split(",")] if args.ellipsoid else None
        if ellipsoid is not None:
            if len(ellipsoid) != 6 or min(ellipsoid[3:]) <= 0:
                raise ValueError("--ellipsoid needs cx,cy,cz,rx,ry,rz with positive radii")
            c, r = ellipsoid[:3], ellipsoid[3:]
            region = [(i, ">", c[i] - r[i]) for i in range(3)] + [(i, "<", c[i] + r[i]) for i in range(3)]
        elif args.region:
            region = parse_region(args.region)
        else:
            raise ValueError("Give --region or --ellipsoid")
        low, high = region_box(region)
        out = _new_dir(args.output)
        out.mkdir(parents=True, exist_ok=True)
        donor_path, alignment = args.donor.resolve(), None
        if not args.no_align:
            match = parse_region(args.align_region) if args.align_region else None
            donor, alignment = align(weld(read_mesh(args.donor)), weld(read_mesh(args.base)), region, match=match)
            donor_path = (out / "donor_aligned.ply").resolve()
            write_ply(donor, donor_path)
        report = run_worker("swap", {"base": str(args.base.resolve()), "donor": str(donor_path),
                                     "box_min": low, "box_max": high, "ellipsoid": ellipsoid, "voxel_m": args.voxel_m,
                                     "output": str(out.resolve())}, blender=args.blender)
        report["alignment"] = alignment
        _write_json(out / "report.json", report)
        print(json.dumps({k: report[k] for k in ("alignment", "pieces", "result")}, indent=2))
        return 0
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} does not exist")
    out = _new_dir(args.output)
    if args.command == "blender-setup":
        report = run_worker("setup", {"handoff": str(args.input.resolve()), "output": str(out.resolve()),
                                      "render": args.render, "render_blockout": args.render_blockout,
                                      "save": args.save}, blender=args.blender, blend_file=args.blend)
    elif args.command == "assemble":
        from .assembly import load_assembly

        report = run_worker("assemble", {"plan": load_assembly(args.input), "output": str(out.resolve())},
                            blender=args.blender)
    elif args.command == "retopo":
        colour = json.loads(args.colour_from.read_text(encoding="utf-8")) if args.colour_from else None
        report = run_worker("retopo", {"mesh": str(args.input.resolve()), "output": str(out.resolve()),
                                       "target_faces": args.target_faces, "bake_size": args.bake_size,
                                       "cage_m": args.cage_m, "colour_from": colour}, blender=args.blender)
    else:
        from .rig import validate_joints

        joints = json.loads(args.joints.read_text(encoding="utf-8"))
        validate_joints(joints)
        report = run_worker("rig", {"mesh": str(args.input.resolve()), "joints": joints, "output": str(out.resolve()),
                                    "rom_deg": args.rom_deg, "render_rom": args.render_rom,
                                    "preserve_volume": args.preserve_volume}, blender=args.blender)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
