"""Command-line entry point for headless asset validation and building."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from .asset_catalog import (CatalogError, find_asset, load_catalog,
                            revision_path, search_assets, validate_sources)
from .asset_import import AssetImportError, inspect_source_glb, load_asset_provenance
from .architecture_details import classical_column
from .architecture_geometry import GeometryError, world_vertices
from .export_audit import ExportAuditError
from .facade import FacadeError, make_facade_study
from .egyptian_monuments import MonumentError, make_monument_study
from .egypt_map import CITY_MARKERS, LABELS, RIVER_CHANNELS, EgyptMapSpec
from .geography.fuji import load_dem
from .glb_audit import AuditError, audit_glb
from .glb_motion import MotionExportError, write_glb_motion
from .physics_plan import PhysicsPlanError, load_physics_plan
from .deformation_plan import DeformationPlanError, load_deformation_plan
from .lattice import make_lattice_tower
from .landmark_scan import prepare_sphinx_scan
from .material_overrides import load_material_overrides, verify_material_overrides
from .material_packs import load_texture_packs, verify_texture_targets
from .portico import PorticoError, make_portico_study
from .recipe import RecipeError, load_recipe
from .render_profiles import RenderProfileError
from .revision_audit import RevisionAuditError, audit_revision
from .surfaces import SurfaceError
from .reconstruction.head_base import HeadFitError
from .reconstruction.completion_image_generation import ImageGenerationError
from .head_sources.makehuman import MakeHumanSourceError
from .vintage_biplane import make_biplane


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kumiki-blender")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "build"):
        subcommand = subcommands.add_parser(command)
        subcommand.add_argument("--recipe", type=Path, required=True)
        if command == "build":
            subcommand.add_argument("--output", type=Path, required=True)
    for command in ("plan-asset", "compile-asset"):
        subcommand = subcommands.add_parser(command, help="Validate or compile an AssetSpec v2 assembly")
        subcommand.add_argument("--spec", type=Path, required=True)
        if command == "compile-asset":
            subcommand.add_argument("--output", type=Path, required=True)
    composer_build = subcommands.add_parser("build-asset", help="Render an AssetSpec v2 assembly")
    composer_input = composer_build.add_mutually_exclusive_group(required=True)
    composer_input.add_argument("--spec", type=Path, help="Host Python validates and launches Blender")
    composer_input.add_argument("--plan", type=Path, help="Precompiled, dependency-free Blender input")
    composer_build.add_argument("--output", type=Path, required=True)
    composer_build.add_argument("--blender", default="blender", help="Host Blender executable path")
    composer_build.add_argument("--texture-packs", type=Path,
                                help="Reviewed local PBR pack recipe, mapped to AssetSpec material roles")
    subcommands.add_parser("asset-schema", help="Print the agent-facing AssetSpec v2 JSON Schema")
    for command in ("plan-portico", "build-portico"):
        subcommand = subcommands.add_parser(command)
        subcommand.add_argument("--bays", type=int, default=3)
        subcommand.add_argument("--order", choices=("tuscan", "doric", "ionic"), default="ionic")
        subcommand.add_argument("--arch-style", choices=("round", "pointed"), default="round")
        subcommand.add_argument("--balustrade", action="store_true")
        if command == "build-portico":
            subcommand.add_argument("--output", type=Path, required=True)
    for command in ("plan-facade", "build-facade"):
        subcommand = subcommands.add_parser(command)
        subcommand.add_argument("--bays", type=int, default=3)
        if command == "build-facade":
            subcommand.add_argument("--output", type=Path, required=True)
            subcommand.add_argument("--texture-packs", type=Path)
    for command in ("plan-monument", "build-monument"):
        subcommand = subcommands.add_parser(command)
        subcommand.add_argument("--kind", choices=("pyramid", "sphinx", "giza-study"),
                                default="giza-study")
        subcommand.add_argument("--pyramid-preset", choices=("illustrative", "khufu-petrie"),
                                default="illustrative")
        subcommand.add_argument("--pyramid-base", type=float, default=36.0)
        subcommand.add_argument("--pyramid-height", type=float, default=23.0)
        subcommand.add_argument("--pyramid-courses", type=int, default=18)
        subcommand.add_argument("--casing-fraction", type=float, default=0.04)
        subcommand.add_argument("--sphinx-length", type=float, default=10.5)
        subcommand.add_argument("--restored-nose", action="store_true")
        if command == "build-monument":
            subcommand.add_argument("--output", type=Path, required=True)
            subcommand.add_argument("--sphinx-source", type=Path)
            subcommand.add_argument("--sphinx-provenance", type=Path)
            subcommand.add_argument("--sphinx-front-axis", choices=("-X", "+X", "-Y", "+Y"), default="-X")
    subcommands.add_parser("plan-biplane")
    airplane_parser = subcommands.add_parser("build-biplane")
    airplane_parser.add_argument("--output", type=Path, required=True)
    catalog_parser = subcommands.add_parser("catalog", help="Find, browse and build categorized 3D assets")
    catalog_actions = catalog_parser.add_subparsers(dest="catalog_action", required=True)
    catalog_actions.add_parser("categories")
    for action in ("list", "search"):
        item = catalog_actions.add_parser(action)
        if action == "search":
            item.add_argument("query", nargs="+", help="Any words in the name, tags or description")
        item.add_argument("--category")
        item.add_argument("--tag")
        item.add_argument("--rigged", action="store_true")
    for action in ("show", "path", "build"):
        item = catalog_actions.add_parser(action)
        item.add_argument("asset_id")
        if action in ("path", "build"):
            item.add_argument("--revision", default="r01")
            item.add_argument("--root", type=Path, default=Path(".review"))
    catalog_actions.add_parser("validate")
    for command in ("plan-egypt-map", "build-egypt-map"):
        subcommand = subcommands.add_parser(command)
        subcommand.add_argument("--detail", choices=("standard", "high"), default="standard")
        if command == "build-egypt-map":
            subcommand.add_argument("--output", type=Path, required=True)
    for command in ("plan-fuji", "build-fuji"):
        subcommand = subcommands.add_parser(command)
        subcommand.add_argument("--style", choices=("realistic", "iconic"), default="realistic",
                                help="Surveyed terrain or compact stylized Japan diorama")
        subcommand.add_argument("--dem", type=Path, default=Path(".review/fuji-dem/fuji-heightmap.bin"))
        subcommand.add_argument("--detail", choices=("preview", "standard", "high"), default="standard")
        subcommand.add_argument("--season", choices=("winter", "summer"), default="winter",
                                help="Realistic Fuji: snowy winter or fully snow-free late summer")
        subcommand.add_argument("--detail-patch", type=Path, action="append", default=[],
                                help="Optional native GSI z15 crater patch (repeat for summit and Hoei)")
        if command == "build-fuji":
            subcommand.add_argument("--output", type=Path, required=True)
    column_parser = subcommands.add_parser("plan-classical-column")
    column_parser.add_argument("--order", choices=("tuscan", "doric", "ionic"), default="ionic")
    column_parser.add_argument("--height", type=float, default=4.55)
    column_parser.add_argument("--radius", type=float, default=0.36)
    audit_parser = subcommands.add_parser("audit")
    audit_parser.add_argument("--model", type=Path, required=True)
    audit_parser.add_argument("--require-pbr-textures", action="store_true")
    motion_parser = subcommands.add_parser(
        "export-motion", help="Convert Blender-baked glTF node clips into Kumiki PM2 motion JSON")
    motion_parser.add_argument("--model", type=Path, required=True, help="Animated Blender-exported GLB")
    motion_parser.add_argument("--clip", help="Clip name; optional when GLB has exactly one animation")
    motion_parser.add_argument("--output", type=Path, required=True, help="New PM2 baked-actions.json")
    motion_parser.add_argument("--action-id", help="Optional Kumiki action ID; defaults to <clip>-baked")
    motion_parser.add_argument("--contacts", type=Path, help="Optional validated contact cue array JSON")
    motion_parser.add_argument("--merge-clips", action="store_true", help="Combine disjoint per-object simulation clips into one PM2 action")
    motion_parser.add_argument("--resample-mixed", action="store_true", help="Permit mixed glTF STEP/LINEAR channels via frame-exact 30-fps resampling")
    for command in ("plan-physics", "build-physics"):
        physics_parser = subcommands.add_parser(command, help="Validate or bake a declarative Blender rigid-body shot")
        physics_parser.add_argument("--spec", type=Path, required=True)
        if command == "build-physics":
            physics_parser.add_argument("--output", type=Path, required=True,
                                        help="New, immutable motion bundle (requires Blender background with a loaded .blend)")
    for command in ("plan-deformation", "build-deformation"):
        deform_parser = subcommands.add_parser(command, help="Validate or bake one topology-stable deforming Blender mesh into glTF morph animation")
        deform_parser.add_argument("--spec", type=Path, required=True)
        if command == "build-deformation":
            deform_parser.add_argument("--output", type=Path, required=True,
                                      help="New GLB/manifest bundle (requires Blender background with a saved .blend)")
    revision_parser = subcommands.add_parser("audit-revision")
    revision_parser.add_argument("--root", type=Path, required=True)
    visual_parser = subcommands.add_parser("visual-qa", help="Contact sheet and optional render regression")
    visual_parser.add_argument("--revision", type=Path, required=True)
    visual_parser.add_argument("--baseline", type=Path, help="Verified prior revision of the same asset")
    visual_parser.add_argument("--output", type=Path, required=True)
    visual_parser.add_argument("--max-rms", type=float, default=0.02)
    visual_parser.add_argument("--max-changed", type=float, default=0.05)
    visual_parser.add_argument("--pixel-threshold", type=int, default=16)
    visual_parser.add_argument("--idiff", action="store_true", help="Also run optional OpenImageIO idiff")
    parity_parser = subcommands.add_parser("bridge-parity", help="Compare native Blender/WebGL silhouette capture receipts")
    parity_parser.add_argument("--plan", type=Path, required=True, help="Byte-identical capture plan used by both renderers")
    parity_parser.add_argument("--blender-dir", type=Path, required=True, help="Native Blender frame/receipt directory")
    parity_parser.add_argument("--web-dir", type=Path, required=True, help="WebGL frame/receipt directory")
    parity_parser.add_argument("--output", type=Path, required=True, help="New review report/contact-sheet directory")
    parity_parser.add_argument("--min-iou", type=float, default=.90)
    parity_parser.add_argument("--max-center-delta", type=float, default=.035)
    for command in ("reference-audit", "reconstruct-photos"):
        photo_parser = subcommands.add_parser(
            command, help="Audit statue photo evidence or build a sparse COLMAP camera reconstruction")
        photo_input = photo_parser.add_mutually_exclusive_group(required=True)
        photo_input.add_argument("--manifest", type=Path,
                                 help="Local reference-only image dossier JSON")
        if command == "reconstruct-photos":
            photo_input.add_argument("--photos", type=Path,
                                     help="Folder of 8–256 local photos; skip dense meshing")
            photo_parser.add_argument("--asset-id", help="Required with --photos")
            photo_parser.add_argument("--output", type=Path, required=True,
                                      help="New immutable reconstruction revision directory")
            photo_parser.add_argument("--colmap", default="colmap", help="Installed COLMAP executable")
            photo_parser.add_argument("--engine", choices=("colmap", "pycolmap"), default="colmap",
                                      help="COLMAP CLI (default) or pip-installed CPU PyCOLMAP")
            photo_parser.add_argument("--matching-preset",
                                      choices=("sift", "sift-lightglue", "aliked-lightglue", "loma", "loma128"),
                                      default="sift", help="Learned matching requires ONNX-enabled COLMAP 4.2+")
            photo_parser.add_argument("--matching-device", choices=("colmap-cpu", "mps"),
                                      default="colmap-cpu",
                                      help="COLMAP CPU matching or Apple Silicon PyTorch MPS LightGlue")
            photo_parser.add_argument("--feature-threads", type=int,
                                      help="CPU feature workers (learned default: 1)")
            photo_parser.add_argument("--matching-threads", type=int,
                                      help="CPU matching workers (learned default: 1)")
            photo_parser.add_argument("--sift-first-octave", type=int, choices=(-1, 0),
                                      help="SIFT pyramid start (learned SIFT default: 0, keeps input pixels)")
            photo_parser.add_argument("--max-colmap-rss-mb", type=int,
                                      help="Abort native COLMAP over RSS budget; 0 disables (learned default: 12288 MiB)")
            photo_parser.add_argument("--matching-block-size", type=int,
                                      help="Exhaustive pair-cache block size (learned default: 5)")
            photo_parser.add_argument("--matching-max-features", type=int,
                                      help="Limit actual SIFT keypoints per image without downscaling (learned default: 4096)")
            photo_parser.add_argument("--matching-pairs-per-process", type=int,
                                      help="Run this many image pairs per COLMAP worker, checkpoint each batch")
            photo_parser.add_argument("--matching-pair-list", type=Path,
                                      help="Optional file of staged image-name pairs, one per line; reduces pair count")
            photo_parser.add_argument("--resume", action="store_true",
                                      help="Resume matching in the same verified output after interruption")
            photo_parser.add_argument("--timeout-seconds", type=int, default=3600)
            photo_parser.add_argument("--undistort", action="store_true",
                                      help="Rectify camera intrinsics and image pixels for accurate Blender projection")
            photo_parser.add_argument("--max-image-size", type=int, default=4000)
            photo_parser.add_argument("--registration-profile", choices=("standard", "high-recall"),
                                      default="standard", help="Optional high-recall CPU PyCOLMAP matching")
    for command in ("plan-photo-model", "build-photo-model"):
        photo_model_parser = subcommands.add_parser(
            command, help="From one folder of overlapping object photographs to a private mesh GLB")
        photo_model_input = photo_model_parser.add_mutually_exclusive_group(required=True)
        photo_model_input.add_argument("--photos", type=Path,
                                       help="Folder of 8–256 photos of the same object")
        photo_model_input.add_argument("--manifest", type=Path,
                                       help="Curated multi-view ReferenceSet JSON manifest")
        photo_model_parser.add_argument("--asset-id", help="Required with --photos")
        photo_model_parser.add_argument("--output", type=Path, required=True,
                                        help="New private, immutable output revision")
        photo_model_parser.add_argument("--engine", choices=("pycolmap", "colmap", "mapanything"),
                                        default="pycolmap",
                                        help="Sparse registration engine; mapanything predicts poses feed-forward (Mac MPS, CUDA or CPU)")
        photo_model_parser.add_argument("--mapanything-model", default="facebook/map-anything-apache",
                                        help="Hugging Face MapAnything checkpoint (default Apache-2.0)")
        photo_model_parser.add_argument("--mapanything-device", choices=("auto", "cuda", "mps", "cpu"),
                                        default="auto", help="Torch device for --engine mapanything")
        photo_model_parser.add_argument("--mapanything-depth-tolerance", type=float, default=0.30,
                                        help="Relative depth agreement (0.01–1.0) for views to count as mutually consistent")
        photo_model_parser.add_argument("--matching-preset",
                                        choices=("sift", "sift-lightglue", "aliked-lightglue", "loma", "loma128"),
                                        default="sift", help="Learned matching requires --engine colmap and COLMAP 4.2+")
        photo_model_parser.add_argument("--matching-device", choices=("colmap-cpu", "mps"),
                                        default="colmap-cpu",
                                        help="COLMAP CPU matching or Apple Silicon PyTorch MPS LightGlue")
        photo_model_parser.add_argument("--feature-threads", type=int,
                                        help="CPU feature workers (learned default: 1)")
        photo_model_parser.add_argument("--matching-threads", type=int,
                                        help="CPU matching workers (learned default: 1)")
        photo_model_parser.add_argument("--sift-first-octave", type=int, choices=(-1, 0),
                                        help="SIFT pyramid start (learned SIFT default: 0)")
        photo_model_parser.add_argument("--max-colmap-rss-mb", type=int,
                                        help="Native COLMAP RSS limit, 0 disables (learned default: 12288 MiB)")
        photo_model_parser.add_argument("--matching-block-size", type=int,
                                        help="Exhaustive pair-cache block size (learned default: 5)")
        photo_model_parser.add_argument("--matching-max-features", type=int,
                                        help="Limit actual SIFT keypoints per image (learned default: 4096)")
        photo_model_parser.add_argument("--matching-pairs-per-process", type=int,
                                        help="Restart COLMAP after this many image pairs to bound memory")
        photo_model_parser.add_argument("--backend", choices=("openmvs", "colmap"),
                                        default="openmvs", help="Dense meshing backend")
        photo_model_parser.add_argument("--colmap", default="colmap")
        photo_model_parser.add_argument("--blender", default="blender")
        photo_model_parser.add_argument("--min-registered-images", type=int, default=8)
        photo_model_parser.add_argument("--min-registered-fraction", type=float, default=0.25)
        photo_model_parser.add_argument("--sparse-timeout-seconds", type=int, default=3600)
        photo_model_parser.add_argument("--dense-timeout-seconds", type=int, default=7200)
        photo_model_parser.add_argument("--max-image-size", type=int, default=3200)
        photo_model_parser.add_argument("--no-texture", action="store_true")
        photo_model_parser.add_argument("--no-renders", action="store_true")
    statue_glb = subcommands.add_parser(
        "build-statue-glb", help="One-command MPS statue reconstruction with angle/mask QA gates and resumable GLB export")
    statue_glb.add_argument("--photos", type=Path,
                           help="Photo folder; reuse the last folder for this asset if omitted")
    statue_glb.add_argument("--asset-id", required=True)
    statue_glb.add_argument("--output", type=Path,
                           help="Output revision; defaults to .review/ASSET/statue-hq-rNNN")
    statue_glb.add_argument("--front-photo",
                           help="Optional frontal image override; otherwise infer from registered image metadata")
    statue_glb.add_argument("--openmvs-dir", type=Path, default=Path("/Applications/OpenMVS-2.4.0"))
    statue_glb.add_argument("--blender", default="/Applications/Blender.app/Contents/MacOS/Blender")
    statue_glb.add_argument("--sam2-checkpoint", type=Path,
                           help="Optional SAM2.1 checkpoint (discover locally or download from Meta if omitted)")
    statue_glb.add_argument("--no-auto-model-download", action="store_true",
                           help="Never download missing model weights; search local caches only")
    statue_glb.add_argument("--sam2-size", choices=("tiny", "small", "base_plus", "large"), default="small")
    statue_glb.add_argument("--sam2-device", choices=("auto", "mps", "cpu"), default="mps")
    statue_glb.add_argument("--mask-guides", type=Path,
                           help="Optional manual overrides for automatically inferred segmentation prompts")
    statue_glb.add_argument("--masks-dir", type=Path,
                           help="Previously reviewed rectified masks; skips automatic SAM2 masking")
    statue_glb.add_argument("--mask-cache-dir", type=Path,
                           help="Persistent per-image mask cache (defaults to a sibling of the output revision)")
    statue_glb.add_argument("--ai-bridges", action="store_true",
                           help="Opt in to synthetic ComfyUI reference previews for unregistered photo pairs")
    statue_glb.add_argument("--ai-bridge-workflow", type=Path,
                           help="Optional local ComfyUI API workflow; otherwise plan previews only")
    statue_glb.add_argument("--min-registered-fraction", type=float, default=0.80)
    statue_glb.add_argument("--max-mixed-mask-fraction", type=float, default=0.12)
    statue_glb.add_argument("--max-colmap-rss-mb", type=int, default=12288)
    statue_glb.add_argument("--timeout-seconds", type=int, default=14400)
    statue_glb.add_argument("--resume", action="store_true")
    completion_image_parser = subcommands.add_parser(
        "generate-completion-image",
        help="Generate a private ComfyUI image candidate with an existing local API workflow")
    completion_image_parser.add_argument("--image", type=Path, required=True)
    completion_image_parser.add_argument("--workflow", type=Path, required=True)
    completion_image_parser.add_argument("--image-node", required=True)
    completion_image_parser.add_argument("--output-node", required=True)
    completion_image_parser.add_argument("--output", type=Path, required=True)
    completion_image_parser.add_argument("--server", default="http://127.0.0.1:8188")
    completion_image_parser.add_argument("--require-mps", action="store_true")
    completion_image_parser.add_argument("--timeout-seconds", type=int, default=360)
    camera_compare_parser = subcommands.add_parser(
        "compare-camera-registrations",
        help="Read-only before/after registration coverage for two COLMAP revisions")
    camera_compare_parser.add_argument("--baseline", type=Path, required=True)
    camera_compare_parser.add_argument("--candidate", type=Path, required=True)
    for command in ("plan-sfm-cohort-recovery", "recover-sfm-cohorts", "diagnose-sfm-coverage"):
        cohort_parser = subcommands.add_parser(
            command, help="Recover missed camera views as independent SfM cohorts with guarded alignment")
        cohort_parser.add_argument("--revision", type=Path, required=True,
                                   help="Existing sparse photo revision with original COLMAP database")
        cohort_parser.add_argument("--model", type=int, default=0,
                                   help="Baseline COLMAP sparse model index")
        cohort_parser.add_argument("--targets", type=Path,
                                   help="JSON list of staged filenames; default is all unregistered photos")
        cohort_parser.add_argument("--anchor", action="append", default=[],
                                   help="Already registered bridge filename (repeatable)")
        cohort_parser.add_argument("--max-anchors", type=int, default=6)
        cohort_parser.add_argument("--max-trials", type=int, default=5)
        cohort_parser.add_argument("--min-inliers", type=int, default=25)
        cohort_parser.add_argument("--min-new-cameras", type=int, default=2)
        cohort_parser.add_argument("--min-seed-angle", type=float, default=1.0,
                                   help="Exclude confidently near-zero-parallax calibrated seed pairs")
        if command == "diagnose-sfm-coverage":
            cohort_parser.add_argument("--prior-recovery", type=Path,
                                       help="Optional existing cohort-recovery-report.json")
            cohort_parser.add_argument("--format", choices=("json", "markdown"), default="json",
                                       help="JSON for automation, Markdown for a readable missing-angle checklist")
            cohort_parser.add_argument("--guide-dir", type=Path,
                                       help="New folder for original-filename guide and side-by-side photo contact sheet")
        if command == "recover-sfm-cohorts":
            cohort_parser.add_argument("--output", type=Path, required=True,
                                       help="New independent output; never alters source models")
            cohort_parser.add_argument("--colmap", default="colmap")
            cohort_parser.add_argument("--trial-timeout", type=int, default=1200)
    bridge_parser = subcommands.add_parser(
        "prepare-sfm-bridge-photos",
        help="Stage new real overlapping photos with original staged images, privately and without modifying source")
    bridge_parser.add_argument("--revision", type=Path, required=True)
    bridge_parser.add_argument("--photos", type=Path, required=True,
                               help="Folder of new real viewpoint photographs")
    bridge_parser.add_argument("--output", type=Path, required=True,
                               help="Unused directory for the new combined photo set")
    bridge_parser.add_argument("--model", type=int, default=0)
    bridge_parser.add_argument("--manifest", type=Path,
                               help="Optional source attribution and license JSON for new photographs")
    alignment_parser = subcommands.add_parser(
        "inspect-sfm-cohort-alignment", help="Audit one solved cohort using shared cameras and optionally 3D object landmarks")
    alignment_parser.add_argument("--revision", type=Path, required=True)
    alignment_parser.add_argument("--model", type=int, default=0)
    alignment_parser.add_argument("--candidate", type=Path, required=True,
                                  help="Exact independent model directory (trial-NN/sparse/0)")
    alignment_parser.add_argument("--landmarks", type=Path,
                                  help="JSON object landmarks in the two specific sparse coordinate frames")
    for command in ("plan-dense-photos", "build-dense-photos"):
        dense_parser = subcommands.add_parser(
            command, help="Review or reconstruct a private dense statue mesh from registered cameras")
        dense_parser.add_argument("--reconstruction", type=Path, required=True,
                                  help="Successful reconstruct-photos --undistort revision")
        dense_parser.add_argument("--rectified-masks", type=Path,
                                  help="Private binary grayscale <photo>.mask.png files matching every rectified image")
        dense_parser.add_argument("--output", type=Path, required=True,
                                  help="New private, immutable dense-mesh revision")
        dense_parser.add_argument("--backend", choices=("openmvs", "colmap"), default="openmvs",
                                  help="OpenMVS CPU (default) or COLMAP CUDA GPU")
        dense_parser.add_argument("--min-registered-images", type=int, default=8)
        dense_parser.add_argument("--min-registered-fraction", type=float, default=0.60)
        dense_parser.add_argument("--min-mesh-faces", type=int, default=1000)
        dense_parser.add_argument("--resolution-level", type=int, default=0,
                                  help="OpenMVS depth map downscale (0 preserves input pixels)")
        dense_parser.add_argument("--max-resolution", type=int, default=3200,
                                  help="OpenMVS maximum depth map image dimension")
        dense_parser.add_argument("--number-views", type=int, default=5,
                                  help="OpenMVS neighboring views per depth estimate")
        dense_parser.add_argument("--refine-decimate", type=float, default=1.0,
                                  help="OpenMVS pre-refinement face retention; 1 preserves high-poly detail")
        dense_parser.add_argument("--texture-decimate", type=float, default=1.0,
                                  help="OpenMVS texture OBJ face retention; lower for faster atlas LOD while PLY stays high-poly")
        dense_parser.add_argument("--no-global-seam-leveling", action="store_true",
                                  help="OpenMVS: disable global texture seam equalization if it creates dark artifacts")
        dense_parser.add_argument("--no-local-seam-leveling", action="store_true",
                                  help="OpenMVS: disable local texture seam blending if it creates dark artifacts")
        dense_parser.add_argument("--timeout-seconds", type=int, default=7200)
        dense_parser.add_argument("--no-refine", action="store_true",
                                  help="OpenMVS: skip refined topology")
        dense_parser.add_argument("--no-texture", action="store_true",
                                  help="Skip the private image-to-mesh texture transfer")
        if command == "build-dense-photos":
            dense_parser.add_argument("--resume", action="store_true",
                                      help="Resume an incomplete OpenMVS revision without repeating completed stages")
    for command in ("plan-splats", "build-splats"):
        splat_parser = subcommands.add_parser(
            command, help="Train private Gaussian splats from multi-angle photos or COLMAP cameras")
        splat_input = splat_parser.add_mutually_exclusive_group(required=True)
        splat_input.add_argument("--photos", type=Path,
                                 help="Directory of 8+ overlapping photos of one object")
        splat_input.add_argument("--references", type=Path,
                                 help="Local multi-view ReferenceSet manifest JSON")
        splat_input.add_argument("--reconstruction", type=Path,
                                 help="Successful reconstruct-photos --undistort revision")
        splat_parser.add_argument("--asset-id", help="Required slug for --photos")
        splat_parser.add_argument("--output", type=Path, required=True,
                                  help="New immutable, private splat revision folder")
        splat_parser.add_argument("--opensplat", default="opensplat",
                                  help="OpenSplat executable, built with Metal/MPS or CUDA")
        splat_parser.add_argument("--iterations", type=int, default=7000)
        splat_parser.add_argument("--downscale", type=float, default=1.0,
                                  help="OpenSplat image downscale factor (1–16), useful for limited GPU memory")
        splat_parser.add_argument("--format", choices=("ply", "spz"), default="ply")
        splat_parser.add_argument("--timeout-seconds", type=int, default=14_400,
                                  help="Maximum OpenSplat training seconds")
        splat_parser.add_argument("--engine", choices=("pycolmap", "colmap"),
                                  default="pycolmap", help="Camera registration for --photos/--references")
        splat_parser.add_argument("--colmap", default="colmap",
                                  help="COLMAP binary when --engine colmap")
        splat_parser.add_argument("--sparse-timeout-seconds", type=int, default=3600)
        splat_parser.add_argument("--max-image-size", type=int, default=4000)
        splat_parser.add_argument("--registration-profile", choices=("standard", "high-recall"),
                                  default="standard", help="Optional high-recall CPU PyCOLMAP matching")
        splat_parser.add_argument("--rectified-masks", type=Path,
                                  help="Optional reviewed masks matching existing --reconstruction rectified photos")
    for command in ("plan-splat-video", "render-splat-video"):
        video_parser = subcommands.add_parser(
            command, help="GPU-render Gaussian splat orbit frames and encode private H.264 MP4")
        video_parser.add_argument("--splat", type=Path, required=True,
                                  help="Candidate splat revision directory or Gaussian .ply/.spz")
        video_parser.add_argument("--output", type=Path, required=True,
                                  help="New immutable video revision directory")
        video_parser.add_argument("--camera-track", type=Path,
                                  help="Existing SplatTransform/SuperSplat camera track (instead of auto orbit)")
        video_parser.add_argument("--fps", type=int, default=30)
        video_parser.add_argument("--seconds", type=int, default=6)
        video_parser.add_argument("--width", type=int, default=1920)
        video_parser.add_argument("--height", type=int, default=1080)
        video_parser.add_argument("--fov", type=float, default=50.0)
        video_parser.add_argument("--start-degrees", type=float, default=-65.0)
        video_parser.add_argument("--orbit-degrees", type=float, default=115.0)
        video_parser.add_argument("--elevation-degrees", type=float, default=12.0)
        video_parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"))
        video_parser.add_argument("--distance", type=float)
        video_parser.add_argument("--hero-photo",
                                  help="Registered photo filename to anchor an observed-angle orbit (e.g. photo-0001.png)")
        video_parser.add_argument("--shutter", type=float, default=0.0,
                                  help="Motion blur shutter fraction 0–1 (0 disables)")
        video_parser.add_argument("--viewer", action="store_true",
                                  help="Also build a self-contained private splat viewer.html")
        video_parser.add_argument("--splat-transform", default="splat-transform")
        video_parser.add_argument("--ffmpeg", default="ffmpeg")
        video_parser.add_argument("--ffprobe", default="ffprobe")
        video_parser.add_argument("--timeout-seconds", type=int, default=7200)
    for command in ("plan-splat-view-qa", "compare-splat-views"):
        qa_parser = subcommands.add_parser(
            command, help="GPU-render COLMAP-matched splat views and compare them with pinned photos")
        qa_parser.add_argument("--splat", type=Path, required=True,
                               help="Complete splat revision with registered source cameras/photos")
        qa_parser.add_argument("--output", type=Path, required=True,
                               help="New private, immutable multi-angle QA revision")
        qa_parser.add_argument("--max-views", type=int, default=12,
                               help="Pose-spread selection of 3–48 registered photographs")
        qa_parser.add_argument("--max-dimension", type=int, default=1280,
                               help="Limit longest rendered photo edge to 256–4096 pixels")
        qa_parser.add_argument("--splat-transform", default="splat-transform")
        qa_parser.add_argument("--timeout-seconds", type=int, default=900)
    photo_mask_parser = subcommands.add_parser(
        "prepare-dense-masks", help="Create inspectable rectified foreground mask candidates with CPU GrabCut")
    photo_mask_parser.add_argument("--reconstruction", type=Path, required=True)
    photo_mask_parser.add_argument("--output", type=Path, required=True)
    photo_mask_parser.add_argument("--guides", type=Path, help="Optional per-photo normalized boxes and seed points JSON")
    photo_mask_parser.add_argument("--center-box", type=float, nargs=4,
                                   default=(0.12, 0.04, 0.88, 0.96),
                                   metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    photo_mask_parser.add_argument("--max-side", type=int, default=1280)
    photo_mask_parser.add_argument("--iterations", type=int, default=5)
    photo_mask_parser.add_argument("--segmenter", choices=("grabcut", "rembg", "sam2"), default="grabcut",
                                   help="CPU GrabCut, semantic IS-Net, or offline guided SAM 2.1")
    photo_mask_parser.add_argument("--sam2-checkpoint", type=Path,
                                   help="Existing official SAM 2.1 weights (.pt); no automatic download")
    photo_mask_parser.add_argument("--sam2-size", choices=("tiny", "small", "base_plus", "large"),
                                   default="small")
    photo_mask_parser.add_argument("--sam2-device", choices=("auto", "mps", "cpu"), default="auto")
    photo_mask_parser.add_argument("--cache-dir", type=Path,
                                   help="Optional content-addressed cache of masks for identical rectified photos")
    refine_mask_parser = subcommands.add_parser(
        "refine-dense-masks", help="Remove tiny mask islands/holes or apply camera-specific polygon cleanup")
    refine_mask_parser.add_argument("--reconstruction", type=Path, required=True)
    refine_mask_parser.add_argument("--mask-revision", type=Path, required=True)
    refine_mask_parser.add_argument("--output", type=Path, required=True)
    refine_mask_parser.add_argument("--edits", type=Path,
                                    help="Normalized per-photo include/exclude polygons JSON")
    refine_mask_parser.add_argument("--min-island-pixels", type=int, default=0)
    refine_mask_parser.add_argument("--max-hole-pixels", type=int, default=0)
    refine_mask_parser.add_argument("--edge-inset-pixels", type=int, default=0,
                                    help="Optional 1–8 pixel contour inset to reduce background edge halos")
    refine_mask_parser.add_argument("--color-band-pixels", type=int, default=0,
                                    help="Optional 1–32 pixel local GrabCut boundary reclassification (candidate only)")
    refine_mask_parser.add_argument("--color-trim-only", action="store_true",
                                    help="Conservative color cleanup: remove suspected background, never add foreground")
    refine_mask_parser.add_argument("--color-max-side", type=int, default=1280)
    refine_mask_parser.add_argument("--color-iterations", type=int, default=3)
    hybrid_mask_parser = subcommands.add_parser(
        "compose-dense-masks", help="Select the best registered masks photo by photo across two candidate revisions")
    hybrid_mask_parser.add_argument("--reconstruction", type=Path, required=True)
    hybrid_mask_parser.add_argument("--primary-mask-revision", type=Path, required=True)
    hybrid_mask_parser.add_argument("--alternate-mask-revision", type=Path, required=True)
    hybrid_mask_parser.add_argument("--output", type=Path, required=True)
    hybrid_mask_parser.add_argument("--use-alternate", nargs="+", required=True,
                                    metavar="PHOTO", help="Exact filenames selected from the alternate revision")
    mask_consistency_parser = subcommands.add_parser(
        "audit-dense-masks", help="Inspect multi-view COLMAP feature-track disagreements in rectified masks")
    mask_consistency_parser.add_argument("--reconstruction", type=Path, required=True)
    mask_consistency_parser.add_argument("--rectified-masks", type=Path, required=True)
    mask_consistency_parser.add_argument("--output", type=Path, required=True)
    mask_consistency_parser.add_argument("--max-overlay-points", type=int, default=200)
    review_mask_parser = subcommands.add_parser(
        "review-mask-suggestions", help="Accept individual verified COLMAP correction prompts into a new guides revision")
    review_mask_parser.add_argument("--reconstruction", type=Path, required=True)
    review_mask_parser.add_argument("--mask-revision", type=Path, required=True)
    review_mask_parser.add_argument("--audit-revision", type=Path, required=True)
    review_mask_parser.add_argument("--output", type=Path, required=True)
    review_mask_parser.add_argument("--accept", nargs="+", required=True,
                                    metavar="PHOTO:POINT3D_ID",
                                    help="Human-confirmed non-tied feature suggestions; never applied to masks directly")
    mask_comparison_parser = subcommands.add_parser(
        "compare-dense-masks", help="Compare two registered mask revisions using identical COLMAP tracks")
    mask_comparison_parser.add_argument("--reconstruction", type=Path, required=True)
    mask_comparison_parser.add_argument("--baseline-masks", type=Path, required=True)
    mask_comparison_parser.add_argument("--candidate-masks", type=Path, required=True)
    mask_comparison_parser.add_argument("--output", type=Path, required=True)
    mask_comparison_parser.add_argument("--max-overlay-points", type=int, default=200)
    texture_finish_parser = subcommands.add_parser(
        "finish-dense-textures", help="Non-destructive UV-atlas exposure, shadow and cast correction")
    texture_finish_parser.add_argument("--dense-revision", type=Path, required=True)
    texture_finish_parser.add_argument("--output", type=Path, required=True)
    texture_finish_parser.add_argument("--exposure-ev", type=float, default=0.0)
    texture_finish_parser.add_argument("--gamma", type=float, default=1.0)
    texture_finish_parser.add_argument("--saturation", type=float, default=1.0)
    texture_finish_parser.add_argument("--shadow-lift", type=float, default=0.0)
    texture_finish_parser.add_argument("--neutralize-cast", action="store_true")
    texture_finish_parser.add_argument("--protect-black", type=int, default=12)
    texture_finish_parser.add_argument("--balance-atlases", action="store_true",
                                       help="Match valid-texel exposure across multiple UV atlas pages")
    texture_finish_parser.add_argument("--balance-uv-seams", action="store_true",
                                       help="Bounded page exposure from shared mesh vertices across UV atlas pages")
    texture_finish_parser.add_argument("--pad-uv-gutters", action="store_true",
                                       help="Extend existing mapped UV colors into nearby dark/transparent unmapped texels")
    texture_finish_parser.add_argument("--gutter-pixels", type=int, default=4,
                                       help="Maximum UV edge padding distance in source-atlas pixels (1–16)")
    texture_finish_parser.add_argument("--uv-aware-stats", action="store_true",
                                       help="Sample only actual OBJ UV faces; report dark/blank texels on the mesh")
    texture_finish_parser.add_argument("--max-balance-ev", type=float, default=0.75)
    texture_finish_parser.add_argument("--max-seam-ev", type=float, default=0.75)
    texture_finish_parser.add_argument("--min-seam-anchors", type=int, default=8)
    texture_finish_parser.add_argument("--auto-exposure-luma", type=float,
                                       help="Opt-in target median brightness (32–224) across all atlas pages")
    texture_finish_parser.add_argument("--max-auto-exposure-ev", type=float, default=1.0)
    texture_finish_parser.add_argument("--surface-recipe", type=Path,
                                       help="Optional local per-atlas material masks, lighting, cleanup and height JSON")
    texture_finish_parser.add_argument("--surface-delight-strength", type=float, default=0.0,
                                       help="0–1 bounded linear-light photo de-lighting (guided map preferred)")
    texture_finish_parser.add_argument("--surface-max-delight-ev", type=float, default=1.0)
    texture_finish_parser.add_argument("--surface-low-frequency-radius", type=int, default=96,
                                       help="Heuristic illumination blur radius when no guidance map exists")
    texture_finish_parser.add_argument("--surface-cleanup-radius", type=int, default=0,
                                       help="Radius 1–16 for explicitly masked small texture defects")
    texture_finish_parser.add_argument("--surface-multiview-strength", type=float, default=0.0,
                                       help="0–1 confidence-masked fusion of 3–8 UV-reprojected photos")
    texture_finish_parser.add_argument("--surface-multiview-max-ev", type=float, default=1.5,
                                       help="Maximum per-pixel and per-material relative correction (0–2 EV)")
    for command in ("plan-material-separation", "propose-material-separation"):
        separate_parser = subcommands.add_parser(
            command, help="Propose reviewable UV material masks from labeled seeds and registered photo views")
        separate_parser.add_argument("--dense-revision", type=Path, required=True)
        separate_parser.add_argument("--seed-recipe", type=Path, required=True,
                                     help="Normal surface recipe whose region masks are sparse seed strokes")
        separate_parser.add_argument("--output", type=Path, required=True)
        separate_parser.add_argument("--min-margin", type=float, default=.25)
        separate_parser.add_argument("--max-distance", type=float, default=3.5)
        separate_parser.add_argument("--min-evidence-views", type=int, default=2)
        separate_parser.add_argument("--max-side", type=int, default=8192)
        separate_parser.add_argument("--max-mesh-faces", type=int, default=5_000_000)
    defect_parser = subcommands.add_parser(
        "detect-photo-texture-defects",
        help="Detect localized photographic texture inconsistencies as review-only masks")
    defect_parser.add_argument("--dense-revision", type=Path, required=True)
    defect_parser.add_argument("--surface-recipe", type=Path, required=True)
    defect_parser.add_argument("--output", type=Path, required=True)
    defect_parser.add_argument("--color-difference", type=int, default=45)
    defect_parser.add_argument("--local-excess", type=int, default=20)
    defect_parser.add_argument("--spatial-radius", type=int, default=12)
    defect_parser.add_argument("--max-side", type=int, default=8192)
    defect_parser.add_argument("--max-mesh-faces", type=int, default=5_000_000)
    illumination_parser = subcommands.add_parser(
        "estimate-uv-illumination",
        help="Build bounded per-material illumination guides from 3+ registered UV photos")
    illumination_parser.add_argument("--dense-revision", type=Path, required=True)
    illumination_parser.add_argument("--surface-recipe", type=Path, required=True)
    illumination_parser.add_argument("--output", type=Path, required=True)
    illumination_parser.add_argument("--smoothing-radius", type=int, default=32)
    illumination_parser.add_argument("--max-illumination-ev", type=float, default=1.5)
    illumination_parser.add_argument("--max-side", type=int, default=8192)
    illumination_parser.add_argument("--max-mesh-faces", type=int, default=5_000_000)
    polarized_parser = subcommands.add_parser(
        "inspect-polarized-uv",
        help="Inspect controlled parallel/cross polarization pairs; contrast is NOT measured roughness")
    polarized_parser.add_argument("--dense-revision", type=Path, required=True)
    polarized_parser.add_argument("--surface-recipe", type=Path, required=True)
    polarized_parser.add_argument("--captures", type=Path, required=True,
                                  help="JSON with 3–8 UV-aligned parallel/cross photo pairs per atlas")
    polarized_parser.add_argument("--output", type=Path, required=True)
    polarized_parser.add_argument("--max-side", type=int, default=8192)
    polarized_parser.add_argument("--min-views", type=int, default=3)
    polarized_parser.add_argument("--max-mesh-faces", type=int, default=5_000_000)
    projection_parser = subcommands.add_parser(
        "project-registered-uv-views",
        help="Project hash-pinned COLMAP photographs onto OBJ UV atlases with occlusion masks")
    projection_parser.add_argument("--dense-revision", type=Path, required=True)
    projection_parser.add_argument("--output", type=Path, required=True)
    projection_parser.add_argument("--views", type=int, default=3)
    projection_parser.add_argument("--atlas-side", type=int, default=2048)
    projection_parser.add_argument("--camera-side", type=int, default=1536)
    projection_parser.add_argument("--max-triangles", type=int, default=500_000)
    for command in ("plan-scan-displacement", "bake-scan-displacement"):
        scan_bake_parser = subcommands.add_parser(
            command, help="Bake real dense PLY relief into signed 16-bit UV displacement")
        scan_bake_parser.add_argument("--dense-revision", type=Path, required=True)
        scan_bake_parser.add_argument("--output", type=Path, required=True)
        scan_bake_parser.add_argument("--meters-per-unit", type=float, required=True,
                                     help="Measured physical metres per COLMAP world unit")
        scan_bake_parser.add_argument("--max-offset-m", type=float, default=.05)
        scan_bake_parser.add_argument("--max-side", type=int, default=2048)
        scan_bake_parser.add_argument("--max-mesh-faces", type=int, default=5_000_000)
        scan_bake_parser.add_argument("--max-uv-triangles", type=int, default=500_000)
        if command == "bake-scan-displacement":
            scan_bake_parser.add_argument("--blender", default="blender")
    udim_parser = subcommands.add_parser(
        "package-finished-udim", help="Copy a verified finished multi-atlas OBJ to numbered UDIM tiles")
    udim_parser.add_argument("--texture-finish", type=Path, required=True)
    udim_parser.add_argument("--output", type=Path, required=True)
    udim_review = subcommands.add_parser(
        "render-udim-material-review",
        help="Compare one verified UDIM model from three angles under opposing lights")
    udim_review.add_argument("--udim-package", type=Path, required=True)
    udim_review.add_argument("--output", type=Path, required=True)
    udim_review.add_argument("--resolution", type=int, default=512)
    udim_review.add_argument("--samples", type=int, default=24)
    udim_review.add_argument("--matched-views", type=int, default=0,
                             help="Include 1–6 exact registered COLMAP/photo views, if available")
    udim_review.add_argument("--matched-resolution-percent", type=int, default=25)
    udim_review.add_argument("--no-renders", action="store_true")
    udim_review.add_argument("--blender", default="blender")
    for command in ("plan-dense-blender-review", "render-dense-blender-review"):
        review_dense = subcommands.add_parser(
            command, help="Inspect or create a private editable Blender scene from a dense PLY")
        review_dense.add_argument("--dense-revision", type=Path, required=True)
        review_dense.add_argument("--output", type=Path, required=True)
        review_dense.add_argument("--max-mesh-faces", type=int, default=5_000_000)
        review_dense.add_argument("--matched-views", type=int, default=0,
                                  help="Render calibrated dense-mesh views against pinned rectified photos (0–12)")
        review_dense.add_argument("--matched-resolution-percent", type=int, default=100,
                                  help="Photo-matched preview resolution percentage; 100 retains full camera pixels")
        review_dense.add_argument("--textured", action="store_true",
                                  help="Import pinned OpenMVS OBJ/MTL/atlas as private UV-textured Blender/GLB review")
        review_dense.add_argument("--texture-finish", type=Path,
                                  help="Use hash-pinned corrected UV atlas revision (requires --textured)")
        if command == "render-dense-blender-review":
            review_dense.add_argument("--no-renders", action="store_true")
            review_dense.add_argument("--blender", default="blender")
    camera_parser = subcommands.add_parser(
        "prepare-camera-plan", help="Align sparse COLMAP cameras to sculpt-space 3D landmarks")
    camera_parser.add_argument("--reconstruction", type=Path, required=True)
    camera_parser.add_argument("--alignment", type=Path, required=True)
    camera_parser.add_argument("--output", type=Path, required=True)
    camera_parser.add_argument("--max-anchor-rms-m", type=float, default=0.10)
    camera_parser.add_argument("--undistorted-masks", type=Path,
                               help="Optional manually verified silhouette masks for rectified photos")
    review_parser = subcommands.add_parser(
        "render-reference-scene", help="Create a private Blender scene with registered reference cameras")
    review_parser.add_argument("--plan", type=Path, required=True)
    review_parser.add_argument("--model", type=Path, required=True, help="Local aligned sculpt GLB")
    review_parser.add_argument("--output", type=Path, required=True)
    review_parser.add_argument("--render-clay", action="store_true")
    review_parser.add_argument("--max-cameras", type=int, default=8)
    review_parser.add_argument("--blender", default="blender")
    mask_parser = subcommands.add_parser("silhouette-qa", help="Compare registered clay alpha masks to subject masks")
    mask_parser.add_argument("--plan", type=Path, required=True)
    mask_parser.add_argument("--renders", type=Path, required=True)
    mask_parser.add_argument("--output", type=Path, required=True)
    mask_parser.add_argument("--min-iou", type=float, default=0.8)
    landmark_parser = subcommands.add_parser(
        "landmark-qa", help="Measure sculpt 3D landmark reprojection error in registered photos")
    landmark_parser.add_argument("--plan", type=Path, required=True)
    landmark_parser.add_argument("--annotations", type=Path, required=True)
    landmark_parser.add_argument("--output", type=Path, required=True)
    landmark_parser.add_argument("--max-error-px", type=float, default=12.0)
    for command in ("plan-head-base", "build-head-base"):
        head_parser = subcommands.add_parser(
            command, help="Align an existing licensed anatomical human-head GLB to statue landmarks")
        head_parser.add_argument("--manifest", type=Path, required=True,
                                 help="Local source GLB, rights, target and anatomical landmark correspondence")
        if command == "build-head-base":
            head_parser.add_argument("--output", type=Path, required=True)
            head_parser.add_argument("--blender", default="blender")
    makehuman_parser = subcommands.add_parser(
        "prepare-makehuman-head", help="Prepare an official, pinned CC0 MakeHuman head base")
    makehuman_input = makehuman_parser.add_mutually_exclusive_group(required=True)
    makehuman_input.add_argument("--source-obj", type=Path,
                                 help="Previously downloaded official MakeHuman base.obj (exact hash required)")
    makehuman_input.add_argument("--download", action="store_true",
                                 help="Explicitly download the pinned CC0 source to a local cache")
    makehuman_parser.add_argument("--cache", type=Path,
                                  default=Path(".review/source-cache/makehuman/base.obj"))
    makehuman_parser.add_argument("--output", type=Path, required=True)
    makehuman_parser.add_argument("--blender", default="blender")
    source_parser = subcommands.add_parser("inspect-asset")
    source_parser.add_argument("--input", type=Path, required=True)
    texture_parser = subcommands.add_parser("inspect-texture-packs")
    texture_parser.add_argument("--input", type=Path, required=True)
    import_parser = subcommands.add_parser("import-asset")
    import_parser.add_argument("--input", type=Path, required=True)
    import_parser.add_argument("--provenance", type=Path, required=True)
    import_parser.add_argument("--material-overrides", type=Path)
    import_parser.add_argument("--texture-packs", type=Path)
    import_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command in ("plan-deformation", "build-deformation"):
            shot = load_deformation_plan(args.spec)
            if args.command == "plan-deformation":
                print(json.dumps({"asset_id": shot.asset_id, "action_id": shot.action_id,
                                  "target": shot.target, "frames": list(shot.frames),
                                  "morph_targets": len(shot.frames) - 1,
                                  "max_vertices": shot.max_vertices}, indent=2))
                return 0
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise DeformationPlanError("build-deformation needs Blender: blender -b source.blend --python scripts/blender_entry.py -- build-deformation --spec plan.json --output new-revision") from error
            from .blender_deformation import build_deformation_shot
            print(json.dumps(build_deformation_shot(bpy, shot, args.output), indent=2))
            return 0
        if args.command in ("plan-physics", "build-physics"):
            shot = load_physics_plan(args.spec)
            if args.command == "plan-physics":
                print(json.dumps({"asset_id": shot.asset_id, "action_id": shot.action_id,
                                  "duration_frames": shot.duration_frames,
                                  "rules": len(shot.bodies), "active_rules": sum(body.kind == "active" for body in shot.bodies),
                                  "constraints": len(shot.constraints), "ground": shot.ground is not None,
                                  "gravity_mps2": shot.gravity_mps2}, indent=2))
                return 0
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise PhysicsPlanError("build-physics requires Blender: blender --background source.blend --python scripts/blender_entry.py -- build-physics --spec plan.json --output new-revision") from error
            from .blender_physics import build_physics_shot
            print(json.dumps(build_physics_shot(bpy, shot, args.output), indent=2))
            return 0
        if args.command == "export-motion":
            print(json.dumps(write_glb_motion(args.model, args.output, clip_name=args.clip,
                                              action_id=args.action_id,
                                              contacts_path=args.contacts,
                                              merge_clips=args.merge_clips,
                                              resample_mixed=args.resample_mixed), indent=2))
            return 0
        if args.command == "prepare-dense-masks":
            from .reconstruction.masking import generate_rectified_masks
            from .reconstruction.references import ReferenceError
            try:
                report = generate_rectified_masks(
                    args.reconstruction, args.output, guides=args.guides,
                    center_box=tuple(args.center_box), max_side=args.max_side,
                    iterations=args.iterations, segmenter=args.segmenter,
                    sam2_checkpoint=args.sam2_checkpoint, sam2_size=args.sam2_size,
                    sam2_device=args.sam2_device, cache_dir=args.cache_dir)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "refine-dense-masks":
            from .reconstruction.mask_refinement import refine_rectified_masks
            from .reconstruction.references import ReferenceError
            try:
                report = refine_rectified_masks(
                    args.reconstruction, args.mask_revision, args.output,
                    edits=args.edits, min_island_pixels=args.min_island_pixels,
                    max_hole_pixels=args.max_hole_pixels,
                    edge_inset_pixels=args.edge_inset_pixels,
                    color_band_pixels=args.color_band_pixels,
                    color_max_side=args.color_max_side,
                    color_iterations=args.color_iterations,
                    color_trim_only=args.color_trim_only)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "compose-dense-masks":
            from .reconstruction.mask_selection import compose_rectified_masks
            from .reconstruction.references import ReferenceError
            try:
                report = compose_rectified_masks(
                    args.reconstruction, args.primary_mask_revision,
                    args.alternate_mask_revision, args.output,
                    alternate_images=args.use_alternate)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "audit-dense-masks":
            from .reconstruction.mask_consistency import audit_multiview_masks
            from .reconstruction.references import ReferenceError
            try:
                report = audit_multiview_masks(
                    args.reconstruction, args.rectified_masks, args.output,
                    max_overlay_points=args.max_overlay_points)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "review-mask-suggestions":
            from .reconstruction.mask_prompt_review import accept_mask_point_suggestions
            from .reconstruction.references import ReferenceError
            try:
                report = accept_mask_point_suggestions(
                    args.reconstruction, args.mask_revision, args.audit_revision,
                    args.output, accept=args.accept)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "compare-dense-masks":
            from .reconstruction.mask_comparison import compare_multiview_masks
            from .reconstruction.references import ReferenceError
            try:
                report = compare_multiview_masks(
                    args.reconstruction, args.baseline_masks, args.candidate_masks,
                    args.output, max_overlay_points=args.max_overlay_points)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "finish-dense-textures":
            from .reconstruction.references import ReferenceError
            from .reconstruction.texture_finish import (TextureFinishSettings,
                                                       finish_dense_textures)
            try:
                report = finish_dense_textures(args.dense_revision, args.output,
                                                TextureFinishSettings(
                                                    exposure_ev=args.exposure_ev,
                                                    gamma=args.gamma,
                                                    saturation=args.saturation,
                                                    shadow_lift=args.shadow_lift,
                                                    neutralize_cast=args.neutralize_cast,
                                                    protect_black=args.protect_black,
                                                    balance_atlases=args.balance_atlases,
                                                    balance_uv_seams=args.balance_uv_seams,
                                                    pad_uv_gutters=args.pad_uv_gutters,
                                                    gutter_pixels=args.gutter_pixels,
                                                    uv_aware_stats=args.uv_aware_stats,
                                                    max_balance_ev=args.max_balance_ev,
                                                    max_seam_ev=args.max_seam_ev,
                                                    min_seam_anchors=args.min_seam_anchors,
                                                    auto_exposure_luma=args.auto_exposure_luma,
                                                    max_auto_exposure_ev=args.max_auto_exposure_ev,
                                                    surface_recipe=(str(args.surface_recipe)
                                                                    if args.surface_recipe else None),
                                                    surface_delight_strength=args.surface_delight_strength,
                                                    surface_max_delight_ev=args.surface_max_delight_ev,
                                                    surface_low_frequency_radius=args.surface_low_frequency_radius,
                                                    surface_cleanup_radius=args.surface_cleanup_radius,
                                                    surface_multiview_strength=args.surface_multiview_strength,
                                                    surface_multiview_max_ev=args.surface_multiview_max_ev))
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command in ("plan-material-separation", "propose-material-separation"):
            from .reconstruction.material_separation import (
                SeparationOptions, plan_material_separation, propose_material_regions,
            )
            from .reconstruction.references import ReferenceError
            options = SeparationOptions(min_margin=args.min_margin,
                                        max_distance=args.max_distance,
                                        min_evidence_views=args.min_evidence_views,
                                        max_side=args.max_side,
                                        max_mesh_faces=args.max_mesh_faces)
            operation = (plan_material_separation if args.command == "plan-material-separation"
                         else propose_material_regions)
            try:
                report = operation(args.dense_revision, args.seed_recipe, args.output,
                                   options=options)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "detect-photo-texture-defects":
            from .reconstruction.texture_defects import DefectDetectionOptions, detect_texture_defects
            from .reconstruction.references import ReferenceError
            try:
                report = detect_texture_defects(
                    args.dense_revision, args.surface_recipe, args.output,
                    options=DefectDetectionOptions(color_difference=args.color_difference,
                                                   local_excess=args.local_excess,
                                                   spatial_radius=args.spatial_radius,
                                                   max_side=args.max_side,
                                                   max_mesh_faces=args.max_mesh_faces))
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "estimate-uv-illumination":
            from .reconstruction.illumination_guides import (
                IlluminationGuideOptions, estimate_uv_illumination)
            from .reconstruction.references import ReferenceError
            try:
                report = estimate_uv_illumination(
                    args.dense_revision, args.surface_recipe, args.output,
                    options=IlluminationGuideOptions(
                        smoothing_radius=args.smoothing_radius,
                        max_illumination_ev=args.max_illumination_ev,
                        max_side=args.max_side, max_mesh_faces=args.max_mesh_faces))
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "inspect-polarized-uv":
            from .reconstruction.polarized_qa import PolarizedQAOptions, inspect_polarized_uv
            from .reconstruction.references import ReferenceError
            try:
                report = inspect_polarized_uv(
                    args.dense_revision, args.surface_recipe, args.captures, args.output,
                    options=PolarizedQAOptions(max_side=args.max_side,
                                               min_views=args.min_views,
                                               max_mesh_faces=args.max_mesh_faces))
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "project-registered-uv-views":
            from .reconstruction.photo_uv_projection import project_registered_photos_to_uv
            from .reconstruction.references import ReferenceError
            try:
                report = project_registered_photos_to_uv(
                    args.dense_revision, args.output, views=args.views,
                    atlas_side=args.atlas_side, camera_side=args.camera_side,
                    max_triangles=args.max_triangles)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command in ("plan-scan-displacement", "bake-scan-displacement"):
            from .reconstruction.references import ReferenceError
            from .reconstruction.scan_displacement import (bake_scanned_displacement,
                                                           plan_scan_displacement)
            options = dict(meters_per_unit=args.meters_per_unit,
                           max_offset_m=args.max_offset_m, max_side=args.max_side,
                           max_mesh_faces=args.max_mesh_faces,
                           max_uv_triangles=args.max_uv_triangles)
            try:
                plan = plan_scan_displacement(args.dense_revision, args.output, **options)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            if args.command == "plan-scan-displacement":
                print(json.dumps(plan, indent=2))
                return 0
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError:
                executable = shutil.which(args.blender)
                if not executable:
                    raise RuntimeError(f"Blender executable not found: {args.blender}")
                entry = Path(__file__).resolve().parents[2] / "scripts/blender_entry.py"
                command = [executable, "--background", "--factory-startup", "--python-exit-code", "1",
                           "--python", str(entry), "--", "bake-scan-displacement",
                           "--dense-revision", str(args.dense_revision.resolve()),
                           "--output", str(args.output.resolve()),
                           "--meters-per-unit", str(args.meters_per_unit),
                           "--max-offset-m", str(args.max_offset_m),
                           "--max-side", str(args.max_side),
                           "--max-mesh-faces", str(args.max_mesh_faces),
                           "--max-uv-triangles", str(args.max_uv_triangles)]
                return subprocess.run(command, check=False).returncode
            try:
                report = bake_scanned_displacement(bpy, args.dense_revision,
                                                   args.output, **options)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "package-finished-udim":
            from .reconstruction.references import ReferenceError
            from .reconstruction.udim import package_finished_udim
            try:
                report = package_finished_udim(args.texture_finish, args.output)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "render-udim-material-review":
            from .reconstruction.references import ReferenceError
            from .reconstruction.udim_relighting import (build_udim_relighting_review,
                                                          validate_udim_relighting_request)
            try:
                validate_udim_relighting_request(args.udim_package, args.output,
                                                 args.resolution, args.samples,
                                                 args.matched_views,
                                                 args.matched_resolution_percent)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError:
                executable = shutil.which(args.blender)
                if not executable:
                    raise RuntimeError(f"Blender executable not found: {args.blender}")
                entry = Path(__file__).resolve().parents[2] / "scripts/blender_entry.py"
                command = [executable, "--background", "--factory-startup",
                           "--python-exit-code", "1", "--python", str(entry), "--",
                           "render-udim-material-review", "--udim-package",
                           str(args.udim_package.resolve()), "--output", str(args.output.resolve()),
                           "--resolution", str(args.resolution), "--samples", str(args.samples),
                           "--matched-views", str(args.matched_views),
                           "--matched-resolution-percent", str(args.matched_resolution_percent)]
                if args.no_renders:
                    command.append("--no-renders")
                return subprocess.run(command, check=False).returncode
            try:
                report = build_udim_relighting_review(
                    bpy, args.udim_package, args.output, resolution=args.resolution,
                    samples=args.samples, render_previews=not args.no_renders,
                    matched_views=args.matched_views,
                    matched_resolution_percent=args.matched_resolution_percent)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command in ("plan-dense-blender-review", "render-dense-blender-review"):
            from .reconstruction.blender_dense import (build_dense_blender_review,
                                                       plan_dense_blender_review)
            from .reconstruction.references import ReferenceError
            try:
                planned = plan_dense_blender_review(
                    args.dense_revision, args.output, max_mesh_faces=args.max_mesh_faces,
                    matched_views=args.matched_views,
                    matched_resolution_percent=args.matched_resolution_percent,
                    textured=args.textured, texture_finish=args.texture_finish)
                if args.command == "plan-dense-blender-review":
                    print(json.dumps(planned, indent=2))
                    return 0
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError:
                executable = shutil.which(args.blender)
                if not executable:
                    raise RuntimeError(f"Blender executable not found: {args.blender}")
                entry = Path(__file__).resolve().parents[2] / "scripts/blender_entry.py"
                if not entry.is_file():
                    raise RuntimeError("Dense Blender review requires scripts/blender_entry.py")
                command = [executable, "--background", "--factory-startup", "--python-exit-code", "1",
                           "--python", str(entry), "--", "render-dense-blender-review",
                           "--dense-revision", str(args.dense_revision.resolve()),
                           "--output", str(args.output.resolve()),
                           "--max-mesh-faces", str(args.max_mesh_faces),
                           "--matched-views", str(args.matched_views),
                           "--matched-resolution-percent", str(args.matched_resolution_percent)]
                if args.textured:
                    command.append("--textured")
                if args.texture_finish:
                    command.extend(["--texture-finish", str(args.texture_finish.resolve())])
                if args.no_renders:
                    command.append("--no-renders")
                return subprocess.run(command, check=False).returncode
            report = build_dense_blender_review(
                args.dense_revision, args.output, bpy_module=bpy,
                render_previews=not args.no_renders, max_mesh_faces=args.max_mesh_faces,
                matched_views=args.matched_views,
                matched_resolution_percent=args.matched_resolution_percent,
                textured=args.textured, texture_finish=args.texture_finish)
            print(json.dumps(report, indent=2))
            return 0 if report["status"].startswith("private-Blender-mesh-candidate") else 3
        if args.command in ("plan-dense-photos", "build-dense-photos"):
            from .reconstruction.dense import (DenseSettings, build_dense_reconstruction,
                                               plan_dense_reconstruction)
            from .reconstruction.references import ReferenceError
            try:
                settings = DenseSettings(
                    backend=args.backend, min_registered_images=args.min_registered_images,
                    min_registered_fraction=args.min_registered_fraction,
                    timeout_seconds=args.timeout_seconds, min_mesh_faces=args.min_mesh_faces,
                    resolution_level=args.resolution_level,
                    max_resolution=args.max_resolution, number_views=args.number_views,
                    refine_decimate=args.refine_decimate,
                    texture_decimate=args.texture_decimate,
                    texture_global_seam_leveling=not args.no_global_seam_leveling,
                    texture_local_seam_leveling=not args.no_local_seam_leveling,
                    refine=not args.no_refine, texture=not args.no_texture)
                report = (plan_dense_reconstruction(args.reconstruction, args.output, settings,
                                                    rectified_masks=args.rectified_masks)
                          if args.command == "plan-dense-photos" else
                          build_dense_reconstruction(args.reconstruction, args.output, settings,
                                                     rectified_masks=args.rectified_masks,
                                                     resume=args.resume))
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return (0 if args.command == "plan-dense-photos" or
                    report["status"].startswith("dense-mesh-candidate") else 3)
        if args.command == "compare-camera-registrations":
            from .reconstruction.camera_compare import compare_camera_registrations
            from .reconstruction.references import ReferenceError
            try:
                report = compare_camera_registrations(args.baseline, args.candidate)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command in ("plan-sfm-cohort-recovery", "recover-sfm-cohorts", "diagnose-sfm-coverage"):
            from .reconstruction.sfm_cohort_recovery import (
                CohortRecoverySettings, plan_cohort_recovery, recover_cohort,
                annotate_failed_recovery)
            settings = CohortRecoverySettings(
                min_pair_inliers=args.min_inliers, max_anchors=args.max_anchors,
                max_trials=args.max_trials, min_new_cameras=args.min_new_cameras,
                min_seed_ray_angle_degrees=args.min_seed_angle,
                max_seconds_per_trial=getattr(args, "trial_timeout", 1200))
            try:
                selected = None
                if args.targets is not None:
                    if args.targets.is_symlink() or not args.targets.is_file():
                        raise ValueError("--targets must be an existing JSON file, not a symlink")
                    selected = json.loads(args.targets.read_text(encoding="utf-8"))
                    if not isinstance(selected, list) or not all(isinstance(s, str) for s in selected):
                        raise ValueError("--targets must contain a JSON array of staged filenames")
                common = dict(model=args.model, targets=selected,
                              preferred_anchors=args.anchor, settings=settings)
                report = (recover_cohort(args.revision, args.output, **common,
                                         colmap=args.colmap)
                          if args.command == "recover-sfm-cohorts" else
                          plan_cohort_recovery(args.revision, **common))
                if args.command == "diagnose-sfm-coverage":
                    report = annotate_failed_recovery(report, args.prior_recovery)
                    report = {"status": report["status"],
                              "first_photos_to_find": report["first_photos_to_find"],
                              "source_revision": report["source_revision"],
                              "seed_pairs": report["seed_pairs"],
                              "rejected_low_parallax_seed_pairs": report["rejected_low_parallax_seed_pairs"],
                              "coverage_gaps": report["coverage_gaps"],
                              "previous_recovery": report.get("previous_recovery")}
                    if args.guide_dir is not None:
                        from .reconstruction.sfm_gap_guide import write_gap_guide
                        report["guide_files"] = write_gap_guide(report, args.revision, args.guide_dir)
            except (OSError, ValueError, sqlite3.Error, ImportError) as error:
                raise RecipeError(str(error)) from error
            if args.command == "diagnose-sfm-coverage" and args.format == "markdown":
                from .reconstruction.sfm_gap_guide import markdown_gap_guide
                print(markdown_gap_guide(report))
            else:
                print(json.dumps(report, indent=2))
            return (0 if args.command != "recover-sfm-cohorts" or
                    report["status"].startswith("candidate / ") else 3)
        if args.command == "prepare-sfm-bridge-photos":
            from .reconstruction.sfm_bridge_photos import prepare_bridge_photos
            try:
                report = prepare_bridge_photos(
                    args.revision, args.photos, args.output, model=args.model,
                    source_manifest=args.manifest)
            except (OSError, ValueError, sqlite3.Error) as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "inspect-sfm-cohort-alignment":
            from .reconstruction.sfm_cohort_recovery import inspect_cohort_alignment
            try:
                report = inspect_cohort_alignment(
                    args.revision, args.candidate, model=args.model,
                    landmarks_path=args.landmarks)
            except (OSError, ValueError, sqlite3.Error, ImportError) as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0 if report["status"].startswith("alignment-evidence-passes") else 3
        if args.command == "build-statue-glb":
            from .reconstruction.statue_pipeline import StatuePipelineSettings, build_statue_glb
            from .reconstruction.statue_asset_registry import (
                remember_statue_inputs, resolve_statue_inputs,
            )
            from .reconstruction.references import ReferenceError
            try:
                photos, output = resolve_statue_inputs(
                    args.asset_id, args.photos, args.output, resume=args.resume)
                remember_statue_inputs(args.asset_id, photos, output)
                report = build_statue_glb(
                    photos, args.asset_id, output,
                    StatuePipelineSettings(
                        front_photo=args.front_photo, blender=args.blender,
                        openmvs_dir=args.openmvs_dir,
                        sam2_checkpoint=args.sam2_checkpoint,
                        auto_download_model=not args.no_auto_model_download,
                        sam2_size=args.sam2_size, sam2_device=args.sam2_device,
                        mask_guides=args.mask_guides, masks_dir=args.masks_dir,
                        mask_cache_dir=args.mask_cache_dir,
                        ai_bridges=args.ai_bridges or args.ai_bridge_workflow is not None,
                        ai_bridge_workflow=args.ai_bridge_workflow,
                        min_registered_fraction=args.min_registered_fraction,
                        max_mixed_mask_fraction=args.max_mixed_mask_fraction,
                        max_colmap_rss_mb=args.max_colmap_rss_mb,
                        timeout_seconds=args.timeout_seconds), resume=args.resume)
            except (ReferenceError, ValueError, ImportError) as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0 if report["status"].startswith("private-GLB-candidate") else 3
        if args.command == "generate-completion-image":
            from .reconstruction.completion_image_generation import generate_comfyui_image
            result = generate_comfyui_image(
                args.image, args.workflow, args.output,
                image_node=args.image_node, output_node=args.output_node,
                server=args.server, require_mps=args.require_mps,
                timeout_seconds=args.timeout_seconds)
            print(json.dumps(result, indent=2))
            return 0
        if args.command in ("plan-photo-model", "build-photo-model"):
            from .reconstruction.photo_model import (PhotoModelSettings, build_photo_model,
                                                     plan_photo_model)
            from .reconstruction.references import ReferenceError, load_reference_set
            from .reconstruction.splat_input import load_photo_folder
            try:
                if args.photos and not args.asset_id:
                    raise ReferenceError("--asset-id is required with --photos")
                if args.asset_id and not args.photos:
                    raise ReferenceError("--asset-id is only used with --photos")
                references = (load_photo_folder(args.photos, args.asset_id)
                              if args.photos else load_reference_set(args.manifest))
                settings = PhotoModelSettings(
                    engine=args.engine, dense_backend=args.backend, colmap=args.colmap,
                    mapanything_model=args.mapanything_model,
                    mapanything_device=args.mapanything_device,
                    mapanything_depth_tolerance=args.mapanything_depth_tolerance,
                    matching_preset=args.matching_preset,
                    matching_device=args.matching_device,
                    feature_threads=args.feature_threads, matching_threads=args.matching_threads,
                    sift_first_octave=args.sift_first_octave,
                    max_colmap_rss_mb=args.max_colmap_rss_mb,
                    matching_block_size=args.matching_block_size,
                    matching_max_features=args.matching_max_features,
                    matching_pairs_per_process=args.matching_pairs_per_process,
                    blender=args.blender, min_registered_images=args.min_registered_images,
                    min_registered_fraction=args.min_registered_fraction,
                    sparse_timeout_seconds=args.sparse_timeout_seconds,
                    dense_timeout_seconds=args.dense_timeout_seconds,
                    max_image_size=args.max_image_size,
                    texture=not args.no_texture, render_previews=not args.no_renders)
                report = (plan_photo_model(references, args.output, settings)
                          if args.command == "plan-photo-model" else
                          build_photo_model(references, args.output, settings))
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return (0 if args.command == "plan-photo-model" or
                    report["status"].startswith("private-GLB-candidate") else 3)
        if args.command in ("plan-splats", "build-splats"):
            from .reconstruction.references import ReferenceError, load_reference_set
            from .reconstruction.splat_input import load_photo_folder, plan_splat_input
            from .reconstruction.splats import (SplatSettings, build_multiview_splat_revision,
                                                 build_splat_revision, plan_splat_revision,
                                                 validate_camera_settings)
            try:
                settings = SplatSettings(iterations=args.iterations,
                                         timeout_seconds=args.timeout_seconds,
                                         output_format=args.format, opensplat=args.opensplat,
                                         downscale_factor=args.downscale)
                settings.validate()
                if args.photos and not args.asset_id:
                    raise ReferenceError("--asset-id is required with --photos")
                if args.asset_id and not args.photos:
                    raise ReferenceError("--asset-id is only used with --photos")
                if args.rectified_masks and not args.reconstruction:
                    raise ReferenceError("--rectified-masks requires --reconstruction")
                if args.reconstruction:
                    report = (plan_splat_revision(args.reconstruction, args.output, settings,
                                                  rectified_masks=args.rectified_masks)
                              if args.command == "plan-splats" else
                              build_splat_revision(args.reconstruction, args.output, settings,
                                                   rectified_masks=args.rectified_masks))
                else:
                    validate_camera_settings(args.engine, args.sparse_timeout_seconds,
                                             args.max_image_size)
                    references = (load_photo_folder(args.photos, args.asset_id)
                                  if args.photos else load_reference_set(args.references))
                    if args.command == "plan-splats":
                        import importlib.util

                        report = plan_splat_input(references)
                        if args.output.is_symlink() or args.output.exists():
                            raise ReferenceError(f"Output revision already exists: {args.output}")
                        report.update(output=str(args.output.resolve()),
                                      engine=args.engine, registration_profile=args.registration_profile,
                                      output_format=settings.output_format,
                                      opensplat_available=bool(shutil.which(args.opensplat)),
                                      camera_engine_available=(
                                          importlib.util.find_spec("pycolmap") is not None
                                          if args.engine == "pycolmap" else
                                          bool(shutil.which(args.colmap))),
                                      caution="Camera registration requires overlapping photos and an installed SfM engine.")
                    else:
                        report = build_multiview_splat_revision(
                            references, args.output, settings,
                            engine=args.engine, colmap=args.colmap,
                            sparse_timeout_seconds=args.sparse_timeout_seconds,
                            max_image_size=args.max_image_size,
                            registration_profile=args.registration_profile)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0 if args.command == "plan-splats" or report["status"].startswith("candidate") else 3
        if args.command in ("plan-splat-video", "render-splat-video"):
            from .reconstruction.references import ReferenceError
            from .reconstruction.splat_video import (SplatVideoSettings, plan_splat_video,
                                                     render_splat_video)
            try:
                settings = SplatVideoSettings(
                    fps=args.fps, seconds=args.seconds, width=args.width, height=args.height,
                    fov=args.fov, orbit_degrees=args.orbit_degrees,
                    start_degrees=args.start_degrees,
                    elevation_degrees=args.elevation_degrees,
                    target=tuple(args.target) if args.target is not None else None,
                    distance=args.distance, hero_photo=args.hero_photo,
                    shutter=args.shutter, viewer=args.viewer,
                    splat_transform=args.splat_transform, ffmpeg=args.ffmpeg,
                    ffprobe=args.ffprobe, timeout_seconds=args.timeout_seconds)
                report = (plan_splat_video(args.splat, args.output, settings,
                                           camera_track=args.camera_track)
                          if args.command == "plan-splat-video" else
                          render_splat_video(args.splat, args.output, settings,
                                             camera_track=args.camera_track))
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return (0 if args.command == "plan-splat-video" or
                    report["status"].startswith("candidate") else 3)
        if args.command in ("plan-splat-view-qa", "compare-splat-views"):
            from .reconstruction.references import ReferenceError
            from .reconstruction.splat_visual_qa import (
                SplatViewQASettings, plan_splat_view_qa, render_splat_view_qa)
            try:
                settings = SplatViewQASettings(
                    max_views=args.max_views, max_dimension=args.max_dimension,
                    splat_transform=args.splat_transform,
                    timeout_seconds=args.timeout_seconds)
                report = (plan_splat_view_qa(args.splat, args.output, settings)
                          if args.command == "plan-splat-view-qa" else
                          render_splat_view_qa(args.splat, args.output, settings))
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return (0 if args.command == "plan-splat-view-qa" or
                    report["status"].startswith("candidate") else 3)
        if args.command == "prepare-makehuman-head":
            from .head_sources.makehuman import (fetch_makehuman_obj,
                                                 load_body_mesh, verify_makehuman_obj)
            output = args.output.resolve()
            if output.exists() or args.output.is_symlink():
                raise MakeHumanSourceError(f"Output revision already exists: {output}")
            source = (fetch_makehuman_obj(args.cache) if args.download
                      else verify_makehuman_obj(args.source_obj))
            vertices, faces = load_body_mesh(source)
            if len(vertices) < 10000 or len(faces) < 10000:
                raise MakeHumanSourceError("Unexpectedly sparse official body topology")
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError:
                executable = shutil.which(args.blender)
                if executable is None:
                    raise RuntimeError(f"Blender executable not found: {args.blender}")
                entry = Path(__file__).resolve().parents[2] / "scripts/blender_entry.py"
                if not entry.is_file():
                    raise RuntimeError("Blender build requires scripts/blender_entry.py from a source checkout")
                completed = subprocess.run([
                    executable, "--background", "--factory-startup", "--python-exit-code", "1",
                    "--python", str(entry), "--", "prepare-makehuman-head",
                    "--source-obj", str(source.path), "--output", str(output),
                ], check=False)
                return completed.returncode
            from .blender_head_sources import build_makehuman_head
            print(json.dumps(build_makehuman_head(source, output, bpy_module=bpy), indent=2))
            return 0
        if args.command in ("plan-head-base", "build-head-base"):
            from .reconstruction.head_base import (
                fit_head_landmarks, head_deformation_field, load_head_fit_plan,
            )
            plan = load_head_fit_plan(args.manifest)
            fit = fit_head_landmarks(plan)
            head_deformation_field(plan, fit)  # Refuse unstable local corrections before Blender.
            if args.command == "plan-head-base":
                print(json.dumps({**fit.to_report(), "fit_mode": plan.fit_mode,
                                  "mesh_name": plan.mesh_name,
                                  "source_front_axis": plan.source_front_axis,
                                  "head_source_asset_id": plan.provenance.asset_id}, indent=2))
                return 0
            output = args.output.resolve()
            if args.output.is_symlink() or output.exists():
                raise HeadFitError(f"Output revision already exists: {output}")
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError:
                executable = shutil.which(args.blender)
                if executable is None:
                    raise RuntimeError(f"Blender executable not found: {args.blender}")
                entry = Path(__file__).resolve().parents[2] / "scripts/blender_entry.py"
                if not entry.is_file():
                    raise RuntimeError("Blender build requires scripts/blender_entry.py from a source checkout")
                completed = subprocess.run([
                    executable, "--background", "--factory-startup", "--python-exit-code", "1",
                    "--python", str(entry), "--", "build-head-base",
                    "--manifest", str(args.manifest.resolve()), "--output", str(output),
                ], check=False)
                return completed.returncode
            from .blender_import import build_imported_scene
            result = build_imported_scene(plan.source_glb, plan.provenance, output,
                                          head_fit=plan, bpy_module=bpy)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "prepare-camera-plan":
            from .reconstruction.cameras import create_camera_plan
            from .reconstruction.references import ReferenceError
            try:
                report = create_camera_plan(args.reconstruction, args.alignment, args.output,
                                            max_anchor_rms_m=args.max_anchor_rms_m,
                                            undistorted_masks=args.undistorted_masks)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "silhouette-qa":
            from .reconstruction.references import ReferenceError
            from .reconstruction.silhouettes import compare_silhouettes
            try:
                report = compare_silhouettes(args.plan, args.renders, args.output,
                                             min_iou=args.min_iou)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0 if report["status"].startswith("mask-threshold-candidate") else 3
        if args.command == "landmark-qa":
            from .reconstruction.landmarks import check_landmark_reprojection
            from .reconstruction.references import ReferenceError
            try:
                report = check_landmark_reprojection(args.plan, args.annotations, args.output,
                                                     max_error_px=args.max_error_px)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0 if report["status"].startswith("reprojection-threshold-candidate") else 3
        if args.command == "render-reference-scene":
            from .reconstruction.blender_reference import load_camera_plan
            from .reconstruction.references import ReferenceError
            try:
                load_camera_plan(args.plan)
                if not 1 <= args.max_cameras <= 32:
                    raise ReferenceError("max_cameras must be 1–32")
                output = args.output.resolve()
                if args.output.is_symlink() or output.exists():
                    raise ReferenceError(f"Reference scene output already exists: {output}")
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError:
                executable = shutil.which(args.blender)
                if executable is None:
                    raise RuntimeError(f"Blender executable not found: {args.blender}")
                entry = Path(__file__).resolve().parents[2] / "scripts/blender_entry.py"
                if not entry.is_file():
                    raise RuntimeError("Blender rendering requires scripts/blender_entry.py from a source checkout")
                command = [executable, "--background", "--factory-startup", "--python-exit-code", "1",
                           "--python", str(entry), "--", "render-reference-scene",
                           "--plan", str(args.plan.resolve()), "--model", str(args.model.resolve()),
                           "--output", str(output), "--max-cameras", str(args.max_cameras)]
                if args.render_clay:
                    command.append("--render-clay")
                return subprocess.run(command, check=False).returncode
            from .reconstruction.blender_reference import build_reference_scene
            try:
                report = build_reference_scene(args.plan, args.model, output,
                                               render=args.render_clay, max_cameras=args.max_cameras,
                                               bpy_module=bpy)
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 0
        if args.command in ("reference-audit", "reconstruct-photos"):
            from .reconstruction.references import (ReferenceError, audit_reference_set,
                                                    load_reference_set)
            from .reconstruction.splat_input import load_photo_folder
            from .reconstruction.colmap import build_sparse_reconstruction
            try:
                if args.command == "reconstruct-photos":
                    if args.photos and not args.asset_id:
                        raise ReferenceError("--asset-id is required with --photos")
                    if args.asset_id and not args.photos:
                        raise ReferenceError("--asset-id is only used with --photos")
                references = (load_photo_folder(args.photos, args.asset_id)
                              if args.command == "reconstruct-photos" and args.photos else
                              load_reference_set(args.manifest))
                report = (audit_reference_set(references) if args.command == "reference-audit" else
                          build_sparse_reconstruction(references, args.output, colmap=args.colmap,
                                                      engine=args.engine,
                                                      timeout_seconds=args.timeout_seconds,
                                                      undistort=args.undistort,
                                                      max_image_size=args.max_image_size,
                                                      registration_profile=args.registration_profile,
                                                      matching_preset=args.matching_preset,
                                                      matching_device=args.matching_device,
                                                      matching_pair_list=args.matching_pair_list,
                                                      feature_threads=args.feature_threads,
                                                      matching_threads=args.matching_threads,
                                                      sift_first_octave=args.sift_first_octave,
                                                      max_colmap_rss_mb=args.max_colmap_rss_mb,
                                                      matching_block_size=args.matching_block_size,
                                                      matching_max_features=args.matching_max_features,
                                                      matching_pairs_per_process=args.matching_pairs_per_process,
                                                      resume=args.resume))
            except ReferenceError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return (3 if args.command == "reconstruct-photos" and
                    report["status"] == "review-required" else 0)
        if args.command == "bridge-parity":
            from .bridge_parity import BridgeParityError, compare_captures
            try:
                report = compare_captures(args.plan, args.blender_dir, args.web_dir, args.output,
                                          min_iou=args.min_iou, max_center_delta=args.max_center_delta)
            except BridgeParityError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 3 if report["status"] == "review-required" else 0
        if args.command == "visual-qa":
            from .visual_qa import VisualQAError, inspect_visual_revision
            try:
                report = inspect_visual_revision(
                    args.revision, args.output, baseline=args.baseline,
                    max_rms=args.max_rms, max_changed=args.max_changed,
                    pixel_threshold=args.pixel_threshold, use_idiff=args.idiff,
                )
            except VisualQAError as error:
                raise RecipeError(str(error)) from error
            print(json.dumps(report, indent=2))
            return 3 if report["status"] == "review-required" else 0
        if args.command in ("plan-asset", "compile-asset", "build-asset", "asset-schema"):
            if args.command == "build-asset" and args.plan is not None:
                # This branch is intentionally safe inside Blender 4.5's
                # stock Python, where Pydantic and pydantic-core aren't present.
                from .compiled_plan import CompiledPlanError, load_compiled_plan
                try:
                    spec, plan = load_compiled_plan(args.plan)
                except CompiledPlanError as error:
                    raise RecipeError(str(error)) from error
                texture_packs = (load_texture_packs(args.texture_packs)
                                 if args.texture_packs else ())
                verify_texture_targets(list(spec.materials), texture_packs)
                for assignment in texture_packs:
                    if spec.materials[assignment.target].preset == "optical-glass":
                        raise RecipeError("Texture packs cannot replace optical-glass transmission materials")
                output = args.output.resolve()
                if output.exists() or output.is_symlink():
                    raise RecipeError(f"Output already exists: {output}. Choose a new revision directory.")
                try:
                    import bpy  # type: ignore[import-not-found]
                except ImportError as error:
                    raise RuntimeError("--plan rendering must run inside Blender") from error
                from .blender_composer import build_assembly_scene
                print(json.dumps(build_assembly_scene(spec, plan, output,
                    texture_packs=texture_packs, bpy_module=bpy), indent=2))
                return 0
            try:
                from .asset_spec import AssetSpecError, asset_json_schema, load_asset_spec
                from .composer import COMPONENT_REGISTRY, ComposerError, plan_assembly
            except ModuleNotFoundError as error:
                if error.name == "pydantic":
                    raise RuntimeError("AssetSpec v2 requires: pip install 'kumiki-blender[composer]'") from error
                raise
            if args.command == "asset-schema":
                print(json.dumps(asset_json_schema(), indent=2))
                return 0
            try:
                spec = load_asset_spec(args.spec)
                plan = plan_assembly(spec)
            except (AssetSpecError, ComposerError) as error:
                raise RecipeError(str(error)) from error
            if args.command == "plan-asset":
                print(json.dumps({
                    "asset_id": plan.asset_id,
                    "generator": "component-assembly",
                    "component_kinds": sorted(COMPONENT_REGISTRY),
                    "named_parts": dict(plan.named_parts),
                    "instances": len(plan.instances),
                    "bounds_m": [plan.bounds_min, plan.bounds_max],
                    "recipe_sha256": plan.recipe_sha256,
                }, indent=2))
                return 0
            from .compiled_plan import CompiledPlanError, write_compiled_plan
            if args.command == "compile-asset":
                try:
                    write_compiled_plan(spec, plan, args.output)
                except CompiledPlanError as error:
                    raise RecipeError(str(error)) from error
                print(json.dumps({"asset_id": spec.asset_id, "compiled": str(args.output),
                                  "instances": len(plan.instances)}, indent=2))
                return 0
            output = args.output.resolve()
            if output.exists() or output.is_symlink():
                raise RecipeError(f"Output already exists: {output}. Choose a new revision directory.")
            texture_packs = (load_texture_packs(args.texture_packs)
                             if args.texture_packs else ())
            verify_texture_targets(list(spec.materials), texture_packs)
            for assignment in texture_packs:
                if spec.materials[assignment.target].preset == "optical-glass":
                    raise RecipeError("Texture packs cannot replace optical-glass transmission materials")
            entry = Path(__file__).resolve().parents[2] / "scripts/blender_entry.py"
            if not entry.is_file():
                raise RuntimeError("Automatic Blender builds require a source checkout containing scripts/blender_entry.py")
            executable = shutil.which(args.blender)
            if executable is None:
                raise RuntimeError(f"Blender executable not found: {args.blender}. Pass --blender <path>")
            with tempfile.TemporaryDirectory(prefix="kumiki-asset-plan-") as directory:
                compiled = Path(directory) / "compiled-plan.json"
                write_compiled_plan(spec, plan, compiled)
                command = [
                    executable, "--background", "--factory-startup", "--python-exit-code", "1",
                    "--python", str(entry), "--", "build-asset", "--plan", str(compiled),
                    "--output", str(output),
                ]
                if args.texture_packs:
                    command.extend(("--texture-packs", str(args.texture_packs.resolve())))
                completed = subprocess.run(command, check=False)
            return completed.returncode
        if args.command == "catalog":
            entries = load_catalog()
            action = args.catalog_action
            if action == "validate":
                repository_root = Path(__file__).resolve().parents[2]
                checked = (repository_root / "pyproject.toml").is_file()
                if checked:
                    validate_sources(entries, repository_root)
                print(json.dumps({"valid": True, "asset_count": len(entries),
                                  "sources_checked": checked}, indent=2))
                return 0
            if action == "categories":
                counts: dict[str, dict[str, int]] = {}
                for entry in entries:
                    subcategories = counts.setdefault(entry.category, {})
                    subcategories[entry.subcategory] = subcategories.get(entry.subcategory, 0) + 1
                print(json.dumps({"categories": [
                    {"name": category, "count": sum(subcategories.values()),
                     "subcategories": subcategories}
                    for category, subcategories in sorted(counts.items())
                ]}, indent=2))
                return 0
            if action in ("list", "search"):
                query = " ".join(args.query) if action == "search" else ""
                result = search_assets(entries, query, category=args.category,
                                       tag=args.tag, rigged=args.rigged)
                print(json.dumps({"count": len(result), "assets": [
                    {"id": entry.asset_id, "title": entry.title,
                     "category": entry.category, "subcategory": entry.subcategory,
                     "tags": entry.tags, "rigged": entry.rigged,
                     "animated": entry.animated}
                    for entry in result
                ]}, indent=2))
                return 0
            entry = find_asset(entries, args.asset_id)
            if action == "show":
                print(json.dumps(entry.to_dict(), indent=2))
                return 0
            output = revision_path(entry, args.revision, args.root).resolve()
            if action == "path":
                print(json.dumps({"id": entry.asset_id, "output": str(output),
                                  "exists": output.exists() or output.is_symlink()}, indent=2))
                return 0
            if entry.build_command is None:
                raise CatalogError(f"{entry.asset_id} is a reusable component; use its plan API")
            if output.exists() or output.is_symlink():
                raise CatalogError(f"Output already exists: {output}. Choose another revision")
            command = list(entry.build_command)
            if "--recipe" in command:
                index = command.index("--recipe") + 1
                recipe = Path(command[index])
                if not recipe.is_file():
                    recipe = Path(__file__).resolve().parents[2] / recipe
                if not recipe.is_file():
                    raise CatalogError(f"Recipe for {entry.asset_id} not found; run from a source checkout")
                command[index] = str(recipe.resolve())
            return main([*command, "--output", str(output)])
        if args.command in ("plan-biplane", "build-biplane"):
            model = make_biplane()
            if args.command == "plan-biplane":
                print(json.dumps({
                    "asset_id": model.asset_id, "generator": "red-vintage-biplane",
                    "named_parts": model.part_counts, "instances": len(model.parts),
                    "bounds_m": [model.bounds_min, model.bounds_max],
                    "rig_controls": dict(model.pivots),
                    "geometry_origin": "original procedural interpretation of user reference",
                }, indent=2))
                return 0
            output = args.output.resolve()
            if output.exists() or output.is_symlink():
                raise RecipeError(f"Output already exists: {output}. Choose a new revision directory.")
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("Building the airplane requires Blender: blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- build-biplane --output <new-directory>") from error
            from .blender_biplane import build_biplane_scene
            print(json.dumps(build_biplane_scene(model, output, bpy_module=bpy), indent=2))
            return 0
        if args.command == "audit":
            print(json.dumps(audit_glb(args.model, require_pbr_textures=args.require_pbr_textures), indent=2))
            return 0
        if args.command == "audit-revision":
            print(json.dumps(audit_revision(args.root), indent=2))
            return 0
        if args.command == "inspect-asset":
            print(json.dumps(inspect_source_glb(args.input), indent=2))
            return 0
        if args.command == "inspect-texture-packs":
            print(json.dumps({"assignments": [entry.to_manifest() for entry in
                                               load_texture_packs(args.input)]}, indent=2))
            return 0
        if args.command == "plan-classical-column":
            pieces = classical_column(prefix="column-study", height=args.height,
                                      base_radius=args.radius, order=args.order)
            coords = [vertex for piece in pieces for vertex in world_vertices(piece)]
            print(json.dumps({
                "order": args.order, "height_m": args.height, "base_radius_m": args.radius,
                "parts": len(pieces), "named_parts": [part.name for part in pieces],
                "vertices": sum(len(piece.mesh.vertices) for piece in pieces),
                "bounds_m": [[min(p[axis] for p in coords) for axis in range(3)],
                             [max(p[axis] for p in coords) for axis in range(3)]],
            }, indent=2))
            return 0
        if args.command in ("plan-fuji", "build-fuji"):
            if args.style == "iconic":
                if args.detail_patch:
                    raise ValueError("Survey detail patches are only valid for realistic Fuji")
                if args.season != "winter":
                    raise ValueError("--season is only available for realistic Fuji")
                from .geography.fuji.iconic_scene import iconic_spec
                if args.command == "plan-fuji":
                    print(json.dumps(iconic_spec(args.detail), indent=2))
                    return 0
                try:
                    import bpy  # type: ignore[import-not-found]
                except ImportError as error:
                    raise RuntimeError("Iconic Fuji builds require Blender 4.2+") from error
                from .geography.fuji.iconic_scene import build_iconic_fuji_scene
                print(json.dumps(build_iconic_fuji_scene(args.output, detail=args.detail,
                                                        bpy_module=bpy), indent=2))
                return 0
            if not args.dem.is_file():
                raise RuntimeError(f"Mount Fuji DEM not found: {args.dem}. First run: python scripts/prepare_fuji_dem.py {args.dem}")
            dem = load_dem(args.dem)
            divisions = {"preview": 256, "standard": 512, "high": 1024}[args.detail]
            from .geography.fuji.patches import load_patch, validate_patches
            patches=tuple(load_patch(path) for path in args.detail_patch)
            validate_patches(patches,dem,divisions)
            if patches and divisions!=1024:
                raise ValueError("GSI crater patches require --detail high")
            if args.command == "plan-fuji":
                print(json.dumps({"asset_id": "mount-fuji-survey-terrain",
                                  "source": dem.source, "detail": args.detail, "season": args.season,
                                  "snow_cover": "none" if args.season=="summer" else "seasonal",
                                  "terrain_divisions": divisions,
                                  "terrain_vertices": (divisions+1)**2 +
                                      sum(p.spec.rows*p.spec.cols-((p.spec.rows-1)//p.spec.factor+1)*
                                          ((p.spec.cols-1)//p.spec.factor+1) for p in patches),
                                  "detail_patches": [p.spec.name for p in patches],
                                  "statistics": dem.statistics(),
                                  "vertical_exaggeration": 1.0}, indent=2))
                return 0
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("Mount Fuji builds require Blender 4.2+. Run with --background --factory-startup --python scripts/blender_entry.py") from error
            from .geography.fuji.blender_scene import build_fuji_scene
            print(json.dumps(build_fuji_scene(args.output, args.dem, divisions=divisions,
                                             detail_patches=tuple(args.detail_patch),
                                             season=args.season,
                                             bpy_module=bpy), indent=2))
            return 0
        if args.command in ("plan-egypt-map", "build-egypt-map"):
            spec = (EgyptMapSpec(columns=700, rows=420) if args.detail == "high"
                    else EgyptMapSpec())
            if args.command == "plan-egypt-map":
                print(json.dumps({
                    "asset_id": "egypt-relief-atlas", "generator": "egypt-relief-map",
                    "detail": args.detail,
                    "terrain_grid": [spec.columns, spec.rows],
                    "terrain_vertices": (spec.columns+1)*(spec.rows+1),
                    "waterways": len(RIVER_CHANNELS),
                    "city_markers": len(CITY_MARKERS),
                    "geography_labels": len(LABELS),
                    "dimensions_m": [16.0, 11.4],
                    "geographic_fidelity": "illustrated, not survey-grade",
                }, indent=2))
                return 0
            output = args.output.resolve()
            if output.exists() or output.is_symlink():
                raise RecipeError(f"Output already exists: {output}. Choose a new revision directory.")
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("Building the Egypt map requires Blender. Run via: blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- build-egypt-map --output <new-directory>") from error
            from .blender_egypt_map import build_egypt_map_scene

            print(json.dumps(build_egypt_map_scene(output, spec=spec, bpy_module=bpy), indent=2))
            return 0
        if args.command == "import-asset":
            provenance = load_asset_provenance(args.provenance)
            report = inspect_source_glb(args.input, provenance)
            overrides = load_material_overrides(args.material_overrides) if args.material_overrides else ()
            texture_packs = load_texture_packs(args.texture_packs) if args.texture_packs else ()
            verify_material_overrides(report["material_names"], overrides)
            verify_texture_targets(report["material_names"], texture_packs)
            if {entry.material_name for entry in overrides} & {entry.target for entry in texture_packs}:
                raise AssetImportError("One material cannot have both a preset override and a texture pack")
            output = args.output.resolve()
            if output.exists() or output.is_symlink():
                raise AssetImportError(f"Output already exists: {output}. Choose a new revision directory.")
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("Importing a model requires Blender. Run via: blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- import-asset --input model.glb --provenance source.json --output <new-directory>") from error
            from .blender_import import build_imported_scene

            print(json.dumps(build_imported_scene(args.input, provenance, output,
                                                 material_overrides=overrides,
                                                 texture_packs=texture_packs, bpy_module=bpy), indent=2))
            return 0
        if args.command in ("plan-portico", "build-portico"):
            study = make_portico_study(bays=args.bays, order=args.order,
                                      arch_style=args.arch_style, balustrade=args.balustrade)
            if args.command == "plan-portico":
                print(json.dumps({
                    "asset_id": study.asset_id,
                    "generator": "modular-portico",
                    "order": study.order,
                    "arch_style": study.arch_style,
                    "balustrade": study.balustrade,
                    "named_parts": dict(study.part_counts),
                    "instances": len(study.instances),
                    "bounds_m": [study.bounds_min, study.bounds_max],
                }, indent=2))
                return 0
            output = args.output.resolve()
            if output.exists() or output.is_symlink():
                raise RecipeError(f"Output already exists: {output}. Choose a new revision directory.")
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("Building requires Blender. Run via: blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- build-portico --output <new-directory>") from error
            from .blender_portico import build_portico_scene

            print(json.dumps(build_portico_scene(study, output, bpy_module=bpy), indent=2))
            return 0
        if args.command in ("plan-facade", "build-facade"):
            study = make_facade_study(bays=args.bays)
            if args.command == "plan-facade":
                print(json.dumps({
                    "asset_id": study.asset_id,
                    "generator": "architectural-facade-detail-kit",
                    "named_parts": dict(study.part_counts),
                    "instances": len(study.instances),
                    "bounds_m": [study.bounds_min, study.bounds_max],
                }, indent=2))
                return 0
            output = args.output.resolve()
            if output.exists() or output.is_symlink():
                raise FacadeError(f"Output already exists: {output}. Choose a new revision directory.")
            texture_packs = load_texture_packs(args.texture_packs) if args.texture_packs else ()
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("Building a facade requires Blender. Run via: blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- build-facade --bays 3 --output <new-directory>") from error
            from .blender_facade import build_facade_scene

            print(json.dumps(build_facade_scene(study, output, texture_packs=texture_packs,
                                               bpy_module=bpy), indent=2))
            return 0
        if args.command in ("plan-monument", "build-monument"):
            study = make_monument_study(
                kind=args.kind, pyramid_base=args.pyramid_base,
                pyramid_height=args.pyramid_height, pyramid_courses=args.pyramid_courses,
                casing_fraction=args.casing_fraction, sphinx_length=args.sphinx_length,
                restored_nose=args.restored_nose, pyramid_preset=args.pyramid_preset)
            if args.command == "plan-monument":
                print(json.dumps({
                    "asset_id": study.asset_id,
                    "generator": "egyptian-monument-detail-kit",
                    "parameters": dict(study.parameters),
                    "named_parts": dict(study.part_counts),
                    "instances": len(study.instances),
                    "bounds_m": [study.bounds_min, study.bounds_max],
                }, indent=2))
                return 0
            output = args.output.resolve()
            if output.exists() or output.is_symlink():
                raise MonumentError(f"Output already exists: {output}. Choose a new revision directory.")
            if bool(args.sphinx_source) != bool(args.sphinx_provenance):
                raise MonumentError("--sphinx-source and --sphinx-provenance must be supplied together")
            if args.sphinx_source and args.kind == "pyramid":
                raise MonumentError("Sphinx scans are only available for sphinx and giza-study builds")
            scan = (prepare_sphinx_scan(args.sphinx_source, args.sphinx_provenance,
                                       front_axis=args.sphinx_front_axis)
                    if args.sphinx_source else None)
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("Building monuments requires Blender. Run via: blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- build-monument --kind giza-study --output <new-directory>") from error
            from .blender_monuments import build_monument_scene

            print(json.dumps(build_monument_scene(study, output, sphinx_scan=scan,
                                                  bpy_module=bpy), indent=2))
            return 0
        recipe = load_recipe(args.recipe)
        plan = make_lattice_tower(recipe)
        if args.command == "validate":
            print(json.dumps({
                "asset_id": recipe.asset_id,
                "generator": "lattice-tower",
                "beams": len(plan.beams),
                "decks": len(plan.decks),
                "gussets": len(plan.gussets),
                "fasteners": len(plan.fasteners),
                "bounds_m": [plan.bounds_min, plan.bounds_max],
                "recipe_sha256": recipe.recipe_sha256,
            }, indent=2))
            return 0

        output = args.output.resolve()
        if output.exists() or output.is_symlink():
            raise RecipeError(f"Output already exists: {output}. Choose a new revision directory.")
        try:
            import bpy  # type: ignore[import-not-found]  # Blender's bundled interpreter only.
        except ImportError as error:
            raise RuntimeError("Building requires Blender. Run via: blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- build --recipe <recipe.json> --output <new-directory>") from error

        from .blender_scene import build_scene

        manifest = build_scene(recipe, plan, output, bpy_module=bpy)
        print(json.dumps(manifest, indent=2))
        return 0
    except (CatalogError, RecipeError, PorticoError, FacadeError, MonumentError, GeometryError, AssetImportError, HeadFitError, ImageGenerationError, MakeHumanSourceError, RenderProfileError, RevisionAuditError, AuditError, ExportAuditError, MotionExportError, PhysicsPlanError, DeformationPlanError,
            SurfaceError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
