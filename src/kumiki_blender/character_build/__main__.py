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


def _spec(handoff: Path) -> dict:
    return json.loads((handoff / "spec.json").read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kumiki-character-build", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan-character", help="validate CharacterSpec v2 and freeze a source-hashed character plan")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new immutable character plan directory")

    p = sub.add_parser("plan-region", help="isolate one anatomical region and its dependencies")
    p.add_argument("--character", type=Path, required=True, help="directory with character-plan.json")
    p.add_argument("--region", required=True, help="stable semantic region id, e.g. eye.L")
    p.add_argument("--output", type=Path, required=True, help="new immutable region task directory")

    p = sub.add_parser("compare-region", help="compare a region mesh with verified registered source masks")
    p.add_argument("--character", type=Path, required=True, help="directory with character-plan.json")
    p.add_argument("--region", required=True)
    p.add_argument("--mesh", type=Path, required=True, help="aligned OBJ, PLY or GLB candidate")
    p.add_argument("--output", type=Path, required=True, help="new comparison directory")
    p.add_argument("--min-iou", type=float, help="optional per-view geometry threshold; no automatic art approval")
    p.add_argument("--baseline", type=Path, help="prior report.json to reject registered-view regressions")
    p.add_argument("--max-regression", type=float, default=0.01, help="maximum per-view drop in IoU")

    p = sub.add_parser("validate-rig", help="native Blender pose tests for a skinned .blend")
    p.add_argument("--blend", type=Path, required=True, help="existing rigged source .blend")
    p.add_argument("--plan", type=Path, required=True, help="bounded pose battery JSON")
    p.add_argument("--output", type=Path, required=True, help="new immutable report directory")
    p.add_argument("--blender", help="local Blender executable path")

    p = sub.add_parser("refine-region", help="bounded local sculpt edit; preserve all other .blend objects")
    p.add_argument("--blend", type=Path, required=True, help="editable source Blender file")
    p.add_argument("--plan", type=Path, required=True, help="region sculpt correction JSON")
    p.add_argument("--output", type=Path, required=True, help="new revision containing refined.blend and region.obj")
    p.add_argument("--blender")



    p = sub.add_parser("ingest", help="copy generator output into canonical view names")
    p.add_argument("source", type=Path)
    p.add_argument("mapping", type=Path, help='JSON {"file.png": "front", ...}')
    p.add_argument("destination", type=Path)

    p = sub.add_parser("check", help="check reference images alone, against each other and in 3D")
    p.add_argument("folder", type=Path)

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
    p.add_argument("--lattice", help="nx,ny,nz cells: deform only through a coarse lattice (best for generator meshes)")

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
    p.add_argument("handoff", type=Path)
    p.add_argument("mesh", type=Path, help="OBJ with UVs, e.g. retopo/low.obj")
    p.add_argument("--output", type=Path, required=True, help=".png")
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--diagonal-weight", type=float, default=1.0, help="score factor of 3/4 views")
    p.add_argument("--min-cosine", type=float, default=0.35, help="views at a grazing angle below this do not paint")
    p.add_argument("--assume-symmetric", action="store_true",
                   help="paint a side without a drawing from the mirrored opposite side")
    p.add_argument("--extra", type=Path, help="manifest.json from `repaint`: add the repainted views")
    p.add_argument("--power", type=float, default=1.0,
                   help="view score = cos^power; use 4 with --extra so each source keeps the surface it faces")
    p.add_argument("--views", help="comma-separated drawings to use (default: all)")
    p.add_argument("--screen-extra", type=float, metavar="DELTA_E",
                   help="drop a repaint that alone raises any drawn view's error by more than this (e.g. 0.5)")

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
    p.add_argument("--resume", action="store_true",
                   help="reuse finished repaints in --output whose render and settings match; repaint the rest")

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
    if args.command == "compare-region":
        from .region_compare import compare_region
        report = compare_region(args.character, args.region, args.mesh, args.output,
                                min_iou=args.min_iou, baseline=args.baseline,
                                max_regression=args.max_regression)
        print(json.dumps(report, indent=2))
        return 1 if report["failed_views"] else 0
    if args.command == "refine-region":
        from .region_sculpt_plan import load_region_sculpt_plan
        from .blender import run_worker
        if not args.blend.is_file() or args.blend.suffix.lower() != ".blend":
            raise ValueError("--blend must be an existing Blender .blend file")
        if args.output.exists():
            raise FileExistsError(f"Output revision already exists: {args.output}")
        plan = load_region_sculpt_plan(args.plan)
        report = run_worker("region-sculpt", {
            "plan": plan, "output": str(args.output.resolve()),
            "source_blend": str(args.blend.resolve())},
            blender=args.blender, blend_file=args.blend.resolve())
        print(json.dumps(report, indent=2))
        return 0
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
    if args.command == "ingest":
        used = ingest(args.source, args.mapping, args.destination)
        print(json.dumps(used, indent=2))
        return 0
    if args.command == "check":
        from .checks import check_folder

        report = check_folder(args.folder)
        print((args.folder / "check" / "report.md").read_text(encoding="utf-8"))
        return 0 if report["pass"] else 1
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
            result = fit_silhouettes(mesh, masks, spec["height_m"], scale=scale, iterations=args.iterations,
                                     normalise_first=not args.no_normalise, smooth_radius_px=args.smooth_radius_px,
                                     lattice_cells=tuple(int(v) for v in args.lattice.split(",")) if args.lattice else None)
            write_mesh(result.mesh, args.output)
            report = {"input": str(args.mesh), "normalise": result.transform, "best_iteration": result.best_iteration,
                      "folds": result.folds,
                      "start": result.history[0], "best": result.history[result.best_iteration]}
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
        mesh, corner_uv = read_obj_uv(args.mesh)
        extra = json.loads(args.extra.read_text(encoding="utf-8"))["extra"] if args.extra else None
        settings = dict(diagonal_weight=args.diagonal_weight, min_cosine=args.min_cosine,
                        mirror_missing=args.assume_symmetric, power=args.power,
                        views=args.views.split(",") if args.views else None)
        screened = None
        if extra and args.screen_extra is not None:
            from .repaint import screen_extras

            extra, screened = screen_extras(args.handoff, mesh, corner_uv, extra,
                                            max_added_delta_e=args.screen_extra, **settings)
        image, report = bake_texture(args.handoff, mesh, corner_uv, size=args.size, extra=extra, **settings)
        if screened is not None:
            report["screen_extra"] = screened
        Image.fromarray(image).save(args.output)
        _write_json(args.output.with_suffix(".json"), report)
        print(json.dumps(report, indent=2))
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
        manifest = repaint_views(args.handoff, mesh, corner_uv, texture, views, out, _LocalComfyClient(args.server),
                                 subject=args.subject, size=args.size, steps=args.steps, denoise=args.denoise)
        print(json.dumps(manifest["views"], indent=2))
        return 0
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
