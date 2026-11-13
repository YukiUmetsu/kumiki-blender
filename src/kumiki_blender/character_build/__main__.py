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
        if name == "rig":
            p.add_argument("--rom-deg", type=float, default=45.0)
            p.add_argument("--render-rom", action="store_true")
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run(args) -> int:
    from .views import ingest

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

        spec, mesh = _spec(args.handoff), read_mesh(args.mesh)
        _new_file(args.output)
        if args.command == "fit":
            from .fit import fit_silhouettes
            from .handoff import reference_masks

            scale = 0.25
            masks = reference_masks(args.handoff, scale)
            if args.views:
                masks = {n: m for n, m in masks.items() if n in args.views}
            result = fit_silhouettes(mesh, masks, spec["height_m"], scale=scale, iterations=args.iterations,
                                     normalise_first=not args.no_normalise)
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
        report = run_worker("retopo", {"mesh": str(args.input.resolve()), "output": str(out.resolve()),
                                       "target_faces": args.target_faces, "bake_size": args.bake_size,
                                       "cage_m": args.cage_m}, blender=args.blender)
    else:
        from .rig import validate_joints

        joints = json.loads(args.joints.read_text(encoding="utf-8"))
        validate_joints(joints)
        report = run_worker("rig", {"mesh": str(args.input.resolve()), "joints": joints, "output": str(out.resolve()),
                                    "rom_deg": args.rom_deg, "render_rom": args.render_rom}, blender=args.blender)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
