"""Command-line entry point for headless asset validation and building."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .export_audit import ExportAuditError
from .glb_audit import AuditError, audit_glb
from .lattice import make_lattice_tower
from .portico import PorticoError, make_portico_study
from .recipe import RecipeError, load_recipe
from .render_profiles import RenderProfileError
from .revision_audit import RevisionAuditError, audit_revision
from .surfaces import SurfaceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kumiki-blender")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "build"):
        subcommand = subcommands.add_parser(command)
        subcommand.add_argument("--recipe", type=Path, required=True)
        if command == "build":
            subcommand.add_argument("--output", type=Path, required=True)
    for command in ("plan-portico", "build-portico"):
        subcommand = subcommands.add_parser(command)
        subcommand.add_argument("--bays", type=int, default=3)
        if command == "build-portico":
            subcommand.add_argument("--output", type=Path, required=True)
    audit_parser = subcommands.add_parser("audit")
    audit_parser.add_argument("--model", type=Path, required=True)
    revision_parser = subcommands.add_parser("audit-revision")
    revision_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "audit":
            print(json.dumps(audit_glb(args.model), indent=2))
            return 0
        if args.command == "audit-revision":
            print(json.dumps(audit_revision(args.root), indent=2))
            return 0
        if args.command in ("plan-portico", "build-portico"):
            study = make_portico_study(bays=args.bays)
            if args.command == "plan-portico":
                print(json.dumps({
                    "asset_id": study.asset_id,
                    "generator": "modular-portico",
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
    except (RecipeError, PorticoError, RenderProfileError, RevisionAuditError, AuditError, ExportAuditError,
            SurfaceError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
