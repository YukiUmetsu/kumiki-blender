"""Command-line entry point for headless asset validation and building."""

from __future__ import annotations

import argparse
import json
import sys
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
from .glb_audit import AuditError, audit_glb
from .lattice import make_lattice_tower
from .landmark_scan import prepare_sphinx_scan
from .liberty import LibertyError, make_liberty_study
from .liberty_scan import ScanSourceError, verify_scan_source
from .material_overrides import load_material_overrides, verify_material_overrides
from .material_packs import load_texture_packs, verify_texture_targets
from .portico import PorticoError, make_portico_study
from .recipe import RecipeError, load_recipe
from .render_profiles import RenderProfileError
from .revision_audit import RevisionAuditError, audit_revision
from .surfaces import SurfaceError
from .vintage_biplane import make_biplane


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
    for command in ("plan-liberty", "build-liberty"):
        subcommand = subcommands.add_parser(command, help="Procedural Statue of Liberty sculpt and pedestal")
        subcommand.add_argument("--scale", type=float, default=1.0)
        subcommand.add_argument("--no-pedestal", action="store_true")
        if command == "build-liberty":
            subcommand.add_argument("--output", type=Path, required=True)
            subcommand.add_argument("--detailed-torch", action="store_true",
                                    help="Replace legacy flame/gallery with editable cast-detail torch")
            subcommand.add_argument("--detailed-tablet", action="store_true",
                                    help="Use the keystone tablet, serif date and curled supporting hand")
    scan_face_parser = subcommands.add_parser(
        "build-liberty-scan-face", help="Licensed photogrammetry face-detail QA checkpoint")
    scan_face_parser.add_argument("--source", type=Path, required=True)
    scan_face_parser.add_argument("--output", type=Path, required=True)
    column_parser = subcommands.add_parser("plan-classical-column")
    column_parser.add_argument("--order", choices=("tuscan", "doric", "ionic"), default="ionic")
    column_parser.add_argument("--height", type=float, default=4.55)
    column_parser.add_argument("--radius", type=float, default=0.36)
    audit_parser = subcommands.add_parser("audit")
    audit_parser.add_argument("--model", type=Path, required=True)
    audit_parser.add_argument("--require-pbr-textures", action="store_true")
    revision_parser = subcommands.add_parser("audit-revision")
    revision_parser.add_argument("--root", type=Path, required=True)
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
        if args.command in ("plan-liberty", "build-liberty"):
            study = make_liberty_study(scale=args.scale,
                                       include_pedestal=not args.no_pedestal,
                                       detailed_tablet=(args.command == "build-liberty" and args.detailed_tablet))
            if args.command == "plan-liberty":
                print(json.dumps({
                    "asset_id": study.asset_id, "generator": "procedural-statue-of-liberty",
                    "named_parts": dict(study.part_counts),
                    "crown_rays": sum("crown-ray-" in p.name for p in study.instances),
                    "instances": len(study.instances),
                    "bounds_m": [study.bounds_min, study.bounds_max],
                }, indent=2))
                return 0
            output = args.output.resolve()
            if output.exists() or output.is_symlink():
                raise LibertyError(f"Output already exists: {output}. Choose a new revision.")
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("Building Liberty requires Blender 4.2+: run via scripts/blender_entry.py") from error
            from .blender_liberty import build_liberty_scene
            print(json.dumps(build_liberty_scene(
                study, output, bpy_module=bpy,
                detailed_torch=args.detailed_torch), indent=2))
            return 0
        if args.command == "build-liberty-scan-face":
            provenance = verify_scan_source(args.source)
            output = args.output.resolve()
            if output.exists() or output.is_symlink():
                raise ScanSourceError(f"Output already exists: {output}. Choose a new revision.")
            try:
                import bpy  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("Building the scan-assisted face requires Blender 4.2+") from error
            from .blender_liberty_scan import build_liberty_scan_face_scene
            print(json.dumps(build_liberty_scan_face_scene(
                args.source.resolve(), output, provenance, bpy_module=bpy), indent=2))
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
    except (CatalogError, RecipeError, PorticoError, FacadeError, MonumentError, LibertyError,
            ScanSourceError,
            GeometryError, AssetImportError, RenderProfileError, RevisionAuditError, AuditError, ExportAuditError,
            SurfaceError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
