"""Open a hash-pinned finished UDIM package as editable Blender tiled images.

Run from Blender's bundled Python. Only stdlib and bpy are required; the image
processing dependencies used to *create* a package are deliberately excluded.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .blender_dense import _png_metadata
from .dense import _json, _sha256
from .references import ReferenceError


def _validated_package(directory: str | Path) -> tuple[Path, dict[str, Any]]:
    raw = Path(directory)
    root = raw.resolve()
    if raw.is_symlink() or not root.is_dir():
        raise ReferenceError("UDIM package is missing or linked")
    report_path = root / "udim-report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise ReferenceError("UDIM report is missing or linked")
    report = _json(report_path)
    if (report.get("schema_version") != 1
            or report.get("status") != "private-udim-artistic-review-required"
            or report.get("distribution_allowed") is not False):
        raise ReferenceError("UDIM package has an unsupported provenance record")

    def asset(entry: Any, expected: str) -> Path:
        if (not isinstance(entry, dict) or entry.get("path") != expected
                or not isinstance(entry.get("sha256"), str)):
            raise ReferenceError("UDIM asset name or digest is invalid")
        path = root / expected
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root)
                or path.stat().st_size > 512 * 1024 * 1024
                or _sha256(path) != entry["sha256"]):
            raise ReferenceError(f"UDIM asset changed or linked: {expected}")
        return path

    obj = asset(report.get("obj"), "udim.obj")
    material = asset(report.get("mtl"), "udim.mtl")
    if (obj.stat().st_size > 1024 * 1024 * 1024
            or material.stat().st_size > 65_536):
        raise ReferenceError("UDIM mesh or material exceeds import budget")
    lines = obj.read_text(encoding="utf-8").splitlines()
    if [line for line in lines if line.startswith("mtllib ")] != ["mtllib udim.mtl"]:
        raise ReferenceError("UDIM OBJ references an unpinned material library")
    material_lines = material.read_text(encoding="utf-8").splitlines()
    safe_mtl_directives = {
        "newmtl", "Ka", "Kd", "Ks", "Ke", "Ns", "Ni", "d", "Tr",
        "illum", "Tf", "Pr", "Pm", "Ps", "Pc", "Pcr", "aniso", "anisor",
        "map_Kd",
    }
    if (not any(line == "map_Kd albedo.<UDIM>.png" for line in material_lines)
            or any(line.strip() and not line.lstrip().startswith("#") and
                   (line.split(maxsplit=1)[0] not in safe_mtl_directives or
                    (line.startswith("map_Kd") and
                     line != "map_Kd albedo.<UDIM>.png"))
                   for line in material_lines)):
        raise ReferenceError("UDIM MTL references unpinned texture files")

    material_tiles = report.get("material_tiles")
    pages = report.get("tiles")
    if (not isinstance(material_tiles, dict) or not material_tiles
            or not isinstance(pages, list) or not 1 <= len(pages) <= 100):
        raise ReferenceError("UDIM tile assignments are missing")
    tile_numbers: set[int] = set()
    for page in pages:
        if not isinstance(page, dict):
            raise ReferenceError("UDIM tile entry is invalid")
        tile = page.get("tile")
        if (type(tile) is not int or not 1001 <= tile <= 1999 or tile in tile_numbers):
            raise ReferenceError("UDIM tiles must be unique 1001+ numbers")
        tile_numbers.add(tile)
        albedo = asset(page.get("albedo"), f"albedo.{tile}.png")
        dimensions, _, _ = _png_metadata(albedo)
        for channel, expected_depth in (("roughness", 8), ("height", 16)):
            if page.get(channel) is None:
                continue
            data = asset(page[channel], f"{channel}.{tile}.png")
            size, depth, color_type = _png_metadata(data)
            if size != dimensions or depth != expected_depth or color_type != 0:
                raise ReferenceError(f"UDIM {channel} image is not a matching grayscale tile")
        if page.get("height") is not None:
            scale = page.get("displacement_scale_m")
            if (type(scale) not in (int, float) or not math.isfinite(scale)
                    or not .00001 <= scale <= .5):
                raise ReferenceError("UDIM height tile has no valid displacement scale")
        elif page.get("displacement_scale_m") is not None:
            raise ReferenceError("UDIM displacement scale requires a height tile")
    if 1001 not in tile_numbers:
        raise ReferenceError("UDIM package requires a 1001 base tile")
    for channel in ("roughness", "height"):
        if (any(page.get(channel) is None for page in pages)
                and any(page.get(channel) is not None for page in pages)):
            raise ReferenceError("UDIM channels need the same coverage on every tile")
    if (any(not isinstance(name, str) or not name or type(tile) is not int
            or tile not in tile_numbers for name, tile in material_tiles.items())
            or set(line[7:].strip() for line in lines if line.startswith("usemtl "))
            != set(material_tiles)):
        raise ReferenceError("UDIM OBJ material assignments do not match its tile report")
    return root, report


def import_finished_udim_package(bpy: Any, directory: str | Path) -> list[Any]:
    """Import the OBJ and replace ordinary images with true tiled PBR nodes.

    The complete package directory must stay next to the resulting .blend.
    glTF/GLB does not retain UDIM or true displacement; use this Blender scene
    for high-resolution Cycles renders, or bake/split UDIM tiles for GLB.
    """
    root, report = _validated_package(directory)
    tiles = [entry["tile"] for entry in report["tiles"]]
    channel_images: dict[str, Any] = {}
    for channel in ("albedo", "roughness", "height"):
        if channel != "albedo" and report["tiles"][0].get(channel) is None:
            continue
        # A tiled image's tokenized path is essential. Loading the OBJ alone
        # creates regular image nodes in some Blender versions.
        image = bpy.data.images.load(str(root / f"{channel}.1001.png"),
                                     check_existing=False)
        image.name = f"Kumiki / {channel} / {root.name}"
        image.source = "TILED"
        for tile in tiles:
            if tile != 1001:
                image.tiles.new(tile)
        image.filepath = str(root / f"{channel}.<UDIM>.png")
        image.colorspace_settings.name = "sRGB" if channel == "albedo" else "Non-Color"
        image.reload()
        # Blender lazily loads tiled pixels. Querying size forces the first
        # tile to be read; otherwise a valid-looking node can show no texture.
        expected_size, _, _ = _png_metadata(root / f"{channel}.1001.png")
        if tuple(image.size) != expected_size or not image.has_data:
            raise ReferenceError(f"Blender could not load UDIM {channel} pixels")
        channel_images[channel] = image

    previous_objects = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(root / "udim.obj"))
    imported = [obj for obj in bpy.data.objects if obj not in previous_objects
                and obj.type == "MESH"]
    if not imported:
        raise ReferenceError("UDIM package imported no mesh")
    tile_pages = {page["tile"]: page for page in report["tiles"]}
    seen: set[str] = set()
    for obj in imported:
        for slot in obj.material_slots:
            material = slot.material
            if material is None:
                continue
            name = material.name
            # Blender may suffix duplicate material names from a scene that
            # already contained identically named materials.
            if name not in report["material_tiles"]:
                base, dot, suffix = name.rpartition(".")
                if not dot or not suffix.isdigit() or base not in report["material_tiles"]:
                    raise ReferenceError(f"Unknown imported UDIM material: {name}")
                name = base
            if name in seen:
                continue
            seen.add(name)
            material.use_nodes = True
            nodes, links = material.node_tree.nodes, material.node_tree.links
            shader = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
            output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
            if shader is None or output is None:
                raise ReferenceError("UDIM material has no editable PBR shader")
            for node in list(nodes):
                if node.type == "TEX_IMAGE":
                    nodes.remove(node)
            for channel, image in channel_images.items():
                node = nodes.new("ShaderNodeTexImage")
                node.label = f"Kumiki / tiled {channel}"
                node.image = image
                if channel == "albedo":
                    links.new(node.outputs["Color"], shader.inputs["Base Color"])
                elif channel == "roughness":
                    links.new(node.outputs["Color"], shader.inputs["Roughness"])
                else:
                    displacement = nodes.new("ShaderNodeDisplacement")
                    displacement.inputs["Midlevel"].default_value = .5
                    displacement.inputs["Scale"].default_value = tile_pages[
                        report["material_tiles"][name]]["displacement_scale_m"]
                    links.new(node.outputs["Color"], displacement.inputs["Height"])
                    links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])
                    if hasattr(material, "displacement_method"):
                        material.displacement_method = "BOTH"
    if seen != set(report["material_tiles"]):
        raise ReferenceError("Some UDIM materials were not imported")
    return imported
