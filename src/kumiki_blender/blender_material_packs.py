"""Blender adapter for verified, externally authored PBR texture maps.

Only standard glTF-compatible image, Principled BSDF and Normal Map nodes are
used. Images are packed into the editable .blend and embedded in exported GLB.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .asset_import import AssetImportError
from .material_packs import TextureMap, TexturePack


def _verified_image(bpy: Any, source: TextureMap, *, non_color: bool) -> Any:
    # Also verify immediately before Blender opens the file. A source map must
    # not change between recipe preflight and packing a portable revision.
    if hashlib.sha256(source.file_path.read_bytes()).hexdigest() != source.sha256:
        raise AssetImportError(f"Texture changed after verification: {source.relative_path}")
    name = f"Kumiki / scanned {source.name} / {source.sha256[:16]}"
    image = bpy.data.images.get(name)
    if image is not None:
        return image
    try:
        image = bpy.data.images.load(str(source.file_path), check_existing=False)
        image.name = name
        if non_color:
            image.colorspace_settings.name = "Non-Color"
        image.pack()
        if image.packed_file is None:
            raise AssetImportError(f"Texture could not be packed: {source.relative_path}")
        # `image.has_data` can be false for an on-disk PNG that Blender has
        # indexed but not decoded into an in-memory pixel buffer yet. The
        # packed-file check confirms it was really read; dimensions are still
        # bounded before loading by material_packs._validate_image.
        if not all(0 < dimension <= 8192 for dimension in image.size):
            raise AssetImportError(f"Texture has invalid or oversized dimensions: {source.relative_path}")
    except RuntimeError as error:
        raise AssetImportError(f"Blender cannot load texture {source.relative_path}: {error}") from error
    return image


def create_texture_pack_material(bpy: Any, pack: TexturePack, name: str) -> Any:
    """Create a physically based, embedded material from inspected local maps."""
    material = bpy.data.materials.new(f"Kumiki / scanned / {name} / {pack.pack_id}")
    material.use_nodes = True
    shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Metallic"].default_value = pack.metallic
    shader.inputs["Roughness"].default_value = 0.7
    nodes, links = material.node_tree.nodes, material.node_tree.links
    for channel, socket in (("base_color", "Base Color"), ("roughness", "Roughness"),
                            ("metallic", "Metallic")):
        source = pack.file_for(channel)
        if source is None:
            continue
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.label = f"Kumiki / scanned {channel} / {pack.pack_id}"
        image_node.image = _verified_image(bpy, source, non_color=channel != "base_color")
        links.new(image_node.outputs["Color"], shader.inputs[socket])
    source = pack.file_for("normal")
    if source is None:
        raise AssetImportError(f"Texture pack {pack.pack_id} has no normal map")
    normal_image = nodes.new("ShaderNodeTexImage")
    normal_image.label = f"Kumiki / scanned tangent normal / {pack.pack_id}"
    normal_image.image = _verified_image(bpy, source, non_color=True)
    normal_node = nodes.new("ShaderNodeNormalMap")
    normal_node.inputs["Strength"].default_value = pack.normal_strength
    links.new(normal_image.outputs["Color"], normal_node.inputs["Color"])
    links.new(normal_node.outputs["Normal"], shader.inputs["Normal"])
    return material
