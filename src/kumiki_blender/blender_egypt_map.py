"""Backward-compatible Blender entry point for the categorized Egypt atlas.

CLI and external scripts may keep importing ``build_egypt_map_scene`` here.
The actual scene authoring now resides in ``geography.egypt``.
"""

from .geography.egypt.blender_support import (
    _color_atlas_png,
    _label_hover_height,
    _river_ribbon,
    _terrain_contact_z,
    _vertex_islands,
)
from .geography.egypt.blender_scene import (
    build_egypt_map_scene,
    plan_egypt_shots,
)

__all__ = ["build_egypt_map_scene", "plan_egypt_shots"]
