"""Blender-independent preparation of seamless sculptural statue masses.

Procedural landmark authors work with separate, named components; the final
visible stone should not show the intersection seams between the lion torso,
shoulders, paws, haunches and neck. This module selects the appropriate closed
solids and produces one world-space voxel-remesh input without losing the
original editing components or requiring Blender during recipe planning.
"""

from __future__ import annotations

from collections.abc import Sequence

from .architecture_geometry import GeometryError, Mesh, MeshInstance, world_vertices


_BODY_ROLES = frozenset({
    "lion-torso", "chest", "carved-neck", "left-forepaw", "right-forepaw",
    "left-haunch", "right-haunch", "curved-tail",
})


def sphinx_integral_parts(parts: Sequence[MeshInstance], *, prefix: str = "sphinx-") -> tuple[MeshInstance, ...]:
    """Pick only statue body volumes, not facial, ornament or bedrock solids.

    Preserve the order of the recipe so that source-part correspondence is
    stable across Blender runs. The selection intentionally excludes toes,
    whose carved definition would disappear when voxel-unioned with the paws.
    """
    if not prefix or not isinstance(prefix, str) or not prefix.endswith("-"):
        raise GeometryError("Sphinx part prefix must be a nonempty, hyphen-ended string")
    selected = []
    for part in parts:
        if not isinstance(part, MeshInstance):
            raise GeometryError("Sphinx integral inputs must be MeshInstance objects")
        if part.name.startswith(prefix) and part.name[len(prefix):] in _BODY_ROLES:
            if part.material != "sphinx-limestone":
                raise GeometryError(f"Sphinx body source must use limestone: {part.name}")
            selected.append(part)
    if selected:
        roles = [p.name[len(prefix):] for p in selected]
        if len(set(roles)) != len(roles):
            raise GeometryError("Duplicate Sphinx anatomical body role")
        if len(selected) != len(_BODY_ROLES):
            missing = sorted(_BODY_ROLES - set(roles))
            raise GeometryError(f"Incomplete Sphinx body source: {', '.join(missing)}")
    return tuple(selected)


def combine_sculpture_sources(parts: Sequence[MeshInstance]) -> Mesh:
    """Append closed, transformed surfaces as input to Blender voxel union.

    The returned mesh holds overlapping closed shells; it is *not* yet a
    topological union. Blender's voxel Remesh modifier is the separate step
    that removes intersections and produces one manifold sculptural skin.
    """
    if not parts:
        raise GeometryError("Sculpture source union requires at least one closed solid")
    vertices = []
    faces = []
    for part in parts:
        if not isinstance(part, MeshInstance):
            raise GeometryError("Sculpture sources must be MeshInstance objects")
        if len(part.mesh.vertices) < 8 or not part.mesh.faces:
            raise GeometryError(f"Sculpture source is empty: {part.name}")
        offset = len(vertices)
        vertices.extend(world_vertices(part))
        faces.extend(tuple(offset + index for index in face) for face in part.mesh.faces)
    if len(vertices) > 1_000_000:
        raise GeometryError("Sculpture source exceeds safe million-vertex union limit")
    return Mesh(tuple(vertices), tuple(faces))
