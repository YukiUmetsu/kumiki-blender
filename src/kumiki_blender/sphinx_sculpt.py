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
_HEAD_ROLES = frozenset({
    "nemes-hood", "face", "nemes-left-lappet", "nemes-right-lappet",
    "left-ear", "right-ear",
})


def _pick_sphinx_parts(parts: Sequence[MeshInstance], *, prefix: str,
                       roles: frozenset[str], label: str) -> tuple[MeshInstance, ...]:
    if not prefix or not isinstance(prefix, str) or not prefix.endswith("-"):
        raise GeometryError("Sphinx part prefix must be a nonempty, hyphen-ended string")
    selected = []
    for part in parts:
        if not isinstance(part, MeshInstance):
            raise GeometryError("Sphinx integral inputs must be MeshInstance objects")
        if part.name.startswith(prefix) and part.name[len(prefix):] in roles:
            if part.material not in ("sphinx-limestone", "casing-limestone"):
                raise GeometryError(f"Sphinx {label} source must use limestone: {part.name}")
            selected.append(part)
    if selected:
        found_roles = [p.name[len(prefix):] for p in selected]
        if len(set(found_roles)) != len(found_roles):
            raise GeometryError(f"Duplicate Sphinx anatomical {label} role")
        if len(selected) != len(roles):
            missing = sorted(roles - set(found_roles))
            raise GeometryError(f"Incomplete Sphinx {label} source: {', '.join(missing)}")
    return tuple(selected)


def sphinx_integral_parts(parts: Sequence[MeshInstance], *, prefix: str = "sphinx-") -> tuple[MeshInstance, ...]:
    """Pick the eight lion volumes, excluding toes, head and bedrock.

    Preserve recipe order so source correspondence survives Blender runs.
    """
    return _pick_sphinx_parts(parts, prefix=prefix, roles=_BODY_ROLES, label="body")


def sphinx_head_parts(parts: Sequence[MeshInstance], *, prefix: str = "sphinx-") -> tuple[MeshInstance, ...]:
    """Pick six head volumes for one carved face-and-nemes stone skin."""
    return _pick_sphinx_parts(parts, prefix=prefix, roles=_HEAD_ROLES, label="head")


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
