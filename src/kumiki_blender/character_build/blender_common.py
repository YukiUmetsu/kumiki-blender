"""Helpers shared by the character_build Blender workers. Imported only inside Blender."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def arguments() -> dict:
    """Read the JSON argument file that follows '--' on the Blender command line."""
    if "--" not in sys.argv or len(sys.argv) <= sys.argv.index("--") + 1:
        raise SystemExit("Pass the JSON argument file after --")
    return json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))


def clear_startup_scene() -> None:
    """Remove the factory cube, light and camera. Only for a new, unsaved file."""
    if bpy.data.filepath:
        return
    for name in ("Cube", "Light", "Camera"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj)


def import_mesh(path: str, name: str):
    """Import an OBJ (Y up), PLY (Z up) or GLB and join it into one mesh object called ``name``."""
    before = set(bpy.data.objects)
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif suffix == ".glb":
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh file {path}")
    meshes = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
    others = [o for o in set(bpy.data.objects) - before if o.type != "MESH"]
    if not meshes:
        raise ValueError(f"{path} contains no mesh")
    for obj in meshes:  # keep the world transform from parent nodes, then drop helper empties
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
    for obj in others:
        bpy.data.objects.remove(obj)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # glTF and OBJ split vertices at UV seams and hard edges, which leaves open boundary edges.
    # Merge them back, or the mesh is a set of patches (heat weighting and QuadriFlow fail).
    # Only merge a mesh with open edges: on a closed mesh a merge welds separate vertices at
    # pinch points and makes it non-manifold. UVs are per corner, so a merge keeps them.
    if open_edges(obj):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=1e-6)
        bpy.ops.object.mode_set(mode="OBJECT")
    # OBJ and glTF files can carry per-face normals, which override smooth shading and show
    # every quad. Character surfaces are smooth: drop them and shade smooth.
    if obj.data.has_custom_normals:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    bpy.ops.object.shade_smooth()
    obj.name = obj.data.name = name
    return obj


def open_edges(obj) -> int:
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    count = sum(1 for e in bm.edges if e.is_boundary)
    bm.free()
    return count


def remove_debris(obj, keep_fraction: float = 0.01) -> int:
    """Delete connected pieces smaller than ``keep_fraction`` of the faces. Return the count removed."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    seen, pieces = set(), []
    for face in bm.faces:
        if face.index in seen:
            continue
        stack, piece = [face], []
        seen.add(face.index)
        while stack:
            current = stack.pop()
            piece.append(current)
            for edge in current.edges:
                for other in edge.link_faces:
                    if other.index not in seen:
                        seen.add(other.index)
                        stack.append(other)
        pieces.append(piece)
    small = [p for p in pieces if len(p) < keep_fraction * len(bm.faces)]
    doomed = [f for p in small for f in p]
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        bm.to_mesh(obj.data)
    bm.free()
    return len(small)


def activate(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def mesh_stats(obj) -> dict:
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces = len(bm.faces)
    quads = sum(1 for f in bm.faces if len(f.verts) == 4)
    stats = {"vertices": len(bm.verts), "faces": faces, "quad_fraction": round(quads / max(1, faces), 4),
             "non_manifold_edges": sum(1 for e in bm.edges if not e.is_manifold),
             "volume_m3": round(abs(bm.calc_volume()), 8)}
    bm.free()
    return stats


def export_registered_obj(filepath: str | Path, *, selected: bool = True,
                          materials: bool = False) -> None:
    """OBJ in the canonical source character frame: +Z up, -Y forward.

    Blender 4.5 defaults to -Z forward / Y up. That is appropriate for some
    target engines but rotates OBJ vertices away from CharacterSpec v2
    cameras, silently corrupting original-art silhouette comparisons.
    """
    bpy.ops.wm.obj_export(
        filepath=str(filepath), export_selected_objects=selected,
        export_materials=materials, forward_axis="NEGATIVE_Y", up_axis="Z")


def write_report(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
