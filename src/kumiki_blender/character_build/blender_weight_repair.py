"""Blender worker: source-locked, non-destructive anatomical skin-weight repair.

Reads an existing .blend and a validated semantic-region pose contract.
Never edits coordinates, materials, mesh topology or source .blend. Repairs only
bounded, confidently mislabeled influences on explicitly marked part vertices.
"""
from __future__ import annotations

import hashlib
import math
import struct
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402
from kumiki_blender.character_build.weight_zone_metrics import audit_weight_zone  # noqa: E402
from kumiki_blender.character_build.weight_repair import propose_weight_repair  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_fingerprint(obj) -> str:
    """World-space rest geometry invariant; weight edits must not change it."""
    sha = hashlib.sha256(b"kumiki-weight-repair-shape-v1")
    for row in obj.matrix_world:
        sha.update(struct.pack("<4d", *row))
    for vertex in obj.data.vertices:
        sha.update(struct.pack("<3d", *vertex.co))
    for edge in obj.data.edges:
        sha.update(struct.pack("<2I", *edge.vertices))
    for polygon in obj.data.polygons:
        sha.update(struct.pack("<I", len(polygon.vertices)))
        for index in polygon.vertices:
            sha.update(struct.pack("<I", index))
    return sha.hexdigest()


def weight_rows(mesh, region_group: str, deform_names: set[str]):
    """Stream source vertex membership and deform influences, including zeros."""
    names = {group.index: group.name for group in mesh.vertex_groups}
    for vertex in mesh.data.vertices:
        member = 0.0
        weights = {}
        for entry in vertex.groups:
            name = names.get(entry.group)
            if name == region_group:
                member = entry.weight
            if name in deform_names and entry.weight > 0:
                weights[name] = float(entry.weight)
        yield vertex.index, member, weights


def main() -> None:
    args = arguments()
    output = Path(args["output"]).resolve()
    source = Path(args["source_blend"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output revision already exists: {output}")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Source-locked weight repair requires the exact saved source .blend")
    if not source.is_file() or source.suffix.lower() != ".blend":
        raise ValueError("Source .blend is missing")
    if source.is_relative_to(output):
        raise ValueError("Source .blend may not live inside the output revision")
    before_sha = file_sha256(source)
    if before_sha != args["source_blend_sha256"]:
        raise ValueError("Source .blend checksum changed since the weight repair was planned")

    rig = bpy.data.objects.get(args["armature"])
    if rig is None or rig.type != "ARMATURE" or rig.library is not None:
        raise ValueError("Expected an editable saved Blender armature")
    if not args["weight_zones"]:
        raise ValueError("No reviewed semantic weight zones; refusing to infer anatomy")
    deform_names = {bone.name for bone in rig.data.bones if bone.use_deform}
    source_meshes = {obj.name: obj for obj in bpy.data.objects if obj.type == "MESH" and any(
        mod.type == "ARMATURE" and mod.object == rig for mod in obj.modifiers)}
    if not source_meshes:
        raise ValueError("Source has no mesh skinned to requested rig")

    # Perform all checks before any vertex group is changed. A multi-zone job
    # either succeeds as one new revision or leaves the original untouched.
    prepared = []
    claimed = {}
    zones_before = []
    geometry_before = {name: shape_fingerprint(mesh)
                       for name, mesh in source_meshes.items()}
    max_removed = args["max_removed_weight"]
    limit = args["max_changed_vertices"]
    for zone in args["weight_zones"]:
        mesh = source_meshes.get(zone["mesh"])
        if mesh is None or mesh.library or mesh.data.library:
            raise ValueError(f"Zone mesh is missing, unbound or linked: {zone['mesh']}")
        region_group = zone["region_group"]
        membership_group = mesh.vertex_groups.get(region_group)
        if membership_group is None or region_group in deform_names:
            raise ValueError(f"{mesh.name}: semantic marker {region_group} is missing or is a deform bone")
        allowed = set(zone["allowed_bones"])
        if not allowed <= deform_names:
            raise ValueError(f"{mesh.name}/{region_group}: allowed bone missing or nondeforming")
        before = audit_weight_zone(
            weight_rows(mesh, region_group, deform_names), allowed,
            max_outside_weight=zone["max_outside_weight"],
            min_vertices=zone["min_vertices"],
        )
        if before["selected_vertices"] < zone["min_vertices"]:
            raise ValueError(f"{mesh.name}/{region_group}: semantic region lacks reviewed vertex coverage")
        zones_before.append({
            "mesh": mesh.name, "region_group": region_group,
            "allowed_bones": sorted(allowed), "before": before,
        })
        for index, membership, weights in weight_rows(mesh, region_group, deform_names):
            if membership < .5:
                continue
            key = (mesh.name, index)
            if key in claimed:
                raise ValueError(
                    f"{mesh.name} vertex {index} belongs to overlapping semantic regions: "
                    f"{claimed[key]} and {region_group}; choose one unambiguous zone")
            claimed[key] = region_group
            candidate = propose_weight_repair(
                weights, allowed, outside_tolerance=zone["max_outside_weight"],
                max_removed_weight=max_removed,
            )
            if candidate is None:
                continue
            for bone in set(candidate["before"]) | set(candidate["after"]):
                group = mesh.vertex_groups.get(bone)
                if group is None or group.lock_weight:
                    raise ValueError(
                        f"{mesh.name} vertex {index}: missing or locked deform group {bone}")
            prepared.append((mesh, index, candidate))
            if len(prepared) > limit:
                raise ValueError(
                    f"Proposed {len(prepared)} modified vertices exceeds "
                    f"max_changed_vertices={limit}; split repair into smaller regions")

    if file_sha256(source) != before_sha:
        raise ValueError("Saved source .blend changed during preflight")
    dry_run = bool(args["dry_run"])
    if not dry_run:
        for mesh, index, candidate in prepared:
            before_weights, after_weights = candidate["before"], candidate["after"]
            for bone in before_weights:
                if bone not in after_weights:
                    mesh.vertex_groups[bone].remove([index])
            for bone, weight in after_weights.items():
                mesh.vertex_groups[bone].add([index], weight, "REPLACE")
        for mesh in source_meshes.values():
            mesh.data.update()
        for name, mesh in source_meshes.items():
            if shape_fingerprint(mesh) != geometry_before[name]:
                raise ValueError(f"{name}: skin repair unexpectedly changed source geometry")

    zones_after = []
    for zone in args["weight_zones"]:
        mesh = source_meshes[zone["mesh"]]
        after = audit_weight_zone(
            weight_rows(mesh, zone["region_group"], deform_names),
            set(zone["allowed_bones"]),
            max_outside_weight=zone["max_outside_weight"],
            min_vertices=zone["min_vertices"],
        )
        if not dry_run and after["status"] != "pass":
            raise ValueError(
                f"{mesh.name}/{zone['region_group']}: repaired weights still fail region QA")
        zones_after.append(after)

    if file_sha256(source) != before_sha:
        raise ValueError("Original .blend changed during in-memory weight repair")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    status = "weight-repair-proposal-only" if dry_run else (
        "weight-repaired-animation-review-required" if prepared
        else "no-weight-repair-needed")
    if not dry_run and prepared:
        bpy.ops.wm.save_as_mainfile(filepath=str(output / "repaired.blend"))
    details = []
    for before, after in zip(zones_before, zones_after):
        details.append({**before, "after": after,
                        "modified_vertices": sum(
                            mesh.name == before["mesh"] and claimed[(mesh.name, idx)] == before["region_group"]
                            for mesh, idx, _ in prepared)})
    write_report(output, {
        "schema_version": 1, "status": status,
        "source_blend": str(source), "source_blend_sha256": before_sha,
        "armature": rig.name, "preview_only": dry_run,
        "max_removed_weight_per_vertex": max_removed,
        "max_changed_vertices": limit,
        "changed_vertices": len(prepared),
        "max_observed_removed_weight": max(
            (candidate["removed_outside_weight"] for _, _, candidate in prepared), default=0),
        "modified_vertex_examples": [
            {"mesh": mesh.name, "vertex": index,
             "removed_bones": candidate["removed_bones"],
             "removed_weight": candidate["removed_outside_weight"]}
            for mesh, index, candidate in prepared[:20]],
        "zones": details,
        "rest_geometry_sha256": geometry_before,
        "files": ["repaired.blend"] if prepared and not dry_run else [],
        "limitations": [
            "Only marked semantic zones and EXISTING permitted deform influences are used.",
            "Unnormalized weights and large/disconnected wrong influences require artist intervention.",
            "This operation changes Blender bone weights, not geometry or topology.",
            "It does not prove improved motion; run native validate-rig with real wing, paw and tail poses.",
            "Reference silhouettes and facial likeness are unchanged, not improved.",
            "The exact saved source .blend remains unmodified; a repaired candidate is a new revision.",
        ],
    })


main()
