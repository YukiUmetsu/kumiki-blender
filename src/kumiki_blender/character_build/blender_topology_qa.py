"""Blender-native joint-topology review: contour continuity and local face flow.

Does not alter retopology or claim artistic approval. Optionally writes a
new annotated .blend with native editable joint section rings.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402


def _basis(axis: Vector):
    basis = Vector((0, 0, 1)) if abs(axis.z) < .9 else Vector((0, 1, 0))
    u = axis.cross(basis).normalized()
    return u, axis.cross(u).normalized()


def _section(coords, edges, center, axis, u, v, offset, radius):
    cut = center + axis * offset
    angles = []
    pts = set()
    for a, b in edges:
        first, second = coords[a], coords[b]
        da, db = (first - cut).dot(axis), (second - cut).dot(axis)
        if not ((da < 0 <= db) or (db < 0 <= da)):
            continue
        divisor = da - db
        if abs(divisor) < 1e-12:
            continue
        t = da / divisor
        if t < 0 or t > 1:
            continue
        position = first.lerp(second, t) - cut
        radius_at = position.length
        if radius_at < 1e-7 or radius_at > radius:
            continue
        quantized = tuple(round(x, 6) for x in position)
        if quantized in pts:
            continue
        pts.add(quantized)
        angles.append(math.atan2(position.dot(v), position.dot(u)) % (2 * math.pi))
    angles.sort()
    bins = {min(15, int(a / (2 * math.pi) * 16)) for a in angles}
    if len(angles) >= 2:
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(angles[0] + 2 * math.pi - angles[-1])
        max_gap = math.degrees(max(gaps))
    else:
        max_gap = 360.0
    return {
        "offset_m": round(offset, 8),
        "unique_crossing_edges": len(angles),
        "angular_coverage_16_bins": round(len(bins) / 16, 4),
        "largest_angular_gap_deg": round(max_gap, 2),
    }


def _faces_and_poles(obj, coords, center, axis, radius, half_width):
    local = set()
    quads = polygons = triangles = ngons = 0
    for face in obj.data.polygons:
        points = [coords[i] for i in face.vertices]
        centroid = sum(points[1:], points[0].copy()) / len(points)
        radial = centroid - center
        along = radial.dot(axis)
        normal = (radial - axis * along).length
        if abs(along) <= half_width and normal <= radius:
            polygons += 1
            quads += len(face.vertices) == 4
            triangles += len(face.vertices) == 3
            ngons += len(face.vertices) > 4
            local.update(face.vertices)
    if not local:
        return {"faces": 0, "quad_fraction": None, "triangles": 0,
                "ngons": 0, "interior_poles": 0, "sampled_vertices": 0}
    valence = [0] * len(obj.data.vertices)
    for edge in obj.data.edges:
        a, b = edge.vertices
        if a in local:
            valence[a] += 1
        if b in local:
            valence[b] += 1
    # A vertex with a non-quad neighborhood is diagnostic, not an automatic
    # edge-loop failure. Count valence!=4 locally and report fraction.
    poles = sum(valence[i] != 4 for i in local)
    return {
        "faces": polygons, "quad_fraction": round(quads / polygons, 4),
        "triangles": triangles, "ngons": ngons,
        "non_four_valence_vertices": poles,
        "non_four_valence_fraction": round(poles / len(local), 4),
        "sampled_vertices": len(local),
        "limitations": "Local valence includes boundary/non-quad vertices; it does not prove a closed edge loop.",
    }


def _draw_ring(collection, name, center, axis, radius):
    u, v = _basis(axis)
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = max(.0002, radius * .006)
    curve.bevel_resolution = 1
    spline = curve.splines.new("POLY")
    spline.points.add(63)
    for i, point in enumerate(spline.points):
        angle = (i / 64) * 2 * math.pi
        pos = center + radius * (u * math.cos(angle) + v * math.sin(angle))
        point.co = (*pos, 1)
    spline.use_cyclic_u = True
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.hide_render = True
    obj.show_in_front = True


def main():
    args = arguments()
    plan = args["plan"]
    output = Path(args["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output revision exists: {output}")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != Path(args["source_blend"]).resolve():
        raise ValueError("Open the specified reviewed source .blend")
    obj = bpy.data.objects.get(plan["mesh"])
    if obj is None or obj.type != "MESH":
        raise ValueError(f"Mesh object not found: {plan['mesh']}")
    if len(obj.data.polygons) > 500_000 or len(obj.data.edges) > 1_000_000:
        raise ValueError("Topology QA expects a retopologized mesh, not the original high-res sculpt")
    coords = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    edges = [tuple(edge.vertices) for edge in obj.data.edges]
    diagnostics = []
    problems = []
    collection = None
    if args.get("annotate"):
        collection = bpy.data.collections.new("Kumiki Topology Review Guides")
        bpy.context.scene.collection.children.link(collection)
    for zone in plan["zones"]:
        center, axis = Vector(zone["center_m"]), Vector(zone["axis"]).normalized()
        radius, width = zone["search_radius_m"], zone["half_width_m"]
        u, v = _basis(axis)
        sections = [_section(coords, edges, center, axis, u, v, offset, radius)
                    for offset in (-width, 0.0, width)]
        flow = _faces_and_poles(obj, coords, center, axis, radius, width)
        relevant = [row for row in sections if (row["unique_crossing_edges"] < plan["min_crossing_edges"]
                    or row["angular_coverage_16_bins"] < plan["min_cross_section_coverage"])]
        if relevant:
            problems.append(zone["id"])
        diagnostics.append({
            "zone": zone["id"], "bone": zone["source_bone"],
            "sections": sections, "local_surface": flow,
            "cross_section_threshold_pass": not relevant,
        })
        if collection is not None:
            for i, offset in enumerate((-width, 0, width)):
                _draw_ring(collection, f"{zone['id']}.section.{i}",
                           center + offset * axis, axis, radius * .5)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    files = ["report.json"]
    if collection is not None:
        bpy.ops.wm.save_as_mainfile(filepath=str(output / "topology-reviewed.blend"))
        files.append("topology-reviewed.blend")
    report = {
        "schema_version": 1,
        "source_blend": str(Path(args["source_blend"]).resolve()),
        "mesh": obj.name, "zones": diagnostics,
        "candidate_problem_zones": problems,
        "status": "structural-review-failed" if problems and plan["strict"] else "topology-review-required",
        "files": files,
        "limitations": [
            "A cross-section with sufficient crossings can still have bad animated edge flow.",
            "Joined body/clothing/feather surfaces can inflate contour counts and angular coverage.",
            "Quads, local valence and contour metrics do not certify deformation performance.",
            "An annotated blend contains display-only guide rings; source .blend is not modified.",
        ],
    }
    write_report(output, report)


main()
