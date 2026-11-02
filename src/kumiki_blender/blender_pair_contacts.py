"""Conservative, opt-in rigid-body pair contact candidates from Blender bakes.

Only evaluated Bullet motion plus actual mesh surface proximity may qualify an
impact. Blender Python does not expose a stable Bullet impulse/manifold API.
Dense meshes require an explicit low-poly collision proxy instead of silently
substituting a bounding sphere and fabricating an impact cue.
"""

from __future__ import annotations

from typing import Any

from .impact_measurement import PairRelativeSample, pair_impact_frames
from .physics_plan import PhysicsPlanError, PhysicsShot


_MAX_PAIR_VERTICES = 8000


def measure_pair_contacts(shot: PhysicsShot, assignments: list[tuple[Any, Any]],
                          samples: dict[Any, list[tuple[int, Any, Any | None]]]) -> dict[Any, list[dict[str, Any]]]:
    """Measure only explicitly listed pairs, without any live solver dependency.

    `samples` contains Blender world matrices captured *during* the original
    forward-only Bullet simulation. Geometry is static in mesh-local space;
    vertex-cache deformation must be handled by a different pipeline.
    """
    if not shot.contact_pairs:
        return {}
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    def resolve(ref: str) -> Any:
        kind, label = ref.split(":", 1)
        found = [obj for obj, _ in assignments
                 if (obj.name == label if kind == "name" else obj.get("kumiki_part") == label)]
        if len(found) != 1:
            raise PhysicsPlanError(f"Contact body {ref!r} must select exactly one configured mesh")
        return found[0]

    bodies = {obj: body for obj, body in assignments}
    disabled = {frozenset((resolve(rule.body_a), resolve(rule.body_b)))
                for rule in shot.constraints if rule.disable_collisions}
    geometry: dict[Any, tuple[list[Any], list[tuple[int, ...]]]] = {}

    def mesh(obj: Any) -> tuple[list[Any], list[tuple[int, ...]]]:
        if obj not in geometry:
            data = obj.data
            if any(modifier.show_viewport for modifier in obj.modifiers):
                raise PhysicsPlanError(f"Contact mesh {obj.name!r} has unapplied geometry modifiers; apply them or use a proxy")
            if len(data.vertices) > _MAX_PAIR_VERTICES or not data.polygons or len(data.vertices) < 3:
                raise PhysicsPlanError(f"Contact mesh {obj.name!r} requires a 3–8000 vertex, surfaced collision proxy")
            geometry[obj] = ([vertex.co.copy() for vertex in data.vertices],
                             [tuple(poly.vertices) for poly in data.polygons])
        return geometry[obj]

    # Each (body, frame) tree is shared among explicit pairs where possible.
    tree_cache: dict[tuple[Any, int], tuple[Any, list[Any], tuple[Any, Any]]] = {}

    def tree(obj: Any, index: int) -> tuple[Any, list[Any], tuple[Any, Any]]:
        key = (obj, index)
        if key not in tree_cache:
            local, polygons = mesh(obj)
            matrix = samples[obj][index][1]
            world = [matrix @ vertex for vertex in local]
            bounds = (tuple(min(vertex[axis] for vertex in world) for axis in range(3)),
                      tuple(max(vertex[axis] for vertex in world) for axis in range(3)))
            tree_cache[key] = (BVHTree.FromPolygons(world, polygons), world, bounds)
        return tree_cache[key]

    measured: dict[Any, list[dict[str, Any]]] = {}
    resolved_pairs: set[frozenset[Any]] = set()
    center_cache: dict[Any, Any] = {}

    def local_center(obj: Any) -> Any:
        if obj not in center_cache:
            center_cache[obj] = sum((Vector(corner) for corner in obj.bound_box), Vector()) / 8
        return center_cache[obj]

    for pair in shot.contact_pairs:
        a, b = resolve(pair.body_a), resolve(pair.body_b)
        if a == b or (bodies[a].kind != "active" and bodies[b].kind != "active"):
            raise PhysicsPlanError("Contact pair must contain two different bodies and at least one active body")
        if frozenset((a, b)) in disabled:
            raise PhysicsPlanError("Contact pair has disabled collisions in a configured constraint")
        if frozenset((a, b)) in resolved_pairs:
            raise PhysicsPlanError("Contact pair resolves to a duplicate physical body pair")
        resolved_pairs.add(frozenset((a, b)))
        mesh(a), mesh(b)  # Fail explicitly for unsupported dense/deforming source geometry.
        if a not in samples or b not in samples:
            raise PhysicsPlanError("Contact pair has no evaluated Bullet trajectory")
        if len(samples[a]) != len(samples[b]):
            raise PhysicsPlanError("Contact pair sample counts differ")

        relative = []
        for (frame, matrix_a, _), (frame_b, matrix_b, _) in zip(samples[a], samples[b]):
            if frame != frame_b:
                raise PhysicsPlanError("Contact pair frames differ")
            # Authored object origins can be far from the visible mesh. Use
            # actual object bounds for a stable center-separation direction.
            delta = (matrix_a @ local_center(a)) - (matrix_b @ local_center(b))
            relative.append(PairRelativeSample(frame, delta.x, delta.y, delta.z))

        def near_surface(index: int) -> bool:
            bvh_a, vertices_a, (min_a, max_a) = tree(a, index)
            bvh_b, vertices_b, (min_b, max_b) = tree(b, index)
            gap_sq = sum(max(min_a[axis]-max_b[axis], min_b[axis]-max_a[axis], 0)**2
                         for axis in range(3))
            tolerance = pair.tolerance_m
            if gap_sq > tolerance*tolerance:
                return False
            if bvh_a.overlap(bvh_b):
                return True
            # Bidirectional vertex→surface distances avoid falsely triggering
            # from overlapping AABBs of distant, rotated or hollow geometry.
            for vertices, target in ((vertices_a, bvh_b), (vertices_b, bvh_a)):
                for vertex in vertices:
                    _, _, _, distance = target.find_nearest(vertex, tolerance)
                    if distance is not None and distance <= tolerance:
                        return True
            return False

        hits = pair_impact_frames(relative, near_surface)
        target = a if bodies[a].kind == "active" else b
        measured.setdefault(target, []).extend(
            {"frame": frame, "speedMps": speed, "material": pair.material} for frame, speed in hits)
    return measured
