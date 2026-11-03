"""Conservative, opt-in rigid-body pair contact candidates from Blender bakes.

Only evaluated Bullet motion plus actual mesh surface proximity may qualify an
impact. Blender Python does not expose a stable Bullet impulse/manifold API.
Dense meshes require an explicit low-poly collision proxy instead of silently
substituting a bounding sphere and fabricating an impact cue.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .impact_measurement import PairRelativeSample, pair_impact_frames, surface_impact_speed
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

    # Cap retained geometry for long videos: at most 32 evaluated meshes,
    # regardless of clip duration and the number of explicitly named pairs.
    tree_cache: OrderedDict[tuple[Any, int], tuple[Any, list[Any], tuple[Any, Any]]] = OrderedDict()

    def tree(obj: Any, index: int) -> tuple[Any, list[Any], tuple[Any, Any]]:
        key = (obj, index)
        if key not in tree_cache:
            local, polygons = mesh(obj)
            matrix = samples[obj][index][1]
            world = [matrix @ vertex for vertex in local]
            bounds = (tuple(min(vertex[axis] for vertex in world) for axis in range(3)),
                      tuple(max(vertex[axis] for vertex in world) for axis in range(3)))
            tree_cache[key] = (BVHTree.FromPolygons(world, polygons), world, bounds)
            if len(tree_cache) > 32:
                tree_cache.popitem(last=False)
        else:
            tree_cache.move_to_end(key)
        return tree_cache[key]

    def may_touch(a: Any, b: Any, index: int, tolerance: float) -> bool:
        """Cheap world AABBs from eight bound corners before expensive BVHs."""
        bounds = []
        for obj in (a, b):
            matrix = samples[obj][index][1]
            corners = [matrix @ Vector(corner) for corner in obj.bound_box]
            bounds.append((tuple(min(p[axis] for p in corners) for axis in range(3)),
                           tuple(max(p[axis] for p in corners) for axis in range(3))))
        (min_a, max_a), (min_b, max_b) = bounds
        return all(min_a[axis] - max_b[axis] <= tolerance and
                   min_b[axis] - max_a[axis] <= tolerance for axis in range(3))

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

        # The narrowphase also returns an actual witness on each rigid mesh.
        # Retaining this contact point in *local* coordinates lets us measure
        # angular motion even when both mesh origins stay perfectly still.
        contact_cache: dict[int, tuple[bool, tuple[Any, Any, Any] | None]] = {}

        def contact(index: int) -> tuple[bool, tuple[Any, Any, Any] | None]:
            if index in contact_cache:
                return contact_cache[index]
            tolerance = pair.tolerance_m
            if not may_touch(a, b, index, tolerance):
                contact_cache[index] = (False, None)
                return contact_cache[index]
            bvh_a, vertices_a, (min_a, max_a) = tree(a, index)
            bvh_b, vertices_b, (min_b, max_b) = tree(b, index)
            gap_sq = sum(max(min_a[axis]-max_b[axis], min_b[axis]-max_a[axis], 0)**2
                         for axis in range(3))
            if gap_sq > tolerance*tolerance:
                contact_cache[index] = (False, None)
                return contact_cache[index]
            overlapping = bool(bvh_a.overlap(bvh_b))
            nearest: tuple[Any, Any, Any] | None = None
            best = float("inf")
            # Bidirectional vertex→surface distances reject AABB-only false
            # positives. The closest witnessed faces also yield local-space
            # contact points and an oriented surface normal for angular hits.
            for source_a, vertices, target in ((True, vertices_a, bvh_b),
                                               (False, vertices_b, bvh_a)):
                for vertex in vertices:
                    point, face_normal, _, distance = target.find_nearest(vertex, tolerance)
                    if distance is None or distance > tolerance or distance >= best or point is None:
                        continue
                    point_a, point_b = (vertex, point) if source_a else (point, vertex)
                    # The other mesh's face normal is reversed when it is A.
                    normal = face_normal.copy() if source_a else -face_normal
                    direction = (samples[a][index][1] @ local_center(a)
                                 - samples[b][index][1] @ local_center(b))
                    if normal.dot(direction) < 0:
                        normal.negate()
                    if normal.length > 1e-8:
                        normal.normalize()
                        best = distance
                        nearest = (point_a.copy(), point_b.copy(), normal)
            contact_cache[index] = (overlapping or nearest is not None, nearest)
            return contact_cache[index]

        def near_surface(index: int) -> bool:
            return contact(index)[0]

        hits = pair_impact_frames(relative, near_surface)
        for index in range(2, len(relative) - 1):
            past, previous, current, upcoming = relative[index-2:index+2]
            if (current.frame - previous.frame != 1 or previous.frame - past.frame != 1
                    or upcoming.frame - current.frame != 1 or
                    any(abs(current.frame - frame) < 5 for frame, _ in hits)):
                continue
            _, witness = contact(index)
            if witness is None:
                continue
            point_a, point_b, normal = witness
            # World-space witness converted once into the source meshes' local
            # frames. These same material points are followed across Bullet's
            # evaluated matrices, not approximated using mesh-origin velocity.
            matrix_a, matrix_b = samples[a][index][1], samples[b][index][1]
            try:
                local_a = matrix_a.inverted_safe() @ point_a
                local_b = matrix_b.inverted_safe() @ point_b
            except (ValueError, ZeroDivisionError):
                continue
            positions = []
            for at in range(index - 2, index + 2):
                world_a = samples[a][at][1] @ local_a
                world_b = samples[b][at][1] @ local_b
                delta = world_a - world_b
                positions.append((delta.x, delta.y, delta.z))
            speed = surface_impact_speed(positions, (normal.x, normal.y, normal.z))
            if speed is not None:
                hits.append((current.frame, speed))
        hits.sort(key=lambda item: item[0])
        target = a if bodies[a].kind == "active" else b
        measured.setdefault(target, []).extend(
            {"frame": frame, "speedMps": speed, "material": pair.material} for frame, speed in hits)
    return measured
