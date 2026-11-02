"""Preview-only anatomical head graft with preservation and geometry audits.

Replace the simplified full-figure face with an actual sculptable anatomical
donor head, without modifying body/arms/pedestal. The CC0 Blender Studio
donor is NOT a Christ the Redeemer scan. Likeness stays on HOLD.
"""
from __future__ import annotations

import hashlib
import math


def _modules():
    try:
        import numpy as np
        import trimesh
    except ImportError as error:
        raise RuntimeError("Install kumiki-blender[preview]") from error
    return np, trimesh


def _world_mesh(scene, node):
    transform, geom = scene.graph[node]
    mesh = scene.geometry[geom].copy()
    mesh.apply_transform(transform)
    return mesh


def _head_node(scene, *, donor):
    names = list(scene.graph.nodes_geometry)
    if donor:
        candidates = [name for name in names if (
            name.startswith("Head | anatomical quad-topology clay sculpt")
            or name.endswith("/ GEO-head_sculpting_realistic"))]
    else:
        candidates = [name for name in names if
                      name.startswith("Head | integrated cheekbone")]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one %s head; found %r" %
                         ("donor" if donor else "original", candidates))
    return candidates[0]


def _geometry_digest(scene, nodes):
    np, _ = _modules()
    h = hashlib.sha256()
    for node in sorted(nodes):
        mesh = _world_mesh(scene, node)
        h.update(node.encode("utf-8"))
        h.update(np.asarray(mesh.vertices, dtype="<f8").tobytes())
        h.update(np.asarray(mesh.faces, dtype="<i8").tobytes())
    return h.hexdigest()


def graft_anatomical_head(base_scene, donor_scene, *, alignment="world",
                          replace_hair=False):
    """Return a new provisional GLB scene plus preservation audit.

    world: Blender-generated head study already in the full statue's axes.
    bounds: uniform bounding-box diagnostic alignment for generic donor GLBs;
            never claim that this is photograph or survey registration.
    """
    np, _ = _modules()
    if alignment not in ("world", "bounds"):
        raise ValueError("alignment must be world or bounds")
    base = base_scene.copy()
    donor = donor_scene
    original = _head_node(base, donor=False)
    replacement = _head_node(donor, donor=True)
    old_head = _world_mesh(base, original)
    head = _world_mesh(donor, replacement)
    if len(head.vertices) < 10000 or len(head.faces) < 10000:
        raise ValueError("Donor head must have genuine dense sculptable topology")
    if not np.isfinite(head.vertices).all():
        raise ValueError("Donor head contains invalid coordinates")
    old_min, old_max = old_head.bounds
    donor_min, donor_max = head.bounds
    fit = {"mode": alignment}
    if alignment == "bounds":
        scale = float(min((old_max[0]-old_min[0])/(donor_max[0]-donor_min[0]),
                          (old_max[1]-old_min[1])/(donor_max[1]-donor_min[1])))
        if not math.isfinite(scale) or scale <= 0 or scale > 100:
            raise ValueError("Implausible generic donor scaling")
        old_center = (old_min+old_max)/2
        donor_center = (donor_min+donor_max)/2
        # GLB uses Y up and Z face depth. Preserve the nose-front position.
        target = np.array((old_center[0],old_min[1],old_max[2]-.10))
        source = np.array((donor_center[0],donor_min[1],donor_max[2]))
        head.vertices = (head.vertices-source)*scale+target
        fit.update({"uniform_scale": round(scale,6),
                    "translation_world_m": [round(float(v),6)
                                            for v in target-scale*source]})
    bounds = head.bounds
    if abs(float(bounds[:,0].mean()-old_head.bounds[:,0].mean())) > .40:
        raise ValueError("Head center x differs by >0.4 m from body axis")
    if (abs(float(bounds[1,1]-old_max[1])) > .65 or
            abs(float(bounds[0,1]-old_min[1])) > .65):
        raise ValueError("Donor head height does not match the full figure")
    if abs(float(bounds[1,2]-old_max[2])) > .85:
        raise ValueError("Donor face depth is too far from original sculpt")

    remove = [node for node in base.graph.nodes_geometry if
              node.startswith(("Head | integrated cheekbone","Face |"))
              or (replace_hair and node.startswith("Hair |"))]
    keep = [node for node in base.graph.nodes_geometry if node not in remove]
    original_digest = _geometry_digest(base, keep)
    kept_geometry = {base.graph[node][1] for node in keep}
    removed_geometry = {base.graph[node][1] for node in remove}
    if kept_geometry & removed_geometry:
        raise ValueError("Head/face geometry is shared with a preserved body node")
    # trimesh Scene.delete_geometry also deletes its graph nodes.
    for geom in removed_geometry:
        base.delete_geometry(geom)
    base.add_geometry(head, geom_name="Head | anatomical quad-topology HOLD",
                      node_name="Head | anatomical quad-topology HOLD")
    donor_hair_count = 0
    if replace_hair:
        for node in donor.graph.nodes_geometry:
            if not node.startswith("Hair |"):
                continue
            mesh = _world_mesh(donor, node)
            if alignment == "bounds":
                mesh.vertices = (mesh.vertices-source)*scale+target
            base.add_geometry(mesh,geom_name="Study | "+node,
                              node_name="Study | "+node)
            donor_hair_count += 1
        if donor_hair_count == 0:
            raise ValueError("replace_hair requested but donor has no hair meshes")
    if _geometry_digest(base, keep) != original_digest:
        raise AssertionError("An unrelated body/arm/pedestal mesh was modified")
    audit = {
        "status":"HOLD_photo_likeness_not_verified",
        "original_head_vertices":int(len(old_head.vertices)),
        "donor_head_vertices":int(len(head.vertices)),
        "donor_head_faces":int(len(head.faces)),
        "removed_original_nodes":len(remove),
        "preserved_nodes":len(keep),
        "preserved_body_sha256":original_digest,
        "donor_hair_meshes":donor_hair_count,
        "alignment":fit,
        "head_bounds_world_m":np.round(head.bounds,4).tolist(),
        "provenance":"Generic CC0 Blender Studio anatomy; NOT Christ scan",
        "limitations":"Bounding fit is not photo registration; hair, facial likeness and mesh intersection need Blender/scan QA.",
    }
    return base, audit
