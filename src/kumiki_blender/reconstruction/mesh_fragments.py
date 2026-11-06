"""Remove floating pieces from a reconstructed mesh.

Dense reconstruction can leave small pieces that float away from the subject.
Masks do not detect them well, because few cameras see such a piece. Geometry
does: a piece that does not touch the main body and has little area is not part
of it. The main body is never changed, and a large or touching piece stays.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .references import ReferenceError


REPORT = "mesh-fragment-report.json"
REFINED_MESH = "scene_dense_mesh_refined.ply"


@dataclass(frozen=True)
class FragmentLimits:
    """A piece stays if it touches the body or has this share of the body's area."""

    touching_fraction: float = .005
    keep_area_fraction: float = .05

    def __post_init__(self) -> None:
        if not 0 < self.touching_fraction < 1 or not 0 < self.keep_area_fraction < 1:
            raise ReferenceError("Fragment limits must be between 0 and 1")


def face_components(vertices: Any, faces: Any) -> Any:
    """Return the connected component of each face."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.r_[faces[:, [0, 1]], faces[:, [1, 2]]]
    graph = coo_matrix((np.ones(len(edges), bool), (edges[:, 0], edges[:, 1])),
                       shape=(len(vertices), len(vertices)))
    return connected_components(graph, directed=False)[1][faces[:, 0]]


def choose_faces(vertices: Any, faces: Any, limits: FragmentLimits) -> tuple[Any, list[dict[str, Any]]]:
    """Return a mask of the faces to keep, and a record of each removed piece."""
    import numpy as np
    from scipy.spatial import cKDTree

    labels = face_components(vertices, faces)
    corners = vertices[faces]
    areas = .5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                         corners[:, 2] - corners[:, 0]), axis=1)
    totals = np.bincount(labels, weights=areas)
    main = int(np.argmax(totals))
    body = vertices[np.unique(faces[labels == main])]
    size = float(np.linalg.norm(body.max(axis=0) - body.min(axis=0)))
    tree = cKDTree(body)
    keep = labels == main
    removed = []
    for label in np.unique(labels):
        if label == main:
            continue
        members = labels == label
        points = vertices[np.unique(faces[members])]
        gap = float(tree.query(points, workers=-1)[0].min() / size)
        share = float(totals[label] / totals[main])
        if gap <= limits.touching_fraction or share >= limits.keep_area_fraction:
            keep |= members
        else:
            removed.append({"faces": int(members.sum()), "area_share_of_body": round(share, 5),
                            "gap_fraction_of_body_size": round(gap, 4),
                            "center": [round(float(v), 3) for v in points.mean(axis=0)]})
    return keep, removed


def write_mesh(path: Path, vertices: Any, faces: Any) -> None:
    """Write the kept faces and their vertices as an OpenMVS binary PLY."""
    import numpy as np

    used, remapped = np.unique(faces, return_inverse=True)
    remapped = remapped.reshape(faces.shape)
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(used)}\n"
              "property float32 x\nproperty float32 y\nproperty float32 z\n"
              f"element face {len(faces)}\nproperty list uint8 uint32 vertex_indices\nend_header\n")
    rows = np.zeros(len(faces), np.dtype([("count", "u1"), ("corners", "<u4", 3)]))
    rows["count"], rows["corners"] = 3, remapped
    path.write_bytes(header.encode() + vertices[used].astype("<f4").tobytes() + rows.tobytes())


def remove_mesh_fragments(reconstruction: str | Path, rectified_masks: str | Path,
                          dense_revision: str | Path, output: str | Path, *,
                          limits: FragmentLimits | None = None,
                          executables: dict[str, str] | None = None) -> dict[str, Any]:
    """Write a dense revision without floating pieces, and texture it again."""
    import numpy as np

    from .dense import _json, rerun_dense_from_stage
    from .surface_coverage import _load_triangular_ply

    limits = limits or FragmentLimits()
    source = Path(dense_revision).resolve()
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists() or target.is_relative_to(source):
        raise ReferenceError("Fragment removal needs an unused output outside the dense revision")
    mesh = _json(source / "dense-report.json").get("mesh")
    if not mesh:
        raise ReferenceError(f"Dense revision has no finished mesh: {source}")
    vertices, faces = _load_triangular_ply(source / mesh["ply"], expected_vertices=mesh["vertices"],
                                           expected_faces=mesh["faces"], np=np)
    keep, removed = choose_faces(vertices, faces, limits)
    report: dict[str, Any] = {"source_dense_revision": str(source), "limits": asdict(limits),
                              "removed_pieces": removed, "faces_before": len(faces),
                              "faces_after": int(keep.sum())}

    def write_clean_mesh(workspace: Path) -> None:
        write_mesh(workspace / REFINED_MESH, vertices, faces[keep])

    result = rerun_dense_from_stage(reconstruction, rectified_masks, source, target, stage="texture",
                                    without=("scene_dense_mesh_refined",), prepare=write_clean_mesh,
                                    executables=executables)
    report.update(status=result["status"], failed_stage=result.get("failed_stage"), mesh=result.get("mesh"))
    (target / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
