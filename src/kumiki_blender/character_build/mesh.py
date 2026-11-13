"""Triangle meshes for character builds: read, write, measure and rasterize.

Pure NumPy and SciPy. No Blender. Coordinates are Blender metres (Z up).
OBJ and glTF use Y up; the readers and writers convert. GLB support covers
triangle primitives with float positions and node transforms, which is what
image-to-3D generators export.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .views import CANVAS_H, CANVAS_W, Camera


class MeshError(ValueError):
    """The mesh file cannot be read or is not usable."""


@dataclass
class Mesh:
    vertices: np.ndarray            # (N, 3) float64, Blender axes
    faces: np.ndarray               # (M, 3) int64
    colours: np.ndarray | None = None  # (N, 3) uint8

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if len(self.faces) and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise MeshError("Face index out of range")

    def face_normals(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit face normals and face areas."""
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        cross = np.cross(b - a, c - a)
        length = np.linalg.norm(cross, axis=1)
        return cross / np.maximum(length, 1e-20)[:, None], length / 2

    def vertex_normals(self) -> np.ndarray:
        normals, areas = self.face_normals()
        out = np.zeros_like(self.vertices)
        for i in range(3):
            np.add.at(out, self.faces[:, i], normals * areas[:, None])
        return out / np.maximum(np.linalg.norm(out, axis=1), 1e-20)[:, None]

    def adjacency(self) -> sparse.csr_matrix:
        """Row-normalised vertex adjacency: (A @ x) is the mean of the neighbours."""
        f = self.faces
        rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
        cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
        n = len(self.vertices)
        matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        matrix.data[:] = 1.0  # duplicates sum; reset to a plain 0/1 adjacency
        degree = np.asarray(matrix.sum(axis=1)).ravel()
        return sparse.diags(1.0 / np.maximum(degree, 1)) @ matrix

    def area(self) -> float:
        return float(self.face_normals()[1].sum())

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def weld(mesh: Mesh, tolerance: float = 1e-7) -> Mesh:
    """Merge vertices at the same position (within ``tolerance``) and drop faces that collapse.

    glTF and OBJ split vertices along UV seams and hard edges. Smoothing and fitting
    need the true neighbours, or the seams crack open.
    """
    keys = np.round(mesh.vertices / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    faces = inverse.reshape(-1)[mesh.faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    colours = mesh.colours[first] if mesh.colours is not None else None
    return Mesh(mesh.vertices[first], faces[keep], colours)


def orient_faces(mesh: Mesh) -> tuple[Mesh, int]:
    """Make the face winding consistent and outward. Return the mesh and the count of flipped faces.

    Generator meshes can have nearly random winding (TRELLIS.2: about half the faces
    point in). Signed volume, voxel remeshing and visibility tests all need outward
    faces. Two faces that share a manifold edge must traverse it in opposite
    directions; a breadth-first walk flips faces to agree, then each connected piece
    is turned so its signed volume is positive. Edges with more than two faces do
    not propagate.
    """
    from collections import deque

    faces = mesh.faces.copy()
    count = len(faces)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(count), 3)
    undirected = np.sort(directed, axis=1)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    undirected, directed, owner = undirected[order], directed[order], owner[order]
    starts = np.r_[True, (undirected[1:] != undirected[:-1]).any(axis=1)]
    group = np.cumsum(starts) - 1
    sizes = np.bincount(group)
    pair = sizes[group] == 2
    first = np.nonzero(pair & starts)[0]
    a, b = owner[first], owner[first + 1]
    same = (directed[first] == directed[first + 1]).all(axis=1)  # same direction: one of them must flip
    neighbours = [[] for _ in range(count)]
    for f1, f2, parity in zip(a.tolist(), b.tolist(), same.tolist()):
        neighbours[f1].append((f2, parity))
        neighbours[f2].append((f1, parity))
    flip = np.full(count, -1, dtype=np.int8)
    component = np.full(count, -1)
    for seed in range(count):
        if flip[seed] >= 0:
            continue
        flip[seed], component[seed] = 0, seed
        queue = deque([seed])
        while queue:
            f = queue.popleft()
            for g, parity in neighbours[f]:
                if flip[g] < 0:
                    flip[g] = flip[f] ^ parity
                    component[g] = seed
                    queue.append(g)
    faces[flip == 1] = faces[flip == 1][:, ::-1]
    a_, b_, c_ = (mesh.vertices[faces[:, i]] for i in range(3))
    volume = np.einsum("ij,ij->i", a_, np.cross(b_, c_))
    labels, inverse = np.unique(component, return_inverse=True)
    inward = np.bincount(inverse, weights=volume) < 0
    turn = inward[inverse]
    faces[turn] = faces[turn][:, ::-1]
    flipped = int(((flip == 1) ^ turn).sum())
    return Mesh(mesh.vertices, faces, mesh.colours), flipped


# ---------------------------------------------------------------- reading

def read_mesh(path: Path, *, max_faces: int = 3_000_000) -> Mesh:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        mesh = _read_obj(path)
    elif suffix == ".ply":
        mesh = _read_ply(path)
    elif suffix in (".glb", ".gltf"):
        mesh = _read_glb(path)
    else:
        raise MeshError(f"{path.name}: use .obj, .ply or .glb")
    if not len(mesh.faces):
        raise MeshError(f"{path.name}: the mesh has no faces")
    if len(mesh.faces) > max_faces:
        raise MeshError(f"{path.name}: {len(mesh.faces)} faces exceed the limit of {max_faces}")
    if not np.isfinite(mesh.vertices).all():
        raise MeshError(f"{path.name}: the mesh has NaN or infinite coordinates")
    return mesh


def _y_up_to_z_up(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])


def _z_up_to_y_up(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])


def _read_obj(path: Path) -> Mesh:
    vertices, colours, faces = [], [], []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()[1:]
                vertices.append([float(v) for v in values[:3]])
                if len(values) >= 6:
                    colours.append([float(v) for v in values[3:6]])
            elif line.startswith("f "):
                ids = []
                for token in line.split()[1:]:
                    index = int(token.split("/")[0])
                    ids.append(index - 1 if index > 0 else len(vertices) + index)
                faces.extend([ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1))
    points = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    # Native Blender source-registered exports explicitly preserve Z-up
    # world coordinates. Do not silently rotate these like interchange OBJ;
    # the registered source-camera comparison would become meaningless.
    if not (path.name.endswith("-source.obj") or path.name.endswith(".source.obj")):
        points = _y_up_to_z_up(points)
    rgb = None
    if colours and len(colours) == len(vertices):
        values = np.array(colours)
        rgb = np.clip(values * (255 if values.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    return Mesh(points, np.array(faces, dtype=np.int64), rgb)


_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _read_ply(path: Path) -> Mesh:
    """ASCII and binary little-endian PLY with vertex x y z [red green blue] and a face list."""
    data = path.read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise MeshError(f"{path.name}: not a PLY file")
    header = data[:end].decode("ascii", errors="replace").splitlines()
    body = data[data.index(b"\n", end) + 1:]
    fmt, elements = None, []
    for line in header:
        parts = line.split()
        if parts[:1] == ["format"]:
            fmt = parts[1]
        elif parts[:1] == ["element"]:
            elements.append({"name": parts[1], "count": int(parts[2]), "props": []})
        elif parts[:1] == ["property"]:
            elements[-1]["props"].append(parts[1:])
    if fmt not in ("ascii", "binary_little_endian"):
        raise MeshError(f"{path.name}: PLY format {fmt} is not supported")
    vertices = colours = faces = None
    if fmt == "ascii":
        tokens = body.decode("ascii").split()
        pos = 0
        for element in elements:
            names = [p[-1] for p in element["props"]]
            if element["name"] == "vertex":
                width = len(names)
                table = np.array(tokens[pos:pos + width * element["count"]], dtype=np.float64).reshape(-1, width)
                pos += width * element["count"]
                vertices = table[:, [names.index(a) for a in "xyz"]]
                if {"red", "green", "blue"} <= set(names):
                    colours = table[:, [names.index(c) for c in ("red", "green", "blue")]].astype(np.uint8)
            elif element["name"] == "face":
                faces = []
                for _ in range(element["count"]):
                    n = int(tokens[pos])
                    ids = [int(t) for t in tokens[pos + 1:pos + 1 + n]]
                    pos += 1 + n
                    faces.extend([ids[0], ids[i], ids[i + 1]] for i in range(1, n - 1))
            else:
                pos += len(names) * element["count"]
    else:
        pos = 0
        for element in elements:
            props = element["props"]
            if element["name"] == "face":
                lst = props[0]
                if lst[0] != "list":
                    raise MeshError(f"{path.name}: face element without a list")
                count_t, index_t = np.dtype("<" + _PLY_TYPES[lst[1]]), np.dtype("<" + _PLY_TYPES[lst[2]])
                faces = []
                for _ in range(element["count"]):
                    n = int(np.frombuffer(body, count_t, 1, pos)[0])
                    pos += count_t.itemsize
                    ids = np.frombuffer(body, index_t, n, pos).astype(np.int64)
                    pos += index_t.itemsize * n
                    faces.extend([ids[0], ids[i], ids[i + 1]] for i in range(1, n - 1))
                continue
            if any(p[0] == "list" for p in props):
                raise MeshError(f"{path.name}: list properties are only supported on faces")
            dtype = np.dtype([(p[1], "<" + _PLY_TYPES[p[0]]) for p in props])
            table = np.frombuffer(body, dtype, element["count"], pos)
            pos += dtype.itemsize * element["count"]
            if element["name"] == "vertex":
                vertices = np.column_stack([table[a].astype(np.float64) for a in "xyz"])
                if {"red", "green", "blue"} <= set(dtype.names):
                    colours = np.column_stack([table[c] for c in ("red", "green", "blue")]).astype(np.uint8)
    if vertices is None or faces is None:
        raise MeshError(f"{path.name}: PLY needs vertex and face elements")
    # PLY written by Blender and by this module is Z up.
    return Mesh(vertices, np.array(faces, dtype=np.int64), colours)


_GLTF_COMPONENT = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
_GLTF_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_glb(path: Path) -> Mesh:
    data = path.read_bytes()
    if path.suffix.lower() == ".gltf" or data[:4] != b"glTF":
        raise MeshError(f"{path.name}: only binary .glb files are supported")
    length = struct.unpack_from("<I", data, 8)[0]
    pos, doc, binary = 12, None, b""
    while pos < min(length, len(data)):
        chunk_len, chunk_type = struct.unpack_from("<II", data, pos)
        chunk = data[pos + 8:pos + 8 + chunk_len]
        if chunk_type == 0x4E4F534A:
            doc = json.loads(chunk)
        elif chunk_type == 0x004E4942:
            binary = chunk
        pos += 8 + chunk_len
    if doc is None:
        raise MeshError(f"{path.name}: GLB has no JSON chunk")

    def accessor(index: int) -> np.ndarray:
        acc = doc["accessors"][index]
        if "bufferView" not in acc or acc.get("sparse"):
            raise MeshError(f"{path.name}: sparse or empty accessors are not supported")
        view = doc["bufferViews"][acc["bufferView"]]
        if view.get("buffer", 0) != 0:
            raise MeshError(f"{path.name}: external buffers are not supported")
        dtype = np.dtype("<" + _GLTF_COMPONENT[acc["componentType"]])
        width = _GLTF_WIDTH[acc["type"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride", 0) or dtype.itemsize * width
        rows = np.ndarray((acc["count"], width), dtype, binary, start, (stride, dtype.itemsize))
        values = rows.astype(np.float64)
        if acc.get("normalized") and dtype.kind in "iu":
            values /= np.iinfo(dtype).max
        return values

    def node_matrix(node: dict) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        t = np.eye(4)
        t[:3, 3] = node.get("translation", [0, 0, 0])
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        r = np.eye(4)
        r[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        s = np.diag(list(node.get("scale", [1, 1, 1])) + [1])
        return t @ r @ s

    vertices, faces, colours = [], [], []
    count = 0

    def visit(index: int, parent: np.ndarray, depth: int = 0) -> None:
        nonlocal count
        if depth > 64:
            raise MeshError(f"{path.name}: node hierarchy is too deep")
        node = doc["nodes"][index]
        matrix = parent @ node_matrix(node)
        if "mesh" in node:
            for primitive in doc["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("mode", 4) != 4 or "POSITION" not in primitive["attributes"]:
                    continue
                points = accessor(primitive["attributes"]["POSITION"])
                points = (np.column_stack([points, np.ones(len(points))]) @ matrix.T)[:, :3]
                if "indices" in primitive:
                    tris = accessor(primitive["indices"]).astype(np.int64).reshape(-1, 3)
                else:
                    tris = np.arange(len(points)).reshape(-1, 3)
                rgb = None
                if "COLOR_0" in primitive["attributes"]:
                    c = accessor(primitive["attributes"]["COLOR_0"])[:, :3]
                    rgb = np.clip(c * (255 if c.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
                vertices.append(points)
                faces.append(tris + count)
                colours.append(rgb)
                count += len(points)
        for child in node.get("children", []):
            visit(child, matrix, depth + 1)

    scene = doc.get("scenes", [{}])[doc.get("scene", 0)] if doc.get("scenes") else {"nodes": range(len(doc.get("nodes", [])))}
    for root in scene.get("nodes", []):
        visit(root, np.eye(4))
    if not vertices:
        raise MeshError(f"{path.name}: GLB has no triangle meshes")
    rgb = np.concatenate(colours) if all(c is not None for c in colours) else None
    return Mesh(_y_up_to_z_up(np.concatenate(vertices)), np.concatenate(faces), rgb)


# ---------------------------------------------------------------- writing

def write_obj(mesh: Mesh, path: Path, comment: str = "") -> None:
    points = _z_up_to_y_up(mesh.vertices)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {comment}\n" if comment else "")
        if mesh.colours is not None:
            np.savetxt(handle, np.column_stack([points, mesh.colours / 255.0]), fmt="v %.6f %.6f %.6f %.4f %.4f %.4f")
        else:
            np.savetxt(handle, points, fmt="v %.6f %.6f %.6f")
        np.savetxt(handle, mesh.faces + 1, fmt="f %d %d %d")


def write_ply(mesh: Mesh, path: Path) -> None:
    """Binary little-endian PLY, Z up, with optional vertex colours. Blender imports it directly."""
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(mesh.vertices)}",
              "property float x", "property float y", "property float z"]
    if mesh.colours is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += [f"element face {len(mesh.faces)}", "property list uchar int vertex_indices", "end_header"]
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if mesh.colours is not None:
        fields += [("r", "u1"), ("g", "u1"), ("b", "u1")]
    table = np.zeros(len(mesh.vertices), dtype=fields)
    for i, axis in enumerate("xyz"):
        table[axis] = mesh.vertices[:, i]
    if mesh.colours is not None:
        for i, channel in enumerate("rgb"):
            table[channel] = mesh.colours[:, i]
    faces = np.zeros(len(mesh.faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    faces["n"], faces["i"] = 3, mesh.faces
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(table.tobytes())
        handle.write(faces.tobytes())


def write_mesh(mesh: Mesh, path: Path, comment: str = "") -> None:
    if path.suffix.lower() == ".ply":
        write_ply(mesh, path)
    elif path.suffix.lower() == ".obj":
        write_obj(mesh, path, comment)
    else:
        raise MeshError(f"{path.name}: write .obj or .ply")


# ---------------------------------------------------------------- rasterizing

def surface_samples(mesh: Mesh, metres_per_sample: float) -> tuple[np.ndarray, np.ndarray]:
    """Points that cover every triangle with spacing <= metres_per_sample. Returns points and face ids.

    Each triangle gets a regular barycentric grid with n subdivisions, n from its longest edge.
    """
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    longest = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                                 np.linalg.norm(a - c, axis=1)])
    steps = np.clip(np.ceil(longest / metres_per_sample).astype(int), 1, 64)
    points, owners = [], []
    for n in np.unique(steps):
        ids = np.nonzero(steps == n)[0]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        u, v = (i[keep] / n), (j[keep] / n)
        w = 1 - u - v
        grid = (w[None, :, None] * a[ids, None] + u[None, :, None] * b[ids, None] + v[None, :, None] * c[ids, None])
        points.append(grid.reshape(-1, 3))
        owners.append(np.repeat(ids, len(u)))
    return np.concatenate(points), np.concatenate(owners)


def rasterize(mesh: Mesh, cam: Camera, metres_per_px: float, scale: float = 1.0,
              samples: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic silhouette and depth buffer in the aligned frame (scaled by ``scale``).

    Depth is the largest p . toward per pixel (nearest to the camera), NaN where empty.
    Dense surface samples are splatted, so the result matches a polygon render
    to within about one pixel. Pass ``samples`` to reuse them across cameras.
    """
    height, width = int(CANVAS_H * scale), int(CANVAS_W * scale)
    if samples is None:
        samples = surface_samples(mesh, 0.5 * metres_per_px / scale)
    points = samples[0]
    col, row, depth = cam.project(points, metres_per_px, scale)
    c, r = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
    inside = (c >= 0) & (c < width) & (r >= 0) & (r < height)
    flat = r[inside] * width + c[inside]
    zbuffer = np.full(height * width, -np.inf)
    np.maximum.at(zbuffer, flat, depth[inside])
    zbuffer = zbuffer.reshape(height, width)
    mask = np.isfinite(zbuffer)
    zbuffer[~mask] = np.nan
    return mask, zbuffer
