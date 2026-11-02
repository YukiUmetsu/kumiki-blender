"""Conservative cross-page UV seam exposure estimates for pinned OBJ atlases.

Only vertices shared by faces on *different* texture pages are compared. A
sample is taken just inside each face, avoiding edge padding/bleed. Page gains
are bounded and zero-centered so the overall sculpture exposure stays put.
This does not attempt in-page chart repair or infer absent texture detail.
"""

from __future__ import annotations

import math
import random
import statistics
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any

from .references import ReferenceError
from .uv_coverage import _MAX_UV_COORDS, _MAX_UV_FACES, _material_textures


_MAX_VERTICES = 5_000_000
_MAX_SEAM_ANCHORS = 40_000
_MAX_SAMPLES_PER_PAGE = 3
_MAX_PAGES_PER_VERTEX = 8
_MAX_PAIR_MAD_EV = 0.45
_BOUNDARY_MARKER = 0xFFFFFFFF


def _obj_index(text: str, count: int, kind: str) -> int:
    try:
        index = int(text)
    except ValueError as error:
        raise ReferenceError(f"Textured OBJ has an invalid {kind} index") from error
    offset = index - 1 if index > 0 else count + index
    if index == 0 or not 0 <= offset < count:
        raise ReferenceError(f"Textured OBJ references an unknown {kind} index")
    return offset


def _face_tokens(line: str) -> list[str]:
    tokens = line.split()[1:]
    if not 3 <= len(tokens) <= 32:
        raise ReferenceError("UV face must have 3–32 vertices")
    return tokens


def _shared_vertex_sample(obj: Path, materials: dict[str, str],
                          page_ids: dict[str, int]) -> tuple[set[int], int]:
    """Reservoir-sample shared page-boundary vertices across the entire OBJ."""
    first_page = array("I")
    chosen: list[int] = []
    eligible = 0
    current: str | None = None
    faces = 0
    rng = random.Random(271828)
    with obj.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("v "):
                first_page.append(0)
                if len(first_page) > _MAX_VERTICES:
                    raise ReferenceError("UV seam analysis exceeds safe OBJ vertex budget")
            elif stripped.startswith("usemtl "):
                current = materials.get(stripped[7:].strip())
            elif stripped.startswith("f "):
                faces += 1
                if faces > _MAX_UV_FACES:
                    raise ReferenceError("UV seam analysis exceeds safe OBJ face budget")
                if current is None:
                    raise ReferenceError("OBJ face has no pinned diffuse UV material")
                page = page_ids[current]
                for token in _face_tokens(stripped):
                    vertex = _obj_index(token.split("/", 1)[0], len(first_page), "vertex")
                    previous = first_page[vertex]
                    if previous == 0:
                        first_page[vertex] = page
                    elif previous != page and previous != _BOUNDARY_MARKER:
                        first_page[vertex] = _BOUNDARY_MARKER
                        eligible += 1
                        if len(chosen) < _MAX_SEAM_ANCHORS:
                            chosen.append(vertex)
                        else:
                            position = rng.randrange(eligible)
                            if position < _MAX_SEAM_ANCHORS:
                                chosen[position] = vertex
    if not faces:
        raise ReferenceError("UV seam analysis needs a textured OBJ mesh")
    return set(chosen), eligible


def _sample_uv_near_boundary(obj: Path, materials: dict[str, str],
                             selected: set[int]) -> dict[int, dict[str, list[tuple[float, float]]]]:
    """Find nearby face-interior texels per boundary vertex and atlas page."""
    texcoords: list[tuple[float, float]] = []
    samples: dict[int, dict[str, list[tuple[float, float]]]] = {}
    current: str | None = None
    vertex_count = 0
    with obj.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("v "):
                vertex_count += 1
            elif stripped.startswith("vt "):
                values = stripped.split()
                if len(values) not in (3, 4):
                    raise ReferenceError("OBJ UV texture coordinate must have two or three values")
                try:
                    u, v = float(values[1]), float(values[2])
                except ValueError as error:
                    raise ReferenceError("OBJ has an invalid UV coordinate") from error
                if (not math.isfinite(u) or not math.isfinite(v)
                        or not -0.0001 <= u <= 1.0001 or not -0.0001 <= v <= 1.0001):
                    raise ReferenceError("UV seam analysis cannot use tiled/out-of-range coordinates")
                texcoords.append((max(0.0, min(1.0, u)), max(0.0, min(1.0, v))))
                if len(texcoords) > _MAX_UV_COORDS:
                    raise ReferenceError("UV seam analysis exceeds safe texture-coordinate budget")
            elif stripped.startswith("usemtl "):
                current = materials.get(stripped[7:].strip())
            elif stripped.startswith("f "):
                if current is None:
                    raise ReferenceError("OBJ face has no pinned diffuse UV material")
                vertices: list[int] = []
                coordinates: list[tuple[float, float]] = []
                for token in _face_tokens(stripped):
                    fields = token.split("/")
                    if len(fields) < 2 or not fields[1]:
                        raise ReferenceError("Textured OBJ face is missing a UV index")
                    vertices.append(_obj_index(fields[0], vertex_count, "vertex"))
                    coordinates.append(texcoords[_obj_index(fields[1], len(texcoords), "UV")])
                selected_corners = [(index, vertex) for index, vertex in enumerate(vertices)
                                    if vertex in selected]
                if not selected_corners:
                    continue
                center_u = sum(point[0] for point in coordinates) / len(coordinates)
                center_v = sum(point[1] for point in coordinates) / len(coordinates)
                for index, vertex in selected_corners:
                    pages = samples.setdefault(vertex, {})
                    collected = pages.setdefault(current, [])
                    if len(collected) < _MAX_SAMPLES_PER_PAGE:
                        u, v = coordinates[index]
                        collected.append((0.90*u + 0.10*center_u,
                                          0.90*v + 0.10*center_v))
    return samples


def _sample_luma_pages(atlases: dict[str, Path],
                       coordinates: dict[int, dict[str, list[tuple[float, float]]]],
                       protect_black: int) -> dict[int, dict[str, float]]:
    from PIL import Image

    # Decode one atlas at a time: a large, many-page outdoor scan should not
    # need all the original 8K+ pages resident at once.
    by_page: dict[str, list[tuple[int, list[tuple[float, float]]]]] = defaultdict(list)
    for vertex, pages in coordinates.items():
        for page, points in pages.items():
            by_page[page].append((vertex, points))
    result: dict[int, dict[str, float]] = {}
    for page, path in atlases.items():
        if page not in by_page:
            continue
        with Image.open(path) as photo:
            if photo.format not in ("JPEG", "PNG"):
                raise ReferenceError("UV seam analysis requires a JPEG or PNG atlas")
            photo.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
            gray = photo.convert("L")
            alpha = (photo.convert("RGBA").getchannel("A")
                     if "A" in photo.getbands() or "transparency" in photo.info else None)
            width, height = gray.size
            for vertex, points in by_page[page]:
                values = []
                for u, v in points:
                    x = round(u * (width-1))
                    y = round((1-v) * (height-1))
                    value = gray.getpixel((x, y))
                    if (max(16, protect_black) <= value <= 249
                            and (alpha is None or alpha.getpixel((x, y)) >= 128)):
                        values.append(float(value))
                if values:
                    result.setdefault(vertex, {})[page] = statistics.median(values)
    return result


def _solve_page_offsets(edges: list[tuple[str, str, float, int]],
                        max_ev: float) -> dict[str, float]:
    adjacency: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for a, b, delta, count in edges:
        adjacency[a].append((b, delta, count))
        adjacency[b].append((a, -delta, count))
    result = {}
    seen = set()
    for root in sorted(adjacency):
        if root in seen:
            continue
        component = []
        pending = [root]
        seen.add(root)
        while pending:
            node = pending.pop()
            component.append(node)
            for neighbor, _delta, _weight in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        component.sort()
        values = {page: 0.0 for page in component}
        # The root fixes the gauge while weighted Gauss–Seidel minimizes the
        # inconsistent pairwise exposure constraints in a connected component.
        for _ in range(1000):
            change = 0.0
            for page in component[1:]:
                links = adjacency[page]
                updated = (sum(weight * (values[other] + delta)
                               for other, delta, weight in links)
                           / sum(weight for _other, _delta, weight in links))
                change = max(change, abs(values[page] - updated))
                values[page] = updated
            if change < 0.00001:
                break
        # Project to both constraints *together*: independently clipping
        # zero-centered gains can still shift the component's mean exposure.
        # The sum of clamped(page_gain - shift) is monotone in shift.
        lo = min(values.values()) - max_ev
        hi = max(values.values()) + max_ev
        for _ in range(50):
            shift = (lo + hi) / 2
            total = sum(max(-max_ev, min(max_ev, gain-shift))
                        for gain in values.values())
            if total > 0:
                lo = shift
            else:
                hi = shift
        center = (lo + hi) / 2
        for page in component:
            result[page] = max(-max_ev, min(max_ev, values[page] - center))
    return result


def balance_uv_seams(obj: Path, mtl: Path, atlases: dict[str, Path], *,
                     protect_black: int, max_ev: float,
                     min_anchors: int) -> tuple[dict[str, float], dict[str, Any]]:
    """Return bounded page EVs and auditable seam-pair diagnostics.

    No correction is made without enough consistent shared mesh vertices.
    Duplicated/disconnected OBJ vertex indices are intentionally not guessed
    to be neighbors: that is preferable to altering unrelated materials.
    """
    materials = _material_textures(mtl, set(atlases))
    page_ids = {name: index+1 for index, name in enumerate(sorted(atlases))}
    sampled, candidates = _shared_vertex_sample(obj, materials, page_ids)
    # Validate the UV indexing even for meshes without cross-page anchors.
    # Malformed imports must not silently pass through a requested seam QA.
    coordinates = _sample_uv_near_boundary(obj, materials, sampled)
    luminance = _sample_luma_pages(atlases, coordinates, protect_black)
    differences: dict[tuple[str, str], list[float]] = defaultdict(list)
    for pages in luminance.values():
        if len(pages) > _MAX_PAGES_PER_VERTEX:
            continue  # Non-manifold joins are unreliable seam evidence.
        names = sorted(pages)
        for index, a in enumerate(names):
            for b in names[index+1:]:
                differences[(a, b)].append(math.log2(pages[b] / pages[a]))
    edges = []
    pairs = []
    for (a, b), diffs in sorted(differences.items()):
        median = statistics.median(diffs)
        mad = statistics.median(abs(value-median) for value in diffs)
        reason = None
        if len(diffs) < min_anchors:
            reason = "insufficient-shared-anchors"
        elif mad > _MAX_PAIR_MAD_EV:
            reason = "inconsistent-seam-brightness"
        else:
            edges.append((a, b, median, len(diffs)))
        pairs.append({"atlas_a": a, "atlas_b": b, "valid_anchors": len(diffs),
                      "median_gap_ev_before": round(median, 5),
                      "mad_ev": round(mad, 5), "used": reason is None,
                      "excluded_reason": reason})
    offsets = _solve_page_offsets(edges, max_ev)
    for pair in pairs:
        pair["estimated_gap_ev_after"] = round(
            pair["median_gap_ev_before"]
            - offsets.get(pair["atlas_a"], 0.0)
            + offsets.get(pair["atlas_b"], 0.0), 5)
    warnings = []
    if not edges:
        warnings.append("No reliable shared cross-page UV seams; page exposure unchanged")
    unconnected = len(set(atlases) - set(offsets))
    if edges and unconnected:
        warnings.append(f"{unconnected} atlas page(s) lack supported UV seam connections")
    if candidates > len(sampled):
        warnings.append(f"Analyzed a deterministic reservoir of {len(sampled)} seam vertices")
    if any(not pair["used"] for pair in pairs):
        warnings.append("Some atlas pairs lack consistent/adequate seam evidence; inspect pair diagnostics")
    return offsets, {"enabled": True, "candidate_vertices": candidates,
                     "sampled_vertices": len(sampled), "valid_vertex_samples": len(luminance),
                     "min_anchors_per_pair": min_anchors, "max_page_ev": max_ev,
                     "pairs": pairs, "warnings": warnings}
