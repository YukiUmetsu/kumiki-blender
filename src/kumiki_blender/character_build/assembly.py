"""Validate a part-assembly plan and fit each part's placement.

Plan (JSON, paths relative to the plan file)::

    {"schema_version": 1,
     "body": "blockout_fitted.ply",
     "join_voxel_m": 0.002,
     "parts": [{"id": "glasses", "file": "parts/glasses.glb", "join": "separate",
                "anchors": [{"part": [x, y, z], "target": [x, y, z]}, ...]}]}

Each part needs 3 or more non-collinear anchor pairs: a point on the part, in
the part file's own coordinates (Blender axes), and the same point on the
character. ``join`` is ``separate`` (its own object, e.g. glasses) or
``union`` (merged into the body by voxel remesh, e.g. a head sculpted alone).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..reconstruction.cameras import fit_similarity
from ..reconstruction.references import ReferenceError

MESH_SUFFIXES = (".obj", ".ply", ".glb")


def _file(root: Path, value, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"{label} must be a relative path inside the plan folder")
    path = (root / value).resolve()
    if not path.is_file() or not path.is_relative_to(root.resolve()) or path.suffix.lower() not in MESH_SUFFIXES:
        raise ValueError(f"{label}: missing or unsupported mesh file {value!r}")
    return path


def load_assembly(plan_path: Path, max_rms_m: float = 0.01) -> dict:
    """Validate the plan and return it with resolved paths and fitted 4x4 matrices."""
    root = plan_path.parent
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or not isinstance(raw.get("parts"), list):
        raise ValueError("Assembly plan needs schema_version 1 and a parts list")
    voxel = raw.get("join_voxel_m", 0.002)
    if not isinstance(voxel, (int, float)) or not 0.0002 <= voxel <= 0.05:
        raise ValueError("join_voxel_m must be between 0.0002 and 0.05")
    plan = {"body": str(_file(root, raw.get("body"), "body")), "join_voxel_m": float(voxel), "parts": []}
    seen = set()
    for index, part in enumerate(raw["parts"]):
        if not isinstance(part, dict):
            raise ValueError(f"Part {index} must be an object")
        pid = part.get("id")
        if not isinstance(pid, str) or not pid.replace("_", "").replace("-", "").isalnum() or pid in seen:
            raise ValueError(f"Part {index} needs a unique id of letters, digits, - or _")
        seen.add(pid)
        if part.get("join", "separate") not in ("separate", "union"):
            raise ValueError(f"Part {pid}: join must be separate or union")
        anchors = part.get("anchors")
        if not isinstance(anchors, list) or len(anchors) < 3:
            raise ValueError(f"Part {pid} needs 3 or more anchor pairs")
        pairs = []
        for anchor in anchors:
            if not isinstance(anchor, dict) or set(anchor) != {"part", "target"}:
                raise ValueError(f"Part {pid}: each anchor needs exactly part and target")
            try:
                pair = (tuple(float(v) for v in anchor["part"]), tuple(float(v) for v in anchor["target"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Part {pid}: anchor points need numbers ({error})") from error
            if any(len(p) != 3 or not all(map(np.isfinite, p)) for p in pair):
                raise ValueError(f"Part {pid}: anchor points need 3 finite coordinates")
            pairs.append(pair)
        try:
            scale, rotation, translation, rms = fit_similarity(pairs)
        except ReferenceError as error:
            raise ValueError(f"Part {pid}: {error}") from error
        if rms > max_rms_m:
            raise ValueError(f"Part {pid}: anchors disagree by {rms:.4f} m RMS (limit {max_rms_m}). "
                             "Check that each pair names the same point.")
        matrix = [[scale * rotation[r][c] for c in range(3)] + [translation[r]] for r in range(3)] + [[0, 0, 0, 1]]
        plan["parts"].append({"id": pid, "file": str(_file(root, part.get("file"), f"part {pid}")),
                              "join": part.get("join", "separate"), "matrix": matrix,
                              "scale": scale, "rms_m": rms})
    if not plan["parts"]:
        raise ValueError("Assembly plan has no parts")
    return plan
