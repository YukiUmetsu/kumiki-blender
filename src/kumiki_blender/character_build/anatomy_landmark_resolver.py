"""Resolve verified-view 3D landmark references into native anatomical part plans.

Source landmark positions are inserted only at declared geometric control
point slots. No implicit symmetry, guessed eye depth or arbitrary expression
evaluation. Output retains immutable provenance and requires visual review.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .anatomy_plan import validate_anatomy_spec
from .landmark_triangulation import load_triangulated_landmarks


def _checksum(path: Path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _resolve_one(value, report: dict, evidence: dict, used: dict):
    if not isinstance(value, dict):
        return value
    if set(value) != {"landmark"} or not isinstance(value["landmark"], str):
        raise ValueError("Anatomical 3D control point must be numeric or {landmark: ID}")
    lid = value["landmark"]
    candidates = {row["id"]: row for row in report["landmarks"]}
    if lid not in candidates:
        raise ValueError(f"Missing independently triangulated landmark {lid}")
    row = candidates[lid]
    if row.get("status") != "triangulated-candidate-source-registration-review-required":
        raise ValueError(f"Landmark {lid} has not passed source triangulation checks")
    if evidence.get("status") == "registered-verified":
        # Do not label a geometry control point source-verified if its actual
        # supporting registered camera observations exclude the cited views.
        if not set(evidence.get("source_views", [])) <= set(row["views"]):
            raise ValueError(f"Anatomy's source views do not support landmark {lid}")
    used[lid] = {
        "position_m": row["position_m"],
        "reprojection_error_px": row["max_reprojection_error_px"],
        "source_views": row["views"],
        "camera_condition_pivot_ratio": row["camera_condition_pivot_ratio"],
    }
    return list(row["position_m"])


def resolve_anatomy_landmarks(template: str | Path, landmarks: str | Path,
                             output: str | Path) -> dict:
    source = Path(template).resolve()
    reference = Path(landmarks).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    record = load_triangulated_landmarks(reference)
    if not isinstance(data, dict) or data.get("character_id") != record["character_id"]:
        raise ValueError("Anatomy template and landmark reference have different character IDs")
    if not isinstance(data.get("parts"), list):
        raise ValueError("Anatomy template needs its editable part list")
    resolved = deepcopy(data)
    used: dict = {}
    for part in resolved["parts"]:
        if not isinstance(part, dict):
            raise ValueError("Invalid anatomy part")
        kind = part.get("kind")
        evidence = part.get("evidence", {})
        if kind == "ocular_globe":
            part["center_m"] = _resolve_one(part.get("center_m"), record, evidence, used)
        elif kind == "swept_tube":
            points = part.get("control_points_m")
            if not isinstance(points, list):
                raise ValueError("Tube control points must be a list")
            part["control_points_m"] = [
                _resolve_one(v, record, evidence, used) for v in points]
        elif kind == "contour_loft":
            rings = part.get("sections_m")
            if not isinstance(rings, list) or any(not isinstance(ring, list) for ring in rings):
                raise ValueError("Loft contour stations must be nested point lists")
            part["sections_m"] = [
                [_resolve_one(point, record, evidence, used) for point in ring]
                for ring in rings
            ]
        else:
            raise ValueError(f"Unknown anatomical geometry kind: {kind}")
    validate_anatomy_spec(resolved)
    if not used:
        raise ValueError("Anatomy template contains no landmark references to resolve")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Resolved anatomy revision already exists: {destination}")
    receipt = {
        "schema_version": 1,
        "character_id": record["character_id"],
        "anatomy_template": str(source),
        "anatomy_template_sha256": _checksum(source),
        "landmark_report": str(reference),
        "landmark_report_sha256": _checksum(reference),
        "source_image_sha256": {
            view: meta["sha256"] for view, meta in record["registered_views"].items()
        },
        "used_landmarks": used,
        "status": "resolved-anatomy-candidate-review-required",
        "limitations": [
            "The source image registration is an upstream artist assertion, not automatically verified here.",
            "Only explicitly annotated world-space landmarks are substituted; hidden anatomy is never guessed.",
            "The output is a candidate and still needs original source mask comparison and posed visual review.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    (destination / "resolved-anatomy.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    (destination / "landmark-provenance.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
