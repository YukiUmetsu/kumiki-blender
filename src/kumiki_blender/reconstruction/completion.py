"""Source-bound candidate geometry for historical and cinematic completion.

The planner never runs an AI model or edits a mesh. A generator adapter may
produce a GLB independently; only a pinned local candidate enters this stage.
"""

from __future__ import annotations

import json
import itertools
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..asset_import import AssetImportError, inspect_source_glb
from .head_base import HeadFitError, HeadLandmark, fit_landmark_pairs


class CompletionError(ValueError):
    """Unsafe completion evidence, candidate or geometry alignment."""


Vec3 = tuple[float, float, float]
_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{1,79}\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9_.-]{0,95})?\Z")
_OBSERVED = {"observed-scan", "observed-survey"}
_INFERRED = {"generated", "artist-hypothesis"}


@dataclass(frozen=True)
class CompletionAnchor:
    id: str
    source_xyz_m: Vec3
    target_xyz_m: Vec3


@dataclass(frozen=True)
class CompletionRegion:
    id: str
    candidate_glb: Path
    candidate_sha256: str
    origin: str
    generator: dict[str, str] | None
    source_url: str | None
    license: str | None
    max_error_m: float
    anchors: tuple[CompletionAnchor, ...]


@dataclass(frozen=True)
class CompletionPlan:
    asset_id: str
    mode: str
    base_glb: Path
    base_sha256: str
    base_origin: str
    regions: tuple[CompletionRegion, ...]


def _slug(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) > 96 or not _SLUG.fullmatch(value):
        raise CompletionError(f"{name} requires a lowercase hyphenated slug")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CompletionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _vec3(value: Any, name: str) -> Vec3:
    if (not isinstance(value, list) or len(value) != 3
            or any(type(number) not in (int, float) or abs(number) > 1e7
                   or not math.isfinite(number) for number in value)):
        raise CompletionError(f"{name} must contain three finite metre coordinates")
    return tuple(float(number) for number in value)  # type: ignore[return-value]


def _file(root: Path, value: Any, name: str, sha256: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CompletionError(f"{name} must be a relative file")
    relative = Path(value)
    if relative.is_absolute() or any(part in (".", "..") for part in relative.parts):
        raise CompletionError(f"{name} escapes the manifest directory")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise CompletionError(f"{name} escapes the manifest directory")
    if result.suffix.lower() != ".glb" or not result.is_file():
        raise CompletionError(f"{name} must name an existing GLB")
    try:
        report = inspect_source_glb(result)
    except AssetImportError as error:
        raise CompletionError(f"{name}: {error}") from error
    if report["input_sha256"] != sha256:
        raise CompletionError(f"{name} SHA-256 does not match the source bytes")
    return result


def _embedded_geometry_origins(path: Path) -> set[str]:
    """Inspect source-node glTF extras to prevent evidence-label laundering."""
    with path.open("rb") as stream:
        stream.seek(12)
        length, kind = struct.unpack("<I4s", stream.read(8))
        if kind != b"JSON" or length > 16 * 1024 * 1024:
            raise CompletionError("Base GLB JSON is invalid")
        data = json.loads(stream.read(length))
    origins = set()
    for node in data.get("nodes", []):
        if not isinstance(node, dict) or not isinstance(node.get("extras"), dict):
            continue
        origin = node["extras"].get("kumiki_geometry_origin")
        if origin is None:
            continue
        if origin not in ("observed", "inferred"):
            raise CompletionError("Base GLB has an unknown geometry origin tag")
        origins.add(origin)
    return origins


def _source_url(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise CompletionError("observational candidate needs a source_url")
    try:
        url = urlsplit(value)
    except ValueError as error:
        raise CompletionError(f"Invalid source_url: {error}") from error
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or any(ord(c) < 32 for c in value)):
        raise CompletionError("source_url must be a public HTTPS URL without credentials or query")
    return value


def _exact(raw: Any, required: set[str], optional: set[str], label: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or not required <= raw.keys() or raw.keys() - required - optional:
        raise CompletionError(f"Invalid {label} schema")
    return raw


def load_completion_plan(path: str | Path) -> CompletionPlan:
    """Validate all local sources and candidate metadata without writing files."""
    path = Path(path)
    try:
        if path.stat().st_size > 128 * 1024:
            raise CompletionError("Completion manifest exceeds 128 KiB")
        raw = json.loads(path.read_text(encoding="utf-8"),
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, ValueError, UnicodeError) as error:
        if isinstance(error, CompletionError):
            raise
        raise CompletionError(f"Cannot read completion manifest: {error}") from error
    if isinstance(raw, dict) and "base_origin" not in raw:
        raise CompletionError("base_origin must explicitly identify observed or inferred geometry")
    data = _exact(raw, {"schema_version", "asset_id", "mode", "base_origin", "base_glb",
                        "base_sha256", "regions"}, set(), "completion")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise CompletionError("Invalid completion schema_version")
    asset_id = _slug(data["asset_id"], "asset_id")
    mode = data["mode"]
    if mode not in ("historical", "cinematic"):
        raise CompletionError("mode must be historical or cinematic")
    base_origin = data["base_origin"]
    if base_origin not in ("observed", "inferred"):
        raise CompletionError("base_origin must be observed or inferred")
    if mode == "historical" and base_origin == "inferred":
        raise CompletionError("historical mode cannot use an inferred base GLB")
    base_sha = _sha(data["base_sha256"], "base_sha256")
    base = _file(path.parent, data["base_glb"], "base_glb", base_sha)
    if "inferred" in _embedded_geometry_origins(base):
        if mode == "historical":
            raise CompletionError("historical mode cannot reuse a base containing inferred geometry")
        if base_origin == "observed":
            raise CompletionError("base_origin observed conflicts with embedded inferred geometry")
    items = data["regions"]
    if not isinstance(items, list) or not 1 <= len(items) <= 16:
        raise CompletionError("regions must contain 1–16 candidate regions")
    regions: list[CompletionRegion] = []
    for index, item in enumerate(items):
        item = _exact(item, {"id", "candidate_glb", "candidate_sha256", "origin",
                             "max_error_m", "anchors"}, {"generator", "source_url", "license"},
                      f"region {index}")
        region_id = _slug(item["id"], "region id")
        origin = item["origin"]
        if not isinstance(origin, str) or origin not in _OBSERVED | _INFERRED:
            raise CompletionError(f"{region_id}: unknown origin")
        digest = _sha(item["candidate_sha256"], f"{region_id} candidate_sha256")
        candidate = _file(path.parent, item["candidate_glb"], f"{region_id} candidate_glb", digest)
        if origin in _OBSERVED and "inferred" in _embedded_geometry_origins(candidate):
            raise CompletionError(f"{region_id}: observed candidate contains inferred geometry tags")
        generator = item.get("generator")
        if origin == "generated" and generator is None:
            raise CompletionError(f"{region_id}: generated geometry requires generator provenance")
        if generator is not None:
            generator = _exact(generator, {"adapter", "model_id"}, set(), "generator")
            adapter = _slug(generator["adapter"], "adapter")
            model_id = generator["model_id"]
            if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
                raise CompletionError(f"{region_id}: model_id must be an identifiable local/model repository name")
            generator = {"adapter": adapter, "model_id": model_id}
        source_url = item.get("source_url")
        license_id = item.get("license")
        if origin in _OBSERVED:
            source_url = _source_url(source_url)
            if not isinstance(license_id, str) or not _LICENSE.fullmatch(license_id):
                raise CompletionError(f"{region_id}: observed geometry requires a license")
        elif source_url is not None:
            source_url = _source_url(source_url)
        if license_id is not None and (not isinstance(license_id, str)
                                        or not _LICENSE.fullmatch(license_id)):
            raise CompletionError(f"{region_id}: invalid license")
        tolerance = item["max_error_m"]
        if (type(tolerance) not in (int, float) or not 1e-6 <= tolerance <= 1.0
                or not math.isfinite(tolerance)):
            raise CompletionError(f"{region_id}: max_error_m must be in [1e-6, 1] metres")
        anchors = item["anchors"]
        if not isinstance(anchors, list) or not 4 <= len(anchors) <= 32:
            raise CompletionError(f"{region_id}: 4–32 landmarks required")
        parsed: list[CompletionAnchor] = []
        for anchor in anchors:
            anchor = _exact(anchor, {"id", "source_xyz_m", "target_xyz_m"}, set(), "anchor")
            parsed.append(CompletionAnchor(
                _slug(anchor["id"], "anchor id"),
                _vec3(anchor["source_xyz_m"], "source_xyz_m"),
                _vec3(anchor["target_xyz_m"], "target_xyz_m"),
            ))
        if len({anchor.id for anchor in parsed}) != len(parsed):
            raise CompletionError(f"{region_id}: duplicate landmark ids")
        regions.append(CompletionRegion(region_id, candidate, digest, origin, generator,
                                        source_url, license_id, float(tolerance), tuple(parsed)))
    if len({region.id for region in regions}) != len(regions):
        raise CompletionError("Duplicate region ids")
    return CompletionPlan(asset_id, mode, base, base_sha, base_origin, tuple(regions))


def verify_completion_sources(plan: CompletionPlan) -> None:
    """Recheck actual bytes immediately before passing paths to Blender."""
    sources = [("base_glb", plan.base_glb, plan.base_sha256)]
    sources.extend((region.id, region.candidate_glb, region.candidate_sha256)
                   for region in plan.regions)
    for name, path, digest in sources:
        try:
            actual = inspect_source_glb(path)["input_sha256"]
        except AssetImportError as error:
            raise CompletionError(f"{name}: {error}") from error
        if actual != digest:
            raise CompletionError(f"{name} SHA-256 changed since planning")


def evaluate_completion(plan: CompletionPlan) -> dict[str, Any]:
    """Fit each region and apply mode policy without altering input geometry."""
    reports = []
    for region in plan.regions:
        source = [item.source_xyz_m for item in region.anchors]
        target = [item.target_xyz_m for item in region.anchors]
        for label, points in (("source", source), ("target", target)):
            span = max(math.dist(a, b) for a, b in itertools.combinations(points, 2))
            volume = max(abs(
                (b[0]-a[0]) * ((c[1]-a[1])*(d[2]-a[2])-(c[2]-a[2])*(d[1]-a[1]))
                + (b[1]-a[1]) * ((c[2]-a[2])*(d[0]-a[0])-(c[0]-a[0])*(d[2]-a[2]))
                + (b[2]-a[2]) * ((c[0]-a[0])*(d[1]-a[1])-(c[1]-a[1])*(d[0]-a[0]))
            ) for a, b, c, d in itertools.combinations(points, 4))
            if span <= 1e-7 or volume <= 1e-6 * span**3:
                raise CompletionError(f"{region.id}: {label} landmarks are coplanar or degenerate")
        # Signed tetrahedron orientation is invariant under proper rotation and
        # positive scale, even for thin-but-admitted configurations. The older
        # anatomical fitter uses a looser mirror threshold; do not rely on it.
        def signed_volume(points: list[Vec3], indices: tuple[int, int, int, int]) -> float:
            a, b, c, d = (points[index] for index in indices)
            return (
                (b[0]-a[0])*((c[1]-a[1])*(d[2]-a[2])-(c[2]-a[2])*(d[1]-a[1]))
                + (b[1]-a[1])*((c[2]-a[2])*(d[0]-a[0])-(c[0]-a[0])*(d[2]-a[2]))
                + (b[2]-a[2])*((c[0]-a[0])*(d[1]-a[1])-(c[1]-a[1])*(d[0]-a[0]))
            )
        stable_tetra = max(itertools.combinations(range(len(source)), 4),
                           key=lambda ids: abs(signed_volume(source, ids)))
        if signed_volume(source, stable_tetra) * signed_volume(target, stable_tetra) < 0:
            raise CompletionError(f"{region.id}: target landmarks mirror candidate geometry")
        landmarks = tuple(HeadLandmark(
            anchor.id, region.id, anchor.source_xyz_m, anchor.target_xyz_m, 1.0
        ) for anchor in region.anchors)
        try:
            fit = fit_landmark_pairs(landmarks, asset_id=plan.asset_id,
                                     input_sha256=region.candidate_sha256)
        except HeadFitError as error:
            raise CompletionError(f"{region.id}: {error}") from error
        if fit.max_error_m > region.max_error_m:
            raise CompletionError(
                f"{region.id}: maximum anchor residual {fit.max_error_m:.6f}m exceeds "
                f"the declared tolerance {region.max_error_m:.6f}m"
            )
        is_observed = region.origin in _OBSERVED
        included = plan.mode == "cinematic" or is_observed
        reports.append({
            "id": region.id, "candidate_sha256": region.candidate_sha256,
            "origin": region.origin,
            "geometry_origin": "observed" if is_observed else "inferred",
            "included": included,
            "reason": ("included as sourced observational geometry" if is_observed else
                       "included as inferred cinematic geometry" if included else
                       "excluded: historical mode forbids inferred geometry"),
            "generator": region.generator, "source_url": region.source_url,
            "license": region.license,
            "rms_error_m": fit.rms_error_m, "max_error_m": fit.max_error_m,
            "transform": {"scale": fit.scale, "rotation_3x3": [list(row) for row in fit.rotation],
                          "translation_m": list(fit.translation_m)},
            "anchor_errors": list(fit.landmark_errors),
        })
    return {
        "schema_version": 1, "asset_id": plan.asset_id, "mode": plan.mode,
        "status": "completion-candidate / artistic-review-required",
        "base_glb_sha256": plan.base_sha256, "base_origin": plan.base_origin,
        "historical_fidelity_verified": False,
        "regions": reports,
        "limitations": "Candidate labels and source URLs are declarations, not authentication; visual and historical review required.",
    }
