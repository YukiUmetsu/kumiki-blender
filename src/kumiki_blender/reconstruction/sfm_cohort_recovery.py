"""Recover missing SfM views as isolated cohorts, then validate model alignment.

No object-specific photo names or geometry assumptions. Uses the original COLMAP
database/image IDs and never modifies the source reconstruction. The resulting
merged model is always a *review candidate*, not a production-ready mesh.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .sfm_pair_quality import assess_verified_pairs
from .sfm_track_support import audit_2d3d_support

MAX_IMAGE_ID = 2_147_483_647


@dataclass(frozen=True)
class CohortRecoverySettings:
    min_pair_inliers: int = 25
    max_anchors: int = 6
    max_trials: int = 5
    min_shared_cameras: int = 3
    min_new_cameras: int = 2
    min_anchor_spread_ratio: float = 0.04
    max_camera_residual_ratio: float = 0.08
    max_view_angle_degrees: float = 12.0
    max_landmark_residual_ratio: float = 0.05
    max_seconds_per_trial: int = 1200
    merge_max_reproj_error: float = 4.0
    min_seed_ray_angle_degrees: float = 1.0
    seed_strategy: str = "confidence"

    def validate(self) -> None:
        if (self.min_pair_inliers < 10 or
                self.seed_strategy not in ("confidence", "anchor-first") or
                not 3 <= self.max_anchors <= 100 or
                not 1 <= self.max_trials <= 50 or
                not 3 <= self.min_shared_cameras <= self.max_anchors or
                not 1 <= self.min_new_cameras <= 100 or
                not 0 < self.min_anchor_spread_ratio < 1 or
                not 0 < self.max_camera_residual_ratio < 1 or
                not 0 < self.max_view_angle_degrees < 90 or
                not 0 < self.max_landmark_residual_ratio < 1 or
                not 30 <= self.max_seconds_per_trial <= 86400 or
                not 0.5 <= self.merge_max_reproj_error <= 20 or
                not 0 <= self.min_seed_ray_angle_degrees <= 10):
            raise ValueError("Invalid cohort settings; check bounds and alignment thresholds")


def _sha256(path: Path) -> str:
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def _source_paths(revision: Path, model: int) -> dict[str, Path]:
    if not isinstance(model, int) or not 0 <= model <= 1000:
        raise ValueError("Model index must be 0..1000")
    root = revision.resolve()
    sparse = root / "sparse-reconstruction" if (root / "sparse-reconstruction").is_dir() else root
    paths = {"revision": root, "sparse": sparse, "database": sparse / "database.db",
             "images": sparse / "images", "model": sparse / "sparse" / str(model),
             "text": sparse / "text-models" / str(model) / "images.txt",
             "index": sparse / "staged-reference-index.json"}
    for name in ("database", "text", "index"):
        if paths[name].is_symlink() or not paths[name].is_file():
            raise FileNotFoundError(f"Missing or unsafe {name}: {paths[name]}")
    for name in ("images", "model"):
        if paths[name].is_symlink() or not paths[name].is_dir():
            raise FileNotFoundError(f"Missing or unsafe {name}: {paths[name]}")
    return paths


def _registered_names(text_model: Path) -> set[str]:
    names = set()
    lines = iter(text_model.read_text(encoding="utf-8").splitlines())
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10 or not fields[0].isdigit() or not fields[8].isdigit():
            raise ValueError(f"Malformed COLMAP model: {text_model}")
        name = fields[9]
        if name in names:
            raise ValueError(f"Duplicate registered photo: {name}")
        names.add(name)
        next(lines, None)  # COLMAP's second line contains 2D observations.
    if len(names) < 3:
        raise ValueError("Existing model requires at least three registered cameras")
    return names


def _read_database(database: Path, min_inliers: int) -> tuple[dict[str, int], list[dict]]:
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        by_id = dict(db.execute("SELECT image_id, name FROM images"))
        if len(by_id) != len(set(by_id.values())) or not by_id:
            raise ValueError("Duplicate or absent COLMAP database images")
        names = {name: image_id for image_id, name in by_id.items()}
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='two_view_geometries'").fetchone():
            raise ValueError("COLMAP database has no verified pair geometry")
        edges = []
        for pair_id, count, config in db.execute(
                "SELECT pair_id, rows, config FROM two_view_geometries"):
            if not count or count < min_inliers or config in (0, 1):
                continue
            left, right = divmod(pair_id, MAX_IMAGE_ID)
            if left in by_id and right in by_id and left != right:
                edges.append({"a": by_id[left], "b": by_id[right],
                              "inliers": count, "geometry_config": config})
    return names, edges


def _coverage_gaps(missing: set[str], registered: set[str],
                   anchors: set[str], edges: list[dict],
                   track_support: dict[str, dict]) -> list[dict]:
    """Find missing-view groups and rank evidence-based bridge-shot targets.

    Graph connectivity alone cannot establish camera geometry. Recommendations
    intentionally avoid claiming an inferred camera direction or surface.
    """
    neighbors = {name: set() for name in missing}
    for edge in edges:
        a, b = edge["a"], edge["b"]
        if a in missing and b in missing:
            neighbors[a].add(b)
            neighbors[b].add(a)
    groups = []
    visited = set()
    for name in sorted(missing):
        if name in visited:
            continue
        group = {name}
        todo = [name]
        visited.add(name)
        while todo:
            for adjacent in neighbors[todo.pop()]:
                if adjacent not in visited:
                    visited.add(adjacent)
                    group.add(adjacent)
                    todo.append(adjacent)
        bridge_edges = []
        within = []
        for edge in edges:
            a, b = edge["a"], edge["b"]
            if a in group and b in group:
                within.append(edge)
            elif (a in group and b in registered) or (b in group and a in registered):
                bridge_edges.append(edge)
        bridge_edges.sort(key=lambda pair: (-pair["seed_score"], -pair["inliers"],
                                           pair["a"], pair["b"]))
        registered_neighbors = sorted({
            edge["b"] if edge["a"] in group else edge["a"]
            for edge in bridge_edges})
        best = bridge_edges[0] if bridge_edges else None
        best_missing = (best["a"] if best["a"] in group else best["b"]) if best else sorted(group)[0]
        best_registered = (best["b"] if best["a"] in group else best["a"]) if best else None
        support = {name: track_support.get(name, {"status": "unavailable"})
                   for name in sorted(group)}
        supported = [name for name in sorted(group) if any(
            edge["a"] == name or edge["b"] == name for edge in bridge_edges)]
        if supported:
            best_missing = sorted(supported, key=lambda name: (
                -(support[name].get("conservative_one_to_one_2d3d") or 0), name))[0]
            connecting = [edge for edge in bridge_edges if best_missing in (edge["a"], edge["b"])]
            if connecting:
                best_registered = (connecting[0]["b"] if connecting[0]["a"] == best_missing
                                   else connecting[0]["a"])
        available_counts = [entry.get("conservative_one_to_one_2d3d") for entry in support.values()
                            if isinstance(entry.get("conservative_one_to_one_2d3d"), int)]
        groups.append({
            "unregistered_filenames": sorted(group),
            "missing_cameras": len(group),
            "missing_to_missing_verified_pairs": len(within),
            "verified_pairs_to_registered": len(bridge_edges),
            "registered_2d_match_neighbors": registered_neighbors,
            "selected_anchor_neighbors": sorted(set(registered_neighbors) & anchors),
            "existing_3d_track_support": support,
            "best_verified_bridge_pair": best,
            "bridge_photo_targets": {"registered_reference": best_registered,
                                     "missing_reference": best_missing},
            "assessment": ("no-verified-registered-bridge" if not bridge_edges else
                           "single-registered-image-bridge" if len(registered_neighbors) < 2 else
                           "multiple-2d-bridges / camera-support-unverified"),
            "pnp_support_warning": ("existing-3d-tracks-insufficient-for-default-pnp"
                                    if available_counts and max(available_counts) < 30 else
                                    "existing-3d-track-support-unavailable" if not available_counts
                                    else None),
            "acquisition_guidance": (
                "Find additional real intermediate viewpoints with shared, stable OBJECT "
                "features visible in both the registered-reference and missing-reference "
                "views. Capture a sequence with overlapping views and meaningful baseline; "
                "repeated angles or background-only matches cannot recover missing depth."
            ),
        })
    groups.sort(key=lambda item: (item["verified_pairs_to_registered"],
                                  item["missing_to_missing_verified_pairs"],
                                  item["unregistered_filenames"]))
    for index, item in enumerate(groups, 1):
        item["group"] = f"gap-{index:02d}"
    return groups


def _verify_staged_images(paths: dict[str, Path], db_names: set[str]) -> dict[str, dict]:
    staged = json.loads(paths["index"].read_text(encoding="utf-8"))
    entries = staged.get("images")
    if not isinstance(entries, list):
        raise ValueError("Staged index requires an images list")
    hashes = {}
    records = {}
    for item in entries:
        name = item.get("filename")
        if (not isinstance(name, str) or not name or
                Path(name).name != name or name in hashes or
                not isinstance(item.get("sha256"), str)):
            raise ValueError("Unsafe or duplicate staged photo entry")
        hashes[name] = item["sha256"]
        records[name] = item
    if set(hashes) != db_names:
        raise ValueError("Staged index and database must contain the same images")
    for name in sorted(db_names):
        photo = paths["images"] / name
        if photo.is_symlink() or not photo.is_file() or _sha256(photo) != hashes[name]:
            raise ValueError(f"Missing, linked or modified staged image: {name}")
    return records


def plan_cohort_recovery(revision: Path, *, model: int = 0,
                         targets: list[str] | None = None,
                         preferred_anchors: list[str] | None = None,
                         settings: CohortRecoverySettings = CohortRecoverySettings()) -> dict:
    """Plan from verified feature edges, without importing PyCOLMAP or touching outputs."""
    settings.validate()
    paths = _source_paths(revision, model)
    registered = _registered_names(paths["text"])
    image_ids, edges = _read_database(paths["database"], settings.min_pair_inliers)
    staged = _verify_staged_images(paths, set(image_ids))
    if not registered <= image_ids.keys():
        raise ValueError("Source model cameras do not match the source database")
    chosen = set(image_ids) - registered if targets is None else set(targets)
    if not chosen or not chosen <= image_ids.keys():
        raise ValueError("Target images must exist in the source database")
    missing = chosen - registered
    if not missing:
        raise ValueError("At least one target must be unregistered in the source model")
    requested_anchors = set(preferred_anchors or []) | (chosen & registered)
    if not requested_anchors <= registered:
        raise ValueError("Preferred anchors must already be registered in the source model")
    if len(requested_anchors) > settings.max_anchors:
        raise ValueError("Number of requested anchors exceeds max_anchors")

    # Rank by direct overlap with targets first; use secondary links through
    # already selected anchors to find geometrically useful alignment bridges.
    adjacency: dict[str, dict[str, int]] = {name: {} for name in image_ids}
    for edge in edges:
        a, b, count = edge["a"], edge["b"], edge["inliers"]
        adjacency[a][b] = max(adjacency[a].get(b, 0), count)
        adjacency[b][a] = max(adjacency[b].get(a, 0), count)
    anchors = set(requested_anchors)
    while len(anchors) < settings.max_anchors:
        ranked = []
        for name in registered - anchors:
            direct = [adjacency[name][target] for target in missing if target in adjacency[name]]
            secondary = [adjacency[name][anchor] for anchor in anchors if anchor in adjacency[name]]
            if direct or secondary:
                ranked.append((len(direct), sum(direct), len(secondary), sum(secondary), name))
        if not ranked:
            break
        anchors.add(sorted(ranked, key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4]))[0][-1])

    cohort = missing | anchors
    evidence = assess_verified_pairs(paths["database"], edges)
    track_support = audit_2d3d_support(paths["sparse"], model, registered, missing)
    # Some legacy databases do not retain tentative matches. Do not mistake
    # a missing row for proof that a pair was never attempted.
    from .sfm_pair_diagnostics import diagnose_pairs
    from contextlib import closing
    with closing(sqlite3.connect(paths["database"].resolve().as_uri() + "?mode=ro", uri=True)) as db:
        has_matches = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='matches'").fetchone()
    pair_stages = (diagnose_pairs(paths["database"], paths["sparse"] / "text-models" / str(model) / "points3D.txt",
                                  registered, missing) if has_matches else [])
    coverage_gaps = _coverage_gaps(missing, registered, anchors, evidence, track_support)
    from .sfm_gap_guide import annotate_gap_requests, first_photos_to_find
    annotate_gap_requests(coverage_gaps, staged, anchors)
    seed_pairs = [edge for edge in evidence if edge["a"] in cohort and
                  edge["b"] in cohort and ({edge["a"], edge["b"]} & missing)]
    unsuitable = [edge for edge in seed_pairs if (
        edge["quality"]["median_ray_angle_degrees"] is not None and
        not edge["quality"].get("approximate_distortion", False) and
        edge["quality"]["median_ray_angle_degrees"] < settings.min_seed_ray_angle_degrees)]
    seed_pairs = [edge for edge in seed_pairs if edge not in unsuitable]
    # A strong missing/missing pair may reconstruct a beautiful, completely
    # disconnected model. During alignment retries, first initialize from a
    # verified registered/missing pair so original camera anchors have a chance
    # to join the cohort. Geometry/parallax checks stay unchanged.
    def seed_rank(edge: dict) -> tuple:
        anchored = ((edge["a"] in registered) != (edge["b"] in registered))
        return ((0 if anchored else 1) if settings.seed_strategy == "anchor-first" else 0,
                -edge["seed_score"], -edge["inliers"], edge["a"], edge["b"])

    seed_pairs.sort(key=seed_rank)
    # Diverse seeds avoid spending all trial time on near-duplicate screenshots.
    diverse = []
    seen = set()
    for edge in seed_pairs:
        opposite = {edge["a"], edge["b"]} & missing
        if opposite - seen:
            diverse.append(edge)
            seen |= opposite
        if len(diverse) >= settings.max_trials:
            break
    for edge in seed_pairs:
        if len(diverse) >= settings.max_trials:
            break
        if edge not in diverse:
            diverse.append(edge)
    status = ("no-verified-seed-pairs" if not diverse else
              "independent-only / too-few-shared-anchors" if len(anchors) < settings.min_shared_cameras
              else "ready-to-attempt")
    return {"schema_version": 1, "status": status,
            "first_photos_to_find": first_photos_to_find(coverage_gaps),
            "source_revision": str(paths["revision"]),
            "source_sparse": str(paths["sparse"]), "source_model_index": model,
            "source_database_sha256": _sha256(paths["database"]),
            "source_images": len(image_ids), "registered_seed_images": len(registered),
            "registered_seed_filenames": sorted(registered),
            "registered_seed_image_ids": {name: image_ids[name] for name in sorted(registered)},
            "target_filenames": sorted(chosen), "missing_target_filenames": sorted(missing),
            "anchor_filenames": sorted(anchors), "cohort_filenames": sorted(cohort),
            "verified_pair_count": len(edges), "seed_pairs": diverse,
            "rejected_low_parallax_seed_pairs": unsuitable,
            "coverage_gaps": coverage_gaps, "pair_stage_diagnostics": pair_stages,
            "source_image_ids": {name: image_ids[name] for name in sorted(cohort)},
            "settings": vars(settings),
            "warning": "Even estimated ray angles and 2D match coverage cannot prove a "
                       "consistent camera solve or subject-surface overlap; independently "
                       "reconstructed/merged models still require object-surface review."}


def annotate_failed_recovery(plan: dict, prior_report: Path | None) -> dict:
    """Correlate read-only camera/point outcomes with gaps for targeted acquisition."""
    if prior_report is None:
        return plan
    if (prior_report.is_symlink() or not prior_report.is_file() or
            prior_report.stat().st_size > 5_000_000):
        raise ValueError("Previous recovery must be a regular JSON report under 5 MB")
    previous = json.loads(prior_report.read_text(encoding="utf-8"))
    if (not isinstance(previous, dict) or
            previous.get("source_database_sha256") != plan["source_database_sha256"] or
            previous.get("source_model_index") != plan["source_model_index"]):
        raise ValueError("Previous recovery report does not match this source database/model")
    trials = previous.get("trials", [])
    if not isinstance(trials, list):
        raise ValueError("Previous recovery has invalid trials")
    for gap in plan["coverage_gaps"]:
        filenames = set(gap["unregistered_filenames"])
        outcomes = []
        for trial in trials:
            if not isinstance(trial, dict) or not isinstance(trial.get("seed_pair"), dict):
                continue
            pair = trial["seed_pair"]
            if not {pair.get("a"), pair.get("b")} & filenames:
                continue
            models = trial.get("models", [])
            if not isinstance(models, list):
                continue
            outcomes.append({"trial": trial.get("trial"),
                             "seed_pair": {"a": pair.get("a"), "b": pair.get("b")},
                             "model_registered_cameras": [m.get("registered_cameras")
                                                         for m in models if isinstance(m, dict)],
                             "model_sparse_points": [m.get("points3D") for m in models
                                                     if isinstance(m, dict)],
                             "shared_camera_counts": [m.get("alignment", {}).get("shared_cameras")
                                                      for m in models if isinstance(m, dict) and
                                                      isinstance(m.get("alignment"), dict)]})
        gap["previous_trial_results"] = outcomes
        if outcomes and not any(max((value or 0 for value in item["shared_camera_counts"]), default=0) >= 3
                                for item in outcomes):
            gap["previous_failure_evidence"] = "no-prior-trial-formed-3-shared-camera-anchors"
    plan["previous_recovery"] = {"status": previous.get("status"),
                                 "trials": len(trials),
                                 "report": str(prior_report.resolve())}
    return plan


def _registered_images(reconstruction) -> dict[str, tuple[int, object]]:
    return {reconstruction.images[image_id].name: (image_id, reconstruction.images[image_id])
            for image_id in reconstruction.reg_image_ids()}


def _similarity(source, target, *, min_spread_ratio: float = 0.000001):
    """Proper 3D similarity from candidate points to baseline, with spread checks."""
    import numpy as np

    src = np.asarray(source, dtype=float).reshape((-1, 3))
    dst = np.asarray(target, dtype=float).reshape((-1, 3))
    if len(src) < 3 or src.shape != dst.shape or not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Alignment requires at least three finite paired 3D positions")
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    x, y = src - src_mean, dst - dst_mean
    src_svd = np.linalg.svd(x, compute_uv=False)
    dst_svd = np.linalg.svd(y, compute_uv=False)
    if (src_svd[0] < 1e-9 or dst_svd[0] < 1e-9 or
            src_svd[1] / src_svd[0] < min_spread_ratio or
            dst_svd[1] / dst_svd[0] < min_spread_ratio):
        raise ValueError("Common camera centers are nearly coincident or collinear")
    u, singular, vt = np.linalg.svd(x.T @ y / len(src))
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ sign @ u.T
    scale = float(np.sum(singular * np.diag(sign)) / (np.sum(x * x) / len(src)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid alignment scale")
    translation = dst_mean - scale * (rotation @ src_mean)
    residuals = np.linalg.norm(scale * (src @ rotation.T) + translation - dst, axis=1)
    spread = float(np.sqrt(np.sum(y * y) / len(dst)))
    return {"rotation": rotation, "scale": scale, "translation": translation,
            "residuals": residuals, "spread": spread,
            "center_rms_ratio": float(np.sqrt(np.mean(residuals ** 2)) / spread),
            "center_max_ratio": float(max(residuals) / spread),
            "spread_ratio": float(min(src_svd[1] / src_svd[0], dst_svd[1] / dst_svd[0]))}


def assess_alignment(baseline, cohort, settings: CohortRecoverySettings,
                     landmarks: list[dict] | None = None) -> dict:
    """Fail closed on mixed image IDs, camera geometry and optional object landmarks."""
    import numpy as np

    original, alternate = _registered_images(baseline), _registered_images(cohort)
    common = sorted(original.keys() & alternate.keys())
    gained = sorted(alternate.keys() - original.keys())
    result = {"shared_filenames": common, "shared_cameras": len(common),
              "new_filenames": gained, "new_cameras": len(gained),
              "accepted": False, "reasons": []}
    if len(common) < settings.min_shared_cameras:
        result["reasons"].append("insufficient-shared-registered-cameras")
    if len(gained) < settings.min_new_cameras:
        result["reasons"].append("insufficient-new-cameras")
    if any(original[name][0] != alternate[name][0] for name in common):
        result["reasons"].append("image-id-collision-or-different-database")
    if result["reasons"]:
        return result
    try:
        source = [alternate[name][1].projection_center() for name in common]
        target = [original[name][1].projection_center() for name in common]
        fit = _similarity(source, target,
                          min_spread_ratio=settings.min_anchor_spread_ratio)
        angles = []
        for name in common:
            direction = np.asarray(alternate[name][1].viewing_direction(), dtype=float).reshape(3)
            reference = np.asarray(original[name][1].viewing_direction(), dtype=float).reshape(3)
            predicted = fit["rotation"] @ direction
            denom = np.linalg.norm(predicted) * np.linalg.norm(reference)
            if denom < 1e-9:
                raise ValueError("Camera viewing direction is degenerate")
            angles.append(float(np.degrees(np.arccos(np.clip(predicted @ reference / denom, -1, 1)))))
        result.update({"scale": fit["scale"], "anchor_spread_ratio": fit["spread_ratio"],
                       "camera_rms_residual_ratio": fit["center_rms_ratio"],
                       "camera_max_residual_ratio": fit["center_max_ratio"],
                       "max_view_angle_degrees": max(angles)})
        if fit["spread_ratio"] < settings.min_anchor_spread_ratio:
            result["reasons"].append("camera-anchors-poorly-distributed")
        if fit["center_max_ratio"] > settings.max_camera_residual_ratio:
            result["reasons"].append("common-camera-center-mismatch")
        if max(angles) > settings.max_view_angle_degrees:
            result["reasons"].append("common-camera-orientation-mismatch")
        if landmarks is not None:
            if len(landmarks) < 3 or len({x.get("id") for x in landmarks}) != len(landmarks):
                raise ValueError("At least three uniquely labeled object landmarks required")
            obj_source = [item["cohort_xyz"] for item in landmarks]
            obj_target = [item["baseline_xyz"] for item in landmarks]
            # Verify object points are spatially distributed, not just camera anchors.
            obj_fit = _similarity(obj_source, obj_target, min_spread_ratio=1e-6)
            predicted = fit["scale"] * (np.asarray(obj_source) @ fit["rotation"].T) + fit["translation"]
            errors = np.linalg.norm(predicted - np.asarray(obj_target), axis=1)
            ratio = float(max(errors) / obj_fit["spread"])
            result["object_landmarks"] = {"count": len(landmarks),
                                           "max_residual_ratio": ratio,
                                           "names": sorted(str(item["id"]) for item in landmarks)}
            if ratio > settings.max_landmark_residual_ratio:
                result["reasons"].append("object-landmark-mismatch")
    except (ValueError, KeyError, TypeError, FloatingPointError) as error:
        result["reasons"].append(f"invalid-geometric-evidence: {error}")
    result["accepted"] = not result["reasons"]
    return result


def assess_disconnected_alignment(baseline, cohort, database: Path, qa_index: Path | None,
                                  settings: CohortRecoverySettings) -> dict:
    """Use verified 3D point identities where no common registered camera exists."""
    import numpy as np
    from .sfm_point_alignment import (estimate_point_alignment,
                                      extract_verified_3d_matches, validate_cross_reprojection)

    first, second = _registered_images(baseline), _registered_images(cohort)
    gained = sorted(set(second) - set(first))
    result = {"method": "verified-3d-points", "accepted": False,
              "shared_cameras": 0, "shared_filenames": [],
              "new_cameras": len(gained), "new_filenames": gained, "reasons": []}
    if set(first) & set(second):
        result["reasons"].append("point-alignment-requires-no-common-registered-cameras")
    if len(gained) < settings.min_new_cameras:
        result["reasons"].append("insufficient-new-cameras")
    if qa_index is None:
        result["reasons"].append("subject-confidence-unavailable")
    if result["reasons"]:
        return result
    try:
        evidence_audit: dict = {}
        correspondences = extract_verified_3d_matches(
            database, baseline, cohort, qa_index=qa_index, audit=evidence_audit)
        result["evidence_audit"] = evidence_audit
        first_points = [np.asarray(item.xyz).reshape(3) for item in baseline.points3D.values()]
        other_points = [np.asarray(item.xyz).reshape(3) for item in cohort.points3D.values()]
        fit = estimate_point_alignment(correspondences,
                                       baseline_cloud=first_points, cohort_cloud=other_points)
        result.update(fit)
        if fit["accepted"]:
            reproj = validate_cross_reprojection(correspondences, fit, baseline, cohort)
            result["reprojection"] = reproj
            if not reproj["accepted"]:
                result["reasons"].append(reproj.get("reason") or "cross-model-reprojection-failed")
        # Finite camera positions/orientations are required even when only
        # point tracks generated the Sim(3); do not accept a degenerate pose.
        if fit["accepted"]:
            for name, (_, image) in second.items():
                center = np.asarray(image.projection_center(), dtype=float).reshape(3)
                forward = np.asarray(image.viewing_direction(), dtype=float).reshape(3)
                if (not np.isfinite(center).all() or not np.isfinite(forward).all() or
                        np.linalg.norm(forward) < 1e-9):
                    result["reasons"].append("invalid-cohort-camera-pose")
                    break
    except (ValueError, KeyError, OSError, sqlite3.DatabaseError, TypeError,
            np.linalg.LinAlgError) as error:
        result["reasons"].append(f"invalid-cross-model-evidence: {error}")
    result["accepted"] = not result["reasons"]
    return result


def assess_partial_alignment(baseline, cohort, database: Path, qa_index: Path | None,
                             settings: CohortRecoverySettings) -> dict:
    """Align one/two shared cameras using independently triangulated subject points.

    A single shared camera does not determine SfM scale. It only helps validate
    a Sim(3) estimated from original 2D/3D feature tracks; accepted new camera
    observations must independently reproject to the original statue surface.
    """
    import numpy as np
    from .sfm_point_alignment import (estimate_point_alignment,
                                      extract_verified_3d_matches, _subject_lookup,
                                      validate_cross_reprojection)

    first, second = _registered_images(baseline), _registered_images(cohort)
    common = sorted(set(first) & set(second))
    gained = sorted(set(second) - set(first))
    result = {"method": "verified-hybrid-3d-points", "accepted": False,
              "shared_cameras": len(common), "shared_filenames": common,
              "new_cameras": len(gained), "new_filenames": gained, "reasons": []}
    if not 1 <= len(common) <= 2:
        result["reasons"].append("hybrid-requires-one-or-two-shared-cameras")
    if len(gained) < settings.min_new_cameras:
        result["reasons"].append("insufficient-new-cameras")
    if any(first[name][0] != second[name][0] for name in common):
        result["reasons"].append("image-id-collision-or-different-database")
    if qa_index is None:
        result["reasons"].append("subject-confidence-unavailable")
    if result["reasons"]:
        return result
    try:
        audit: dict = {}
        matches = extract_verified_3d_matches(
            database, baseline, cohort, qa_index=qa_index,
            allow_shared=True, audit=audit)
        result["evidence_audit"] = audit
        fit = estimate_point_alignment(
            matches,
            baseline_cloud=[np.asarray(p.xyz).reshape(3) for p in baseline.points3D.values()],
            cohort_cloud=[np.asarray(p.xyz).reshape(3) for p in cohort.points3D.values()])
        result.update(fit)
        if not fit["accepted"]:
            return result
        reprojection = validate_cross_reprojection(matches, fit, baseline, cohort)
        result["reprojection"] = reprojection
        if not reprojection["accepted"]:
            result["reasons"].append(reprojection.get("reason") or "cross-model-reprojection-failed")
        rotation = np.asarray(fit["rotation"], dtype=float)
        translation = np.asarray(fit["translation"], dtype=float)
        scale = float(fit["scale"])
        base_centers = np.asarray([image.projection_center()
                                   for _, image in first.values()], dtype=float).reshape((-1, 3))
        spread = max(1e-9, float(np.sqrt(np.mean(np.sum(
            (base_centers - base_centers.mean(axis=0)) ** 2, axis=1)))))
        center_ratios, view_angles = [], []
        for name in common:
            base, other = first[name][1], second[name][1]
            mapped_center = scale * (rotation @ np.asarray(
                other.projection_center(), dtype=float).reshape(3)) + translation
            center_ratios.append(float(np.linalg.norm(mapped_center - np.asarray(
                base.projection_center(), dtype=float).reshape(3)) / spread))
            direction = rotation @ np.asarray(other.viewing_direction(), dtype=float).reshape(3)
            expected = np.asarray(base.viewing_direction(), dtype=float).reshape(3)
            denominator = float(np.linalg.norm(direction) * np.linalg.norm(expected))
            if denominator < 1e-9:
                result["reasons"].append("invalid-shared-camera-direction")
                break
            view_angles.append(float(np.degrees(np.arccos(np.clip(
                np.dot(direction, expected) / denominator, -1, 1)))))
        result["max_shared_center_residual_ratio"] = max(center_ratios, default=None)
        result["max_shared_view_angle_degrees"] = max(view_angles, default=None)
        if any(x > settings.max_camera_residual_ratio for x in center_ratios):
            result["reasons"].append("shared-camera-center-mismatch")
        if any(x > settings.max_view_angle_degrees for x in view_angles):
            result["reasons"].append("shared-camera-orientation-mismatch")

        # Independently verify that each new camera observes several *matched*
        # baseline object points. An aligned camera with zero usable tracks is
        # not evidence of another valid reconstruction viewpoint.
        subject = _subject_lookup(qa_index)
        new_ids = set(cohort.reg_image_ids()) - set(baseline.reg_image_ids())
        accepted_pairs = {tuple(ids) for ids in fit["inlier_point_ids"]}
        links, observed = [], {}
        errors_by_camera = {image_id: [] for image_id in new_ids}
        for row in matches:
            base_pid, co_pid = row["baseline_point_id"], row["cohort_point_id"]
            if (base_pid, co_pid) not in accepted_pairs:
                continue
            for item in cohort.points3D[co_pid].track.elements:
                image_id, index = int(item.image_id), int(item.point2D_idx)
                if image_id not in new_ids:
                    continue
                image = cohort.images[image_id]
                if index >= len(image.points2D):
                    continue
                point = image.points2D[index]
                if (not point.has_point3D() or int(point.point3D_id) != co_pid or
                        not subject(image.name, point.xy)):
                    continue
                projected = image.project_point(
                    rotation.T @ ((np.asarray(row["baseline_xyz"]) - translation) / scale))
                if projected is None:
                    continue
                error = float(np.linalg.norm(np.asarray(projected).reshape(2) -
                                             np.asarray(point.xy).reshape(2)))
                if not math.isfinite(error):
                    continue
                observation = (image_id, index)
                if observation in observed and observed[observation] != base_pid:
                    result["reasons"].append("ambiguous-new-camera-observation")
                    continue
                observed[observation] = base_pid
                errors_by_camera[image_id].append(error)
                links.append([base_pid, co_pid, image_id, index])
        support = {str(image_id): {"count": len(errors),
                                   "median_px": float(np.median(errors)) if errors else None,
                                   "p95_px": float(np.percentile(errors, 95)) if errors else None}
                   for image_id, errors in errors_by_camera.items()}
        result["new_camera_3d_support"] = support
        result["verified_new_camera_observations"] = links
        if any(item["count"] < 8 for item in support.values()):
            result["reasons"].append("insufficient-verified-new-camera-3d-support")
        if any(item["count"] and (item["median_px"] > 3.0 or item["p95_px"] > 6.0)
               for item in support.values()):
            result["reasons"].append("new-camera-cross-reprojection-mismatch")
    except (ValueError, KeyError, OSError, sqlite3.DatabaseError, TypeError,
            np.linalg.LinAlgError) as error:
        result["reasons"].append(f"invalid-hybrid-evidence: {error}")
    result["accepted"] = not result["reasons"]
    return result


def _write_report(output: Path, report: dict) -> None:
    (output / "cohort-recovery-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_landmarks(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Landmark manifest is absent or a symbolic link")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("landmarks"), list):
        raise ValueError("Landmark manifest must contain an object with landmarks list")
    landmarks = data["landmarks"]
    if len(landmarks) < 3:
        raise ValueError("At least three physical object landmarks are required")
    seen = set()
    for item in landmarks:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("Every landmark needs a nonempty string id")
        if item["id"] in seen:
            raise ValueError(f"Duplicate object landmark: {item['id']}")
        seen.add(item["id"])
        for key in ("baseline_xyz", "cohort_xyz"):
            coords = item.get(key)
            if (not isinstance(coords, list) or len(coords) != 3 or
                    any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value) for value in coords)):
                raise ValueError(f"Landmark {item['id']} requires finite {key} coordinates")
    return landmarks


def inspect_cohort_alignment(revision: Path, candidate_model: Path, *, model: int = 0,
                             landmarks_path: Path | None = None,
                             settings: CohortRecoverySettings = CohortRecoverySettings(),
                             pycolmap_module=None) -> dict:
    """Review an existing, specific cohort frame with optional physical landmarks.

    Landmark coordinates are tied to a *particular* independently solved model;
    they cannot safely be reused across seeds with different coordinate frames.
    """
    settings.validate()
    paths = _source_paths(revision, model)
    if candidate_model.is_symlink() or not candidate_model.is_dir():
        raise FileNotFoundError("Existing candidate model must be a real directory")
    landmarks = _load_landmarks(landmarks_path)
    if pycolmap_module is None:
        import pycolmap as pycolmap_module
    baseline = pycolmap_module.Reconstruction(paths["model"])
    alternate = pycolmap_module.Reconstruction(candidate_model)
    shared = len(set(_registered_images(baseline)) & set(_registered_images(alternate)))
    qa_index = paths["sparse"] / "subject-feature-index.json"
    if shared == 0:
        qa = assess_disconnected_alignment(baseline, alternate, paths["database"],
                                           qa_index if qa_index.is_file() else None, settings)
    elif shared < settings.min_shared_cameras:
        qa = assess_partial_alignment(baseline, alternate, paths["database"],
                                      qa_index if qa_index.is_file() else None, settings)
    else:
        qa = assess_alignment(baseline, alternate, settings, landmarks)
    qa.update({"schema_version": 1, "baseline": str(paths["model"]),
               "candidate": str(candidate_model.resolve()),
               "landmark_manifest": str(landmarks_path) if landmarks_path else None,
               "status": "alignment-evidence-passes / visual-review-required" if qa["accepted"]
                         else "alignment-rejected"})
    return qa


def recover_cohort(revision: Path, output: Path, *, model: int = 0,
                   targets: list[str] | None = None,
                   preferred_anchors: list[str] | None = None,
                   settings: CohortRecoverySettings = CohortRecoverySettings(),
                   colmap: str = "colmap",
                   pycolmap_module=None, merge_runner=None) -> dict:
    """Reconstruct separate camera cohorts; merge only after independently validated overlap."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a new output directory: {output}")
    planned = plan_cohort_recovery(revision, model=model, targets=targets,
                                   preferred_anchors=preferred_anchors, settings=settings)
    if planned["status"] == "no-verified-seed-pairs":
        raise ValueError("No strong verified pair can initialize the selected cohort")
    if pycolmap_module is None:
        import pycolmap as pycolmap_module
    if not hasattr(pycolmap_module, "incremental_mapping"):
        raise RuntimeError("PyCOLMAP incremental_mapping is unavailable")
    options_probe = pycolmap_module.IncrementalPipelineOptions()
    for property_name in ("image_names", "init_image_id1", "init_image_id2",
                          "max_runtime_seconds", "multiple_models", "ba_use_gpu"):
        if not hasattr(options_probe, property_name):
            raise RuntimeError(f"PyCOLMAP missing required mapper option: {property_name}")
    paths = _source_paths(revision, model)
    if output.resolve().is_relative_to(paths["sparse"]):
        raise ValueError("Recovery output must be outside the source sparse revision")
    baseline = pycolmap_module.Reconstruction(paths["model"])
    baseline_registered = _registered_images(baseline)
    if (set(baseline_registered) != set(planned["registered_seed_filenames"]) or
            any(image_id != planned["registered_seed_image_ids"][name]
                for name, (image_id, _) in baseline_registered.items())):
        raise ValueError("Source sparse binary, text model and database camera IDs disagree")
    if merge_runner is None:
        if not shutil.which(colmap):
            raise FileNotFoundError(f"COLMAP executable not found: {colmap}")
        merge_runner = subprocess.run
    output.mkdir(parents=True, exist_ok=False)
    report = {**planned, "status": "running", "trials": [],
              "landmark_manifest": None,
              "message": "Only object-validated, visually reviewed candidates may be used for dense reconstruction."}
    try:
        working_db = output / "database.db"
        shutil.copy2(paths["database"], working_db)
        if _sha256(working_db) != planned["source_database_sha256"]:
            raise RuntimeError("Copied database hash mismatch")
        best = None
        best_score = None
        for attempt, edge in enumerate(planned["seed_pairs"][:settings.max_trials], 1):
            trial_dir = output / f"trial-{attempt:02d}"
            trial_dir.mkdir()
            options = pycolmap_module.IncrementalPipelineOptions()
            options.image_names = planned["cohort_filenames"]
            options.init_image_id1 = planned["source_image_ids"][edge["a"]]
            options.init_image_id2 = planned["source_image_ids"][edge["b"]]
            options.min_model_size = 3
            options.multiple_models = False
            options.max_runtime_seconds = settings.max_seconds_per_trial
            options.ba_use_gpu = False  # Ceres bundle adjustment, not the neural matcher.
            entry = {"trial": attempt, "seed_pair": edge, "models": [], "status": "running"}
            report["trials"].append(entry)
            _write_report(output, report)
            try:
                maps = pycolmap_module.incremental_mapping(
                    working_db, paths["images"], trial_dir / "sparse", options=options)
                for index, candidate in sorted(maps.items()):
                    alternate = _registered_images(candidate)
                    if not set(alternate) <= set(planned["cohort_filenames"]):
                        raise RuntimeError("Mapper registered images outside selected cohort")
                    if any(planned["source_image_ids"][name] != image_id
                           for name, (image_id, _) in alternate.items()):
                        raise RuntimeError("Mapper used image IDs inconsistent with original database")
                    alternate_common = set(alternate) & set(baseline_registered)
                    qa_file = paths["sparse"] / "subject-feature-index.json"
                    if len(alternate_common) >= settings.min_shared_cameras:
                        alignment = assess_alignment(baseline, candidate, settings)
                    elif alternate_common:
                        alignment = assess_partial_alignment(
                            baseline, candidate, paths["database"],
                            qa_file if qa_file.is_file() else None, settings)
                    else:
                        alignment = assess_disconnected_alignment(
                            baseline, candidate, paths["database"],
                            qa_file if qa_file.is_file() else None, settings)
                    gained_targets = sorted(set(alternate) & set(planned["missing_target_filenames"]))
                    alignment["new_target_filenames"] = gained_targets
                    alignment["new_target_cameras"] = len(gained_targets)
                    if len(gained_targets) < settings.min_new_cameras:
                        alignment["accepted"] = False
                        alignment["reasons"].append("insufficient-new-target-cameras")
                    saved = trial_dir / "sparse" / str(index)
                    saved.mkdir(parents=True, exist_ok=True)
                    candidate.write(saved)
                    entry["models"].append({"index": index, "points3D": candidate.num_points3D(),
                                            "registered_cameras": len(alternate), "alignment": alignment})
                    score = (len(gained_targets), len(alignment["shared_filenames"]), candidate.num_points3D())
                    if alignment["accepted"] and (best_score is None or score > best_score):
                        best_score = score
                        best = (saved, alignment, attempt)
                entry["status"] = "completed"
            except Exception as error:
                entry["status"] = "failed"
                entry["error"] = f"{type(error).__name__}: {error}"[:320]
            _write_report(output, report)
        if best is None:
            report["status"] = (
                "independent-models-only / alignment-not-validated"
                if any(trial["models"] for trial in report["trials"]) else
                "no-independent-model / all-trials-failed")
            return report

        candidate_path, alignment, attempt = best
        # Keep all unvalidated native output in a clearly non-promoted folder.
        merged_path = output / "merge-under-review"
        merged_path.mkdir()
        if alignment.get("method") in ("verified-3d-points", "verified-hybrid-3d-points"):
            from .sfm_point_alignment import fuse_disjoint_models
            try:
                independent = pycolmap_module.Reconstruction(candidate_path)
                joined = fuse_disjoint_models(baseline, independent, alignment,
                                              pycolmap_module=pycolmap_module)
                joined.write(merged_path)
                report["merge"] = {"from_trial": attempt, "alignment": alignment,
                                   "method": "validated-evidence-track-union", "returncode": 0,
                                   "log_excerpt": "Aligned verified real camera tracks without moving originals."}
            except (ValueError, RuntimeError, TypeError, OSError) as error:
                report["merge"] = {"from_trial": attempt, "alignment": alignment,
                                   "method": "validated-evidence-track-union", "returncode": 2,
                                   "error": f"{type(error).__name__}: {error}"[:300]}
                report["status"] = "merge-rejected / disconnected-model-structure"
                return report
        else:
            command = [colmap, "model_merger", "--input_path1", str(paths["model"]),
                       "--input_path2", str(candidate_path), "--output_path", str(merged_path),
                       "--max_reproj_error", str(settings.merge_max_reproj_error)]
            completed = merge_runner(command, capture_output=True, text=True,
                                     timeout=settings.max_seconds_per_trial, check=False)
            report["merge"] = {"from_trial": attempt, "alignment": alignment,
                               "returncode": completed.returncode,
                               "log_excerpt": (completed.stdout + "\n" + completed.stderr)[-2500:]}
            if completed.returncode != 0:
                report["status"] = "merge-rejected / colmap-geometric-check"
                return report
        merged = pycolmap_module.Reconstruction(merged_path)
        before = _registered_images(baseline)
        after = _registered_images(merged)
        if not set(before) <= set(after):
            report["status"] = "merge-rejected / lost-original-cameras"
            return report
        if not set(alignment["new_target_filenames"]) <= set(after):
            report["status"] = "merge-rejected / lost-opposite-cameras"
            return report
        if any(before[name][0] != after[name][0] for name in before):
            report["status"] = "merge-rejected / inconsistent-image-ids"
            return report
        import numpy as np
        base_centers = np.asarray([np.asarray(cam.projection_center()).reshape(3)
                                   for _, cam in before.values()])
        base_spread = max(1e-9, float(np.sqrt(np.mean(np.sum(
            (base_centers - base_centers.mean(axis=0)) ** 2, axis=1)))))
        moved = []
        angles = []
        for name in sorted(before):
            first = before[name][1]
            second = after[name][1]
            moved.append(float(np.linalg.norm(np.asarray(first.projection_center()) -
                                              np.asarray(second.projection_center())) / base_spread))
            a = np.asarray(first.viewing_direction()).reshape(3)
            b = np.asarray(second.viewing_direction()).reshape(3)
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            if denominator < 1e-9:
                report["status"] = "merge-rejected / invalid-camera-orientation"
                return report
            angles.append(float(np.degrees(np.arccos(np.clip(a @ b / denominator, -1, 1)))))
        report["merge"]["max_original_camera_displacement_ratio"] = max(moved)
        report["merge"]["max_original_camera_direction_degrees"] = max(angles)
        if max(moved) > 0.01 or max(angles) > 3.0:
            report["status"] = "merge-rejected / baseline-camera-moved"
            return report
        report["merge"]["registered_cameras"] = len(after)
        report["merge"]["points3D"] = merged.num_points3D()
        merged_path.rename(output / "merged-candidate")
        report["merge"]["candidate_path"] = str(output / "merged-candidate")
        report["status"] = "candidate / needs-object-and-camera-visual-review"
        return report
    except Exception as error:
        report["status"] = "failed / review-required"
        report["error"] = f"{type(error).__name__}: {error}"[:320]
        raise
    finally:
        _write_report(output, report)
