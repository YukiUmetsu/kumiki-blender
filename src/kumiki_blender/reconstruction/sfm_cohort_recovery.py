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

    def validate(self) -> None:
        if (self.min_pair_inliers < 15 or not 3 <= self.max_anchors <= 100 or
                not 1 <= self.max_trials <= 50 or
                not 3 <= self.min_shared_cameras <= self.max_anchors or
                not 1 <= self.min_new_cameras <= 100 or
                not 0 < self.min_anchor_spread_ratio < 1 or
                not 0 < self.max_camera_residual_ratio < 1 or
                not 0 < self.max_view_angle_degrees < 90 or
                not 0 < self.max_landmark_residual_ratio < 1 or
                not 30 <= self.max_seconds_per_trial <= 86400 or
                not 0.5 <= self.merge_max_reproj_error <= 20):
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
                edges.append({"a": by_id[left], "b": by_id[right], "inliers": count})
    return names, edges


def _verify_staged_images(paths: dict[str, Path], db_names: set[str]) -> None:
    staged = json.loads(paths["index"].read_text(encoding="utf-8"))
    entries = staged.get("images")
    if not isinstance(entries, list):
        raise ValueError("Staged index requires an images list")
    hashes = {}
    for item in entries:
        name = item.get("filename")
        if (not isinstance(name, str) or not name or
                Path(name).name != name or name in hashes or
                not isinstance(item.get("sha256"), str)):
            raise ValueError("Unsafe or duplicate staged photo entry")
        hashes[name] = item["sha256"]
    if set(hashes) != db_names:
        raise ValueError("Staged index and database must contain the same images")
    for name in sorted(db_names):
        photo = paths["images"] / name
        if photo.is_symlink() or not photo.is_file() or _sha256(photo) != hashes[name]:
            raise ValueError(f"Missing, linked or modified staged image: {name}")


def plan_cohort_recovery(revision: Path, *, model: int = 0,
                         targets: list[str] | None = None,
                         preferred_anchors: list[str] | None = None,
                         settings: CohortRecoverySettings = CohortRecoverySettings()) -> dict:
    """Plan from verified feature edges, without importing PyCOLMAP or touching outputs."""
    settings.validate()
    paths = _source_paths(revision, model)
    registered = _registered_names(paths["text"])
    image_ids, edges = _read_database(paths["database"], settings.min_pair_inliers)
    _verify_staged_images(paths, set(image_ids))
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
    seed_pairs = [edge for edge in edges if edge["a"] in cohort and
                  edge["b"] in cohort and ({edge["a"], edge["b"]} & missing)]
    seed_pairs.sort(key=lambda edge: (-edge["inliers"], edge["a"], edge["b"]))
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
    return {"schema_version": 1, "status": status, "source_revision": str(paths["revision"]),
            "source_sparse": str(paths["sparse"]), "source_model_index": model,
            "source_database_sha256": _sha256(paths["database"]),
            "source_images": len(image_ids), "registered_seed_images": len(registered),
            "registered_seed_filenames": sorted(registered),
            "registered_seed_image_ids": {name: image_ids[name] for name in sorted(registered)},
            "target_filenames": sorted(chosen), "missing_target_filenames": sorted(missing),
            "anchor_filenames": sorted(anchors), "cohort_filenames": sorted(cohort),
            "verified_pair_count": len(edges), "seed_pairs": diverse,
            "source_image_ids": {name: image_ids[name] for name in sorted(cohort)},
            "settings": vars(settings),
            "warning": "Verified 2D matches do not prove a consistent camera solution. "
                       "Independent models and merged outputs require object-surface review."}


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
                    alignment = assess_alignment(baseline, candidate, settings)
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
