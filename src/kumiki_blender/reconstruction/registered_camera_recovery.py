"""Conservative camera recovery against existing *real* COLMAP observations.

New camera poses are accepted only after COLMAP registration and independent
baseline/track checks. Learned depth/pose predictions may guide future camera
search but are never counted as measured cameras here.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from .references import ReferenceError


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReferenceError(f"Missing or unsafe camera-recovery input: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReferenceError(f"Expected object in {path}")
    return data


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _registered(model) -> dict[str, tuple[int, Any]]:
    return {model.images[image_id].name: (image_id, model.images[image_id])
            for image_id in model.reg_image_ids()}


def prioritize_missing_camera_views(staged: list[dict], registered: set[str]) -> list[str]:
    """Rank existing unregistered elevated source photos before requesting bridges.

    These names are *hints*, never evidence of elevation or a reconstructed pose.
    The downstream measured coverage audit remains the production gate.
    """
    elevated = ("elevated", "overhead", "aerial", "drone", "above", "top", "bird")
    missing = []
    for item in staged:
        if not isinstance(item, dict):
            continue
        name = item.get("filename")
        if not isinstance(name, str) or name in registered:
            continue
        original = str(item.get("original_filename") or name).lower()
        score = sum(3 for word in elevated if word in original)
        score += sum(word in original for word in ("left", "right", "rear", "back"))
        missing.append((-score, name))
    return [name for _, name in sorted(missing)]


def validate_camera_recovery(baseline, candidate, *,
                             min_new_track_points: int = 10,
                             max_baseline_drift: float = 0.01,
                             max_view_angle_degrees: float = 3.0) -> dict[str, Any]:
    """Check native PnP registrations; never accept solely predicted model poses."""
    import numpy as np

    before, after = _registered(baseline), _registered(candidate)
    common = sorted(before.keys() & after.keys())
    gained = sorted(after.keys() - before.keys())
    result: dict[str, Any] = {"accepted": False, "new_filenames": gained,
                              "new_cameras": len(gained), "reasons": []}
    if set(before) - set(after):
        result["reasons"].append("lost-original-registered-camera")
    if not gained:
        result["reasons"].append("no-new-registered-cameras")
    if any(before[name][0] != after[name][0] for name in common):
        result["reasons"].append("original-image-id-mismatch")
    if any(after[name][1].num_points3D < min_new_track_points for name in gained):
        result["reasons"].append("new-camera-lacks-verified-3d-tracks")
    for name in gained:
        center = np.asarray(after[name][1].projection_center(), dtype=float)
        direction = np.asarray(after[name][1].viewing_direction(), dtype=float)
        if (center.shape != (3,) or direction.shape != (3,) or
                not np.isfinite(center).all() or not np.isfinite(direction).all() or
                np.linalg.norm(direction) < 1e-9):
            result["reasons"].append("new-camera-has-invalid-pose")
            break
    if result["reasons"]:
        return result

    centers = np.asarray([before[name][1].projection_center() for name in common], dtype=float)
    if not np.isfinite(centers).all() or len(centers) < 3:
        result["reasons"].append("invalid-original-camera-centers")
        return result
    spread = max(float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).mean()), 1e-9)
    drift, angles = [], []
    for name in common:
        first = before[name][1]
        second = after[name][1]
        shift = float(np.linalg.norm(np.asarray(first.projection_center()) -
                                     np.asarray(second.projection_center()))) / spread
        a = np.asarray(first.viewing_direction(), dtype=float)
        b = np.asarray(second.viewing_direction(), dtype=float)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        angle = (float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1, 1))))
                 if denom > 1e-9 else math.inf)
        drift.append(shift)
        angles.append(angle)
    result["max_original_center_displacement_ratio"] = max(drift)
    result["max_original_direction_change_degrees"] = max(angles)
    result["new_camera_track_points"] = {
        name: int(after[name][1].num_points3D) for name in gained}
    if not all(math.isfinite(number) for number in drift + angles):
        result["reasons"].append("non-finite-recovered-pose")
    if max(drift) > max_baseline_drift:
        result["reasons"].append("baseline-camera-center-drift")
    if max(angles) > max_view_angle_degrees:
        result["reasons"].append("baseline-camera-orientation-drift")
    result["accepted"] = not result["reasons"]
    return result


def recover_registered_cameras(
    reconstruction: Path, output: Path, *, colmap: str = "colmap",
    timeout_seconds: int = 1800,
    runner: Callable[[list[str], Path, int], int] | None = None,
    model_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Attempt verified PnP registration and export an independent dense-ready revision.

    Failed or unchanged candidates are kept separately as diagnostics. Original
    sparse model, source photos and database are never mutated.
    """
    source = reconstruction.resolve()
    target = output.resolve()
    if (reconstruction.is_symlink() or output.is_symlink() or not source.is_dir()
            or target.exists() or target.is_relative_to(source)):
        raise ReferenceError("Camera recovery requires an unused output outside the source revision")
    source_report = _read_json(source / "reconstruction-report.json")
    staged = _read_json(source / "staged-reference-index.json")
    if (not str(source_report.get("status", "")).startswith("candidate")
            or not isinstance(staged.get("images"), list)
            or source_report.get("engine") not in (None, "colmap")):
        raise ReferenceError("Source must be a verified COLMAP camera revision")
    best_model_name = str(source_report.get("source_sparse_model", "0"))
    if not best_model_name.isdigit() or len(best_model_name) > 4:
        raise ReferenceError("Source sparse model index is invalid")
    source_model = source / "sparse" / best_model_name
    if (not source_model.is_dir() or source_model.is_symlink() or
            not source_model.resolve().is_relative_to(source)):
        raise ReferenceError("Source COLMAP sparse model is missing")
    database = source / "database.db"
    images = source / "images"
    if (database.is_symlink() or not database.is_file() or
            images.is_symlink() or not images.is_dir()):
        raise ReferenceError("Source database or normalized photos are missing")
    if not shutil.which(colmap):
        raise ReferenceError(f"COLMAP executable not found: {colmap}")
    if not 60 <= timeout_seconds <= 86400:
        raise ReferenceError("Camera recovery timeout must be 60–86400 seconds")

    if model_loader is None:
        import pycolmap
        model_loader = pycolmap.Reconstruction
    baseline = model_loader(source_model)
    target.mkdir(parents=True)
    (target / "logs").mkdir()
    report: dict[str, Any] = {"schema_version": 1, "status": "running",
                              "source_revision": str(source), "source_database_sha256": _sha(database),
                              "source_registered_images": len(baseline.reg_image_ids()),
                              "registered_images": 0, "steps": [], "candidate_accepted": False,
                              "distribution_allowed": False}

    def save() -> dict[str, Any]:
        (target / "camera-recovery-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    def fail(reason: str) -> dict[str, Any]:
        report.update(status="no-verified-camera-recovery", reason=reason)
        return save()

    def execute(name: str, args: list[str]) -> bool:
        log = target / "logs" / f"{len(report['steps']) + 1:02}-{name}.log"
        argv = [colmap, name] + args
        if runner is None:
            try:
                with log.open("w", encoding="utf-8") as stream:
                    code = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                          timeout=timeout_seconds, check=False).returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                # Preserve native output written before a timeout; it is often
                # the only useful diagnostic for an OpenCV/COLMAP failure.
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(f"\n{type(error).__name__}: {error}\n")
                code = 124 if isinstance(error, subprocess.TimeoutExpired) else 127
        else:
            code = runner(argv, log, timeout_seconds)
        report["steps"].append({"stage": name, "exit_code": code,
                                "log": str(log.relative_to(target))})
        save()
        return code == 0

    def candidate_model(path: Path) -> bool:
        return (path.is_dir() and not path.is_symlink() and all(
            (path / name).is_file() and not (path / name).is_symlink()
            for name in ("cameras.bin", "images.bin", "points3D.bin")))

    def check_candidate(path: Path, label: str) -> dict[str, Any]:
        if not candidate_model(path):
            return {"accepted": False, "reasons": [f"{label}-produced-no-model"]}
        try:
            return validate_camera_recovery(baseline, model_loader(path))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            return {"accepted": False,
                    "reasons": [f"{label}-invalid-model"],
                    "error": f"{type(error).__name__}: {error}"[:240]}

    original_names = set(_registered(baseline))
    priority = prioritize_missing_camera_views(staged["images"], original_names)
    elevated_words = ("elevated", "overhead", "aerial", "drone", "above", "top", "bird")
    elevated_targets = {item["filename"] for item in staged["images"]
                        if isinstance(item, dict) and isinstance(item.get("filename"), str)
                        and any(word in str(item.get("original_filename") or "").lower()
                                for word in elevated_words)}
    report["priority_camera_targets"] = priority
    report["elevated_filename_hints"] = sorted(elevated_targets)

    best: Path | None = None

    def current_names() -> set[str]:
        return set(_registered(model_loader(best))) if best is not None else original_names

    def remaining_elevated() -> list[str]:
        return sorted(elevated_targets - current_names())

    def remaining_targets() -> list[str]:
        return [name for name in priority if name not in current_names()]

    def try_promote(candidate: Path, label: str) -> bool:
        """Merge independent verified models without dropping an accepted pose."""
        nonlocal best
        checked = check_candidate(candidate, label)
        report[f"{label}_validation"] = checked
        if not checked["accepted"]:
            return False
        if best is None:
            best = candidate
            return True
        before = current_names()
        after = set(_registered(model_loader(candidate)))
        if not after - before:
            return False
        if before <= after:
            incremental = validate_camera_recovery(model_loader(best), model_loader(candidate))
            report[f"{label}_incremental_validation"] = incremental
            if not incremental["accepted"]:
                return False
            best = candidate
            return True
        merged = target / f"combined-{label}"
        merged.mkdir()
        if not execute("model_merger", ["--input_path1", str(best),
                                        "--input_path2", str(candidate),
                                        "--output_path", str(merged),
                                        "--max_reproj_error", "4.0"]):
            return False
        validation = check_candidate(merged, f"combined-{label}")
        report[f"combined_{label}_validation"] = validation
        if not validation["accepted"]:
            return False
        combined_names = set(_registered(model_loader(merged)))
        if not before <= combined_names or not combined_names - before:
            return False
        incremental = validate_camera_recovery(model_loader(best), model_loader(merged))
        report[f"combined_{label}_incremental_validation"] = incremental
        if not incremental["accepted"]:
            return False
        best = merged
        return True

    registered = target / "registered"
    registered.mkdir()
    if not execute("image_registrator", ["--database_path", str(database),
                                         "--input_path", str(source_model),
                                         "--output_path", str(registered)]):
        report["direct_registration_warning"] = "image-registrator-command-failed"
    evidence = check_candidate(registered, "image-registrator")
    report["initial_validation"] = evidence
    if evidence["accepted"]:
        best = registered
    save()

    # Newly verified photographs can triangulate additional REAL 3D tracks,
    # enabling PnP for elevated views that were impossible in the first pass.
    # Bound the loop on both work and new evidence; never accept triangulation
    # itself as proof of another registered camera.
    for round_no in range(1, 3):
        if best is None or not remaining_targets():
            break
        triangulated = target / f"triangulated-{round_no:02d}"
        triangulated.mkdir()
        if not execute("point_triangulator", ["--database_path", str(database),
                                               "--image_path", str(images),
                                               "--input_path", str(best),
                                               "--output_path", str(triangulated)]):
            break
        if not candidate_model(triangulated):
            break
        improved = target / f"registered-{round_no + 1:02d}"
        improved.mkdir()
        if not execute("image_registrator", ["--database_path", str(database),
                                              "--input_path", str(triangulated),
                                              "--output_path", str(improved)]):
            break
        accepted = try_promote(improved, f"iterated_pnp_{round_no}")
        report.setdefault("iterative_registration", []).append({
            "round": round_no, "accepted": accepted,
            "remaining_elevated_hints": remaining_elevated()})
        save()
        if not accepted:
            break

    # A successful but non-elevated PnP pass must *not* terminate recovery.
    # Target missing overhead views explicitly and also try the remaining real
    # photos when no filename hints are available.
    if best is None or remaining_targets():
        try:
            from .sfm_cohort_recovery import CohortRecoverySettings, recover_cohort
            targets = remaining_elevated() or remaining_targets()
            alternate = recover_cohort(
                source, target / "cohorts", model=int(best_model_name), targets=targets,
                settings=CohortRecoverySettings(
                    max_trials=4, min_new_cameras=1,
                    max_seconds_per_trial=min(timeout_seconds, 1200)))
            report["cohort_status"] = alternate.get("status")
            report["cohort_report"] = "cohorts/cohort-recovery-report.json"
            if alternate.get("status") == "candidate / needs-object-and-camera-visual-review":
                try_promote(target / "cohorts/merged-candidate", "cohort")
        except (ValueError, OSError, RuntimeError, ImportError, TypeError) as error:
            report["cohort_error"] = f"{type(error).__name__}: {error}"[:250]
        save()

    if best is None or remaining_targets():
        learned: dict[str, Any] = {}
        try:
            from .learned_camera_hypotheses import build_learned_camera_hypotheses
            learned = build_learned_camera_hypotheses(
                source, target / "learned-priors", device="mps",
                timeout_seconds=min(timeout_seconds, 1800))
            report["learned_camera_status"] = learned.get("status")
            report["learned_camera_report"] = "learned-camera-hypotheses.json"
            (target / "learned-camera-hypotheses.json").write_text(
                json.dumps(learned, indent=2) + "\n", encoding="utf-8")
        except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
            report["learned_camera_status"] = "inference-unavailable"
            report["learned_camera_error"] = str(error)[:250]
        hypotheses = learned.get("pose_hypotheses") or []
        guided_targets = [item["filename"] for item in hypotheses
                          if isinstance(item, dict) and isinstance(item.get("filename"), str)
                          and item["filename"] not in current_names()]
        if guided_targets:
            try:
                from .sfm_cohort_recovery import CohortRecoverySettings, recover_cohort
                guided = recover_cohort(
                    source, target / "learned-guided-cohorts", model=int(best_model_name),
                    targets=guided_targets, settings=CohortRecoverySettings(
                        max_trials=3, min_new_cameras=1,
                        max_seconds_per_trial=min(timeout_seconds, 1200)))
                report["learned_guided_cohort_status"] = guided.get("status")
                if guided.get("status") == "candidate / needs-object-and-camera-visual-review":
                    try_promote(target / "learned-guided-cohorts/merged-candidate",
                                "learned_guided_cohort")
            except (ValueError, OSError, RuntimeError, ImportError, TypeError) as error:
                report["learned_guided_cohort_error"] = str(error)[:250]
        save()
    if best is None:
        return fail("no-geometrically-validated-new-cameras")
    report["remaining_elevated_filename_hints"] = remaining_elevated()
    report["remaining_unregistered_photo_targets"] = remaining_targets()
    registered = best

    # Global BA refines real-image poses. If original cameras drift excessively,
    # reject BA and retain only the independently validated PnP model.
    refined = target / "refined"
    refined.mkdir()
    selected = registered
    if execute("bundle_adjuster", ["--input_path", str(registered),
                                   "--output_path", str(refined)]):
        if candidate_model(refined):
            refined_validation = check_candidate(refined, "bundle-adjuster")
            report["refined_validation"] = refined_validation
            if refined_validation["accepted"]:
                selected = refined
    report["selected_model"] = selected.name

    # Export exact registered camera/rectified photo revision consumed by SAM2
    # and OpenMVS. Never present an inferred pose as an accepted registration.
    (target / "images").mkdir()
    originals = {}
    for entry in staged["images"]:
        name, expected = entry.get("filename"), entry.get("sha256")
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or len(expected) != 64):
            return fail("unsafe-staged-image-index")
        source_file = images / name
        if source_file.is_symlink() or not source_file.is_file() or _sha(source_file) != expected:
            return fail("source-photo-integrity-failed")
        shutil.copyfile(source_file, target / "images" / name)
        originals[name] = entry
    shutil.copyfile(source / "staged-reference-index.json", target / "staged-reference-index.json")
    shutil.copyfile(database, target / "database.db")

    saved_model = target / "sparse" / "0"
    saved_model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        item = selected / name
        if item.is_symlink() or not item.is_file():
            return fail(f"missing-registered-model-{name}")
        shutil.copyfile(item, saved_model / name)
    text = target / "text-models" / "0"
    text.mkdir(parents=True)
    if not execute("model_converter", ["--input_path", str(saved_model),
                                       "--output_path", str(text), "--output_type", "TXT"]):
        return fail("model-converter-failed")

    rectified = target / "undistorted"
    rectified.mkdir()
    if not execute("image_undistorter", ["--image_path", str(target / "images"),
                                        "--input_path", str(saved_model),
                                        "--output_path", str(rectified),
                                        "--output_type", "COLMAP", "--max_image_size", "3200"]):
        return fail("image-undistorter-failed")
    undistorted_text = rectified / "text-model"
    undistorted_text.mkdir()
    if not execute("model_converter", ["--input_path", str(rectified / "sparse"),
                                       "--output_path", str(undistorted_text),
                                       "--output_type", "TXT"]):
        return fail("undistorted-model-converter-failed")
    from .cameras import _read_intrinsics, _read_poses
    from PIL import Image

    intrinsics = _read_intrinsics(undistorted_text / "cameras.txt")
    poses = _read_poses(undistorted_text / "images.txt")
    registered_names = set(_registered(model_loader(saved_model)))
    rectified_index = []
    for _, camera_id, name, _, _ in poses:
        if name not in originals or name not in registered_names or camera_id not in intrinsics:
            return fail("invalid-rectified-camera-provenance")
        photo = rectified / "images" / name
        if photo.is_symlink() or not photo.is_file():
            return fail("rectified-photo-missing")
        with Image.open(photo) as image:
            size = list(image.size)
            image.verify()
        camera = intrinsics[camera_id]
        if size != [camera["width"], camera["height"]]:
            return fail("rectified-camera-dimensions-mismatch")
        rectified_index.append({**originals[name], "dimensions": size,
                                "sha256": _sha(photo), "mask_requires_rectification": True})
    if len(rectified_index) != len(registered_names):
        return fail("undistorted-camera-count-mismatch")
    (target / "undistorted-reference-index.json").write_text(
        json.dumps({"images": rectified_index}, indent=2) + "\n", encoding="utf-8")
    recovered = sorted(registered_names - set(_registered(baseline)))
    source_report.update({"source_sparse_model": "0", "registered_images": len(registered_names),
                          "registered_filenames": sorted(registered_names),
                          "unregistered_filenames": sorted(set(originals) - registered_names),
                          "undistorted_images": len(rectified_index),
                          "models": [{"model": "0", "registered_images": len(registered_names),
                                      "registered_filenames": sorted(registered_names)}],
                          "engine": "colmap", "status": "candidate / visual-review-required",
                          "undistorted_model": "undistorted/text-model",
                          "camera_recovery_source": str(source),
                          "camera_recovery_added": recovered})
    (target / "reconstruction-report.json").write_text(
        json.dumps(source_report, indent=2) + "\n", encoding="utf-8")
    report.update(status="candidate / verified-real-photo-cameras",
                  candidate_accepted=True, registered_images=len(registered_names),
                  new_filenames=recovered, new_cameras=len(recovered),
                  recovered_revision=str(target))
    return save()
