"""Stage cameras that COLMAP registered from predicted matches.

`prepare-roma-candidate-database` adds predicted 2D-3D matches for photos that
measured features could not register. This module registers those photos
against the unchanged baseline model, checks that no baseline camera moved,
and writes a dense-ready revision. Each camera that depends on predicted
matches is named in the reports. A predicted-match camera is a hypothesis:
compare the existing mesh with its photo before you trust a new mesh.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Callable

from .references import ReferenceError
from .registered_camera_recovery import (RevisionExportError, _read_json, _registered, _sha,
                                         export_dense_ready_revision, remove_implausible_cameras,
                                         run_colmap_stage, validate_camera_recovery)


STAGING_REPORT = "predicted-camera-staging-report.json"
STAGED_STATUS = "candidate / predicted-match-cameras-need-visual-review"


def keypoint_counts(database: Path) -> dict[str, int]:
    """Return the keypoint count of each photo in a COLMAP database."""
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return dict(db.execute(
            "SELECT i.name, k.rows FROM images i JOIN keypoints k ON k.image_id = i.image_id"))


def photos_with_predicted_keypoints(source_counts: dict[str, int],
                                    candidate_counts: dict[str, int]) -> set[str]:
    """Name the photos whose keypoints include predictions that the source did not measure."""
    if set(source_counts) != set(candidate_counts) or any(
            candidate_counts[name] < count for name, count in source_counts.items()):
        raise ReferenceError("Candidate database does not extend the database of this revision; "
                             "create it with prepare-roma-candidate-database")
    return {name for name, count in source_counts.items() if candidate_counts[name] > count}


def stage_predicted_cameras(
        revision: str | Path, candidate_database: str | Path, output: str | Path, *,
        colmap: str = "colmap", timeout_seconds: int = 1800,
        runner: Callable[[list[str], Path, int], int] | None = None,
        model_loader: Callable[[Path], Any] | None = None) -> dict[str, Any]:
    """Register predicted-match photos and write a labelled dense-ready revision."""
    source = Path(revision).resolve()
    candidate = Path(candidate_database).resolve()
    target = Path(output).resolve()
    if (Path(revision).is_symlink() or not source.is_dir() or Path(output).is_symlink() or
            target.exists() or target.is_relative_to(source) or source.is_relative_to(target)):
        raise ReferenceError("Predicted-camera staging needs a camera revision and an unused "
                             "output folder outside it")
    if Path(candidate_database).is_symlink() or not candidate.is_file():
        raise ReferenceError(f"Candidate database is missing: {candidate}")
    if not 60 <= timeout_seconds <= 86400:
        raise ReferenceError("Staging timeout must be 60 to 86400 seconds")
    if runner is None and not shutil.which(colmap):
        raise ReferenceError(f"COLMAP executable not found: {colmap}")
    source_report = _read_json(source / "reconstruction-report.json")
    staged = _read_json(source / "staged-reference-index.json")
    model_name = str(source_report.get("source_sparse_model", "0"))
    source_model = source / "sparse" / model_name
    rectified_max_side = source_report.get("max_image_size", 3200)
    if (not model_name.isdigit() or not source_model.is_dir() or
            type(rectified_max_side) is not int or not 256 <= rectified_max_side <= 16384):
        raise ReferenceError("Source revision has no usable sparse model or max_image_size")
    predicted = photos_with_predicted_keypoints(keypoint_counts(source / "database.db"),
                                                keypoint_counts(candidate))
    if not predicted:
        raise ReferenceError("Candidate database has no predicted keypoints; "
                             "use recover-registered-cameras for measured features")
    if model_loader is None:
        import pycolmap
        model_loader = pycolmap.Reconstruction
    baseline = model_loader(source_model)

    target.mkdir(parents=True)
    (target / "logs").mkdir()
    report: dict[str, Any] = {
        "schema_version": 1, "status": "running", "source_revision": str(source),
        "candidate_database_sha256": _sha(candidate), "steps": [],
        "photos_with_predicted_keypoints": sorted(predicted),
        "approved_for_reconstruction": False, "distribution_allowed": False}

    def save() -> dict[str, Any]:
        (target / STAGING_REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    def fail(reason: str) -> dict[str, Any]:
        report.update(status="no-staged-predicted-cameras", reason=reason)
        return save()

    def execute(name: str, args: list[str]) -> bool:
        log = target / "logs" / f"{len(report['steps']) + 1:02}-{name}.log"
        code = run_colmap_stage(colmap, name, args, log, timeout_seconds, runner)
        report["steps"].append({"stage": name, "exit_code": code,
                                "log": str(log.relative_to(target))})
        save()
        return code == 0

    # COLMAP writes verification results into the database, so use a private copy.
    database = target / "database.db"
    shutil.copyfile(candidate, database)
    if not execute("geometric_verifier", ["--database_path", str(database), "--num_threads", "1"]):
        return fail("geometric-verifier-failed")
    registered = target / "registered"
    registered.mkdir()
    if not execute("image_registrator", ["--database_path", str(database),
                                         "--input_path", str(source_model),
                                         "--output_path", str(registered)]):
        return fail("image-registrator-failed")
    try:
        validation = validate_camera_recovery(baseline, model_loader(registered))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        report["error"] = f"{type(error).__name__}: {error}"[:240]
        return fail("image-registrator-produced-no-valid-model")
    selected = remove_implausible_cameras(baseline, registered, target / "plausible", report,
                                          model_loader)
    if selected != registered:
        validation = validate_camera_recovery(baseline, model_loader(selected))
    report["validation"] = validation
    predicted_cameras = sorted(predicted & set(validation["new_filenames"]))
    report["predicted_match_cameras"] = predicted_cameras
    report["measured_feature_cameras"] = sorted(set(validation["new_filenames"]) - predicted)
    report["unregistered_predicted_photos"] = sorted(predicted - set(validation["new_filenames"]))
    if not validation["accepted"]:
        return fail("baseline-or-track-validation-failed")
    if not predicted_cameras:
        return fail("no-predicted-photo-was-registered")
    try:
        exported = export_dense_ready_revision(
            source, target, selected, set(_registered(baseline)), staged=staged,
            # A staged revision can be the source of the next staging; keep its labels.
            source_report={**source_report, "predicted_match_cameras": sorted(
                set(source_report.get("predicted_match_cameras", [])) | set(predicted_cameras))},
            database=database, rectified_max_side=rectified_max_side, execute=execute,
            model_loader=model_loader)
    except RevisionExportError as error:
        return fail(str(error))
    report.update(
        status=STAGED_STATUS, registered_images=exported["registered_images"],
        new_filenames=exported["new_filenames"], staged_revision=str(target),
        limitations=("Predicted keypoints are model output, not measured features. A low "
                     "reprojection error on them does not prove the pose. Project the existing "
                     "mesh into each predicted-match camera and compare it with the photo. "
                     "The database has placeholder descriptors: never match features with it."))
    return save()
