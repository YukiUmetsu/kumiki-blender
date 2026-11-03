#!/usr/bin/env python3
"""Non-destructive seeded COLMAP recovery from an existing sparse model.

First triangulate additional points between already registered views, then
resume incremental mapping from that single seed. All output is a new revision;
source photographs, database and sparse models remain unchanged. A successful
mapper run does NOT certify statue geometry or license a splat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _registered_names(reconstruction) -> set[str]:
    return {reconstruction.images[image_id].name
            for image_id in reconstruction.reg_image_ids()}


def _report(path: Path, data: dict) -> None:
    with (path / "recovery-report.json").open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


MAX_IMAGE_ID = 2_147_483_647


def _database_match_evidence(database: Path, seed_names: set[str],
                             *, min_inliers: int = 15) -> dict:
    """Read-only verification of match edges connecting the seed to missing photos.

    This is database evidence, not proof that PnP or incremental mapping will
    succeed: geometric spread, camera calibration and 3D tracks still matter.
    """
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        names = dict(db.execute("SELECT image_id, name FROM images"))
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "two_view_geometries" not in tables:
            return {
                "status": "no-two-view-geometry-table",
                "verified_pairs": 0, "seed_to_missing_pairs": 0,
                "unregistered_photo_advice": [],
                "interpretation": "Cannot inspect verified pair geometry in this database.",
            }
        stats = {
            name: {"filename": name, "verified_pairs_to_seed": 0,
                   "inliers_to_seed": 0, "strong_pairs_to_seed": 0,
                   "max_pair_inliers_to_seed": 0, "strong_pairs_to_other_missing": 0}
            for name in names.values() if name not in seed_names
        }
        raw_pairs = db.execute("SELECT COUNT(*) FROM matches").fetchone()[0] if (
            "matches" in tables) else None
        geometry_rows = 0
        verified_pairs = 0
        strong_pairs = 0
        seed_to_missing_pairs = 0
        strong_seed_to_missing_pairs = 0
        seed_to_seed_pairs = 0
        missing_to_missing_pairs = 0
        for pair_id, rows, config in db.execute(
            "SELECT pair_id, rows, config FROM two_view_geometries"
        ):
            geometry_rows += 1
            if not isinstance(rows, int) or rows < 1 or config in (0, 1):
                continue
            first, second = divmod(pair_id, MAX_IMAGE_ID)
            if first not in names or second not in names:
                continue
            verified_pairs += 1
            if rows >= min_inliers:
                strong_pairs += 1
            left, right = names[first], names[second]
            left_seed, right_seed = left in seed_names, right in seed_names
            if left_seed and right_seed:
                seed_to_seed_pairs += 1
            elif left_seed != right_seed:
                seed_to_missing_pairs += 1
                if rows >= min_inliers:
                    strong_seed_to_missing_pairs += 1
                missing = right if left_seed else left
                evidence = stats[missing]
                evidence["verified_pairs_to_seed"] += 1
                evidence["inliers_to_seed"] += rows
                evidence["max_pair_inliers_to_seed"] = max(
                    evidence["max_pair_inliers_to_seed"], rows)
                if rows >= min_inliers:
                    evidence["strong_pairs_to_seed"] += 1
            else:
                missing_to_missing_pairs += 1
                if rows >= min_inliers:
                    stats[left]["strong_pairs_to_other_missing"] += 1
                    stats[right]["strong_pairs_to_other_missing"] += 1
        for entry in stats.values():
            if entry["strong_pairs_to_seed"] == 0:
                entry["assessment"] = "no-strong-seed-overlap"
                entry["next_action"] = (
                    "Find or capture intermediate overlapping views between this photo "
                    "and a registered seed camera; compare its strongest match neighbors.")
            elif entry["strong_pairs_to_seed"] == 1:
                entry["assessment"] = "single-seed-connection"
                entry["next_action"] = (
                    "Add views bridging this photo to a second registered seed camera; "
                    "one matched pair may not provide enough 2D-to-3D PnP support.")
            else:
                entry["assessment"] = "matching-but-unregistered"
                entry["next_action"] = (
                    "Check 2D-to-3D track support, parallax, feature distribution and "
                    "camera calibration; do not assume more photos alone will fix it.")
        advice = sorted(stats.values(), key=lambda item: (
            item["strong_pairs_to_seed"], item["inliers_to_seed"], item["filename"]))
        if not verified_pairs:
            status = "no-verified-pair-correspondences"
        elif not strong_seed_to_missing_pairs:
            status = "no-strong-seed-to-missing-bridges"
        else:
            status = "some-verified-seed-to-missing-bridges"
        return {
            "status": status,
            "min_pair_inliers": min_inliers,
            "database_images": len(names),
            "raw_match_pairs": raw_pairs,
            "geometry_rows": geometry_rows,
            "verified_pairs": verified_pairs,
            "strong_verified_pairs": strong_pairs,
            "seed_to_seed_pairs": seed_to_seed_pairs,
            "seed_to_missing_pairs": seed_to_missing_pairs,
            "strong_seed_to_missing_pairs": strong_seed_to_missing_pairs,
            "missing_to_missing_pairs": missing_to_missing_pairs,
            "unregistered_photo_advice": advice,
            "interpretation": (
                "COLMAP may filter verified matches by camera/geometry/track criteria. "
                "A zero 'Loading matches' log is not evidence that the whole database "
                "contains zero matches. Only seed-to-missing verified edges can "
                "directly support seeded registration; even those do not guarantee PnP."
            ),
        }


def recover(revision: Path, output: Path, *, model: int = 0,
            retriangulate: bool = True, fix_seed_poses: bool = True,
            pycolmap_module=None) -> dict:
    if not isinstance(model, int) or not 0 <= model <= 1000:
        raise ValueError("model must be an integer in 0..1000")
    source = revision.resolve()
    sparse = source / "sparse-reconstruction" if (
        source / "sparse-reconstruction").is_dir() else source
    source_model = sparse / "sparse" / str(model)
    database = sparse / "database.db"
    images = sparse / "images"
    if not source_model.is_dir() or source_model.is_symlink():
        raise FileNotFoundError(f"Missing original sparse model: {source_model}")
    if not database.is_file() or database.is_symlink() or not images.is_dir():
        raise FileNotFoundError("Missing original COLMAP database or staged images")
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"Output must be an unused directory: {output}")
    if pycolmap_module is None:
        import pycolmap as pycolmap_module
    required = ("Reconstruction", "IncrementalPipelineOptions",
                "triangulate_points", "incremental_mapping")
    for symbol in required:
        if not hasattr(pycolmap_module, symbol):
            raise RuntimeError(f"Installed PyCOLMAP lacks {symbol}")
    options = pycolmap_module.IncrementalPipelineOptions()
    if not hasattr(options, "fix_existing_frames"):
        raise RuntimeError("Installed PyCOLMAP lacks fix_existing_frames option")
    options.min_model_size = 3
    options.fix_existing_frames = fix_seed_poses
    seed = pycolmap_module.Reconstruction(source_model)
    seed_names = _registered_names(seed)
    if len(seed_names) < 3 or seed.num_points3D() < 1:
        raise ValueError("Seed model must have >=3 registered views and 3D points")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        total_images = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        imported_names = {row[0] for row in db.execute("SELECT name FROM images")}
    if len(imported_names) != total_images or not seed_names <= imported_names:
        raise ValueError("Seed camera names do not match COLMAP database")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "source_revision": str(source),
        "source_model": str(source_model),
        "source_database_sha256": _sha256(database),
        "source_model_index": model,
        "source_images": total_images,
        "seed_registered_images": len(seed_names),
        "seed_points3D": seed.num_points3D(),
        "seed_registered_filenames": sorted(seed_names),
        "retriangulate": retriangulate,
        "fix_seed_poses": fix_seed_poses,
        "match_evidence": _database_match_evidence(database, seed_names),
        "status": "running",
        "interpretation": "Seeded registration and extra triangulation are experiments; "
                          "only registered cameras from ONE model count. "
                          "Even a high camera count does not verify Sphinx fidelity.",
    }
    try:
        # A private database copy avoids any modification to the original run.
        working_db = output / "database.db"
        shutil.copy2(database, working_db)
        staged_index = sparse / "staged-reference-index.json"
        if staged_index.is_file() and not staged_index.is_symlink():
            shutil.copy2(staged_index, output / staged_index.name)
        if _sha256(working_db) != report["source_database_sha256"]:
            raise RuntimeError("COLMAP database copy checksum mismatch")
        if retriangulate:
            pycolmap_module.triangulate_points(
                seed, working_db, images, output / "triangulation",
                clear_points=False, options=options, refine_intrinsics=False)
        report["seed_after_triangulation_points3D"] = seed.num_points3D()
        seed_path = output / "seed"
        seed_path.mkdir()
        seed.write(seed_path)
        maps = pycolmap_module.incremental_mapping(
            working_db, images, output / "sparse",
            options=options, input_path=seed_path)
        results = []
        for index, reconstruction in sorted(maps.items()):
            registered = _registered_names(reconstruction)
            if not registered <= imported_names:
                raise RuntimeError("Recovered model contains images absent from source database")
            target = output / "sparse" / str(index)
            target.mkdir(parents=True, exist_ok=True)
            reconstruction.write(target)
            text_target = output / "text-models" / str(index)
            text_target.mkdir(parents=True, exist_ok=False)
            reconstruction.write_text(text_target)
            results.append({
                "model": str(index),
                "registered_images": len(registered),
                "points3D": reconstruction.num_points3D(),
                "new_registered_filenames": sorted(registered - seed_names),
                "lost_seed_filenames": sorted(seed_names - registered),
                "registration_fraction": round(len(registered) / total_images, 4),
                "camera_gate_met": len(registered) >= 8 and
                                   len(registered) / total_images >= 0.6,
            })
        report["models"] = results
        best = max(results, key=lambda r: (r["registered_images"], r["points3D"]),
                   default=None)
        report["best_model"] = best["model"] if best else None
        report["status"] = (
            "candidate / needs-geometric-review" if best and
            best["registered_images"] > len(seed_names)
            else "no-registration-improvement"
        )
        report["camera_gate_met"] = bool(best and best["camera_gate_met"])
    except Exception as error:
        report["status"] = "failed / review-required"
        report["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        _report(output, report)
        raise
    _report(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--skip-retriangulation", action="store_true")
    parser.add_argument("--allow-seed-pose-refinement", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = recover(
            args.revision, args.output, model=args.model,
            retriangulate=not args.skip_retriangulation,
            fix_seed_poses=not args.allow_seed_pose_refinement)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Seed: {report['seed_registered_images']}/{report['source_images']} cameras, "
          f"{report['seed_points3D']} points; "
          f"after retriangulation: {report['seed_after_triangulation_points3D']} points")
    for model in report["models"]:
        print(f"  model {model['model']}: {model['registered_images']}/"
              f"{report['source_images']} cameras, {model['points3D']} points, "
              f"new photos: {', '.join(model['new_registered_filenames']) or '(none)'}, "
              f"camera gate: {model['camera_gate_met']}")
    evidence = report["match_evidence"]
    print(f"Verified database pairs: {evidence['verified_pairs']}; "
          f"strong seed-to-missing pairs: {evidence['strong_seed_to_missing_pairs']}")
    print(f"Seed match graph: {evidence['status']}")
    if report["status"] == "no-registration-improvement":
        print("No new cameras. Do not repeat recovery unchanged; inspect "
              "the per-photo seed-match evidence and run photo-advice.")
        for photo in evidence["unregistered_photo_advice"][:10]:
            print(f"  {photo['filename']}: {photo['assessment']} "
                  f"({photo['strong_pairs_to_seed']} strong seed links); "
                  f"{photo['next_action']}")
    print(f"Status: {report['status']}")
    print(f"Report: {args.output / 'recovery-report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
