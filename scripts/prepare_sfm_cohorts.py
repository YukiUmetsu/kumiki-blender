#!/usr/bin/env python3
"""Split an existing failed COLMAP revision into independent, evidence-based photo cohorts.

The cohorts are candidates for NEW camera-registration attempts, not merged
reconstructions. Only verified-match graph components are used; source photos
are copied from the immutable staged inputs, never from fabricated geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path


def _diagnostic_module():
    script = Path(__file__).with_name("diagnose_splat_registration.py")
    spec = importlib.util.spec_from_file_location("diagnose_splat_registration", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load registration diagnostic")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plan(revision: Path, *, min_inliers: int = 15, min_photos: int = 3) -> dict:
    if min_inliers < 1 or not 3 <= min_photos <= 256:
        raise ValueError("min-inliers must be positive and min-photos must be 3–256")
    revision = revision.resolve()
    sparse = revision / "sparse-reconstruction" if (revision / "sparse-reconstruction").is_dir() else revision
    index_path = sparse / "staged-reference-index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Staged reference index is required: {index_path}")
    staged = json.loads(index_path.read_text(encoding="utf-8"))
    index = {entry["filename"]: entry for entry in staged["images"]}
    if len(index) != len(staged["images"]):
        raise ValueError("Duplicate staged filenames in reference index")
    diagnostic = _diagnostic_module().analyze(revision, min_inliers)
    cohorts = []
    for group in diagnostic["connected_components"]:
        if len(group) < min_photos:
            continue
        filenames = sorted(group)
        for filename in filenames:
            if filename not in index:
                raise ValueError(f"Verified image missing from staged index: {filename}")
            photo = sparse / "images" / filename
            if not photo.is_file() or photo.is_symlink():
                raise FileNotFoundError(f"Missing or unsafe staged photo: {photo}")
            if _sha256(photo) != index[filename]["sha256"]:
                raise ValueError(f"Staged photo hash mismatch: {filename}")
        models = [
            {"model": item["model"],
             "registered_in_cohort": len(set(item["filenames"]) & set(filenames)),
             "filenames": sorted(set(item["filenames"]) & set(filenames))}
            for item in diagnostic["sparse_models"]
            if set(item["filenames"]) & set(filenames)
        ]
        size = len(filenames)
        cohorts.append({
            "id": f"cohort-{len(cohorts) + 1:02d}",
            "filenames": filenames,
            "photos": size,
            "registered_in_existing_models": models,
            "required_cameras_for_splat": max(8, math.ceil(0.6 * size)),
            "can_attempt_production_splat": size >= 8,
            "warning": "Connected verified-feature pairs do not prove consistent camera poses or object-only matches.",
        })
    return {
        "schema_version": 1, "source_revision": str(revision),
        "source_sparse": str(sparse), "min_inliers": min_inliers,
        "source_photos": diagnostic["photos"],
        "registered_largest_model": diagnostic["registered"],
        "cohorts": cohorts,
        "excluded_filenames": sorted(set(diagnostic["staged_filenames"]) -
                                     {name for item in cohorts for name in item["filenames"]}),
        "limitations": "Cohorts are new SfM inputs only. Never merge disconnected camera frames or assume a mesh exists.",
    }


def prepare(revision: Path, output: Path, *, min_inliers: int = 15,
            min_photos: int = 3) -> dict:
    report = plan(revision, min_inliers=min_inliers, min_photos=min_photos)
    if not report["cohorts"]:
        raise ValueError("No connected components meet the minimum photo count")
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    output = output.resolve()
    source = Path(report["source_sparse"])
    index = {
        entry["filename"]: entry for entry in json.loads(
            (source / "staged-reference-index.json").read_text(encoding="utf-8")
        )["images"]
    }
    output.mkdir(parents=True, exist_ok=False)
    for cohort in report["cohorts"]:
        photos = output / cohort["id"] / "photos"
        photos.mkdir(parents=True)
        manifest = []
        for filename in cohort["filenames"]:
            original = source / "images" / filename
            if original.is_symlink() or not original.is_file():
                raise FileNotFoundError(f"Source disappeared: {original}")
            destination = photos / filename
            shutil.copyfile(original, destination)
            expected = index[filename]["sha256"]
            if _sha256(destination) != expected:
                raise ValueError(f"Staged photo changed during cohort creation: {filename}")
            manifest.append({
                "filename": filename, "staged_sha256": expected,
                "original_filename": index[filename].get("original_filename"),
                "original_sha256": index[filename].get("original_sha256"),
                "source_url": index[filename].get("source_url"),
            })
        (photos.parent / "source-index.json").write_text(
            json.dumps({"source_revision": report["source_revision"],
                        "images": manifest,
                        "distribution_allowed": False}, indent=2) + "\n",
            encoding="utf-8",
        )
        cohort["photos_dir"] = str(photos)
    (output / "cohort-plan.json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", type=Path, help="Failed splat revision with COLMAP match database")
    parser.add_argument("--output", type=Path, required=True, help="New immutable cohort directory")
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--min-photos", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        report = prepare(args.revision, args.output,
                         min_inliers=args.min_inliers, min_photos=args.min_photos)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Photo cohorts: {len(report['cohorts'])}; excluded/isolated: {len(report['excluded_filenames'])}")
    for cohort in report["cohorts"]:
        print(f"  {cohort['id']}: {cohort['photos']} photos; "
              f"previously registered: {[(m['model'], m['registered_in_cohort']) for m in cohort['registered_in_existing_models']]}")
        if cohort["can_attempt_production_splat"]:
            print(f"    Re-run with: REGISTRATION_PROFILE=high-recall MAX_IMAGE_SIZE=3200 "
                  f"bash scripts/mac_splat_test.sh run '{cohort['photos_dir']}' sphinx-{cohort['id']}")
            print(f"    Needs {cohort['required_cameras_for_splat']} registered cameras; "
                  "success is not guaranteed.")
        else:
            print("    Too few input photos for production splat; sparse SfM only:")
            print("    python3 scripts/experiment_sphinx_commons_2010.py "
                  f"--photos-dir '{cohort['photos_dir']}' "
                  f'--output ".review/sphinx-{cohort["id"]}-$(date -u +%Y%m%dT%H%M%SZ)" '
                  "--profile high-recall --max-image-size 3200")
    print(f"Plan: {args.output / 'cohort-plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
