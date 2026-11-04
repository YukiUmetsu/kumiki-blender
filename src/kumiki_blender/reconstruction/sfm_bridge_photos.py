"""Stage additional *real* overlapping photographs for a new SfM revision.

Produces a private, provenance-aware combined photo folder. No original
reconstruction/database is modified and no unobserved image is generated.
Reconstructing the combined dataset uses the existing photo pipeline.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from .sfm_cohort_recovery import _read_database, _source_paths, _verify_staged_images
from .references import MAX_IMAGE_BYTES, MAX_IMAGES, _decode

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_BRIDGE_PHOTOS = MAX_IMAGES


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Bridge source manifest must be an existing regular JSON file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError("Bridge manifest must have an images array")
    entries = {}
    for record in data["images"]:
        if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
            raise ValueError("Invalid bridge manifest image record")
        filename = record["filename"]
        if Path(filename).name != filename or filename in entries:
            raise ValueError("Manifest has an unsafe or duplicate bridge filename")
        for key in ("source_url", "license", "view", "notes"):
            if key in record and (not isinstance(record[key], str) or len(record[key]) > 2048):
                raise ValueError(f"Invalid bridge manifest {key} for {filename}")
        if "source_url" in record and record["source_url"] != "local-unattributed":
            parsed = urlsplit(record["source_url"])
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError(f"Bridge photo URL must be HTTPS: {filename}")
        entries[filename] = record
    return entries


def prepare_bridge_photos(revision: Path, photos: Path, output: Path, *,
                          source_manifest: Path | None = None, model: int = 0) -> dict:
    """Combine verified originals and new local views into a NEW photo folder.

    This deliberately rebuilds photo matching in the next reconstruction rather
    than trying to append potentially inconsistent features to an old database.
    """
    paths = _source_paths(revision, model)
    if photos.is_symlink() or not photos.is_dir():
        raise FileNotFoundError("Additional photo source must be an existing directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a new bridge-photo output directory: {output}")
    if output.resolve().is_relative_to(paths["sparse"]):
        raise ValueError("Bridge-photo output must be outside the original sparse revision")
    original_index = json.loads(paths["index"].read_text(encoding="utf-8"))
    original = original_index.get("images")
    if not isinstance(original, list):
        raise ValueError("Original staged index must contain an images list")
    database_images, _ = _read_database(paths["database"], 15)
    _verify_staged_images(paths, set(database_images))
    original_hashes = {record["sha256"] for record in original}
    known = _manifest(source_manifest)
    new_photos = sorted((path for path in photos.iterdir()
                         if path.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name)
    if not 1 <= len(new_photos) <= MAX_BRIDGE_PHOTOS or len(new_photos) + len(original) > MAX_IMAGES:
        raise ValueError("Provide at least one new photo; combined folder must contain at most 256")
    old_names = {record["filename"] for record in original}
    staged_names = {f"bridge-{i:04d}{photo.suffix.lower()}"
                    for i, photo in enumerate(new_photos, 1)}
    if old_names & staged_names:
        raise ValueError("Bridge staged filename conflicts with an original staged photo")
    if source_manifest is not None and set(known) != {path.name for path in new_photos}:
        raise ValueError("Bridge manifest filenames do not match the supplied photos")
    validated = []
    seen_hashes = set(original_hashes)
    for photo in new_photos:
        if photo.is_symlink() or not photo.is_file() or not 0 < photo.stat().st_size <= MAX_IMAGE_BYTES:
            raise ValueError(f"Unsafe or oversized new photograph: {photo}")
        sha = _digest(photo)
        if sha in seen_hashes:
            raise ValueError(f"Duplicate photo does not add a new viewpoint: {photo.name}")
        seen_hashes.add(sha)
        dimensions, _ = _decode(photo, photo.name)
        width, height = dimensions
        validated.append((photo, sha, width, height))

    output.mkdir(parents=True, exist_ok=False)
    combined = output / "photos"
    combined.mkdir()
    manifest = {"schema_version": 1, "source_revision": str(paths["revision"]),
                "source_database_sha256": _digest(paths["database"]),
                "source_images": len(original), "new_bridge_images": len(validated),
                "distribution_allowed": False, "images": [],
                "note": "Private photo sources only. No generated geometry or invented views. "
                        "Run a NEW reconstruction; previous matches are not reused."}
    for record in original:
        filename = record["filename"]
        original_photo = paths["images"] / filename
        dest = combined / filename
        shutil.copyfile(original_photo, dest)
        if _digest(dest) != record["sha256"]:
            raise RuntimeError(f"Original photo changed during bridge preparation: {filename}")
        manifest["images"].append({"filename": filename, "kind": "original",
                                   "sha256": record["sha256"],
                                   "original_filename": record.get("original_filename", filename),
                                   "source_url": record.get("source_url", "local-unattributed"),
                                   "license": record.get("license", "unknown")})
    for index, (photo, sha, width, height) in enumerate(validated, 1):
        name = f"bridge-{index:04d}{photo.suffix.lower()}"
        dest = combined / name
        if dest.exists():
            raise ValueError(f"Bridge filename conflicts with staged input: {name}")
        shutil.copyfile(photo, dest)
        if _digest(dest) != sha:
            raise RuntimeError(f"Bridge photo changed during staging: {photo.name}")
        metadata = known.get(photo.name, {})
        manifest["images"].append({"filename": name, "kind": "bridge",
                                   "source_filename": photo.name, "sha256": sha,
                                   "dimensions": [width, height],
                                   "source_url": metadata.get("source_url", "local-unattributed"),
                                   "license": metadata.get("license", "unknown"),
                                   "view": metadata.get("view", "unknown"),
                                   "notes": metadata.get("notes", "")})
    (output / "bridge-photo-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status": "bridge-photos-staged / new-matching-required",
            "output": str(output.resolve()), "photos_dir": str(combined.resolve()),
            "source_images": len(original), "new_bridge_images": len(validated),
            "total_images": len(manifest["images"]),
            "manifest": str(output.resolve() / "bridge-photo-manifest.json"),
            "warning": "Only actual overlapping photographs support geometric recovery. "
                       "Review provenance and source rights; do not substitute generated views."}
