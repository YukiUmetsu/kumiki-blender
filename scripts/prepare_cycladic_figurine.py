"""Index a CC BY 4.0 multi-view sculpture dataset for private photo QA.

The official University of Wyoming Dataverse contains 72 photos of a replica
Cycladic figurine. Download/unpack it outside Git before running this script.
It creates only source-attributed SHA-pinned manifests, never copied images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_PAGE = "https://dataverse.arcc.uwyo.edu/dataset.xhtml?persistentId=doi:10.15786/16543554"
DOI = "10.15786/16543554"
PHOTO_COUNT = 72
# Published by Wyoming Data Repository for cycladic_figurine_photo_dataset.zip.
PUBLISHED_ARCHIVE_MD5 = "8344c56d98a8e80f9ab5c063c4195b42"
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root: Path, *, archive: Path | None = None) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected an unpacked, non-symlinked Cycladic dataset directory")
    root = root.resolve()
    images = sorted(path for path in root.rglob("*")
                    if path.suffix.lower() in PHOTO_EXTENSIONS)
    if (len(images) != PHOTO_COUNT or any(
            not path.is_file() or path.is_symlink()
            or any((root / parent).is_symlink() for parent in path.relative_to(root).parents
                   if parent != Path("."))
            for path in images)):
        raise ValueError(f"Expected exactly {PHOTO_COUNT} original Cycladic JPEG/PNG/WebP photos")
    archive_sha = None
    if archive is not None:
        if archive.is_symlink() or not archive.is_file():
            raise ValueError("Source archive is missing or a symlink")
        if _hash(archive, "md5") != PUBLISHED_ARCHIVE_MD5:
            raise ValueError("Source ZIP does not match the publisher's recorded MD5")
        archive_sha = _hash(archive, "sha256")
    paths = (root / "reference-set.json", root / "reference-source.json")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("Refusing to overwrite an existing photo dossier or source record")
    entries = []
    for index, path in enumerate(images, 1):
        entries.append({
            "id": f"cycladic-view-{index:03d}",
            "file": path.relative_to(root).as_posix(),
            "source_url": SOURCE_PAGE,
            "view": "unknown", "regions": ["sculpture"],
            "expected_sha256": _hash(path, "sha256"),
        })
    manifest = {
        "schema_version": 1, "asset_id": "wyoming-cycladic-figurine-cc-by-4",
        "subject": "Replica Greek Cycladic figurine, University of Wyoming dataset",
        "required_views": ["unknown"], "required_regions": ["sculpture"],
        "images": entries,
    }
    provenance = {
        "provider": "University of Wyoming Dataverse",
        "source_page": SOURCE_PAGE, "source_archive": SOURCE_PAGE,
        "source_doi": DOI, "published_archive_md5": PUBLISHED_ARCHIVE_MD5,
        "archive_sha256": archive_sha, "photo_count": PHOTO_COUNT,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": ("Hutchens, Chad Everett; Koenig, Charles (2021), "
                        "Greek Cycladic Figurine photogrammetry dataset, "
                        "Wyoming Data Repository, doi:10.15786/16543554, CC BY 4.0"),
        "license_note": "Reuse including commercial use requires creator attribution, "
                        "a license link and identification of changes.",
        "distribution_allowed": False,
        "scope": "Private reconstruction benchmark until derived photos/models/renders "
                 "are independently reviewed and attribution is fulfilled.",
    }
    for path, value in zip(paths, (manifest, provenance)):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(paths[0]), "provenance": str(paths[1]),
            "images": len(images), "distribution_allowed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unpacked", type=Path, required=True)
    parser.add_argument("--archive", type=Path,
                        help="Optional original ZIP, validated against the publisher's MD5")
    args = parser.parse_args()
    print(json.dumps(prepare(args.unpacked, archive=args.archive), indent=2))
