"""Index Pix4D's real eagle-statue training photos without republishing them.

Get the official example_eagle.zip from the Pix4D documentation, unzip it
outside the git repository, then run this script against the extracted folder.
All output stays beside the source photos, and the dataset remains subject to
Pix4D's training-use / attribution terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_URL = "https://data.pix4d.com/misc/example_datasets/example_eagle.zip"
SOURCE_PAGE = "https://support.pix4d.com/hc/en-us/articles/360000235126"
KNOWN_ARCHIVE_SHA256 = "69e70f18981ebf2d023e9723f0cf6da53bca4a8fa665a2d425f15255d7fe45ef"


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root: Path, *, archive: Path | None = None) -> dict:
    root = root.resolve()
    images = sorted((root / "images").glob("IMG_*.JPG"))
    if len(images) != 44 or any(not path.is_file() or path.is_symlink() for path in images):
        raise ValueError("Expected the 44 original Pix4D eagle-statue JPG photographs")
    if not (root / "README.txt").is_file():
        raise ValueError("Expected Pix4D example archive README.txt")
    if archive is not None and _hash(archive) != KNOWN_ARCHIVE_SHA256:
        raise ValueError("Downloaded Pix4D archive SHA-256 differs from the verified example")
    paths = (root / "reference-set.json", root / "reference-source.json")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("Refusing to overwrite an existing photo dossier or source record")
    manifest = {
        "schema_version": 1,
        "asset_id": "pix4d-eagle-statue-training",
        "subject": "Pix4D example eagle sculpture, Switzerland (training dataset)",
        # These are the same documented object; actual camera angles and
        # region coverage must be inferred from camera poses, not filenames.
        "required_views": ["unknown"],
        "required_regions": ["sculpture"],
        "images": [{"id": path.stem.lower().replace("_", "-"),
                    "file": "images/" + path.name,
                    "source_url": DATASET_URL,
                    "view": "unknown", "regions": ["sculpture"],
                    "expected_sha256": _hash(path)} for path in images],
    }
    provenance = {
        "provider": "Pix4D", "source_page": SOURCE_PAGE,
        "source_archive": DATASET_URL,
        "archive_sha256": KNOWN_ARCHIVE_SHA256,
        "photo_count": 44,
        "license_note": (
            "Pix4D allows free personal or professional training. Commercial or promotional "
            "use requires linked 'Courtesy of Pix4D / pix4d.com' attribution; "
            "contact Pix4D marketing for further usage. This is not a CC0/CC BY dataset."),
        "distribution_allowed": False,
        "scope": "Local technical reconstruction and visual QA only; no source photos, derived "
                 "textures, mesh, or previews are included in the public repository.",
    }
    for path, value in zip(paths, (manifest, provenance)):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(paths[0]), "provenance": str(paths[1]),
            "images": len(images), "distribution_allowed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unpacked", required=True, type=Path,
                        help="Directory containing the original Pix4D images/ and README.txt")
    parser.add_argument("--archive", type=Path,
                        help="Optional original ZIP, checked against a recorded SHA-256")
    args = parser.parse_args()
    print(json.dumps(prepare(args.unpacked, archive=args.archive), indent=2))
