"""Index EPFL's research-only multi-view monument photos outside the repo.

Supported scenes are a large outdoor Freddie Mercury figure and a sculpted
facial mask. EPFL permits free research use, but no commercial use of data,
derived models or images without written permission. Do not publish outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_PAGE = "https://lgg.epfl.ch/statues.php"
SETS = {
    "freddie-mercury": {
        "folder": "Photos", "extension": ".JPG", "count": 37,
        "subject": "Freddie Mercury outdoor standing statue, Montreux",
        "url": "https://lgg.epfl.ch/old/statues_data/dataset2013/mercury_pics.zip",
        "archive_sha256": "4268e25a7fef21ca94479e2aabca2c2e9694f0750f8ca2f94dab654907037e98",
    },
    "sculpted-mask": {
        "folder": "africanhead_photos", "extension": ".jpg", "count": 32,
        "subject": "EPFL outdoor monumental sculpted face (Mask dataset)",
        "url": "https://lgg.epfl.ch/old/statues_data/dataset2013/head_pics.zip",
        "archive_sha256": "0549f43547f2f5e47b6dea1dcb2082d73c280964adcf97a4b693f17baab9c7f0",
    },
}


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root: Path, scene: str, *, archive: Path | None = None) -> dict:
    if scene not in SETS:
        raise ValueError("Unsupported EPFL photo scene")
    config = SETS[scene]
    root = root.resolve()
    folder = root / config["folder"]
    images = sorted(folder.glob("*" + config["extension"]))
    if (len(images) != config["count"] or
            any(not path.is_file() or path.is_symlink() for path in images)):
        raise ValueError(f"Expected {config['count']} original EPFL {scene} photos")
    if archive is not None and _hash(archive) != config["archive_sha256"]:
        raise ValueError("EPFL archive SHA-256 does not match the verified source")
    manifest_path = root / "reference-set.json"
    rights_path = root / "reference-source.json"
    if any(path.exists() or path.is_symlink() for path in (manifest_path, rights_path)):
        raise ValueError("Refusing to overwrite existing evidence")
    entries = []
    for index, path in enumerate(images, 1):
        entries.append({"id": f"epfl-view-{index:03d}",
                        "file": f"{config['folder']}/{path.name}",
                        "source_url": config["url"],
                        "view": "unknown", "regions": ["sculpture"],
                        "expected_sha256": _hash(path)})
    manifest = {"schema_version": 1, "asset_id": f"epfl-{scene}-research",
                "subject": config["subject"], "required_views": ["unknown"],
                "required_regions": ["sculpture"], "images": entries}
    provenance = {"provider": "EPFL Geometric Computing Laboratory",
                  "source_page": SOURCE_PAGE, "source_archive": config["url"],
                  "archive_sha256": config["archive_sha256"],
                  "photo_count": len(images),
                  "license_note": "Free research use with credit to EPFL; no commercial use of images, "
                                  "derived geometry or renderings without written permission.",
                  "distribution_allowed": False,
                  "scope": "Private research/technical testing only. Nothing derived from this "
                           "scene may be distributed as a public model or preview."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rights_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "provenance": str(rights_path),
            "images": len(images), "distribution_allowed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unpacked", type=Path, required=True)
    parser.add_argument("--scene", choices=tuple(SETS), required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.unpacked, args.scene, archive=args.archive), indent=2))
