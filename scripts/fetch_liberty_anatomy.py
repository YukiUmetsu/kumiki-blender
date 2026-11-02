"""Fetch Blender Studio's specific verified CC0 realistic-head sculpt base.

The 49 MB original CC0 asset bundle is cached locally, never tracked in git.
The entire archive SHA-256 AND extracted .blend SHA-256 must match the record.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kumiki_blender.liberty_anatomy_source import (AnatomySourceError,
                                                   load_anatomy_source,
                                                   verify_anatomy_source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True,
                        help="Local path of the extracted .blend donor (not committed)")
    args = parser.parse_args()
    target = args.output.resolve()
    record = load_anatomy_source()
    if target.exists() or target.is_symlink():
        verify_anatomy_source(target, record)
        print(f"Previously verified CC0 donor: {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix="head-source-") as temp:
        folder = Path(temp)
        archive = folder / "bundle.zip"
        try:
            req = urllib.request.Request(record["download_url"], headers={
                "User-Agent": "kumiki-blender-cc0-donor/1"})
            with urllib.request.urlopen(req, timeout=180) as response, archive.open("wb") as out:
                digest = hashlib.sha256()
                byte_count = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(chunk)
                    byte_count += len(chunk)
                    if byte_count > record["archive_size_bytes"]:
                        raise AnatomySourceError("Asset bundle exceeded reviewed size")
                    out.write(chunk)
            if (byte_count != record["archive_size_bytes"] or
                    digest.hexdigest() != record["archive_sha256"]):
                raise AnatomySourceError("Asset bundle SHA-256/size mismatch")
            extracted = folder / "reviewed-head.blend"
            # Extract only one explicitly named, reviewed member. No zip slip.
            with zipfile.ZipFile(archive) as zipped, \
                    zipped.open(record["member"]) as source, extracted.open("wb") as out:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    out.write(chunk)
            verify_anatomy_source(extracted, record)
            os.replace(extracted, target)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            print(f"Verified donor download failed: {error}", file=sys.stderr)
            return 2
    print(f"Verified CC0 Blender Studio anatomical head: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
