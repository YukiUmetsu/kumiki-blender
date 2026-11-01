"""Fetch the exact CC BY 4.0 Sketchfab scan, mirrored by Objaverse.

Only fetches the recorded model ID; all other GLBs are rejected by checksum.
The file is placed in a caller-specified cache, never committed to the repo.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kumiki_blender.liberty_scan import (ScanSourceError, load_scan_source,
                                         verify_scan_source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True,
                        help="New local scan-cache path, e.g. .review/_sources/liberty-bronze.glb")
    args = parser.parse_args()
    destination = args.output.resolve()
    record = load_scan_source()
    if destination.exists() or destination.is_symlink():
        verify_scan_source(destination, record)
        print(f"Already verified: {destination}")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(record["download_url"],
                                         headers={"User-Agent": "kumiki-blender-source-fetch/1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part",
                                             prefix="liberty-scan-", delete=False) as handle:
                temporary = Path(handle.name)
                remaining = record["size_bytes"]
                while True:
                    chunk = response.read(min(1024 * 1024, remaining + 1))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise ScanSourceError("Download exceeds the licensed source size")
                    handle.write(chunk)
        verify_scan_source(temporary, record)
        os.replace(temporary, destination)
        print(f"Verified CC BY 4.0 scan: {destination}\nCredit: {record['attribution']}")
        return 0
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"Source download failed: {error}", file=sys.stderr)
        return 2
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
