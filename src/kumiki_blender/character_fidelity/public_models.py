"""Explicit public checkpoint acquisition; no Hugging Face API or token.

Only models on the hard-coded publisher URL allowlist are supported. Downloads
are never automatic during image comparison. Run this CLI deliberately.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "sam2.1-tiny": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_tiny.pt"),
    "sam2.1-small": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sam2.1_hiera_small.pt"),
    "sam2.1-base-plus": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_base_plus.pt"),
}
MAX_BYTES = 1024 * 1024 * 1024


def download_public_checkpoint(name: str, destination: str | Path, *, opener=None) -> Path:
    """Stream official, ungated Meta weights to a new file, atomically.

    URL is never user-supplied. Does not overwrite, send credentials, follow
    arbitrary model registry identifiers or contact Hugging Face.
    """
    if name not in CHECKPOINTS:
        raise ValueError(f"Unsupported public checkpoint: {name}")
    url, filename = CHECKPOINTS[name]
    folder = Path(destination)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Checkpoint output must be an existing, non-symlink directory")
    target = folder / filename
    if target.exists() or target.is_symlink():
        raise ValueError(f"Checkpoint already exists: {target}")
    downloader = opener if opener is not None else urllib.request.urlopen
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=folder)
    size = 0
    try:
        with os.fdopen(fd, "wb") as stream, downloader(url, timeout=60) as response:
            final_url = response.geturl() if hasattr(response, "geturl") else url
            if not final_url.startswith("https://dl.fbaipublicfiles.com/segment_anything_2/"):
                raise ValueError("Checkpoint download redirected outside official Meta host")
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("Server returned an HTML document instead of checkpoint")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("Checkpoint exceeds maximum size")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if size < 1024 * 1024:
            raise ValueError("Checkpoint is unexpectedly small or incomplete")
        os.link(temporary, target)
        return target
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Get public, ungated SAM2.1 weights (no Hugging Face)")
    parser.add_argument("model", choices=tuple(CHECKPOINTS))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(download_public_checkpoint(args.model, args.directory))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
