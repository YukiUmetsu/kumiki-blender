"""Emit compact, chunked QA thumbnails for remote visual review.

Run only on self-generated CI PNGs, after artifact integrity checks. JPEG data
is grouped into tagged log chunks so a human or connected review agent can
reassemble a rendered image without relying on flaky artifact mirrors.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: emit_qa_jpegs.py <image.png> [more image.png...]", file=sys.stderr)
        return 2
    from PIL import Image

    for path_str in argv:
        path = Path(path_str)
        if path.suffix.lower() != ".png" or not path.is_file():
            raise ValueError(f"QA image must exist and be a PNG: {path}")
        if path.stat().st_size > 10_000_000:
            raise ValueError(f"QA image is unexpectedly large: {path}")
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            # Large enough to inspect paw anatomy and erosion facets while
            # staying small enough for GitHub's job-log transport.
            source.thumbnail((720, 450), Image.Resampling.LANCZOS)
            frame = source.convert("RGB")
            output = io.BytesIO()
            frame.save(output, format="JPEG", quality=79, optimize=True)
        payload = base64.b64encode(output.getvalue()).decode("ascii")
        # Include the artifact directory: otherwise procedural and imported
        # Sphinx previews both appear as "previews-sphinx-face" in CI logs.
        label = (path.parent.parent.name + "-" + path.stem
                 if path.parent.name == "previews"
                 else path.parent.name + "-" + path.stem)
        print(f"KUMIKI_QA_BEGIN {label}")
        for offset in range(0, len(payload), 2048):
            print("KUMIKI_QA_DATA " + payload[offset:offset+2048])
        print(f"KUMIKI_QA_END {label} bytes={len(output.getvalue())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
