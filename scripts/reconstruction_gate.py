"""Hard release gate for a photo-faithful registered multi-view reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumiki_blender.reconstruction_quality import review_reconstruction  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = review_reconstruction(args.review)
    except (OSError, ValueError, TypeError, ImportError) as error:
        result = {"status": "HOLD", "reasons": [str(error)]}
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output+"\n", encoding="utf-8")
    print(output)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
