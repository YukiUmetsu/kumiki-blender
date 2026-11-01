"""Generate a local MIT-licensed texture-pack fixture for Blender CI.

This is generated from our own procedural PBR code, never downloaded from an
external asset site. It tests real scanned-file ingestion and GLB embedding.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumiki_blender.procedural_pbr import make_pbr_maps  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/generate_first_party_texture_pack.py <new-directory>")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    maps = make_pbr_maps("natural-stone", "#AA9986", 0.82, resolution=128)
    outputs = {
        "base_color": maps.base_color_png,
        "roughness": maps.roughness_png,
        "normal": maps.normal_png,
    }
    hashes = {}
    for channel, payload in outputs.items():
        (root / f"{channel}.png").write_bytes(payload)
        hashes[channel] = hashlib.sha256(payload).hexdigest()
    recipe = {
        "schema_version": 1,
        "packs": [{
            "id": "first-party-limestone", "creator": "kumiki-blender contributors",
            "source_url": "https://github.com/YukiUmetsu/kumiki-blender",
            "license": "MIT", "attribution": "Generated from kumiki-blender procedural_pbr.py",
            "redistribution_allowed": True, "tile_size_m": 1.3,
            "maps": {channel: f"{channel}.png" for channel in outputs},
            "sha256": hashes,
        }],
        "assignments": [{"target": "Architectural stone.004", "pack": "first-party-limestone"}],
    }
    (root / "imported-portico.json").write_text(json.dumps(recipe, indent=2) + "\n")
    recipe["assignments"] = [
        {"target": "wall-stone", "pack": "first-party-limestone"},
        {"target": "wall-stone-light", "pack": "first-party-limestone"},
    ]
    (root / "facade.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"Created first-party texture pack: {root}")


if __name__ == "__main__":
    main()
