"""Assert that the biplane GLB is not merely a static mesh masquerading as rigged."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from kumiki_blender.glb_audit import audit_glb
from kumiki_blender.revision_audit import audit_revision


def verify_revision(root: Path) -> dict[str, object]:
    root = root.resolve()
    audit_revision(root)
    report = audit_glb(root / "model.glb", require_pbr_textures=True)
    with (root / "model.glb").open("rb") as stream:
        stream.seek(12)
        length, kind = struct.unpack("<I4s", stream.read(8))
        if kind != b"JSON":
            raise RuntimeError("GLB is missing its JSON scene graph")
        graph = json.loads(stream.read(length))

    expected = {
        "propeller", "upper-left-aileron", "upper-right-aileron",
        "lower-left-aileron", "lower-right-aileron", "elevator",
        "rudder", "left-wheel", "right-wheel",
    }
    nodes = graph["nodes"]
    rig_node_ids = {
        index: node["extras"]["kumiki_rig"]
        for index, node in enumerate(nodes)
        if isinstance(node.get("extras"), dict) and
        isinstance(node["extras"].get("kumiki_rig"), str)
    }
    tracks = graph.get("animations", [])
    animated = {
        rig_node_ids[target]
        for track in tracks for channel in track.get("channels", [])
        if channel.get("target", {}).get("path") == "rotation"
        for target in (channel["target"]["node"],)
        if target in rig_node_ids
    }
    if not expected <= animated:
        raise RuntimeError(f"GLB lacks animatable biplane controls: {sorted(expected-animated)}")
    if not expected <= set(rig_node_ids.values()):
        raise RuntimeError("GLB omits one or more source rig control empties")
    if report["semantic_nodes"] < 190:
        raise RuntimeError("Biplane GLB lost too many named component meshes")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["rig_controls"]) != expected:
        raise RuntimeError("Manifest control list disagrees with the exported rig")
    required = {"front", "side", "top", "rear", "cockpit"}
    if any(not (root / "previews" / f"{name}.png").is_file() for name in required):
        raise RuntimeError("One or more aircraft orthographic inspection shots are missing")
    return {"animated_controls": sorted(animated), "animation_clips": len(tracks),
            "named_meshes": report["semantic_nodes"], "views": sorted(required)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: PYTHONPATH=src python scripts/verify_biplane_export.py <revision-root>")
    print(json.dumps(verify_revision(Path(sys.argv[1])), indent=2))
