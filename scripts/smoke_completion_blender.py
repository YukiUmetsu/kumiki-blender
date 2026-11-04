"""CI smoke: genuine glTF imports, candidate alignment, and Blender/GLB export.

Run with Blender in factory-startup mode, passing a fresh output directory after
`--`. Test objects are generic cubes, never substitutes for real statue QA.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from kumiki_blender.cli import main  # noqa: E402
from kumiki_blender.glb_audit import audit_glb  # noqa: E402


def source_glb(destination: Path, name: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    mesh = bpy.context.object
    mesh.name = name
    bpy.ops.export_scene.gltf(
        filepath=str(destination), export_format="GLB", use_selection=True,
        export_extras=True, export_animations=False, export_cameras=False,
        export_lights=False,
    )
    bpy.data.objects.remove(mesh, do_unlink=True)


def run(destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"Completion smoke destination exists: {destination}")
    destination.mkdir(parents=True)
    base = destination / "base.glb"
    part = destination / "part.glb"
    source_glb(base, "Measured test cube")
    source_glb(part, "Inferred test part")
    anchors = ((-1, -1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    manifest = destination / "completion.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "asset_id": "completion-ci-cubes",
        "mode": "cinematic",
        "base_origin": "observed",
        "base_glb": "base.glb",
        "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "regions": [{
            "id": "inferred-part", "candidate_glb": "part.glb",
            "candidate_sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
            "origin": "generated",
            "generator": {"adapter": "supplied-glb", "model_id": "unit-test"},
            "max_error_m": 0.0001,
            "anchors": [
                {"id": f"anchor-{index}", "source_xyz_m": list(point),
                 "target_xyz_m": [point[0] + 1.5, point[1], point[2]]}
                for index, point in enumerate(anchors)
            ],
        }],
    }, indent=2), encoding="utf-8")
    if main(["plan-completion", "--manifest", str(manifest)]) != 0:
        raise RuntimeError("Completion planning failed")
    revision = destination / "revision"
    if main(["build-completion", "--manifest", str(manifest),
             "--output", str(revision)]) != 0:
        raise RuntimeError("Completion Blender build failed")
    for filename in ("source.blend", "model.glb", "preview.png", "manifest.json"):
        if not (revision / filename).is_file():
            raise RuntimeError(f"Missing Blender completion output: {filename}")
    audit_glb(revision / "model.glb")
    result = json.loads((revision / "manifest.json").read_text(encoding="utf-8"))
    if (result["mode"] != "cinematic" or result["regions"][0]["included"] is not True
            or result["regions"][0]["max_error_m"] > 0.0001):
        raise RuntimeError("Completion provenance or alignment report changed")
    with (revision / "model.glb").open("rb") as stream:
        stream.seek(12)
        import struct
        length, kind = struct.unpack("<I4s", stream.read(8))
        if kind != b"JSON":
            raise RuntimeError("Exported GLB has no JSON chunk")
        data = json.loads(stream.read(length))
    origins = {node.get("extras", {}).get("kumiki_geometry_origin")
               for node in data["nodes"] if isinstance(node, dict)
               and isinstance(node.get("extras"), dict)}
    if not {"observed", "inferred"} <= origins:
        raise RuntimeError(f"Export lost base/candidate evidence labels: {origins}")
    print("Completion Blender smoke: observed and inferred meshes exported")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("Pass one new output directory after --")
    run(Path(args[0]).resolve())
