#!/usr/bin/env python3
"""End-to-end Blender smoke test for kumiki-character-build on a synthetic figure.

    PYTHONPATH=src:tests python scripts/smoke_character_build_blender.py [--blender PATH]

Checks the camera and axis conventions (Blender renders against the NumPy
rasterizer), QuadriFlow retopology, rigging with heat weights, and assembly.
Needs Blender 4.2 or newer. Exit code 1 means a check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import hashlib  # noqa: E402

from character_build_fixtures import HEIGHT_M, ellipsoid_mesh, glb_bytes, write_views  # noqa: E402
from kumiki_blender.character_build.__main__ import main as cli  # noqa: E402
from kumiki_blender.character_build.handoff import build_handoff, compare  # noqa: E402
from kumiki_blender.character_build.mesh import read_mesh, write_mesh  # noqa: E402


def run(*args: str) -> None:
    if cli(list(args)) != 0:
        raise SystemExit(f"command failed: {' '.join(args)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender")
    path = parser.parse_args().blender
    blender = ["--blender", path] if path else []
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        refs = root / "refs"
        write_views(refs)
        build_handoff(refs, HEIGHT_M)
        handoff = refs / "handoff"

        run("blender-setup", str(handoff), "--output", str(root / "renders"), "--render", "--render-blockout", *blender)
        blender_iou = compare(handoff, root / "renders")["mean_iou"]
        numpy_iou = compare(handoff, handoff / "blockout.obj")["mean_iou"]
        if blender_iou < 0.95 or abs(blender_iou - numpy_iou) > 0.01:
            failures.append(f"camera conventions: Blender {blender_iou} vs NumPy {numpy_iou}")

        # A loaded user .blend must stay byte-identical when references are added and saved.
        run("blender-setup", str(handoff), "--output", str(root / "refs1"), "--save", *blender)
        user_blend = root / "refs1" / "references.blend"
        before = hashlib.sha256(user_blend.read_bytes()).hexdigest()
        run("blender-setup", str(handoff), "--output", str(root / "refs2"), "--blend", str(user_blend), "--save", *blender)
        if hashlib.sha256(user_blend.read_bytes()).hexdigest() != before:
            failures.append("blender-setup --blend --save changed the user's .blend")
        if not (root / "refs2" / "references.blend").is_file():
            failures.append("blender-setup --blend --save wrote no copy")

        # A GLB whose mesh sits under a translated parent node must keep that offset.
        nested = root / "nested.glb"
        nested.write_bytes(glb_bytes(ellipsoid_mesh((0, 0, 0), (.3, .2, .4), 3), translation=(0, 0.5, 0), nested=True))
        run("retopo", str(nested), "--output", str(root / "nested_retopo"), "--target-faces", "1500",
            "--bake-size", "256", *blender)
        low_z = read_mesh(root / "nested_retopo" / "low.obj").vertices[:, 2]
        if abs((low_z.min() + low_z.max()) / 2 - 0.5) > 0.01:
            failures.append(f"nested GLB import lost its parent offset: centre z {(low_z.min() + low_z.max()) / 2:.3f}")

        run("retopo", str(handoff / "blockout.obj"), "--output", str(root / "retopo"), "--target-faces", "4000",
            "--bake-size", "512", *blender)
        retopo = json.loads((root / "retopo" / "report.json").read_text())
        if retopo["low"]["quad_fraction"] < 0.99:
            failures.append(f"retopo quads {retopo['low']['quad_fraction']}")

        run("joints", str(handoff), str(handoff / "blockout.obj"), "--output", str(root / "joints.json"))
        run("rig", str(root / "retopo" / "low.obj"), str(root / "joints.json"), "--output", str(root / "rig"), *blender)
        rig = json.loads((root / "rig" / "report.json").read_text())
        if rig["nearest_bone_fraction"] > 0.01 or rig["bones_without_weights"]:
            failures.append(f"rig weights: {rig['nearest_bone_fraction']}, {rig['bones_without_weights']}")

        parts = root / "parts"
        parts.mkdir()
        write_mesh(ellipsoid_mesh((0, 0, 0), (.1, .1, .06), 2), parts / "hat.obj")
        (parts / "body.obj").write_bytes((handoff / "blockout.obj").read_bytes())
        corners = ([0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1])
        plan = {"schema_version": 1, "body": "body.obj", "join_voxel_m": 0.006,
                "parts": [{"id": "hat", "file": "hat.obj", "join": "union",
                           "anchors": [{"part": p, "target": [p[0], p[1], p[2] + 1.0]} for p in corners]}]}
        (parts / "plan.json").write_text(json.dumps(plan))
        run("assemble", str(parts / "plan.json"), "--output", str(root / "assembled"), *blender)
        assembled = json.loads((root / "assembled" / "report.json").read_text())
        if assembled["union_parts"] != ["Part_hat"]:
            failures.append(f"assembly union {assembled['union_parts']}")
    print("\n".join(failures) or f"SMOKE PASS (Blender {blender_iou}, NumPy {numpy_iou})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
